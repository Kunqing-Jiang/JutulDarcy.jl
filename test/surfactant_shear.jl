module SurfactantShearTests
using Test, Jutul, JutulDarcy, LinearAlgebra
using Jutul.StaticArrays: SVector
const TR = JutulDarcy.Tracers
const JD = JutulDarcy
const FD = Jutul.ForwardDiff
struct TestPermeabilityMultiplier <: Jutul.ScalarVariable end
config(e=1.6) = FoamShearThinning(fmcap=1e-6, epcap=e, sigma=.03)
function model_before_install(; general_ad=false, extra=nothing, dims=(3,3,1), lengths=Float64.(dims), mesh=UnstructuredMesh(CartesianMesh(dims, lengths)), sys=nothing)
    d = reservoir_domain(mesh; permeability=1e-12, porosity=.2)
    sys === nothing && (sys = ImmiscibleSystem((AqueousPhase(), VaporPhase()); reference_densities=[1000.,500.]))
    m = setup_reservoir_model(d, sys; general_ad, block_backend=false)
    ts = extra === nothing ? (SurfactantTracer(sys;k_d=1.),) : (SurfactantTracer(sys;k_d=1.), extra)
    add_tracers_to_model!(m, ts)
    return reservoir_model(m)
end
function installed(; shear=config(), kw...)
    r = model_before_install(; kw...)
    set_surfactant_model!(r; shear)
    return r
end
function state_for(r)
    nc = number_of_cells(r.domain)
    p = setup_parameters(r)
    s = setup_state(r; Pressure=1e7 .+ 1e5.*collect(1:nc), Saturations=[.5,.5], TracerConcentrations=.005)
    return p, s
end
function throws_with(f, reason::AbstractString)
    err = try f(); nothing catch e; e end
    return err isa ArgumentError && occursin(reason, sprint(showerror, err))
end
function model_with(r; system=r.system)
    return SimulationModel(r.domain,system,r.context,r.formulation,r.data_domain,
        deepcopy(r.primary_variables),deepcopy(r.secondary_variables),deepcopy(r.parameters),
        deepcopy(r.equations),copy(r.output_variables),deepcopy(r.extra),r.optimization_level)
end
function rejected_without_change(r, reason; shear=config())
    before = deepcopy((r.primary_variables,r.secondary_variables,r.parameters,r.equations,r.output_variables))
    @test_throws ArgumentError set_surfactant_model!(r; shear)
    after = (r.primary_variables,r.secondary_variables,r.parameters,r.equations,r.output_variables)
    @test repr(before) == repr(after)
    # Check the diagnostic separately to distinguish the intended guard.
    err = try set_surfactant_model!(r; shear); nothing catch e; e end
    @test err isa ArgumentError && occursin(reason, sprint(showerror,err))
end

@testset "Shear configuration and installation" begin
    @test occursin("pressure-gradient", string(@doc JutulDarcy.FoamShearThinning))
    @test_throws UndefKeywordError FoamShearThinning(fmcap=1e-6, epcap=1.6)
    for (b,e,s) in ((0.,1.,.03),(-1.,1.,.03),(1.,-1.,.03),(1.,1.,0.),(Inf,1.,.03),(1.,NaN,.03),(1.,1.,Inf))
        @test_throws ArgumentError FoamShearThinning(fmcap=b,epcap=e,sigma=s)
    end
    rejected_without_change(model_before_install(), "shear must"; shear=1)
    for shear in (nothing,config(0.))
        r = installed(;shear,extra=MultiPhaseTracer((1,2)))
        @test !haskey(r.secondary_variables,:FoamShearGradient)
    end
    @test haskey(installed(extra=SinglePhaseTracer(1)).secondary_variables,:FoamShearGradient)
    rejected_without_change(model_before_install(general_ad=true), "TwoPointPotentialFlowHardCoded")
    # Black oil assembles component fluxes without the shear hook.
    bo = StandardBlackOilSystem(phases=(LiquidPhase(),VaporPhase()),reference_densities=[800.,100.])
    rejected_without_change(model_with(model_before_install(); system=bo), "immiscible or CO2-brine")
    r = model_before_install(); r.parameters[:PermeabilityMultiplier] = TestPermeabilityMultiplier()
    rejected_without_change(r,"PermeabilityMultiplier")
    r = model_before_install(); r.data_domain[:permeability] = ones(6,9)
    rejected_without_change(r,"diagonal")
    # Models derived after installation are rejected when storage is allocated.
    r = installed(); v = r.secondary_variables[:FoamShearGradient]
    sub = Jutul.submodel(r, [1,2,3,4])
    @test throws_with(() -> setup_state(sub; Pressure=1e7, Saturations=[.5,.5], TracerConcentrations=.005), "partitioned")
    for pressure in (true,false)
        seq = JD.Sequential.convert_to_sequential(r;pressure)
        @test throws_with(() -> Jutul.values_per_entity(seq, v), "FullyImplicitFormulation")
        # The assembly entry also checks, independent of storage allocation.
        @test throws_with(() -> Jutul.update_equations!((state=Dict(),),seq,1.), "FullyImplicitFormulation")
    end
end

@testset "Single-layer coordinates and deficient stencils" begin
    r = model_before_install(lengths=(1.,1.,1.))
    @test_logs min_level=Base.CoreLogging.Warn set_surfactant_model!(r; shear=config())
    v = r.secondary_variables[:FoamShearGradient]
    @test TR.least_squares_gradient_weights(r.data_domain).active == [1,2]
    @test count(v.bad) == 0
    for height in (1e-6, 1e-9, 1e-13)
        r = model_before_install(dims=(3,3,2), lengths=(3.,3.,height))
        g = try TR.least_squares_gradient_weights(r.data_domain) catch err; err end
        @test g isa NamedTuple
        g isa NamedTuple || continue
        @test g.active == [1,2,3]
        @test all(all(all(isfinite,w) for (_,w) in ws) for ws in g.weights)
        @test height >= 1e-9 ? !any(g.bad) : all(g.bad)
    end
    # Apply a controlled span to the otherwise constant coordinate. The
    # threshold is relative to the domain, independent of its length unit.
    for scale in (1e-3, 1., 1e3), ratio in (0., 0.5e-8, 1e-8, 2e-8)
        r = model_before_install()
        cc = r.data_domain[:cell_centroids]
        cc .*= scale
        cc[3,:] .= range(0., 2scale*ratio; length=size(cc,2))
        g = TR.least_squares_gradient_weights(r.data_domain)
        @test g.active == (ratio <= 1e-8 ? [1,2] : [1,2,3])
    end
    mesh = extract_submesh(UnstructuredMesh(CartesianMesh((2,2,1))), [1,2,3])
    r = model_before_install(; mesh)
    @test_logs (:warn, r"face-normal fallback") set_surfactant_model!(r; shear=config())
    @test count(r.secondary_variables[:FoamShearGradient].bad) == 2
    for cells in ([1,2,3,4,5,6,7,9,10,11], [1,2,3,4,5,6,7,9,10])
        mesh = extract_submesh(UnstructuredMesh(CartesianMesh((4,3,1), (4.,3.,1.))), cells)
        r = model_before_install(; mesh)
        if length(cells) == 10
            @test_logs min_level=Base.CoreLogging.Warn set_surfactant_model!(r; shear=config())
        else
            @test_logs (:warn, r"face-normal fallback") set_surfactant_model!(r; shear=config())
        end
        @test count(r.secondary_variables[:FoamShearGradient].bad) == 1
    end
    r = model_before_install()
    delete!(r.data_domain.data, :boundary_neighbors)
    delete!(r.data_domain.data, :boundary_centroids)
    @test_logs min_level=Base.CoreLogging.Warn set_surfactant_model!(r; shear=config())
    @test !any(r.secondary_variables[:FoamShearGradient].bad)
end

@testset "Potential values are live primal views" begin
    pressure = [FD.Dual(1.0,2.0),FD.Dual(3.0,4.0)]
    transmissibility = [2.0]
    state = (Pressure=pressure,Transmissibilities=transmissibility)
    pv = TR.primal_potential_values(state)
    @test eltype(pv.Pressure) == Float64
    @test pv.Pressure == [1.,3.]
    @test pv.Transmissibilities === transmissibility
    @test FD.partials(pressure[1])[1] == 2.0 && FD.partials(pressure[2])[1] == 4.0
    pressure[1] = FD.Dual(5.,6.)
    @test pv.Pressure[1] === 5.0
end

@testset "Adjoint entry rejects enabled shear before assembly" begin
    for shear in (nothing, config(0.))
        r = installed(; shear); p, s = state_for(r)
        a = Jutul.setup_adjoint_storage(r; state0=s, parameters=p,
            targets=[:Transmissibilities], linear_solver=nothing, use_sparsity=false)
        @test haskey(a, :forward) && haskey(a, :parameter)
    end
    r = installed(); p, s = state_for(r)
    # state_for includes an explicit cache: rejection must not rely on defaults.
    err, frames = try
        Jutul.setup_adjoint_storage(r; state0=s, parameters=p,
            targets=[:Transmissibilities], linear_solver=nothing, use_sparsity=false)
        (nothing, [])
    catch e
        (e, stacktrace(catch_backtrace()))
    end
    @test err isa ArgumentError && occursin("adjoint or sensitivity", sprint(showerror, err))
    @test any(f -> f.func == :values_per_entity, frames)
    @test !any(f -> f.func == :update_equations!, frames)
    v = r.secondary_variables[:FoamShearGradient]
    for context in (Jutul.DefaultContext(), Jutul.ParallelCSRContext())
        adj = Jutul.adjoint_model_copy(r; context)
        @test throws_with(() -> Jutul.values_per_entity(adj, v), "adjoint or sensitivity")
    end
    # Sensitivity storage converts a complete state without allocating the
    # cache, so the rejection must come from the assembly entry.
    adj = Jutul.adjoint_model_copy(r)
    @test throws_with("adjoint or sensitivity") do
        sim = Simulator(adj; state0=deepcopy(s), parameters=deepcopy(p), mode=:sensitivities)
        Jutul.update_state_dependents!(sim.storage, adj, 1.0, setup_forces(adj))
    end
end

# Exercise actual mass, volume, surfactant and compositional entry points.
# The independent oracle is the ratio of payload and mass-flux derivatives.
function face_state(r, Q; mobility=one(Q), fm=.1)
    v = r.secondary_variables[:FoamShearGradient]
    nc = size(v.kd,2); nf = length(v.areas)
    l,rgt = v.neighbors[:,1]
    pressure = fill(zero(Q)+1e7,nc); pressure[rgt] = 1e7-Q
    rho = repeat([1000.,500.],1,nc)
    mob = fill(zero(Q)+zero(mobility)+1.,2,nc); mob[2,rgt]=mobility
    P = fill(.002,3,nc); P[2,rgt]=.008
    X = repeat([.4,.6],1,nc); Y = repeat([.9,.1],1,nc); Y[:,rgt]=[.7,.3]
    return (;Pressure=pressure,PhaseMassDensities=rho,PhaseMobilities=mob,
        Saturations=fill(.5,2,nc),Transmissibilities=ones(nf),TwoPointGravityDifference=zeros(nf),
        FmFactor=fill(fm,nc),FoamShearGradient=zeros(3,nc),SurfactantPartition=P,
        TracerConcentrations=fill(.005,1,nc),LiquidMassFractions=X,VaporMassFractions=Y)
end
@testset "Local upwind gas loads and disabled Jacobian" begin
    r = installed(); v=r.secondary_variables[:FoamShearGradient]
    l,rr = v.neighbors[:,1]; tracer = r.equations[:tracers].flux_type.tracers[1]
    for half in (1,-1), seed in (1.,-1.)
        kg = half==1 ? TPFA(l,rr,1) : TPFA(rr,l,-1)
        up = SPU(kg.left,kg.right)
        Q = FD.Dual(0.,seed)
        st=face_state(r,Q)
        st.FmFactor[l]=.1; st.FmFactor[rr]=.3
        normal=v.normals[:,1]
        st.FoamShearGradient .= 1e6 .* [-normal[2],normal[1],0.]
        off=installed(shear=nothing)
        qoff=JD.darcy_phase_mass_flux(1,2,st,off,nothing,kg,up,half*Q)
        expected_c=upwind(up,cell->st.SurfactantPartition[2,cell],qoff)
        expected_y=upwind(up,cell->st.VaporMassFractions[:,cell],qoff)
        qm=JD.darcy_phase_mass_flux(1,2,st,r,nothing,kg,up,half*Q)
        qv=JD.darcy_phase_volume_flux(1,2,st,r,nothing,kg,up,half*Q)
        @test FD.value(qm)==0 && FD.value(qv)==0
        @test FD.partials(qm)[1] != 0
        @test qm == 500qv
        fm=upwind(up,cell->st.FmFactor[cell],half*Q)
        velocity=(v.kd[:,l].*st.FoamShearGradient[:,l]+v.kd[:,rr].*st.FoamShearGradient[:,rr])/2
        tangent=velocity-normal*dot(velocity,normal)
        n=norm(tangent)/v.config.sigma
        f=(v.config.fmcap/n)^v.config.epcap
        m=1/(fm+(1-fm)*f)
        slope=v.config.epcap*(1-fm)*f/(fm+(1-fm)*f)
        expected_dq=half*seed*m*(1+slope/2)
        @test abs(FD.partials(qv)[1]-expected_dq)<=1e-12*abs(expected_dq)
        qt=TR.tracer_face_flux(tracer,r,st,(zero(qm),qm),1,up,kg,1)
        @test abs(FD.partials(qt)[1]/FD.partials(qm)[1]-expected_c)<=1e-12
        qc=JD.compositional_fluxes!(SVector(zero(qm),zero(qm)),st.LiquidMassFractions,
            st.VaporMassFractions,(zero(qm),qm),up,Val(false),(1,2),Val(2))
        @test maximum(abs.([FD.partials(x)[1]/FD.partials(qm)[1] for x in qc]-expected_y))<=1e-12
        @test JD.darcy_phase_mass_flux(1,1,st,r,nothing,kg,up,half*Q) == 1000half*Q
        # A zero exponent must preserve the original local-upwind Jacobian.
        off=installed(shear=nothing); e0=installed(shear=config(0.))
        for phase in (1,2)
            @test JD.darcy_phase_mass_flux(1,phase,st,off,nothing,kg,up,half*Q) ==
                JD.darcy_phase_mass_flux(1,phase,st,e0,nothing,kg,up,half*Q)
        end
        @test TR.tracer_face_flux(tracer,off,st,(zero(qm),qm),1,up,kg,1) ==
            TR.tracer_face_flux(tracer,e0,st,(zero(qm),qm),1,up,kg,1)
    end
    for half in (1,-1), seed in (1.,-1.)
        kg=half==1 ? TPFA(l,rr,1) : TPFA(rr,l,-1); up=SPU(kg.left,kg.right)
        st=face_state(r,-1.;mobility=FD.Dual(0.,seed))
        # Inject mobility directly; do not let local_ad discard its derivative.
        mob=copy(st.PhaseMobilities)
        mob[2,rr]=FD.Dual(0.,seed); st=merge(st,(PhaseMobilities=mob,))
        off=installed(shear=nothing)
        qoff=JD.darcy_phase_mass_flux(1,2,st,off,nothing,kg,up,-1.0half)
        expected_c=upwind(up,cell->st.SurfactantPartition[2,cell],qoff)
        expected_y=upwind(up,cell->st.VaporMassFractions[:,cell],qoff)
        qm=JD.darcy_phase_mass_flux(1,2,st,r,nothing,kg,up,-1.0half)
        qt=TR.tracer_face_flux(tracer,r,st,(zero(qm),qm),1,up,kg,1)
        @test FD.partials(qm)[1]!=0
        @test abs(FD.partials(qt)[1]/FD.partials(qm)[1]-expected_c)<=1e-12
        qc=JD.compositional_fluxes!(SVector(zero(qm),zero(qm)),st.LiquidMassFractions,
            st.VaporMassFractions,(zero(qm),qm),up,Val(false),(1,2),Val(2))
        @test maximum(abs.([FD.partials(x)[1]/FD.partials(qm)[1] for x in qc]-expected_y))<=1e-12
    end
    st=face_state(r,1.)
    @test JD.darcy_phase_mass_flux(1,1,st,r,nothing,TPFA(l,rr,1),SPU(l,rr),1.) == 1000.
end

@testset "Magnitude and approximate tangent" begin
    r=installed();v=r.secondary_variables[:FoamShearGradient];st=face_state(r,1.)
    c=v.config; fm=.1
    for n in (c.fmcap*.5,c.fmcap,c.fmcap*2), eta in (0.,.3,1.)
        Q=FD.Dual(1.,1.); f= n<=c.fmcap ? 1. : (c.fmcap/n)^c.epcap
        m=1/(fm+(1-fm)*f); slope=n<=c.fmcap ? 0. : c.epcap*(1-fm)*f/(fm+(1-fm)*f)
        qs=TR.foam_corrected_potential(Q,fm,n,eta,c)
        @test abs(FD.value(qs)-m)<=1e-12
        @test abs(FD.partials(qs)[1]-m*(1+slope*(1+eta)/2))<=1e-12
        qfm=TR.foam_corrected_potential(1.,FD.Dual(fm,1.),n,eta,c)
        @test abs(FD.partials(qfm)[1]+(1-f)*m^2)<=1e-12
    end
    l,rr=v.neighbors[:,1]
    v.bad[l]=true
    # The good neighbor can have a tangential gradient even when the other
    # cell's stencil is deficient. The fallback must ignore that tangent.
    qval=2*c.fmcap*v.areas[1]*c.sigma
    st=face_state(r,qval)
    st.FoamShearGradient[2,rr]=2*(qval/v.areas[1])/v.kd[2,rr]
    Q=FD.Dual(qval,1.)
    qs=JD.face_potential_difference(v,1,2,st,r,SPU(l,rr),Q)
    # Independent derivative of Q/[fm+(1-fm)*(b*A*sigma/Q)^e].
    f=(c.fmcap*v.areas[1]*c.sigma/qval)^c.epcap; den=fm+(1-fm)*f
    @test abs(FD.partials(qs)[1]-(1/den+c.epcap*(1-fm)*f/den^2))<=1e-12
end

@testset "Assembly cache and face capillary numbers" begin
    r=installed();p,s=state_for(r);sim=Simulator(r;state0=s,parameters=p)
    forces=setup_forces(r)
    Jutul.update_state_dependents!(sim.storage,r,1.,forces)
    st=Jutul.evaluation_state(sim.storage);v=r.secondary_variables[:FoamShearGradient]
    @test !any(v.bad)
    oracle=zeros(size(st.FoamShearGradient))
    primal = (; (k => Jutul.value.(st[k]) for k in
        (:Pressure, :PhaseMassDensities, :Saturations, :CapillaryPressure,
         :Transmissibilities, :TwoPointGravityDifference) if haskey(st,k))...)
    TR.fill_least_squares_gradients!(oracle,v.weights,v.bad,v.neighbors,primal,r,v.gas,axes(oracle,2))
    @test maximum(abs.(Jutul.value.(st.FoamShearGradient)-oracle))<=1e-12
    @test all(x->all(iszero,FD.partials(x)),st.FoamShearGradient)
    for f in axes(v.neighbors,2)
        l,rr=v.neighbors[:,f];kg=TPFA(l,rr,1);up=SPU(l,rr)
        Q=JD.darcy_permeability_potential_difference(f,st,r,nothing,kg,up,2)
        vel=(v.kd[:,l].*Jutul.value.(st.FoamShearGradient[:,l])+v.kd[:,rr].*Jutul.value.(st.FoamShearGradient[:,rr]))/2
        tangent=vel-v.normals[:,f]*dot(vel,v.normals[:,f])
        expected_n=sqrt((Jutul.value(Q)/v.areas[f])^2+sum(abs2,tangent))/v.config.sigma
        n,=TR.foam_ls_magnitude(f,Q,st.FoamShearGradient,v)
        @test abs(n-expected_n)<=1e-12*expected_n
        u=Jutul.value(Q)<0 ? rr : l
        qm=Jutul.value(JD.darcy_phase_mass_flux(f,2,st,r,nothing,kg,up,Q))
        qv=Jutul.value(JD.darcy_phase_volume_flux(f,2,st,r,nothing,kg,up,Q))
        @test qm ≈ Jutul.value(st.PhaseMassDensities[2,u])*qv rtol=1e-12
    end
    r2=installed(shear=FoamShearThinning(fmcap=1e-5,epcap=.59,sigma=.03))
    p2,s2=state_for(r2);sim2=Simulator(r2;state0=s2,parameters=p2)
    @test r.secondary_variables[:FoamShearGradient] !== r2.secondary_variables[:FoamShearGradient]
    @test sim.storage.state.FoamShearGradient !== sim2.storage.state.FoamShearGradient
end
@testset "Frozen shear cache storage and ownership" begin
    r = installed()
    p, s = state_for(r)
    fill!(s[:FoamShearGradient], 0.25)
    original = deepcopy(s)
    sim = Simulator(r; state0=s, parameters=p)
    sim2 = Simulator(r; state0=s, parameters=p)
    G = sim.storage.state.FoamShearGradient
    G0 = sim.storage.state0.FoamShearGradient
    @test eltype(G) === Float64
    @test eltype(G0) === Float64
    @test G == original[:FoamShearGradient]
    @test !Base.mightalias(G, G0)
    @test !Base.mightalias(G, s[:FoamShearGradient])
    @test !Base.mightalias(G, sim2.storage.state.FoamShearGradient)
    ad = Jutul.convert_state_ad(r, s)
    @test eltype(ad[:FoamShearGradient]) === Float64
    @test eltype(ad[:Pressure]) <: Jutul.ForwardDiff.Dual
    @test ad[:FoamShearGradient] == original[:FoamShearGradient]
    @test !Base.mightalias(ad[:FoamShearGradient], s[:FoamShearGradient])
    forces = setup_forces(r)
    Jutul.update_state_dependents!(sim.storage, r, 1.0, forces)
    accepted = copy(G)
    Jutul.update_after_step!(sim.storage, r, 1.0, forces)
    @test G0 == accepted
    sim.storage.primary_variables.Pressure[1] += 2e5
    Jutul.update_state_dependents!(sim.storage, r, 1.0, forces)
    @test G != accepted
    @test G0 == accepted
    Jutul.reset_state_to_previous_state!(sim.storage, r)
    Jutul.update_state_dependents!(sim.storage, r, 1.0, forces; update_secondary=false)
    @test G == accepted
    @test s == original
    primary = Dict(k => deepcopy(s[k]) for k in keys(r.primary_variables))
    helper = HelperSimulator(r; state0=primary, parameters=p)
    Jutul.update_state_dependents!(helper.storage, r, 1.0, forces)
    @test eltype(helper.storage.state.FoamShearGradient) === Float64
    @test helper.storage.state.FoamShearGradient == accepted
    # A stale saved cache is recomputed, not trusted.
    saved = deepcopy(primary)
    saved[:FoamShearGradient] = fill(1e100, size(G))
    stale = HelperSimulator(r; state0=saved, parameters=p)
    Jutul.update_state_dependents!(stale.storage, r, 1.0, forces)
    @test stale.storage.state.FoamShearGradient == accepted
    @test all(==(1e100), saved[:FoamShearGradient])
end

@testset "Supplied shear cache conversion" begin
    r = installed()
    p, s = state_for(r)
    forces = setup_forces(r)
    reference = Simulator(r; state0=s, parameters=p)
    Jutul.update_state_dependents!(reference.storage, r, 1.0, forces)
    expected = reference.storage.state.FoamShearGradient
    n = Jutul.degrees_of_freedom_per_entity(r, Cells())
    dual32 = Jutul.get_ad_entity_scalar(0.25f0, n, 1; tag=Cells())
    dual64 = Jutul.get_ad_entity_scalar(0.25, n, 1; tag=Cells())
    for value in (0.25f0, dual32, dual64)
        input = deepcopy(s)
        input[:FoamShearGradient] = fill(value, size(expected))
        original = deepcopy(input)
        ad = Jutul.convert_state_ad(r, input)
        G = ad[:FoamShearGradient]
        @test eltype(G) <: FD.Dual
        @test eltype(Jutul.value.(G)) === Float64
        @test all(==(0.25), Jutul.value.(G))
        # Jutul copies partials when the supplied Dual type already matches.
        initial_partials = ntuple(i -> value === dual64 && i == 1 ? 1.0 : 0.0, n)
        @test all(x -> Tuple(FD.partials(x)) == initial_partials, G)
        @test !Base.mightalias(G, input[:FoamShearGradient])
        sim = Simulator(r; state0=input, parameters=p)
        Jutul.update_state_dependents!(sim.storage, r, 1.0, forces)
        @test Jutul.value.(sim.storage.state.FoamShearGradient) == expected
        @test input == original
    end
end

end
