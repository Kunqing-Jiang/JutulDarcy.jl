@doc raw"""
    FoamShearThinning(; fmcap, epcap, sigma)

Least-squares gas shear thinning with the TPFA normal potential retained.
`fmcap > 0` and `epcap >= 0` define the capillary-number response; surface
tension `sigma > 0` is in N/m and must match the calibration of `fmcap`.
All three values must be finite. An exponent of zero disables shear entirely.

For an internal face, let `Q` be the gas permeability-potential difference
before multiplying by mobility, `A` its area, and `n̂` its unit normal. Bars
denote primal values. The cell least-squares fit `g` uses oriented `Q/T`
drops, with inverse squared distance weights in the active coordinates.
With diagonal permeability `K`,
```math
\mathbf t=(I-\hat{\mathbf n}\hat{\mathbf n}^{T})
    (K_l\mathbf g_l+K_r\mathbf g_r)/2,\qquad
N=\frac{\sqrt{(\bar Q/A)^2+|\mathbf t|^2}}{\sigma},\qquad
\eta=\frac{(\bar Q/A)^2}{(\bar Q/A)^2+|\mathbf t|^2}.
```
Set `η=0` when both components vanish. This is a dimensionless,
pressure-gradient capillary number: `darcy_permeability_potential_difference`
returns `-T*(Δp + Δpc + ρgΔz)`, so both `Q/A` and `K*g` have units Pa m,
matching `sigma` in N/m. It is not a velocity divided by surface tension;
`fmcap` must be calibrated for this definition and the supplied `sigma`.

Writing `b=fmcap`, `e=epcap` and `FM` for the upstream zero-shear foam factor,
```math
F_{cap}(N)=\begin{cases}1 & N\le b\\(b/N)^e & N>b,\end{cases}\qquad
m=\frac{1}{FM+(1-FM)F_{cap}},
```
```math
s=\begin{cases}0 & N\le b\\
e(1-\overline{FM})F_{cap}/[\overline{FM}+(1-\overline{FM})F_{cap}] & N>b,
\end{cases}\qquad
\widehat Q=m(FM^{AD},\bar N)Q+\bar m\,s\,\frac{1+\eta}{2}(Q-\bar Q).
```
The zero-valued compensation changes only the local approximate Jacobian;
the AD dependence of `FM` is retained. If either cell stencil is rank deficient
(`rtol=1e-12`), use `N=abs(Q)/(A*sigma)` with its AD dependence and `Q̂=m*Q`
without compensation. Coordinate spans at most `1e-8` times the largest span
are treated as inactive unless they dominate an internal neighbor displacement.
This retains real thin-layer connections while ignoring constant-coordinate
roundoff. Installation warns when more than 10% of cells are rank deficient.
Fallback diagnostics report `η=1` for nonzero `Q` and `η=0` for zero `Q`.

Install through `set_surfactant_model!` before constructing parameters, states
or simulators. Supports fixed scalar/diagonal permeability, Float64 CPU fully
implicit two-phase immiscible or isothermal CO2-brine models with TPFA/SPU.
The gradient is frozen in the local approximate Jacobian. Boundary sources
and perforations retain their cell foam factors. Rebuild the model and
simulator to change the configuration or geometry. Adjoint sensitivities are
rejected because this Jacobian is only a local approximation.
"""
struct FoamShearThinning
    fmcap::Float64
    epcap::Float64
    sigma::Float64
    function FoamShearThinning(; fmcap, epcap, sigma)
        b, e, sig = Float64(fmcap), Float64(epcap), Float64(sigma)
        all(isfinite, (b, e, sig)) && b > 0 && e >= 0 && sig > 0 ||
            throw(ArgumentError("FoamShearThinning requires finite fmcap > 0, epcap >= 0 and sigma > 0"))
        return new(b, e, sig)
    end
end

foam_shear_enabled(::Nothing) = false
foam_shear_enabled(c::FoamShearThinning) = c.epcap > 0

struct FoamShearGradient <: Jutul.VectorVariables
    config::FoamShearThinning
    gas::Int
    neighbors::Matrix{Int}
    areas::Vector{Float64}
    weights::Vector{Vector{Tuple{Int, Vector{Float64}}}}
    bad::BitVector
    normals::Matrix{Float64}
    kd::Matrix{Float64}
    active::Vector{Int}
end

function check_foam_shear_forward_model(model)
    if model.context isa Jutul.CPUJutulContext &&
            Jutul.represented_as_adjoint(Jutul.matrix_layout(model.context))
        throw(ArgumentError("Foam shear does not support adjoint or sensitivity models"))
    end
    return nothing
end

function Jutul.values_per_entity(model, v::FoamShearGradient)
    # Adjoint storage initializes secondary values before any equation assembly.
    # Check even when the caller supplies a previously allocated gradient cache.
    check_foam_shear_forward_model(model)
    return size(v.kd, 1)
end
Jutul.degrees_of_freedom_per_entity(model, v::FoamShearGradient) = Jutul.values_per_entity(model, v)
Jutul.associated_entity(::FoamShearGradient) = Cells()
function Jutul.get_dependencies(::FoamShearGradient, model)
    return Tuple(k for k in (:Pressure, :Saturations, :PhaseMassDensities, :CapillaryPressure)
        if haskey(model.primary_variables, k) || haskey(model.secondary_variables, k))
end
# Neighbor properties are not synchronized between secondary-variable batches.
# Refresh this cache only after all batches finish, at the equation entry point.
Jutul.update_secondary_variable!(G, ::FoamShearGradient, model, state, ix) = G
function Jutul.subvariable(::FoamShearGradient, ::Jutul.FiniteVolumeGlobalMap)
    throw(ArgumentError("Foam shear does not support partitioned models or domain decomposition"))
end

function check_foam_shear_model(model)
    sys = model.system
    sys isa Union{ImmiscibleSystem, JutulDarcy.LVCompositional2PhaseSystem} ||
        throw(ArgumentError("Foam shear does not support system $(typeof(sys))"))
    phases = get_phases(sys)
    length(phases) == 2 && count(==(VaporPhase()), phases) == 1 ||
        throw(ArgumentError("Foam shear requires two phases with one VaporPhase"))
    model.formulation isa Jutul.FullyImplicitFormulation ||
        throw(ArgumentError("Foam shear requires FullyImplicitFormulation, got $(typeof(model.formulation))"))
    model.context isa Jutul.CPUJutulContext ||
        throw(ArgumentError("Foam shear requires a CPUJutulContext, got $(typeof(model.context))"))
    check_foam_shear_forward_model(model)
    model.domain.global_map isa Jutul.TrivialGlobalMap ||
        throw(ArgumentError("Foam shear does not support partitioned models"))
    haskey(model.data_domain, :nnc) && throw(ArgumentError("Foam shear does not support NNCs"))
    variables = (model.primary_variables, model.secondary_variables, model.parameters)
    thermal = JutulDarcy.model_is_thermal(model) || any(v -> haskey(v, :FluidEnthalpy), variables) ||
        any(eq -> eq isa ConservationLaw{:TotalThermalEnergy}, values(model.equations))
    thermal && throw(ArgumentError("Foam shear requires an isothermal model without FluidEnthalpy or energy equations"))
    any(v -> haskey(v, :PermeabilityMultiplier), variables) &&
        throw(ArgumentError("Foam shear does not support PermeabilityMultiplier"))
    for key in (:mass_conservation, :tracers)
        haskey(model.equations, key) || throw(ArgumentError("Foam shear requires equation $key"))
        eq = model.equations[key]
        eq.flow_discretization isa TwoPointPotentialFlowHardCoded ||
            throw(ArgumentError("Foam shear requires TwoPointPotentialFlowHardCoded for $key, got $(typeof(eq.flow_discretization))"))
    end
    gas = findfirst(==(VaporPhase()), phases)
    tracers = model.equations[:tracers].flux_type.tracers
    count(t -> t isa SurfactantTracer, tracers) == 1 ||
        throw(ArgumentError("Foam shear requires exactly one SurfactantTracer"))
    for t in tracers
        if !(t isa SurfactantTracer) && gas in tracer_phase_indices(t)
            throw(ArgumentError("Foam shear supports gas tracer loads only for SurfactantTracer, got $(typeof(t))"))
        end
    end
    for (key, v) in pairs(model.secondary_variables)
        v isa FoamShearGradient && continue
        :FoamShearGradient in Jutul.get_dependencies(v, model) &&
            throw(ArgumentError("Secondary variable $key cannot depend on the late FoamShearGradient cache"))
    end
    return nothing
end

shear_number_supported(::Type) = false
shear_number_supported(::Type{Float64}) = true
shear_number_supported(::Type{<:Jutul.ForwardDiff.Dual{T, Float64, N}}) where {T, N} = true
function check_shear_number(::Type{T}) where T
    shear_number_supported(T) || throw(ArgumentError("Foam shear requires Float64 or local Float64 ForwardDiff Dual, got $T"))
    return nothing
end
@inline function check_shear_state(state)
    for k in (:PermeabilityMultiplier, :FluidEnthalpy)
        haskey(state, k) && throw(ArgumentError("Foam shear does not support state field $k"))
    end
    # Literal keys let local AD views resolve field types without allocating
    # a dynamically selected view for every field on every face.
    haskey(state, :Pressure) && check_shear_number(eltype(state[:Pressure]))
    haskey(state, :PhaseMassDensities) && check_shear_number(eltype(state[:PhaseMassDensities]))
    haskey(state, :Saturations) && check_shear_number(eltype(state[:Saturations]))
    haskey(state, :CapillaryPressure) && check_shear_number(eltype(state[:CapillaryPressure]))
    haskey(state, :Transmissibilities) && check_shear_number(eltype(state[:Transmissibilities]))
    haskey(state, :TwoPointGravityDifference) && check_shear_number(eltype(state[:TwoPointGravityDifference]))
    haskey(state, :FmFactor) && check_shear_number(eltype(state[:FmFactor]))
    haskey(state, :PhaseMobilities) && check_shear_number(eltype(state[:PhaseMobilities]))
    haskey(state, :PhaseMassMobilities) && check_shear_number(eltype(state[:PhaseMassMobilities]))
    haskey(state, :SurfactantPartition) && check_shear_number(eltype(state[:SurfactantPartition]))
    haskey(state, :LiquidMassFractions) && check_shear_number(eltype(state[:LiquidMassFractions]))
    haskey(state, :VaporMassFractions) && check_shear_number(eltype(state[:VaporMassFractions]))
    haskey(state, :TracerConcentrations) && check_shear_number(eltype(state[:TracerConcentrations]))
    haskey(state, :OverallMoleFractions) && check_shear_number(eltype(state[:OverallMoleFractions]))
    haskey(state, :FoamShearGradient) && check_shear_number(eltype(state[:FoamShearGradient]))
    return nothing
end

# Fit the oriented potential drops on all internal neighbor connections.
function foam_shear_geometry(d)
    cc=d[:cell_centroids]; N=d[:neighbors]; dim,nc=size(cc); nf=size(N,2)
    normals=copy(d[:normals]); A=d[:areas]; k=d[:permeability]
    ndims(k)==1 || size(k,1) in (1,dim) || throw(ArgumentError("Only diagonal permeability is supported"))
    all(isfinite,k) && minimum(k)>0 || throw(ArgumentError("Positive finite diagonal permeability is required"))
    kd=ndims(k)==1 ? repeat(reshape(k,1,:),dim,1) : size(k,1)==1 ? repeat(k,dim,1) : Matrix(k)
    size(kd)==(dim,nc) || throw(ArgumentError("Permeability shape mismatch"))
    all(isfinite,cc) && all(isfinite,normals) || throw(ArgumentError("Finite geometry required"))
    actual=JutulDarcy.reservoir_transmissibility(d)
    length(actual)==nf && all(x->isfinite(x)&&x>0,actual) || throw(ArgumentError("Positive finite TPFA transmissibilities without NNC required"))
    spans = vec(maximum(cc; dims=2) - minimum(cc; dims=2))
    # Ignore roundoff in nominally constant coordinates on single-layer grids.
    active = findall(>(1e-8*maximum(spans)), spans)
    isempty(active) && throw(ArgumentError("No active coordinate direction"))
    connections=[Tuple{Int,Int}[] for _ in 1:nc]; seen=Set{Tuple{Int,Int}}()
    for f in 1:nf
        l,r=N[:,f]; pair=minmax(l,r)
        1<=l<=nc && 1<=r<=nc && l!=r && !(pair in seen) || throw(ArgumentError("Invalid or duplicate connection"))
        push!(seen,pair);push!(connections[l],(f,r));push!(connections[r],(f,l))
        nn=norm(normals[:,f]);nn>0 && isfinite(nn) && isfinite(A[f]) && A[f]>0 || throw(ArgumentError("Invalid face geometry"))
        normals[:,f]./=nn
        # A small domain span can be a real thin layer. Retain any direction
        # that dominates an internal connection so its projected distance
        # cannot collapse solely because of the global span cutoff.
        j = argmax(abs.(cc[:,r] - cc[:,l]))
        spans[j] > 0 && !(j in active) && push!(active, j)
    end
    sort!(active)
    weights=[Tuple{Int,Vector{Float64}}[] for _ in 1:nc];bad=falses(nc)
    for c in 1:nc
        cs=connections[c]
        D=hcat([cc[active,j]-cc[active,c] for (_,j) in cs]...)
        if isempty(cs) || rank(D;rtol=1e-12)<length(active)
            bad[c]=true;continue
        end
        w=1.0./vec(sum(abs2,D;dims=1)); M=(D.*reshape(w,1,:))*D'
        B=M\(D.*reshape(w,1,:))
        for (i,(f,j)) in enumerate(cs)
            v=zeros(dim);v[active].=B[:,i].*(N[1,f]==c ? 1 : -1)
            push!(weights[c],(f,v))
        end
    end
    return (;weights,bad,normals,kd,active,actual)
end

function FoamShearGradient(model, config::FoamShearThinning)
    check_foam_shear_model(model)
    d = model.data_domain
    g = foam_shear_geometry(d)
    areas = Vector{Float64}(d[:areas])
    nbad = count(g.bad)
    if nbad > 0.1*length(g.bad)
        @warn "Foam shear uses the face-normal fallback next to rank-deficient cells" fallback_cells=nbad total_cells=length(g.bad)
    end
    gas = findfirst(==(VaporPhase()), get_phases(model.system))
    return FoamShearGradient(config, gas, Matrix{Int}(d[:neighbors]), areas,
        g.weights, g.bad, g.normals, g.kd, g.active)
end

# Strip all supported Dual tags without copying the underlying arrays.
foam_primal_view(A::AbstractArray{Float64}) = A
foam_primal_view(A::AbstractArray) = Jutul.mappedarray(primal_value, A)
function foam_potential_values(state)
    fields = (:Pressure, :PhaseMassDensities, :Saturations, :CapillaryPressure,
              :Transmissibilities, :TwoPointGravityDifference)
    return (; (k => foam_primal_view(state[k]) for k in fields if haskey(state, k))...)
end

function fill_foam_shear_gradients!(G, v::FoamShearGradient, pv, model, ix)
    dim = size(v.kd, 1)
    g = zeros(dim)
    for c in ix
        fill!(g, 0.0)
        if !v.bad[c]
            for (f, w) in v.weights[c]
                l = v.neighbors[1, f]; r = v.neighbors[2, f]
                q = JutulDarcy.darcy_permeability_potential_difference(f, pv, model, nothing, TPFA(l, r, 1), SPU(l, r), v.gas)
                T = pv.Transmissibilities[f]
                isfinite(T) && T > 0 || throw(ArgumentError("Foam shear requires positive finite runtime transmissibility"))
                g .+= w .* (q/T)
            end
        end
        for j in 1:dim
            G[j, c] = g[j]
        end
    end
    return G
end

function prepare_foam_shear!(state, model, v::FoamShearGradient)
    check_foam_shear_model(model)
    check_shear_state(state)
    pv = foam_potential_values(state)
    G = state.FoamShearGradient
    nb = clamp(size(G, 2) ÷ Jutul.minbatch(model.context), 1, Jutul.nthreads(model.context))
    if nb == 1
        fill_foam_shear_gradients!(G, v, pv, model, axes(G, 2))
    else
        Threads.@threads for batch in 1:nb
            ix = Jutul.entity_eachindex(G, batch, nb)
            fill_foam_shear_gradients!(G, v, pv, model, ix)
        end
    end
    return G
end

function prepare_foam_shear!(state, model)
    v = get(model.secondary_variables, :FoamShearGradient, nothing)
    v === nothing && return nothing
    return prepare_foam_shear!(state, model, v)
end

function Jutul.update_equations!(storage, model::JutulDarcy.DarcyFlowModel, dt)
    prepare_foam_shear!(Jutul.evaluation_state(storage), model)
    Jutul.update_equations!(storage, storage.equations, model.equations, model, dt)
    return nothing
end

foam_fcap(n, b, e) = primal_value(n) <= b ? one(n) : (b/n)^e
foam_shear_multiplier(fm, n, c) = inv(fm + (1 - fm)*foam_fcap(n, c.fmcap, c.epcap))

function foam_ls_magnitude(f, Q, state, v::FoamShearGradient)
    G = state.FoamShearGradient
    l = v.neighbors[1, f]; r = v.neighbors[2, f]
    dim = size(v.kd, 1)
    vn = 0.0
    @inbounds for j in 1:dim
        vj = (v.kd[j, l]*primal_value(G[j, l]) + v.kd[j, r]*primal_value(G[j, r]))/2
        vn += vj*v.normals[j, f]
    end
    t2 = 0.0
    @inbounds for j in 1:dim
        vj = (v.kd[j, l]*primal_value(G[j, l]) + v.kd[j, r]*primal_value(G[j, r]))/2
        t2 += (vj - vn*v.normals[j, f])^2
    end
    un = primal_value(Q)/v.areas[f]
    mag2 = un^2 + t2
    return sqrt(mag2)/v.config.sigma, mag2 > 0 ? un^2/mag2 : 0.0
end

function foam_face_magnitude(f, Q, state, v::FoamShearGradient)
    l = v.neighbors[1, f]; r = v.neighbors[2, f]
    fallback = v.bad[l] || v.bad[r]
    if fallback
        # Keep the potential derivative on a rank-deficient stencil.
        n = abs(Q)/(v.areas[f]*v.config.sigma)
        eta = iszero(primal_value(Q)) ? 0.0 : 1.0
    else
        n, eta = foam_ls_magnitude(f, Q, state, v)
    end
    return n, eta, fallback
end

function foam_corrected_potential(Q, fm, n, eta, c::FoamShearThinning)
    m = foam_shear_multiplier(fm, n, c)
    F = foam_fcap(n, c.fmcap, c.epcap)
    fm0 = primal_value(fm)
    s = n <= c.fmcap ? 0.0 : c.epcap*(1 - fm0)*F/(fm0 + (1 - fm0)*F)
    # This zero-valued term supplies a local approximate Jacobian, without
    # adding neighbor-of-neighbor entries to the TPFA sparsity pattern.
    return m*Q + primal_value(m)*s*((1 + eta)/2)*(Q - primal_value(Q))
end

@inline function check_shear_face_inputs(model, state, kgrad, upw)
    model.formulation isa Jutul.FullyImplicitFormulation ||
        throw(ArgumentError("Foam shear requires FullyImplicitFormulation, got $(typeof(model.formulation))"))
    check_shear_state(state)
    kgrad isa TPFA && upw isa SPU || throw(ArgumentError("Foam shear requires TPFA and SPU, got $(typeof(kgrad)) and $(typeof(upw))"))
    return nothing
end

function foam_canonical_upstream(v::FoamShearGradient, face, Q, model, state, kgrad, upw)
    check_shear_face_inputs(model, state, kgrad, upw)
    return foam_canonical_upstream(v, face, Q, kgrad, upw)
end

function foam_canonical_upstream(v::FoamShearGradient, face, Q, kgrad, upw)
    check_shear_number(typeof(Q))
    l = v.neighbors[1, face]; r = v.neighbors[2, face]
    signhalf = if kgrad.left == l && kgrad.right == r
        1
    elseif kgrad.left == r && kgrad.right == l
        -1
    else
        throw(ArgumentError("Foam shear half-face cells do not match the installed geometry"))
    end
    kgrad.face_sign == signhalf || throw(ArgumentError("Foam shear half-face gravity orientation does not match the installed geometry"))
    upw.left == kgrad.left && upw.right == kgrad.right || throw(ArgumentError("Foam shear has inconsistent TPFA upwind cells"))
    q = signhalf*Q
    u = primal_value(q) < 0 ? r : l
    return u, q, signhalf
end

function JutulDarcy.foam_shear_flux(v::FoamShearGradient, face, phase, state, model, kgrad, upw, Q)
    (foam_shear_enabled(v.config) && phase == v.gas) || return nothing
    u, q, signhalf = foam_canonical_upstream(v, face, Q, model, state, kgrad, upw)
    n, eta, fallback = foam_face_magnitude(face, q, state, v)
    fm = state.FmFactor[u]
    qs = fallback ? foam_shear_multiplier(fm, n, v.config)*q : foam_corrected_potential(q, fm, n, eta, v.config)
    return u, signhalf*qs
end

function JutulDarcy.foam_shear_upstream(v::FoamShearGradient, face, phase, state, model, ft, kgrad, upw)
    (foam_shear_enabled(v.config) && phase == v.gas) || return nothing
    # Validate before computing the potential, including generic tracer inputs.
    check_shear_face_inputs(model, state, kgrad, upw)
    Q = JutulDarcy.darcy_permeability_potential_difference(face, state, model, ft, kgrad, upw, phase)
    return first(foam_canonical_upstream(v, face, Q, kgrad, upw))
end

"""
    foam_shear_diagnostics(model, parameters, saved_state)

Reconstruct ordinary secondary properties and the shear cache from a saved
reservoir state, then return internal-face diagnostics. Disabled shear returns
only `(enabled=false,)`. This does not evaluate boundary or perforation shear.
"""
function foam_shear_diagnostics(model::SimulationModel, parameters, saved_state)
    v = get(model.secondary_variables, :FoamShearGradient, nothing)
    (v !== nothing && foam_shear_enabled(v.config)) || return (enabled = false,)
    # Build from primary variables only: saved secondary caches may be stale.
    initial = Dict(k => deepcopy(saved_state[k]) for k in keys(model.primary_variables))
    sim = HelperSimulator(model; state0 = initial, parameters)
    Jutul.update_state_dependents!(sim.storage, model, 1.0, setup_forces(model))
    state = Jutul.evaluation_state(sim.storage)
    return foam_shear_diagnostics(state, model, v)
end
function foam_shear_diagnostics(state, model, v::FoamShearGradient)
    rows = map(axes(v.neighbors, 2)) do f
        l, r = v.neighbors[:, f]
        kg = TPFA(l, r, 1); up = SPU(l, r)
        Q = JutulDarcy.darcy_permeability_potential_difference(f, state, model, nothing, kg, up, v.gas)
        u, _, _ = foam_canonical_upstream(v, f, Q, model, state, kg, up)
        n, eta, fallback = foam_face_magnitude(f, Q, state, v)
        fm = state.FmFactor[u]
        m = foam_shear_multiplier(fm, n, v.config)
        qm = JutulDarcy.darcy_phase_mass_flux(f, v.gas, state, model, nothing, kg, up, Q)
        qv = JutulDarcy.darcy_phase_volume_flux(f, v.gas, state, model, nothing, kg, up, Q)
        return (normal_capillary_number = abs(primal_value(Q))/(v.areas[f]*v.config.sigma),
            capillary_number = primal_value(n), eta = eta, fallback = fallback,
            foam_factor = primal_value(fm), multiplier = primal_value(m),
            effective_foam_factor = primal_value(fm*m),
            gas_mass_flux = primal_value(qm), gas_volume_flux = primal_value(qv))
    end
    return (enabled = true, faces = rows)
end
