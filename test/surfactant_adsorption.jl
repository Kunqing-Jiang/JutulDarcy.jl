module SurfactantAdsorptionTests
using JutulDarcy, Jutul, Test
import JutulDarcy.Tracers: tracer_total_mass_outer, tracer_face_flux,
    perforation_production_concentration, ReservoirFromWellTracerCT
const FD = Jutul.ForwardDiff
const TABLE = ([0.0, 1, 2, 4], [0.0, 2e-4, 3e-4, 3e-4])
isotherms() = (LinearSurfactantAdsorption(.0002), LangmuirSurfactantAdsorption(.001, 2), TabulatedSurfactantAdsorption(TABLE...))
closemass(x, y) = abs(x-y) <= 1e-12 + 1e-12*abs(y)

function model_with(a; order=(LiquidPhase(), VaporPhase()), n=3, kd=.5, K=nothing,
        L=Inf, phi=.25, rho_s=2400, widths=fill(1.0, n), wells=false, mixed=false,
        compressible_volume=false, D=nothing, general_ad=false, domain_kw...)
    mesh = UnstructuredMesh(CartesianMesh((n,1,1), (widths, [1.0], [1.0])))
    domain = reservoir_domain(mesh; porosity=phi, rock_density=rho_s, permeability=1e-14, domain_kw...)
    gas = findfirst(==(VaporPhase()), order); liquid = 3-gas
    rho = [ph isa VaporPhase ? 500.0 : 1000.0 for ph in order]
    beta = [ph isa VaporPhase ? 2e-8 : 1e-8 for ph in order]
    sys = ImmiscibleSystem(order; reference_densities=rho)
    ws = wells ? [setup_well(domain,1;name=:Injector,simple_well=true,volumes=[1e-8])] : []
    model = setup_reservoir_model(domain, sys; wells=ws, general_ad=general_ad, block_backend=false)
    r = reservoir_model(model)
    replace_variables!(r, PhaseMassDensities=ConstantCompressibilityDensities(sys,1e7,rho,beta))
    if compressible_volume
        delete!(r.parameters, :FluidVolume)
        set_secondary_variables!(r, FluidVolume=JutulDarcy.LinearlyCompressiblePoreVolume(reference_pressure=1e7, expansion=1e-9))
        set_parameters!(r, StaticFluidVolume=JutulDarcy.FluidVolume())
    end
    coef = K === nothing ? (;k_d=kd) : (;K=K)
    t = SurfactantTracer(sys;coef...,L=L,adsorption=a,D=D)
    add_tracers_to_model!(model, mixed ? [MultiPhaseTracer(sys),t] : t)
    set_surfactant_model!(model)
    return (;model,r,t,gas,liquid)
end

function local_state(b, prm; c=.001, sl=.5, p=1e7)
    S = zeros(2, Jutul.number_of_entities(b.r, b.r.primary_variables[:Pressure]))
    S[b.liquid,:] .= sl; S[b.gas,:] .= 1 .- sl
    C = b.r.primary_variables[:TracerConcentrations].flux_type.tracers
    cvals = length(C) == 2 ? [fill(.3,1,size(S,2)); reshape(fill(c,size(S,2)),1,:)] : c
    s0 = setup_state(b.r; Pressure=p, Saturations=S, TracerConcentrations=cvals)
    sim = Simulator(b.r; state0=s0, parameters=prm)
    Jutul.update_state_dependents!(sim.storage, b.r, 1.0, setup_forces(b.r))
    return sim.storage[:state]
end

# Independent density, closure and inventory derivatives at fixed saturation.
function hand_inventory(c,p,sl,ratio,cap,kind; variable_volume=false, MR=1800.0)
    rl = 1000exp(1e-8*(p-1e7)); rg = 500exp(2e-8*(p-1e7))
    drl = 1e-8rl; drg = 2e-8rg
    ml = sl*rl; mg=(1-sl)*rg; mf=ml+mg; dmf=sl*drl+(1-sl)*drg
    if ratio == :K
        B = rl*(sl+.25*(1-sl)); dB = drl*(sl+.25*(1-sl))
    elseif ratio == :floor
        B = 1e-8mf; dB=1e-8dmf
    else
        B = ml+.5mg; dB=sl*drl+.5*(1-sl)*drg
    end
    clfree=c*mf/B
    if clfree >= cap
        cl=cap; dc=0.0; dp=0.0
    else
        cl=clfree; dc=mf/B; dp=c*(dmf*B-mf*dB)/B^2
    end
    C=rl*cl
    if kind == 1
        a=.0002C; da=.0002
    elseif kind == 2
        a=.001*2C/(1+2C); da=.002/(1+2C)^2
    else
        if C <= 1
            a=.0002C; da=.0002
        elseif C <= 2
            a=.0002+.0001*(C-1); da=.0001
        elseif C <= 4
            a=.0003; da=0.0
        else
            a=.0003; da=0.0
        end
    end
    V = .25*(variable_volume ? 1+1e-9*(p-1e7)+.5*(1e-9*(p-1e7))^2 : 1)
    dV = variable_volume ? .25e-9*(1+1e-9*(p-1e7)) : 0.0
    mass = c*mf*V+MR*a
    return (;mass,cl,C,dc=mf*V+MR*da*rl*dc,dp=c*(dmf*V+mf*dV)+MR*da*(drl*cl+rl*dp))
end

function test_isotherms()
@testset "Equilibrium isotherms and concentration domain" begin
    a=LinearSurfactantAdsorption(.0002)
    @test a(1.2) ≈ .00024 rtol=1e-12
    @test FD.derivative(a,0.0) == .0002
    @test_throws DomainError LinearSurfactantAdsorption(2)(floatmax(Float64))
    for bad in (-1.0,NaN,Inf,1+im,"1",[1.0],big"1e1000")
        @test_throws ArgumentError LinearSurfactantAdsorption(bad)
        @test_throws ArgumentError LangmuirSurfactantAdsorption(bad,1)
        @test_throws ArgumentError LangmuirSurfactantAdsorption(1,bad)
    end
    @test_throws ArgumentError LangmuirSurfactantAdsorption(1e308,1e308)
    for (Q,b) in ((.001,2.0),(1e-308,1e308),(1e308,1e-308)), C in (0.0, .2, .5, 1.2, 2.0,1e308)
        a=LangmuirSurfactantAdsorption(Q,b)
        ref=Float64(BigFloat(Q)*BigFloat(b)*BigFloat(C)/(1+BigFloat(b)*BigFloat(C)))
        slope=Float64(BigFloat(Q)*BigFloat(b)/(1+BigFloat(b)*BigFloat(C))^2)
        @test isapprox(a(C),ref;rtol=1e-12,atol=0)
        @test isapprox(FD.derivative(a,C),slope;rtol=1e-10,atol=1e-14)
    end
    X,Y=deepcopy(TABLE); tab=TabulatedSurfactantAdsorption(X,Y); X[2]=.2; Y[2]=5
    for (C,val,slope) in ((0.0,0.0,.0002),(1.0,.0002,.0002),(1.2,.00022,.0001),
            (2.0,.0003,.0001),(4.0,.0003,0.0),(5.0,.0003,0.0))
        @test tab(C) ≈ val rtol=1e-12
        @test FD.derivative(tab,C) ≈ slope rtol=1e-12
    end
    # A nonzero last segment distinguishes the knot slope from the outer plateau.
    kink=TabulatedSurfactantAdsorption([0,1,2],[0,.0002,.0005])
    @test FD.derivative(kink,2.0) ≈ .0003
    @test FD.derivative(kink,2.1) == 0
    for (C,left,right) in ((1.0,.0002,.0003),(2.0,.0003,0.0))
        h=1e-7
        @test (kink(C)-kink(C-h))/h ≈ left rtol=1e-7
        @test (kink(C+h)-kink(C))/h ≈ right rtol=1e-7
    end
    for (X,Y) in (([0],[0]),([0,1],[0]),([0,0],[0,1]),([1,2],[0,1]),([0,1],[1,2]),
            ([0,1],[0,-1]),([0,1,2],[0,1,.5]),([0,Inf],[0,1]),([0,1],[0,NaN]),
            (zeros(2,1),zeros(2,1)),([0,nextfloat(0.0)],[0,1]))
        @test_throws ArgumentError TabulatedSurfactantAdsorption(X,Y)
    end
    for a in (isotherms()..., LinearSurfactantAdsorption(0),LangmuirSurfactantAdsorption(0,2),
            LangmuirSurfactantAdsorption(1,0),TabulatedSurfactantAdsorption([0,1],[0,0]))
        for C in (-1.0,NaN,Inf)
            @test_throws DomainError a(C)
        end
        @test isfinite(a(0.0))
    end
end
end

function test_rock()
@testset "Static rock mass initialization" begin
    b=model_with(first(isotherms());n=2,phi=[.25,.4],rho_s=[2400,2500],widths=[1.0,2.0])
    prm=setup_parameters(b.r)
    @test all(isapprox.(prm[:SurfactantRockMass], [1800,3000]; rtol=1e-12))
    inp=Dict(:SurfactantRockMass=>[12.0,0.0]); p=setup_parameters(b.r;inp...)
    @test p[:SurfactantRockMass] == [12,0]
    inp[:SurfactantRockMass][1]=9
    @test p[:SurfactantRockMass] == [12,0]
    @test setup_parameters(b.r;SurfactantRockMass=3)[:SurfactantRockMass] == [3,3]
    scalar_init=Dict(:SurfactantRockMass=>3.0)
    @test setup_parameters(b.model;Reservoir=scalar_init)[:Reservoir][:SurfactantRockMass] == [3,3]
    @test scalar_init[:SurfactantRockMass] == 3.0
    storage_init=Jutul.JutulStorage((SurfactantRockMass=3.0,))
    storage_parameters=Dict{Symbol,Any}()
    Jutul.initialize_parameter_value!(storage_parameters,b.r.data_domain,b.r,
        SurfactantRockMass(),:SurfactantRockMass,storage_init)
    @test storage_parameters[:SurfactantRockMass] == [3,3]
    @test storage_init[:SurfactantRockMass] == 3.0
    for x in (-1.0,NaN,Inf,[-1,1],[NaN,0],[Inf,0],zeros(2,1),[1],[1,2,3],big"1e1000")
        @test_throws ArgumentError setup_parameters(b.r;SurfactantRockMass=x)
    end
    for (key,values) in ((:rock_density,[0.0,2500]),(:rock_density,[-1.0,2500]),
            (:rock_density,[NaN,2500]),(:porosity,[1.1,.4]),(:porosity,[-.1,.4]),
            (:volumes,[0.0,2]),(:volumes,[Inf,2]))
        domain=deepcopy(b.r.data_domain); domain[key,Cells()]=values
        @test_throws ArgumentError Jutul.default_parameter_values(domain,b.r,SurfactantRockMass(),:SurfactantRockMass)
    end
    b.r.data_domain[:rock_density,Cells()] .= -1
    @test setup_parameters(b.r;SurfactantRockMass=3)[:SurfactantRockMass] == [3,3]
    for ntg in (.5,1.0)
        n=model_with(first(isotherms());n=2,net_to_gross=ntg,pore_volume=[.1,.2])
        pp=setup_parameters(n.r)
        @test all(isapprox.(pp[:SurfactantRockMass], [1800,1800]; rtol=1e-12))
        @test pp[:FluidVolume] == [.1,.2]
    end
    v=model_with(first(isotherms());n=2,compressible_volume=true)
    pp=setup_parameters(v.r); saved=copy(pp[:SurfactantRockMass])
    st1=local_state(v,pp;p=1e7); st2=local_state(v,pp;p=2e7)
    @test Jutul.value(st1[:FluidVolume]) != Jutul.value(st2[:FluidVolume])
    @test Jutul.value(st1[:SurfactantRockMass]) == Jutul.value(st2[:SurfactantRockMass]) == saved
end
end

function test_inventory()
@testset "Reservoir inventory, dependencies and well fluid storage" begin
    for order in ((LiquidPhase(),VaporPhase()),(VaporPhase(),LiquidPhase()))
        for (a,expected) in zip(isotherms(),(.5475,1.2463235294117647,.5175))
            b=model_with(a;order=order,mixed=true,wells=true,rho_s=2000)
            prm=setup_parameters(b.model); pp=prm[:Reservoir]
            st=local_state(b,pp)
            @test :SurfactantPartition in Jutul.get_dependencies(b.r.secondary_variables[:TracerMasses],b.r)
            @test Jutul.get_dependencies(b.t,b.r) == (:SurfactantPartition,:PhaseMassDensities,:SurfactantRockMass)
            @test closemass(Jutul.value(st[:TracerMasses][2,1]),expected)
            @test closemass(Jutul.value(st[:TracerMasses][1,1]),.3*750*.25)
            for (c,sl,p) in ((.0006,.7,1.2e7),(.002,.3,1.4e7))
                s=local_state(b,pp;c=c,sl=sl,p=p)
                rl=1000exp(1e-8*(p-1e7));rg=500exp(2e-8*(p-1e7));mf=sl*rl+(1-sl)*rg
                C=rl*c*mf/(sl*rl+.5*(1-sl)*rg)
                av=a isa LinearSurfactantAdsorption ? .0002C : a isa LangmuirSurfactantAdsorption ? .002C/(1+2C) : C<=1 ? .0002C : C<=2 ? .0002+.0001*(C-1) : .0003
                @test closemass(Jutul.value(s[:TracerMasses][2,1]),c*mf*.25+1500av)
            end
            w=b.model[:Injector]
            @test Jutul.get_dependencies(b.t,w) == ()
            @test !haskey(w.parameters,:SurfactantRockMass)
            @test !haskey(w.secondary_variables,:SurfactantPartition)
            @test tracer_total_mass_outer(b.t,w,(;),.001,750.0,.25,1,2) == .1875
        end
        cap=model_with(first(isotherms());order=order,L=.001,rho_s=2000)
        st=local_state(cap,setup_parameters(cap.r))
        @test Jutul.value(st[:SurfactantPartition][3,1]) ≈ .125
        @test closemass(Jutul.value(st[:TracerMasses][1,1]),.4875)
        for a in (nothing,LinearSurfactantAdsorption(0),LangmuirSurfactantAdsorption(0,2),
                LangmuirSurfactantAdsorption(1,0),TabulatedSurfactantAdsorption([0,1],[0,0]))
            b=model_with(a;order=order)
            @test b.t.adsorption === nothing
            @test !haskey(b.r.parameters,:SurfactantRockMass)
            @test Jutul.get_dependencies(b.t,b.r) == ()
            @test tracer_total_mass_outer(b.t,b.r,(;),.001,750.0,.25,1,1) == .1875
        end
    end
    b=model_with(first(isotherms()))
    c=Jutul.get_ad_entity_scalar(.001,1,1;tag=Cells())
    tiny=JutulDarcy.Tracers.TRACER_TOL/2
    st=(SurfactantPartition=reshape([.001,0,0],3,1),PhaseMassDensities=reshape([tiny,tiny],2,1),SurfactantRockMass=[1800.0])
    m=tracer_total_mass_outer(b.t,b.r,st,c,tiny,.25,1,1)
    @test Jutul.value(m) ≈ 1800*.0002*.001*tiny rtol=1e-12
    @test FD.partials(m)[1] == .25
end
end

function test_ad()
@testset "Concentration and pressure derivatives through simulator storage" begin
    for order in ((LiquidPhase(),VaporPhase()),(VaporPhase(),LiquidPhase())), ratio in (:kd,:K),
            branch in (:free,:cap,:floor), (kind,a) in enumerate(isotherms())
        floor=branch==:floor; sl=floor ? 1e-12 : .5; c0=floor ? 1e-12 : .001
        cap=branch==:cap ? .0009 : floor ? .1 : Inf
        b=model_with(a;order=order,kd=floor ? 0 : .5,K=ratio==:K ? (floor ? 0 : .25) : nothing,L=cap,compressible_volume=true)
        pp=setup_parameters(b.r); st=local_state(b,pp;c=c0,sl=sl)
        mr=1800.0; ref=hand_inventory(c0,1e7,sl,floor ? :floor : ratio,cap,kind;variable_volume=true,MR=mr)
        @test closemass(Jutul.value(st[:TracerMasses][1,1]),ref.mass)
        if branch==:free
            @test Jutul.value(st[:SurfactantPartition][1,1]) < cap
        elseif branch==:cap
            @test c0*750/(500+.5*250) > cap
            @test Jutul.value(st[:SurfactantPartition][1,1]) == cap
        else
            @test sl*1000 < 1e-8*(sl*1000+(1-sl)*500)
            @test Jutul.value(st[:SurfactantPartition][1,1]) < cap
        end
        for (sym,norm,analytic,hs) in ((:TracerConcentrations,c0,ref.dc,(1e-5,5e-6)),(:Pressure,1e7,ref.dp,(1e-6,5e-7)))
            pv=b.r.primary_variables[sym]; scale=Jutul.variable_scale(pv); scale=scale===nothing ? 1.0 : scale
            entry=sym==:Pressure ? st[sym][1] : st[sym][1,1]
            i=findfirst(!iszero,FD.partials(entry).values)
            @test FD.partials(entry)[i] == scale
            ad=FD.partials(st[:TracerMasses][1,1])[i]/scale*norm
            Mscale=max(abs(ref.mass),1e-12)
            @test abs(ad-analytic*norm) <= 1e-10*Mscale+1e-6*abs(analytic*norm)
            for h in hs
                plus=local_state(b,pp;c=c0*(sym==:TracerConcentrations ? 1+h : 1),sl=sl,p=1e7*(sym==:Pressure ? 1+h : 1))
                minus=local_state(b,pp;c=c0*(sym==:TracerConcentrations ? 1-h : 1),sl=sl,p=1e7*(sym==:Pressure ? 1-h : 1))
                fd=(Jutul.value(plus[:TracerMasses][1,1])-Jutul.value(minus[:TracerMasses][1,1]))/(2h)
                @test abs(ad-fd) <= 1e-10*Mscale+1e-6*abs(fd)
            end
        end
        @test all(isapprox.(1800,Jutul.value(st[:SurfactantRockMass]);rtol=1e-12))
    end
end
end

function test_flux()
@testset "Adsorbed inventory stays outside dissolved fluxes" begin
    for order in ((LiquidPhase(),VaporPhase()),(VaporPhase(),LiquidPhase()))
        b=model_with(first(isotherms());order=order,D=(liquid=1e-7,gas=5e-7),wells=true)
        pp=setup_parameters(b.r); st=local_state(b,pp)
        P=[.001 .002 .003; .0005 .001 .0015; .2 .3 .4]
        s=(SurfactantPartition=P,TracerConcentrations=fill(.003,1,3),
            SurfactantRockMass=pp[:SurfactantRockMass],Saturations=fill(.5,2,3),
            PhaseMassDensities=Jutul.value(st[:PhaseMassDensities]),SurfactantDiffusivities=pp[:SurfactantDiffusivities])
        q=zeros(2);q[b.liquid]=2;q[b.gas]=-3
        hand=2*.001-3*.001
        for MR in (pp[:SurfactantRockMass],10pp[:SurfactantRockMass])
            ss=(s...,SurfactantRockMass=MR)
            @test tracer_face_flux(SurfactantTracer(b.r.system;k_d=.5,adsorption=first(isotherms())),b.r,ss,q,1,SPU(1,2),TPFA(1,2,1),1) ≈ hand rtol=1e-12
            hd=hand-.25e-7*.5*1000*.001-1.25e-7*.5*500*.0005
            @test tracer_face_flux(b.t,b.r,ss,q,1,SPU(1,2),TPFA(1,2,1),1) ≈ hd rtol=1e-12
            for ph in 1:2
                expected=ph==b.liquid ? .001 : .0005
                @test perforation_production_concentration(b.t,b.r,ss,1,1,ph) == expected
            end
        end
        for gad in (false,true)
            m=model_with(first(isotherms());order=order,D=b.t.D,general_ad=gad)
            eq=m.r.equations[:tracers];disc=m.r.domain.discretizations.mass_flow
            massmob=copy(s.PhaseMassDensities)
            massmob[b.liquid,:].*=2;massmob[b.gas,:].*=5
            sf=(s...,Pressure=[2e7,1e7,1e7],PhaseMassMobilities=massmob,
                Transmissibilities=fill(1e-14,2),TwoPointGravityDifference=zeros(2))
            expected=1e-7*(massmob[b.liquid,1]*.001+massmob[b.gas,1]*.0005)-.25e-7*.5*1000*.001-1.25e-7*.5*500*.0005
            for MR in (1800.0,18000.0)
                ss=(sf...,SurfactantRockMass=fill(MR,3))
                Q=Jutul.StaticArrays.SVector(0.0)
                got=gad ? Jutul.face_flux!(Q,1,eq,ss,m.r,1.0,disc,(face_disc=f->(TPFA(1,2,1),SPU(1,2)),)) :
                    Jutul.face_flux!(Q,1,2,1,1,eq,ss,m.r,1.0,disc)
                @test isapprox(only(got),expected;rtol=1e-12,atol=0)
            end
        end
        for pw in (1.9e7,2.1e7), scaleMR in (1.0,10.0)
            sr=(Pressure=[2e7],PhaseMobilities=reshape([2.0,5.0],2,1),
                PhaseMassDensities=reshape([1000.0,500.0],2,1),
                TracerConcentrations=fill(.003,1,1),SurfactantPartition=P[:,1:1],
                SurfactantRockMass=[1800scaleMR])
            sw=(Pressure=[pw],WellIndices=[1e-12],PerforationGravityDifference=[0.0],
                TracerConcentrations=fill(.004,1,1),Saturations=reshape([.5,.5],2,1),
                PhaseMassDensities=reshape([1000.0,500.0],2,1))
            conn=(dp=pw-2e7,WI=1e-12,gdz=0.0,well=1,perforation=1,reservoir=1)
            qperf=[JutulDarcy.perforation_phase_mass_flux(7.0,conn,sr,sw,ph) for ph in 1:2]
            expected=sum(qperf[ph]*(qperf[ph]>0 ? (ph==b.liquid ? .001 : .0005) : .004) for ph in 1:2)
            out=zeros(1)
            Jutul.update_cross_term_in_entity!(out,1,sr,sr,sw,sw,b.r,b.model[:Injector],ReservoirFromWellTracerCT([1],[1]),b.r.equations[:tracers],1.0)
            @test isapprox(only(out),expected;rtol=1e-12,atol=0)
        end
        zero=model_with(first(isotherms());order=order,kd=0,L=.1)
        zs=(s...,SurfactantPartition=[.001 .002 .003; zeros(1,3); .2 .3 .4])
        @test perforation_production_concentration(zero.t,zero.r,zs,1,1,b.gas) == 0
    end
end
end

# The transport diagnostics (Fritis drivers, tracer block correction) and the
# closed-pressure report live in the validation repository, foam/tests/.
function run_tests()
    group=get(ENV,"FOAM_ADSORPTION_TEST_GROUP","all")
    for (name,f) in (("isotherms",test_isotherms),("rock",test_rock),("inventory",test_inventory),("ad",test_ad),("flux",test_flux))
        group in ("all",name) && f()
    end
end
end
SurfactantAdsorptionTests.run_tests()
