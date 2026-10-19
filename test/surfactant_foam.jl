using JutulDarcy, Jutul, Test
import JutulDarcy.Tracers: surfactant_partition, partition_mass_ratio, MassPartitionRatio,
    VolumePartitionRatio, SurfactantPartition, update_surfactant_partition!,
    SURFACTANT_PARTITION_FLOOR

const FD = Jutul.ForwardDiff
const εB = SURFACTANT_PARTITION_FLOOR

# Expected values below come from the closure equations
#   m_l C_l + m_g C_g + E = M,  C_g = k_d C_l,  0 <= C_l <= L,  E >= 0,  E (L - C_l) = 0
# evaluated by hand for each case, not from the branch under test.
close_kgkg(x, ref) = abs(x - ref) <= 1e-12*abs(ref) + 1e-15

function central_difference(f, x, i; rel = 1e-6)
    h = rel*abs(x[i])
    xp = copy(x); xp[i] += h
    xm = copy(x); xm[i] -= h
    return (f(xp) .- f(xm)) ./ (2h)
end

rel_close(a, b; tol = 1e-6) = all(abs.(a .- b) .<= tol .* max.(abs.(b), 1e-300))

@testset "Local partition closure" begin
    ρ = (956.2875239261082, 543.1990853377915) # Zeng (2016) case, liquid and CO2
    @testset "positive k_d, no cap" begin
        c = 1e-3
        for kd in (0.088, 1.76, 7.0), sl in (1.0, 0.7, 0.3, 0.05)
            ml, mg = sl*ρ[1], (1 - sl)*ρ[2]
            M = c*(ml + mg)
            Cl, Cg, E = surfactant_partition(c, ml, mg, kd, Inf)
            @test E == 0
            @test close_kgkg(Cg, kd*Cl)
            @test abs(ml*Cl + mg*Cg - M) <= 1e-12*M
        end
    end
    @testset "zero k_d, capped liquid" begin
        ml, mg, c, L = 10.0, 100.0, 0.05, 0.1
        M = c*(ml + mg)
        @test M > ml*L # the cap is active
        Cl, Cg, E = surfactant_partition(c, ml, mg, 0.0, L)
        @test Cl == L
        @test Cg == 0
        @test abs(E - (M - ml*L)) <= 1e-12*M
    end
    @testset "positive k_d, capped liquid" begin
        # The gas holds k_d L as well, so the deposit is M - (m_l + k_d m_g) L
        # = 5.5 - (10 + 20) 0.1 = 2.5, not M - m_l L = 4.5.
        ml, mg, kd, c, L = 10.0, 100.0, 0.2, 0.05, 0.1
        Cl, Cg, E = surfactant_partition(c, ml, mg, kd, L)
        @test Cl == L
        @test close_kgkg(Cg, 0.02)
        @test abs(E - 2.5) <= 1e-12*c*(ml + mg)
    end
    @testset "zero k_d, dry cell" begin
        mg, L = 100.0, 0.1
        @test surfactant_partition(0.0, 0.0, mg, 0.0, L) == (0.0, 0.0, 0.0)
        c = 1e-3
        Cl, Cg, E = surfactant_partition(c, 0.0, mg, 0.0, L)
        @test (Cl, Cg) == (L, 0.0)
        @test abs(E - c*mg) <= 1e-12*c*mg
        c = 1e-12 # below floor * L: liquid concentration rises linearly with c
        Cl, Cg, E = surfactant_partition(c, 0.0, mg, 0.0, L)
        @test close_kgkg(Cl, c/εB)
        @test Cg == 0
        @test abs(E - c*mg) <= 1e-12*c*mg
        @test all(isfinite, (Cl, Cg, E))
    end
    @testset "no gas" begin
        kd, ml, c = 1.76, 500.0, 1e-3
        Cl, Cg, E = surfactant_partition(c, ml, 0.0, kd, Inf)
        @test close_kgkg(Cl, c)
        @test close_kgkg(Cg, kd*c)
        @test E == 0
    end
    @testset "zero k_d, nearly dry cell below the cap" begin
        ml, mg, L = 2e-9, 100.0, 0.1
        mf = ml + mg
        for c in (1e-13, 5e-10, 9e-10)
            M = c*mf
            Cl, Cg, E = surfactant_partition(c, ml, mg, 0.0, L)
            @test close_kgkg(Cl, M/(εB*mf))
            @test Cl < L
            @test abs(E - (M - ml*Cl)) <= 1e-12*M
            @test 0 < E <= εB*mf*L
        end
    end
    @testset "small positive k_d is not floored" begin
        mg, c = 500.0, 1e-3
        for kd in (1e-10, 5e-9)
            Cl, Cg, E = surfactant_partition(c, 0.0, mg, kd, Inf)
            @test E == 0
            @test close_kgkg(Cl, c/kd)
        end
    end
    @testset "no rounding deposit" begin
        # M - B (M / B) is 1.1e-16 here; the uncapped branch must give exactly 0.
        ml, mg, kd, c = 927.9845455248775, 333.3647836659588, 1.76, 6.813526022852874e-4
        M = c*(ml + mg)
        B = ml + kd*mg
        @test M - B*(M/B) != 0
        @test surfactant_partition(c, ml, mg, kd, Inf)[3] == 0
    end
end

@testset "Partition coefficient input" begin
    K, ρl, ρg = 0.5, 1000.0, 400.0
    @test abs(partition_mass_ratio(VolumePartitionRatio(K), ρl, ρg) - K*ρl/ρg) <= 1e-14*K*ρl/ρg
    @test partition_mass_ratio(VolumePartitionRatio(K), ρl, 2ρg) != partition_mass_ratio(VolumePartitionRatio(K), ρl, ρg)
    @test partition_mass_ratio(MassPartitionRatio(1.3), ρl, ρg) == partition_mass_ratio(MassPartitionRatio(1.3), ρl, 2ρg)
end

function surfactant_test_model(; n = 3, tracers = nothing, wells = false, foam = FmFactor(), install = true, kwarg...)
    mesh = UnstructuredMesh(CartesianMesh((n, 1, 1), (n*1.0, 1.0, 1.0)))
    domain = reservoir_domain(mesh; permeability = 1e-12, porosity = 0.25)
    sys = ImmiscibleSystem((LiquidPhase(), VaporPhase()); reference_densities = [1000.0, 500.0])
    w = wells ? [setup_well(domain, 1; name = :Injector, simple_well = true)] : []
    model = setup_reservoir_model(domain, sys; wells = w)
    if isnothing(tracers)
        tracers = SurfactantTracer(sys; kwarg...)
    end
    add_tracers_to_model!(model, tracers)
    install && set_surfactant_model!(model; foam)
    return model
end

@testset "Volume partition coefficient follows density derivatives" begin
    # Both densities depend on pressure with different compressibilities, so
    # k_d = K ρ_l / ρ_g changes with pressure; dropping that dependence would
    # make the derivatives below disagree with the difference quotient. The
    # pressure derivative is seeded with a Jutul cell tag, as in a simulation,
    # so that taking Jutul.value of the densities would remove it.
    K, c = 0.5, 1e-3
    model = reservoir_model(surfactant_test_model(n = 1, K = K))
    def = SurfactantPartition(1, only(model.equations[:tracers].flux_type.tracers))
    @test Jutul.subvariable(def, Jutul.FiniteVolumeGlobalMap([1], Int[])) === def
    ρ(p) = [1000.0*exp(4.5e-10*(p - 1e7)), 500.0*exp(1e-8*(p - 1e7))]
    function phase_fractions(p)
        dens = reshape(ρ(p), 2, 1)
        vals = zeros(eltype(dens), 3, 1)
        update_surfactant_partition!(vals, def, model, fill(c, 1, 1), fill(0.5, 2, 1), dens, 1:1)
        return [vals[1, 1], vals[2, 1]]
    end
    p0 = 2e7
    p_ad = Jutul.get_ad_entity_scalar(p0, 1, 1; tag = Cells())
    ad = [FD.partials(v)[1] for v in phase_fractions(p_ad)]
    fd = central_difference(x -> phase_fractions(x[1]), [p0], 1)
    @test rel_close(ad, fd)
    @test all(!iszero, ad)
end

@testset "Partition derivatives" begin
    f(kd, L) = x -> collect(surfactant_partition(x[1], x[2], x[3], kd, L))
    for kd in (0.088, 1.76, 7.0)
        x = [1e-3, 500.0, 200.0] # c, m_l, m_g
        J = FD.jacobian(f(kd, Inf), x)
        for i in 1:3
            @test rel_close(J[1:2, i], central_difference(f(kd, Inf), x, i)[1:2])
        end
        c, ml, mg = x
        B = ml + kd*mg
        dCl_dml = -c*mg*(1 - kd)/B^2
        @test abs(J[1, 2] - dCl_dml) <= 1e-12*abs(dCl_dml)
        @test sign(J[1, 2]) == (kd < 1 ? -1 : 1)
    end
    # Capped branch: C_l is constant and dE = dM - L dB, B = m_l + k_d m_g.
    x = [0.05, 10.0, 100.0]
    L = 0.1
    c, ml, mg = x
    for kd in (0.0, 0.2)
        J = FD.jacobian(f(kd, L), x)
        @test all(==(0), J[1, :])
        @test all(==(0), J[2, :])
        dE = [ml + mg, c - L, c - kd*L]
        @test all(abs.(J[3, :] .- dE) .<= 1e-12 .* abs.(dE))
    end
end

@testset "Surfactant model installation" begin
    model = surfactant_test_model(k_d = 1.76, wells = true)
    rmodel = reservoir_model(model)
    Jutul.sort_secondary_variables!(rmodel)
    order = collect(keys(rmodel.secondary_variables))
    pos(k) = findfirst(==(k), order)
    @test pos(:SurfactantPartition) < pos(:ClVolumetricConcentration) < pos(:FmFactor) < pos(:PhaseViscosities)
    @test !haskey(model[:Injector].secondary_variables, :SurfactantPartition)
    @test !haskey(rmodel.parameters, :SurfactantDiffusivities)
    diffusive = surfactant_test_model(k_d = 1.76, wells = true, D = (gas = 5e-7, liquid = 1e-7))
    rdiff = reservoir_model(diffusive)
    @test haskey(rdiff.parameters, :SurfactantDiffusivities)
    @test !haskey(diffusive[:Injector].parameters, :SurfactantDiffusivities)
    defaults = setup_parameters(diffusive)[:Reservoir][:SurfactantDiffusivities]
    @test size(defaults) == (2, Jutul.number_of_faces(rdiff.domain))
    T = [1e-9*(phase + face) for phase in 1:2, face in axes(defaults, 2)]
    @test T != defaults
    @test setup_parameters(diffusive, Reservoir = Dict(:SurfactantDiffusivities => T))[:Reservoir][:SurfactantDiffusivities] == T
    @test setup_parameters(rdiff, SurfactantDiffusivities = T)[:SurfactantDiffusivities] == T
    @test !haskey(rdiff.parameters, :Diffusivities)


    c = [0.0, 1e-3, 2e-3]
    sl = [1.0, 0.6, 0.2]
    state0 = setup_state(rmodel; Pressure = 1e7, Saturations = [sl'; (1 .- sl)'], TracerConcentrations = reshape(c, 1, :))
    prm = setup_parameters(rmodel)
    state = Jutul.evaluate_all_secondary_variables(rmodel, state0, prm)
    P = state[:SurfactantPartition]
    for i in 1:3
        ρl, ρg = state[:PhaseMassDensities][:, i]
        ml, mg = sl[i]*ρl, (1 - sl[i])*ρg
        Cl = c[i]*(ml + mg)/(ml + 1.76*mg)
        @test close_kgkg(P[1, i], Cl)
        @test close_kgkg(P[2, i], 1.76*Cl)
        @test P[3, i] == 0
        @test abs(state[:ClVolumetricConcentration][i] - ρl*Cl) <= 1e-12*ρl*Cl
    end
    # Without surfactant there is no foam: unit foam factor and base gas viscosity.
    state0 = setup_state(rmodel; Pressure = 1e7, Saturations = [0.5, 0.5], TracerConcentrations = 0.0)
    state = Jutul.evaluate_all_secondary_variables(rmodel, state0, prm)
    @test all(==(1), state[:FmFactor])
    @test state[:PhaseViscosities] == prm[:BasePhaseViscosities]

    sys = rmodel.system
    @test_throws ArgumentError surfactant_test_model(tracers = [SurfactantTracer(sys; k_d = 1.0), SurfactantTracer(sys; k_d = 2.0)])
    mixed = surfactant_test_model(tracers = [MultiPhaseTracer(sys, [1, 2]), SurfactantTracer(sys; k_d = 1.0)])
    rmixed = reservoir_model(mixed)
    state0 = setup_state(rmixed; Pressure = 1e7, Saturations = [0.5, 0.5], TracerConcentrations = [0.3, 1e-3])
    state = Jutul.evaluate_all_secondary_variables(rmixed, state0, setup_parameters(rmixed))
    @test all(isfinite, state[:SurfactantPartition])
    @test state[:SurfactantPartition][1, 1] > 0
end

@testset "SurfactantTracer arguments" begin
    sys = ImmiscibleSystem((LiquidPhase(), VaporPhase()))
    @test SurfactantTracer(sys; k_d = 1.0).D === nothing
    @test SurfactantTracer(sys; k_d = 1.0, D = (gas = 2, liquid = 1)).D === (gas = 2.0, liquid = 1.0)
    @test SurfactantTracer(sys; k_d = 1.0, D = (liquid = 1, gas = 2)).D === (gas = 2.0, liquid = 1.0)
    @test SurfactantTracer(sys; k_d = 1.0, D = (gas = 0, liquid = 0)).D === (gas = 0.0, liquid = 0.0)
    for D in (1.0, (gas = 1.0,), (gas = 1.0, liquid = 1.0, other = 1.0),
            (gas = 1.0, water = 1.0))
        @test_throws ArgumentError SurfactantTracer(sys; k_d = 1.0, D = D)
    end
    for value in (-1.0, NaN, Inf, 1 + im, "1", [1.0], big"1e1000")
        @test_throws ArgumentError SurfactantTracer(sys; k_d = 1.0, D = (gas = value, liquid = 1.0))
        @test_throws ArgumentError SurfactantTracer(sys; k_d = 1.0, D = (gas = 1.0, liquid = value))
    end
    @test_throws ArgumentError SurfactantTracer(sys)
    @test_throws ArgumentError SurfactantTracer(sys; k_d = 1.0, K = 1.0)
    @test_throws ArgumentError SurfactantTracer(sys; k_d = -1.0)
    @test_throws ArgumentError SurfactantTracer(sys; K = -1.0)
    @test_throws ArgumentError SurfactantTracer(sys; k_d = Inf)
    @test_throws ArgumentError SurfactantTracer(sys; k_d = 1.0, L = 0.0)
    @test_throws ArgumentError SurfactantTracer(sys; k_d = 1.0, L = -0.1)
    @test_throws ArgumentError SurfactantTracer(sys; k_d = 0.0, L = Inf)
    @test_throws ArgumentError SurfactantTracer(sys; K = 0.0, L = Inf)
    @test_throws ArgumentError SurfactantTracer(sys, [1]; k_d = 1.0)
    @test_throws ArgumentError SurfactantTracer(ImmiscibleSystem((LiquidPhase(), AqueousPhase())); k_d = 1.0)
    for kw in ((Concentration_limit = "Liquid",), (C_max = 0.1,))
        err = try
            SurfactantTracer(sys; k_d = 1.0, kw...)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("L", err.msg)
    end
    # Boundary inputs of the shared scalar check: negative zero (kept as
    # -0.0) and Bool are accepted, a negative subnormal and a BigFloat that
    # only rounds to -0.0 are rejected on the value as given.
    @test SurfactantTracer(sys; k_d = 1.0, D = (gas = -0.0, liquid = true)).D == (gas = 0.0, liquid = 1.0)
    for value in (prevfloat(0.0), big"-1e-400")
        @test_throws ArgumentError SurfactantTracer(sys; k_d = 1.0, D = (gas = value, liquid = 1.0))
        @test_throws ArgumentError LinearSurfactantAdsorption(value)
        @test_throws ArgumentError FmFactor(epdry = value)
    end
    @test FmFactor(epdry = -0.0).epdry == 0.0
    @test LinearSurfactantAdsorption(true).kappa === 1.0
    @test SurfactantTracer(sys; k_d = 0.0).L == 0.1
    @test SurfactantTracer(sys; k_d = 1.0).L == Inf
    @test SurfactantTracer(sys; K = 2.0, L = 0.05).L == 0.05
end

# Surfactant factor as in JutulDarcy 507d81ce, src/Tracers/tracer_surfactant.jl,
# before the linear segment for epsurf < 1 was added. The branch compares the
# value, as ForwardDiff 0.10 did then; ForwardDiff 1 would also compare the
# partials and take the constant branch at C_sw = fmsurf.
function reference_F_surfactant(C_sw, fmsurf, epsurf)
    if FD.value(C_sw) <= fmsurf
        return (C_sw/fmsurf)^epsurf
    else
        return 1.0
    end
end

@testset "Foam surfactant factor" begin
    F = JutulDarcy.Tracers.foam_surfactant_factor
    dF(f, C, fmsurf, epsurf) = FD.derivative(x -> f(x, fmsurf, epsurf), C)
    fmsurf = 2.0
    @testset "unchanged for epsurf = 1" begin
        # Includes C_sw = fmsurf, where both versions use the power law.
        for C in (0.0, 1e-9, 1e-6*fmsurf, 0.3, fmsurf, 2.5)
            @test abs(F(C, fmsurf, 1.0) - reference_F_surfactant(C, fmsurf, 1.0)) <= 1e-14
            @test abs(dF(F, C, fmsurf, 1.0) - dF(reference_F_surfactant, C, fmsurf, 1.0)) <= 1e-14/fmsurf
        end
    end
    @testset "finite slope at zero for epsurf < 1" begin
        epsurf = 0.196
        @test isinf(dF(reference_F_surfactant, 0.0, fmsurf, epsurf))
        @test isfinite(dF(F, 0.0, fmsurf, epsurf))
        @test F(0.0, fmsurf, epsurf) == 0
        C_eps = 1e-6*fmsurf
        below, at = F(prevfloat(C_eps), fmsurf, epsurf), F(C_eps, fmsurf, epsurf)
        @test abs(below - at) <= 1e-12*at
        @test abs(at - (C_eps/fmsurf)^epsurf) <= 1e-14*at
        # Inside the segment the factor is the chord: linear in C_sw.
        slope = (C_eps/fmsurf)^epsurf/C_eps
        @test abs(F(C_eps/2, fmsurf, epsurf) - at/2) <= 1e-14*at
        @test abs(dF(F, C_eps/2, fmsurf, epsurf) - slope) <= 1e-12*slope
        @test abs(F(0.5, fmsurf, epsurf) - (0.5/fmsurf)^epsurf) <= 1e-14
    end
end

import JutulDarcy.Tracers: perforation_production_concentration, ReservoirFromWellTracerCT

function producer_test_model(tracers; n = 1, perforation = n)
    mesh = UnstructuredMesh(CartesianMesh((n, 1, 1), (n*1.0, 1.0, 1.0)))
    domain = reservoir_domain(mesh; permeability = 1e-12, porosity = 0.25)
    sys = ImmiscibleSystem((LiquidPhase(), VaporPhase()); reference_densities = [1000.0, 500.0])
    well = setup_well(domain, perforation; name = :Producer, simple_well = true)
    model = setup_reservoir_model(domain, sys; wells = [well])
    add_tracers_to_model!(model, tracers isa Function ? tracers(sys) : tracers)
    set_surfactant_model!(model)
    return model
end

@testset "Production perforation carries phase concentrations" begin
    @testset "hand-computed examples" begin
        # Phase-wise outflow q_l C_l + q_g k_d C_l, compared with (q_l + q_g) c
        # that the perforation produced before.
        model = reservoir_model(producer_test_model(sys -> SurfactantTracer(sys; k_d = 1.0)))
        for (ml, mg, kd, ql, qg, c, expected) in (
                (100.0, 10.0, 10.0, 10.0, 100.0, 1.0, 555.5),
                (10.0, 100.0, 0.02, 1.0, 100.0, 1e-3, 0.0275))
            Cl, Cg, E = surfactant_partition(c, ml, mg, kd, Inf)
            state = (SurfactantPartition = reshape([Cl, Cg, E], 3, 1), TracerConcentrations = fill(c, 1, 1))
            C_liq = perforation_production_concentration(model.equations[:tracers].flux_type.tracers[1], model, state, 1, 1, 1)
            C_gas = perforation_production_concentration(model.equations[:tracers].flux_type.tracers[1], model, state, 1, 1, 2)
            @test abs(ql*C_liq + qg*C_gas - expected) <= 1e-12*expected
            @test abs((ql + qg)*c - expected) > 0.1*expected
        end
    end

    function perforation_tracer_flux(model, state_res, state_well)
        rmodel, wmodel = model[:Reservoir], model[:Producer]
        out = zeros(promote_type(eltype(state_res.Pressure), eltype(state_well.Pressure),
                eltype(state_well.Saturations)),
            number_of_tracers(rmodel.equations[:tracers].flux_type))
        ct = ReservoirFromWellTracerCT([1], [1])
        Jutul.update_cross_term_in_entity!(out, 1, state_res, state_res, state_well, state_well,
            rmodel, wmodel, ct, rmodel.equations[:tracers], 1.0)
        return out
    end
    function phase_fluxes(state_res, state_well)
        conn = (dp = state_well.Pressure[1] - state_res.Pressure[1], WI = state_well.WellIndices[1],
            gdz = 0.0, well = 1, perforation = 1, reservoir = 1)
        λ_t = sum(state_res.PhaseMobilities[:, 1])
        return [JutulDarcy.perforation_phase_mass_flux(λ_t, conn, state_res, state_well, ph) for ph in 1:2]
    end
    well_state(p, c) = (WellIndices = [1e-12], PerforationGravityDifference = [0.0], Pressure = [p],
        TracerConcentrations = fill(c, 1, 1), Saturations = reshape([0.3, 0.7], 2, 1),
        PhaseMassDensities = reshape([1000.0, 500.0], 2, 1))

    @testset "cross term" begin
        ρ, λ, c, c_well = [1000.0, 500.0], [2.0, 5.0], 1e-3, 3e-3
        reservoir_state(P) = (Pressure = [2e7], PhaseMobilities = reshape(λ, 2, 1),
            PhaseMassDensities = reshape(ρ, 2, 1), SurfactantPartition = reshape(P, 3, 1),
            TracerConcentrations = fill(c, 1, 1))
        model = producer_test_model(sys -> SurfactantTracer(sys; k_d = 1.76))
        ml, mg = 0.6ρ[1], 0.4ρ[2]
        P = collect(surfactant_partition(c, ml, mg, 1.76, Inf))
        sr, sw = reservoir_state(P), well_state(1.9e7, c_well)
        q = phase_fluxes(sr, sw)
        @test all(>(0), q) # producing both phases
        @test abs(only(perforation_tracer_flux(model, sr, sw)) - (q[1]*P[1] + q[2]*P[2])) <= 1e-12*abs(q[1]*P[1] + q[2]*P[2])
        # Injection takes the well-node concentration.
        sw_inj = well_state(2.1e7, c_well)
        q = phase_fluxes(sr, sw_inj)
        @test all(<(0), q)
        @test abs(only(perforation_tracer_flux(model, sr, sw_inj)) - sum(q)*c_well) <= 1e-12*abs(sum(q)*c_well)
        # k_d = 0, dry cell with deposit, producing gas only: nothing leaves.
        model0 = producer_test_model(sys -> SurfactantTracer(sys; k_d = 0.0))
        P0 = collect(surfactant_partition(c, 0.0, ρ[2], 0.0, 0.1))
        @test P0[3] > 0
        sr0 = (reservoir_state(P0)..., PhaseMobilities = reshape([0.0, 5.0], 2, 1))
        q0 = phase_fluxes(sr0, sw)
        @test q0[1] == 0 && q0[2] > 0
        @test only(perforation_tracer_flux(model0, sr0, sw)) == 0
        # An ordinary multiphase tracer keeps producing at the cell concentration.
        modelm = producer_test_model(sys -> [MultiPhaseTracer(sys, [1, 2]), SurfactantTracer(sys; k_d = 1.76)])
        srm = (reservoir_state(P)..., TracerConcentrations = reshape([0.3, c], 2, 1))
        swm = (well_state(1.9e7, c_well)..., TracerConcentrations = reshape([0.0, c_well], 2, 1))
        q = phase_fluxes(srm, swm)
        out = perforation_tracer_flux(modelm, srm, swm)
        @test abs(out[1] - sum(q)*0.3) <= 1e-12*abs(sum(q)*0.3)
        @test abs(out[2] - (q[1]*P[1] + q[2]*P[2])) <= 1e-12*abs(q[1]*P[1] + q[2]*P[2])
    end

    @testset "derivatives through the simulator storage" begin
        # Evaluate the reservoir state with Jutul's AD on a 3-cell model and
        # compare the perforation tracer flux derivatives with respect to the
        # perforated cell's c and liquid saturation with central differences.
        # Liquid is the first phase, so S_l is the independent saturation and
        # S_g = 1 - S_l moves with it.
        model = producer_test_model(sys -> SurfactantTracer(sys; k_d = 1.76); n = 3, perforation = 3)
        prm = setup_parameters(model)
        forces = setup_reservoir_forces(model; control = Dict(:Producer => ProducerControl(BottomHolePressureTarget(1.9e7))))
        function state0_with(dc, ds)
            s0 = setup_reservoir_state(model; Pressure = 2e7, Saturations = [0.6, 0.4], TracerConcentrations = 1e-3)
            s0[:Reservoir][:TracerConcentrations][1, 3] += dc
            s0[:Reservoir][:Saturations][1, 3] += ds
            s0[:Reservoir][:Saturations][2, 3] -= ds
            s0[:Producer][:Pressure] .= 1.9e7
            return s0
        end
        function flux_at(s0)
            sim = Simulator(model; state0 = s0, parameters = prm)
            Jutul.update_state_dependents!(sim.storage, model, 1.0, forces)
            st_res = sim.storage[:Reservoir][:state]
            st_w = sim.storage[:Producer][:state]
            w = (WellIndices = Jutul.value(st_w[:WellIndices])[end:end],
                PerforationGravityDifference = [0.0], Pressure = Jutul.value(st_w[:Pressure]),
                TracerConcentrations = Jutul.value(st_w[:TracerConcentrations]),
                Saturations = Jutul.value(st_w[:Saturations]), PhaseMassDensities = Jutul.value(st_w[:PhaseMassDensities]))
            r = (Pressure = st_res[:Pressure][3:3], PhaseMobilities = st_res[:PhaseMobilities][:, 3:3],
                PhaseMassDensities = st_res[:PhaseMassDensities][:, 3:3],
                SurfactantPartition = st_res[:SurfactantPartition][:, 3:3],
                TracerConcentrations = st_res[:TracerConcentrations][:, 3:3])
            return only(perforation_tracer_flux(model, r, w)), st_res
        end
        v, st = flux_at(state0_with(0.0, 0.0))
        ic = findfirst(!iszero, FD.partials(st[:TracerConcentrations][1, 3]).values)
        is = findfirst(!iszero, FD.partials(st[:Saturations][1, 3]).values)
        @test Jutul.value(v) > 0
        for (k, dc, ds) in ((ic, 1e-9, 0.0), (is, 0.0, 1e-6))
            fd = (Jutul.value(flux_at(state0_with(dc, ds))[1]) - Jutul.value(flux_at(state0_with(-dc, -ds))[1]))/(2*(dc + ds))
            ad = FD.partials(v)[k]
            @test abs(ad - fd) <= 1e-6*abs(fd)
        end
    end

    @testset "balanced perforation" begin
        # Well and reservoir at the same pressure: every phase flux is zero, so
        # the injection/production branch is a tie. perforation_phase_mass_flux
        # settles it on the value and returns the production flux, so the cross
        # term has to carry the reservoir concentrations. Branching on the dual
        # instead took the well concentration whenever the seeded derivative of
        # the flux was negative, pairing a production flux with an injection
        # concentration. Differentiating with respect to the well pressure gives
        # that negative seed, since dpsi/dp_well = -WI.
        model = producer_test_model(sys -> SurfactantTracer(sys; k_d = 1.76); n = 3, perforation = 3)
        prm = setup_parameters(model)
        forces = setup_reservoir_forces(model;
            control = Dict(:Producer => ProducerControl(BottomHolePressureTarget(2e7))))
        s0 = setup_reservoir_state(model; Pressure = 2e7, Saturations = [0.6, 0.4],
            TracerConcentrations = 1e-3)
        s0[:Producer][:Pressure] .= 2e7
        s0[:Producer][:TracerConcentrations] .= 3e-3
        sim = Simulator(model; state0 = s0, parameters = prm)
        Jutul.update_state_dependents!(sim.storage, model, 1.0, forces)
        st_res, st_w = sim.storage[:Reservoir][:state], sim.storage[:Producer][:state]
        WI = Jutul.value(st_w[:WellIndices])[end]
        # Only the well pressure keeps its derivatives, so the cross term's
        # partials are the perforation's dependence on p_well alone.
        w = (WellIndices = [WI], PerforationGravityDifference = [0.0],
            Pressure = st_w[:Pressure],
            TracerConcentrations = Jutul.value(st_w[:TracerConcentrations]),
            Saturations = Jutul.value(st_w[:Saturations]),
            PhaseMassDensities = Jutul.value(st_w[:PhaseMassDensities]))
        r = (Pressure = Jutul.value(st_res[:Pressure])[3:3],
            PhaseMobilities = Jutul.value(st_res[:PhaseMobilities])[:, 3:3],
            PhaseMassDensities = Jutul.value(st_res[:PhaseMassDensities])[:, 3:3],
            SurfactantPartition = Jutul.value(st_res[:SurfactantPartition])[:, 3:3],
            TracerConcentrations = Jutul.value(st_res[:TracerConcentrations])[:, 3:3])
        v = only(perforation_tracer_flux(model, r, w))
        k = findfirst(!iszero, FD.partials(w.Pressure[1]).values)
        @test count(!iszero, FD.partials(w.Pressure[1]).values) == 1
        # Jutul scales the pressure seed, so divide it out to get dv/dp_well.
        seed = FD.partials(w.Pressure[1])[k]
        @test seed > 0
        @test FD.value(v) == 0.0
        rhoL(ph) = r.PhaseMassDensities[ph, 1]*r.PhaseMobilities[ph, 1]
        d_prod = -WI*sum(r.SurfactantPartition[ph, 1]*rhoL(ph) for ph in 1:2)
        d_well = -WI*w.TracerConcentrations[1, 1]*sum(rhoL(ph) for ph in 1:2)
        @test abs(d_prod - d_well) > 0.1*abs(d_prod)
        @test abs(FD.partials(v, k)/seed - d_prod) <= 1e-12*abs(d_prod)
    end

    @testset "injecting perforation with an absent phase" begin
        # An injector carrying only liquid has s_w == 0 for the gas phase, so
        # that phase's mass flux is zero although the perforation is injecting.
        # Its derivative with respect to the well saturation is not zero, so the
        # concentration still matters: both phases have to take the injection
        # concentration, as perforation_phase_mass_flux took the injection
        # formula for both. Branching on the flux rather than on the potential
        # difference gave the absent phase a production concentration.
        model = producer_test_model(sys -> SurfactantTracer(sys; k_d = 1.76); n = 3, perforation = 3)
        prm = setup_parameters(model)
        forces = setup_reservoir_forces(model;
            control = Dict(:Producer => ProducerControl(BottomHolePressureTarget(2e7))))
        s0 = setup_reservoir_state(model; Pressure = 2e7, Saturations = [0.6, 0.4],
            TracerConcentrations = 1e-3)
        s0[:Producer][:Pressure] .= 2.1e7
        s0[:Producer][:Saturations][1, 1] = 1.0
        s0[:Producer][:Saturations][2, 1] = 0.0
        s0[:Producer][:TracerConcentrations] .= 3e-3
        sim = Simulator(model; state0 = s0, parameters = prm)
        Jutul.update_state_dependents!(sim.storage, model, 1.0, forces)
        st_res, st_w = sim.storage[:Reservoir][:state], sim.storage[:Producer][:state]
        WI = Jutul.value(st_w[:WellIndices])[end]
        # Only the well saturation keeps its derivatives this time. The well
        # node's two phases come out at the same mass density here, which would
        # make the correct injection derivative vanish identically, so the
        # densities are set apart to keep both answers away from zero.
        w = (WellIndices = [WI], PerforationGravityDifference = [0.0],
            Pressure = Jutul.value(st_w[:Pressure]),
            TracerConcentrations = Jutul.value(st_w[:TracerConcentrations]),
            Saturations = st_w[:Saturations],
            PhaseMassDensities = reshape([1000.0, 500.0], 2, 1))
        r = (Pressure = Jutul.value(st_res[:Pressure])[3:3],
            PhaseMobilities = Jutul.value(st_res[:PhaseMobilities])[:, 3:3],
            PhaseMassDensities = Jutul.value(st_res[:PhaseMassDensities])[:, 3:3],
            SurfactantPartition = Jutul.value(st_res[:SurfactantPartition])[:, 3:3],
            TracerConcentrations = Jutul.value(st_res[:TracerConcentrations])[:, 3:3])
        @test Jutul.value(w.Saturations[1, 1]) == 1.0 && Jutul.value(w.Saturations[2, 1]) == 0.0
        v = only(perforation_tracer_flux(model, r, w))
        k = findfirst(!iszero, FD.partials(w.Saturations[1, 1]).values)
        @test count(!iszero, FD.partials(w.Saturations[1, 1]).values) == 1
        ds(ph) = FD.partials(w.Saturations[ph, 1])[k]
        c_w = w.TracerConcentrations[1, 1]
        λ_t = sum(r.PhaseMobilities[:, 1])
        ψ = -WI*(w.Pressure[1] - r.Pressure[1])
        @test ψ < 0 # injecting
        # Injection: q_ph = s_w ρ_w ψ λ_t with the well concentration throughout.
        d_inj = c_w*ψ*λ_t*sum(w.PhaseMassDensities[ph, 1]*ds(ph) for ph in 1:2)
        # What branching on the flux gave: phase 2 has value(q) == -0.0, so it
        # fell through to the reservoir concentration.
        d_mixed = ψ*λ_t*(c_w*w.PhaseMassDensities[1, 1]*ds(1)
            + r.SurfactantPartition[2, 1]*w.PhaseMassDensities[2, 1]*ds(2))
        @test abs(d_inj - d_mixed) > 0.1*abs(d_mixed)
        # The value is the same either way, only the derivative separates them.
        v_expected = c_w*ψ*λ_t*Jutul.value(w.Saturations[1, 1])*w.PhaseMassDensities[1, 1]
        @test abs(FD.value(v) - v_expected) <= 1e-12*abs(v_expected)
        @test abs(FD.partials(v, k) - d_inj) <= 1e-12*abs(d_mixed)
    end
end

# Foam interface fixtures also support focused checks without transport runs.
function foam_interface_model(; n=3, phases=(LiquidPhase(), VaporPhase()),
        split_wells=false, mixed=false, install=true, foam=FmFactor(), kwarg...)
    mesh = UnstructuredMesh(CartesianMesh((n, 1, 1), (Float64(n), 1.0, 1.0)))
    domain = reservoir_domain(mesh; permeability=1e-12, porosity=.25, rock_density=2400.0)
    rho = [ph == VaporPhase() ? 500.0 : 1000.0 for ph in phases]
    sys = ImmiscibleSystem(phases; reference_densities=rho)
    wells = [setup_well(domain, 1; name=:Injector, simple_well=true),
        setup_well(domain, n; name=:Producer, simple_well=true)]
    model = setup_reservoir_model(domain, sys; wells, split_wells)
    surf = SurfactantTracer(sys; kwarg...)
    tracers = mixed ? [MultiPhaseTracer(sys, [1, 2]), surf] : surf
    add_tracers_to_model!(model, tracers)
    install && set_surfactant_model!(model; foam)
    return model
end

function foam_configuration_checks()
    @testset "Foam configuration and installation" begin
        for (name, value) in ((:epdry, -1.0), (:fmdry, 1.1), (:fmsurf, 0.0),
                (:epsurf, 0.0), (:fommb, [-1.0]), (:epdry, NaN),
                (:fmsurf, big"1e1000"), (:fommb, Float64[]),
                (:dry_correction, 1), (:residual_water, 1.1))
            err = try FmFactor(; (name=>value,)...); catch e; e; end
            @test err isa ArgumentError
            @test occursin(string(name), sprint(showerror, err))
        end
        @test_throws ArgumentError FmFactor(dry_correction=true)
        @test FmFactor(0, 1, 2, 1, 0).fommb === 0.0
        @test FmFactor(fmdry=0).fmdry === 0.0
        input = [0.0, 100.0, 500.0]
        foam = FmFactor(fommb=input)
        input[2] = 999.0
        @test foam.fommb == [0.0, 100.0, 500.0]
        @test foam.fommb !== input
        model = foam_interface_model(install=false, k_d=1.0,
            D=(gas=1e-7, liquid=0.0), adsorption=LinearSurfactantAdsorption(.0002))
        r = reservoir_model(model)
        before = (copy(r.secondary_variables), copy(r.parameters))
        @test_throws ArgumentError set_surfactant_model!(model; foam=FmFactor(fommb=[1.0]))
        @test (r.secondary_variables, r.parameters) == before
        @test_throws ArgumentError set_surfactant_model!(model; foam=1.0)
        @test (r.secondary_variables, r.parameters) == before
        @test set_surfactant_model!(model; foam) === model
        @test r.secondary_variables[:FmFactor] === foam
        @test haskey(r.parameters, :SurfactantDiffusivities)
        @test haskey(r.parameters, :SurfactantRockMass)
        @test all(w -> !haskey(model[w].secondary_variables, :FmFactor) &&
            !haskey(model[w].parameters, :SurfactantRockMass) &&
            !haskey(model[w].parameters, :SurfactantDiffusivities), (:Injector, :Producer))
        before = (copy(r.secondary_variables), copy(r.parameters))
        @test_throws ArgumentError set_surfactant_model!(model; foam=FmFactor())
        @test (r.secondary_variables, r.parameters) == before
        prm = setup_parameters(model)
        @test all(>(0), prm[:Reservoir][:SurfactantRockMass])
        replace_variables!(r, FmFactor=FmFactor(fommb=[1.0]))
        @test_throws ArgumentError setup_state(r; Pressure=1e7, Saturations=[.5,.5], TracerConcentrations=0.0)
        # A partition alone does not mean that viscosity has already been adjusted.
        partial = foam_interface_model(install=false, k_d=1.0)
        set_secondary_variables!(reservoir_model(partial), SurfactantPartition=SurfactantPartition(1,
            only(reservoir_model(partial).equations[:tracers].flux_type.tracers)))
        @test set_surfactant_model!(partial) === partial
        # Without a viscosity source the installer rejects the model before
        # changing anything.
        noviscosity = foam_interface_model(install=false, k_d=1.0)
        rn = reservoir_model(noviscosity)
        delete!(rn.parameters, :PhaseViscosities)
        delete!(rn.secondary_variables, :PhaseViscosities)
        before = (copy(rn.secondary_variables), copy(rn.parameters), copy(rn.output_variables))
        @test_throws ArgumentError set_surfactant_model!(noviscosity)
        @test (rn.secondary_variables, rn.parameters, rn.output_variables) == before
        # The installer checks the target system itself, not only the tracer.
        three = ImmiscibleSystem((AqueousPhase(), LiquidPhase(), VaporPhase()); reference_densities=[1000.0, 800.0, 500.0])
        mesh3 = UnstructuredMesh(CartesianMesh((3, 1, 1), (3.0, 1.0, 1.0)))
        model3 = setup_reservoir_model(reservoir_domain(mesh3; permeability=1e-12, porosity=.25), three)
        raw = JutulDarcy.Tracers.SurfactantTracer{JutulDarcy.Tracers.MassPartitionRatio, Nothing}(
            3, 2, JutulDarcy.Tracers.MassPartitionRatio(1.0), Inf, nothing, nothing)
        add_tracers_to_model!(model3, raw)
        @test_throws ArgumentError set_surfactant_model!(model3)
        # A tracer built for the opposite phase order is rejected by the installer.
        sys_gl = ImmiscibleSystem((VaporPhase(), LiquidPhase()); reference_densities=[500.0, 1000.0])
        model_gl = setup_reservoir_model(reservoir_domain(mesh3; permeability=1e-12, porosity=.25), sys_gl)
        forward = SurfactantTracer(ImmiscibleSystem((LiquidPhase(), VaporPhase())); k_d=1.0)
        @test (forward.gas_index, forward.liquid_index) == (2, 1)
        @test (SurfactantTracer(sys_gl; k_d=1.0).gas_index, SurfactantTracer(sys_gl; k_d=1.0).liquid_index) == (1, 2)
        add_tracers_to_model!(model_gl, forward)
        @test_throws ArgumentError set_surfactant_model!(model_gl)
    end
end

function foam_field_checks()
    @testset "Foam strength per cell and subdomain" begin
        model = reservoir_model(foam_interface_model(k_d=1.0))
        C = fill(2.5, 3); S = fill(.5, 2, 3)
        update = JutulDarcy.Tracers.update_foam_factor!
        viscosity = JutulDarcy.Tracers.update_gas_viscosity!
        fm_scalar = zeros(3); fm_uniform = zeros(3); fm_field = zeros(3)
        update(fm_scalar, FmFactor(), model, C, S, 1:3)
        update(fm_uniform, FmFactor(fommb=fill(500.0,3)), model, C, S, 1:3)
        @test isequal(fm_scalar, fm_uniform)
        foam = FmFactor(fommb=[0.0, 100.0, 500.0])
        update(fm_field, foam, model, C, S, 1:3)
        D = .5 + atan(500.0*(.5-.25))/pi
        @test all(abs.(fm_field .- [1/(1+m*D) for m in (0,100,500)]) .<= 1e-14)
        mu = repeat([1e-3, 6e-5], 1, 3)
        mu_scalar = zeros(2,3); mu_uniform = zeros(2,3)
        def = JutulDarcy.Tracers.SurfactantAdjustedViscosities()
        viscosity(mu_scalar, def, model, mu, fm_scalar, 1:3)
        viscosity(mu_uniform, def, model, mu, fm_uniform, 1:3)
        @test isequal(mu_scalar, mu_uniform)
        @test mu_scalar[1,:] == mu[1,:]
        map = Jutul.FiniteVolumeGlobalMap([3,1], Int[])
        local_foam = Jutul.subvariable(foam, map)
        @test local_foam.fommb == [500.0, 0.0]
        fm_local = zeros(2)
        update(fm_local, local_foam, model, C[[3,1]], S[:,[3,1]], 1:2)
        @test fm_local == fm_field[[3,1]]
        scalar = FmFactor()
        @test Jutul.subvariable(scalar, map) === scalar
    end
end

function foam_dry_checks()
    @testset "Residual water dry correction values and derivatives" begin
        model = reservoir_model(foam_interface_model(n=1, k_d=1.0))
        update = JutulDarcy.Tracers.update_foam_factor!
        function factor(foam, Sw)
            fm = zeros(typeof(Sw), 1)
            update(fm, foam, model, [2.5], reshape([Sw,1-Sw],2,1), 1:1)
            return only(fm)
        end
        @test isequal(factor(FmFactor(), .4), 1/(1+500*(.5+atan(500*(.4-.25))/pi)))
        @test isequal(factor(FmFactor(), .4), factor(FmFactor(residual_water=.2), .4))
        foam = FmFactor(epdry=10,fmdry=.3,fommb=2,dry_correction=true,residual_water=.2)
        for Sw in (.1,.2)
            @test factor(foam, Sw) == 1.0
            ad = Jutul.get_ad_entity_scalar(Sw,1,1; tag=Cells())
            @test FD.partials(factor(foam, ad))[1] == 0
        end
        @test isapprox(factor(foam,.4), .5; atol=1e-14,rtol=0)
        ad = Jutul.get_ad_entity_scalar(.4,1,1; tag=Cells())
        # Dnew(.4)=.5, D'(.4)=10/(2pi), Fm'=-2D'/(1+2Dnew)^2.
        @test isapprox(FD.partials(factor(foam, ad))[1], -2*(10/(2pi))/4; rtol=1e-10,atol=1e-12)
        @test factor(FmFactor(epdry=0,dry_correction=true,residual_water=.2),.4) == 1
        @test factor(FmFactor(dry_correction=true,residual_water=1),1.0) == 1
        mu = zeros(2,1)
        JutulDarcy.Tracers.update_gas_viscosity!(mu,
            JutulDarcy.Tracers.SurfactantAdjustedViscosities(), model,
            reshape([1e-3,6e-5],2,1), [factor(foam,.1)], 1:1)
        @test mu == reshape([1e-3,6e-5],2,1)
    end
end

function foam_injection_checks()
    @testset "Static surfactant injection stages" begin
        function forces(model, mix, c; producer=true)
            control = Dict(:Injector=>InjectorControl(TotalRateTarget(1e-6), mix; tracers=c,check=false),
                :Producer=>producer ? ProducerControl(BottomHolePressureTarget(1e7)) : DisabledControl())
            return setup_reservoir_forces(model; control, set_default_limits=false)
        end
        model = foam_interface_model(k_d=0,L=.1)
        @test check_surfactant_injection(model, forces(model,[0.,1.],[0.])) === nothing
        @test check_surfactant_injection(model, forces(model,[0.,1.],missing)) === nothing
        @test check_surfactant_injection(model, forces(model,[.5,.5],[.001])) === nothing
        @test_throws ArgumentError check_surfactant_injection(model, forces(model,[0.,1.],[.001]))
        @test_throws ArgumentError check_surfactant_injection(model, forces(model,[1.,0.],[.1]))
        stages = [forces(model,[1.,0.],[.001]), forces(model,[0.,1.],[.001])]
        err = try check_surfactant_injection(model, stages); catch e; e; end
        @test err isa ArgumentError
        @test occursin("Stage 2", sprint(showerror,err)) && occursin("Injector", sprint(showerror,err))
        @test check_surfactant_injection(model,nothing) === nothing
        @test check_surfactant_injection(model,Any[]) === nothing
        @test_throws ArgumentError check_surfactant_injection(model, t->stages[1])
        @test_throws ArgumentError check_surfactant_injection(model, Dict(:Facility=>(control=nothing,)))
        @test_throws ArgumentError check_surfactant_injection(model, forces(model,[.6,.6],[.001]))
        @test_throws ArgumentError check_surfactant_injection(model, forces(model,[1.,0.],[-.001]))
        @test_throws ArgumentError check_surfactant_injection(model, forces(model,[1.,0.],[0.,.001]))
        reverse = foam_interface_model(phases=(VaporPhase(),LiquidPhase()),mixed=true,k_d=0,L=.1)
        @test check_surfactant_injection(reverse, forces(reverse,[0.,1.],[.3,.001])) === nothing
        @test_throws ArgumentError check_surfactant_injection(reverse, forces(reverse,[1.,0.],[0.,.001]))
        split = foam_interface_model(split_wells=true,k_d=0,L=.1)
        @test check_surfactant_injection(split, forces(split,[1.,0.],[.001];producer=false)) === nothing
        @test_throws ArgumentError check_surfactant_injection(split, forces(split,[0.,1.],[.001]))
        volume = foam_interface_model(K=1.,L=.1)
        @test_logs (:warn,r"Stage 1, well Injector:.*independent of phase densities") check_surfactant_injection(volume,forces(volume,[1.,0.],[.1]))
        @test check_surfactant_injection(volume,forces(volume,[0.,1.],[.001])) === nothing
        volume0 = foam_interface_model(K=0.,L=.1)
        @test_logs (:warn,r"zero dissolved carrying capacity") check_surfactant_injection(volume0,forces(volume0,[0.,1.],[.001]))
        @test_logs check_surfactant_injection(volume0,forces(volume0,[0.,1.],[0.]))
        volume_overflow = foam_interface_model(K=floatmax(Float64),L=.1)
        @test_logs (:warn,r"estimate overflow") check_surfactant_injection(volume_overflow,forces(volume_overflow,[1.,0.],[.001]))
        sys = reservoir_model(model).system
        @test_logs (:warn,r"finite L") SurfactantTracer(sys;k_d=1e-7)
        @test_logs SurfactantTracer(sys;k_d=1e-7,L=.1)
        @test_logs SurfactantTracer(sys;K=1e-7)
        # A compositional system (co2brine k-values) carries component mass
        # fractions, not phase fractions, so the static check skips it: pure
        # gas with surfactant and k_d = 0 is rejected for the immiscible model
        # above but passes here.
        cmesh = UnstructuredMesh(CartesianMesh((3, 1, 1), (3.0, 1.0, 1.0)))
        cdomain = reservoir_domain(cmesh; permeability=1e-12, porosity=.25, temperature=393.15)
        cwells = [setup_well(cdomain, 1; name=:Injector, simple_well=true),
            setup_well(cdomain, 3; name=:Producer, simple_well=true)]
        compositional = setup_reservoir_model(cdomain, :co2brine; wells=cwells, co2_physics=:kvalue)
        add_tracers_to_model!(compositional, SurfactantTracer(reservoir_model(compositional).system; k_d=0, L=.1))
        set_surfactant_model!(compositional)
        @test check_surfactant_injection(compositional, forces(compositional, [0., 1.], [.001])) === nothing
    end
end

# Run interface assertions as part of the existing registered foam suite.
foam_configuration_checks()
foam_field_checks()
foam_dry_checks()
foam_injection_checks()
