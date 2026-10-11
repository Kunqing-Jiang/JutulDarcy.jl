# Evaporation and rewetting column with k_d = 0 (dry CO2 evaporates the
# water near the injector and leaves surfactant as immobile deposit;
# surfactant-free water then redissolves it). CO2-brine k-value model, as the
# historical Zeng case. Shared by the registered deposit test and the slow
# validation item. The full driver with CSV and JSON output lives in the
# validation repository.
using Test
import JutulDarcy.Tracers: SURFACTANT_PARTITION_FLOOR

const DEFAULTS = (nx=10, pv_gas=1.0, pv_water=2.0, max_dt=2.0, c0=0.003)

const LENGTH = 0.3048
const AREA = (0.01LENGTH)^2
const PRESSURE = 27.579e6
const TEMPERATURE = 393.15
const RATE = 5.66336e-5/86400 # m3/s, as the legacy Zeng case


value_state(st, keys) = (; (k => Jutul.value(st[k]) for k in keys if haskey(st, k))...)

# Surfactant flux through each phase of a perforation, from the step-end state,
# with the same connection and phase fluxes as ReservoirFromWellTracerCT.
function perforation_fluxes(sres, swell, cell)
    conn = (dp = swell.Pressure[1] - sres.Pressure[cell], WI = swell.WellIndices[1],
        gdz = swell.PerforationGravityDifference[1], well = 1, perforation = 1, reservoir = cell)
    λ_t = sum(sres.PhaseMobilities[:, cell])
    return [JutulDarcy.perforation_phase_mass_flux(λ_t, conn, sres, swell, ph) for ph in 1:2]
end

function setup_column(L; nx = DEFAULTS.nx, pv_gas = DEFAULTS.pv_gas,
        pv_water = DEFAULTS.pv_water, max_dt = DEFAULTS.max_dt, c0 = DEFAULTS.c0, kd = 0.0)
    nx isa Integer && nx >= 2 || throw(ArgumentError("nx must be an integer >= 2"))
    all(isfinite, (L, kd, c0, pv_gas, pv_water, max_dt)) && L > 0 && kd >= 0 && c0 >= 0 &&
        pv_gas > 0 && pv_water >= 0 && max_dt > 0 || throw(ArgumentError("Invalid column inputs"))
    mesh = UnstructuredMesh(CartesianMesh((nx, 1, 1), (LENGTH, sqrt(AREA), sqrt(AREA))))
    domain = reservoir_domain(mesh; permeability = si_unit(:darcy), porosity = 0.25, temperature = TEMPERATURE)
    inj = setup_well(domain, 1; name = :Injector, radius = 0.0005, simple_well = true)
    prod = setup_well(domain, nx; name = :Producer, radius = 0.0005, simple_well = true)
    model = setup_reservoir_model(domain, :co2brine; wells = [inj, prod], co2_physics = :kvalue)
    rmodel = reservoir_model(model)
    @assert JutulDarcy.get_phases(rmodel.system) == (LiquidPhase(), VaporPhase())
    # Injected stream densities at the initial pressure and temperature, as in
    # the Zeng example.
    rho = rmodel.secondary_variables[:PhaseMassDensities].tab(PRESSURE, TEMPERATURE)
    add_tracers_to_model!(model, SurfactantTracer(rmodel.system; k_d = kd, L = L))
    set_surfactant_model!(model; foam = FmFactor(epdry=500.0, fmdry=0.25,
        fmsurf=2.5, epsurf=1.0, fommb=500.0, dry_correction=false, residual_water=nothing))
    prm = setup_parameters(model)
    Vp = prm[:Reservoir][:FluidVolume]
    pv = sum(Vp)
    T1, T2 = pv_gas*pv/RATE, pv_water*pv/RATE
    producer = ProducerControl(BottomHolePressureTarget(PRESSURE))
    gas = InjectorControl(TotalRateTarget(RATE), [0.0, 1.0]; density = rho[2], tracers = [0.0])
    water = InjectorControl(TotalRateTarget(RATE), [1.0, 0.0]; density = rho[1], tracers = [0.0])
    forces = [setup_reservoir_forces(model; control = Dict(:Injector => gas, :Producer => producer)),
        setup_reservoir_forces(model; control = Dict(:Injector => water, :Producer => producer))]
    state0 = setup_reservoir_state(model; Pressure = PRESSURE, Saturations = [1.0, 0.0],
        OverallMoleFractions = [1.0, 0.0], TracerConcentrations = c0)

    return (; model, prm, Vp, pv, T1, T2, rho, gas, water, producer, forces, state0,
        inputs=(; nx, pv_gas, pv_water, max_dt, c0, kd, L))
end

function run_column(L; nx = DEFAULTS.nx, pv_gas = DEFAULTS.pv_gas,
        pv_water = DEFAULTS.pv_water, max_dt = DEFAULTS.max_dt, c0 = DEFAULTS.c0)
    (; model, prm, Vp, pv, T1, T2, forces, state0) = setup_column(L; nx, pv_gas, pv_water, max_dt, c0)

    facility = model.models[:Facility]
    ip = JutulDarcy.get_well_position(facility.domain, :Producer)
    d = (
        t = Ref(0.0), total_time = T1 + T2, initial = Ref(NaN), inventory = Ref(0.0), produced_facility = Ref(0.0),
        max_balance_error = Ref(0.0),
        cap_ok = Ref(true), nonneg_ok = Ref(true), complementarity_ok = Ref(true),
        producer_ok = Ref(true), max_release_margin = Ref(-Inf),
        max_R = Ref(0.0), max_split_error = Ref(0.0), TM0 = Ref(0.0),
        E_end_gas = Ref(NaN), deposit_cells_ok = Ref(true), first_deposit = Ref(NaN),
        state_end_gas = Ref{Any}(nothing),
        E0 = Ref(0.0), Mf0 = Ref(0.0), Ql = Ref(0.0), Ps = Ref(0.0), Is = Ref(0.0),
        newton = [0, 0], accepted = [0, 0], failed = [0, 0], failed_at_kink = [0, 0],
        kink_steps = [0, 0], E_final = Ref(NaN),
    )
    prev_branch = Ref(falses(nx))
    rkeys = (:Pressure, :PhaseMobilities, :PhaseMassDensities, :PhaseMassMobilities,
        :PermeabilityMultiplier, :Saturations, :SurfactantPartition)
    wkeys = (:Pressure, :WellIndices, :PerforationGravityDifference, :Saturations,
        :PhaseMassDensities, :TracerConcentrations, :ConnectionPressureDrop)
    on_cap(P) = [P[1, i] == L for i in axes(P, 2)]

    function hook(done, report, sim, dt, forces, max_iter, cfg)
        phase = d.t[] < T1*(1 - 1e-12) ? 1 : 2
        d.newton[phase] += length(report[:steps])
        rs = sim.storage[:Reservoir]
        if isnan(d.initial[])
            d.initial[] = sum(sum(Jutul.value(sim.storage[n][:state0][:TracerMasses]))
                for n in (:Reservoir, :Injector, :Producer))
        end
        if !done
            d.failed[phase] += 1
            # Did the rejected iterate move a cell onto or off the cap?
            P = Jutul.value(rs[:state][:SurfactantPartition])
            d.failed_at_kink[phase] += on_cap(P) != prev_branch[]
            return (done, report)
        end
        d.t[] += dt
        d.accepted[phase] += 1
        sr = value_state(rs[:state], rkeys)
        P = sr.SurfactantPartition
        ml = sr.Saturations[1, :] .* sr.PhaseMassDensities[1, :]
        mg = sr.Saturations[2, :] .* sr.PhaseMassDensities[2, :]
        mf = ml .+ mg
        E = P[3, :] .* Vp
        # The partition must account for the whole cell inventory, which the
        # accumulation term computes from c alone.
        TM = Jutul.value(rs[:state][:TracerMasses][1, :])
        split = (ml .* P[1, :] .+ mg .* P[2, :] .+ P[3, :]) .* Vp
        d.max_split_error[] = max(d.max_split_error[], maximum(abs.(split .- TM) ./ max.(TM, eps())))
        branch = on_cap(P)
        d.kink_steps[phase] += branch != prev_branch[]
        prev_branch[] = branch

        d.cap_ok[] &= all(P[1, :] .<= L*(1 + 1e-15))
        d.nonneg_ok[] &= all(P[3, :] .>= 0)
        floor_inactive = ml .>= SURFACTANT_PARTITION_FLOOR .* mf
        d.complementarity_ok[] &= all(P[3, floor_inactive] .* (L .- P[1, floor_inactive]) .== 0)
        if isnan(d.first_deposit[]) && sum(E) > 0
            d.first_deposit[] = d.t[]
        end

        # Global balance with facility rates, as the Zeng example audits it.
        st = sim.storage[:Facility][:state]
        q = Jutul.value(st[:TotalSurfaceMassRate])
        c_prod = Jutul.value(sim.storage[:Producer][:state][:TracerConcentrations][1, 1])
        d.produced_facility[] -= dt*q[ip]*c_prod
        d.inventory[] = sum(sum(Jutul.value(sim.storage[n][:state][:TracerMasses]))
            for n in (:Reservoir, :Injector, :Producer))
        d.max_balance_error[] = max(d.max_balance_error[],
            abs(d.inventory[] + d.produced_facility[] - d.initial[]))

        Mf = sum(ml .* P[1, :] .* Vp) + sum(mg .* P[2, :] .* Vp)
        if phase == 1
            d.deposit_cells_ok[] &= all(i -> P[3, i] == 0 || P[1, i] == L || !floor_inactive[i], 1:nx)
            if d.t[] >= T1*(1 - 1e-12)
                d.E_end_gas[] = sum(E)
                d.E0[] = sum(E)
                d.Mf0[] = Mf
                d.TM0[] = sum(TM)
                d.state_end_gas[] = (P = copy(P), ml = copy(ml), mg = copy(mg), c = Jutul.value(rs[:state][:TracerConcentrations][1, :]))
            end
        else
            for (well, cell) in ((:Injector, 1), (:Producer, nx))
                sw = value_state(sim.storage[well][:state], wkeys)
                qph = perforation_fluxes(sr, sw, cell)
                C = (P[1, cell], P[2, cell])
                for ph in 1:2
                    if qph[ph] < 0
                        d.Is[] -= dt*qph[ph]*sw.TracerConcentrations[1, 1]
                    else
                        d.Ps[] += dt*qph[ph]*C[ph]
                    end
                end
                d.Ql[] += dt*max(qph[1], 0)
            end
            d.producer_ok[] &= P[1, nx] <= L*(1 + 1e-15) && P[2, nx] == 0
            # Reservoir surfactant balance from the accumulation term and the
            # perforation fluxes, independent of the partition.
            R = (sum(TM) - d.TM0[]) + d.Ps[] - d.Is[]
            d.max_R[] = max(d.max_R[], abs(R))
            # Release bound from the liquid capacity. Given C_l <= L and C_g = 0
            # it follows from the balance above, so it is reported, not tested.
            lhs = d.E0[] - sum(E)
            rhs = L*(sum(ml .* Vp) + d.Ql[]) - d.Mf0[] - d.Is[] + abs(R) + 1e-12*(d.E0[] + d.Mf0[])
            d.max_release_margin[] = max(d.max_release_margin[], lhs - rhs)
            d.E_final[] = sum(E)
        end
        return (done, report)
    end
    simulate_reservoir(state0, model, [T1, T2]; parameters = prm, forces = forces,
        max_timestep = max_dt, timesteps = :none, info_level = -1,
        post_ministep_hook = hook, output_path = nothing)
    return d
end

# Checks shared with the slow validation item, which runs the other caps.
function deposit_column_checks(L, d)
    @testset "L = $L" begin
        @test isapprox(d.t[], d.total_time; rtol = 1e-9)
        # The K-value model runs with the default solver tolerances (it
        # does not converge its first step with the strict ones), which
        # leave a relative surfactant balance error of 2.3e-5 in the
        # legacy Zeng case and about 3e-6 here.
        @test d.max_balance_error[] <= 1e-4*d.initial[]
        @test d.cap_ok[]
        @test d.nonneg_ok[]
        @test d.complementarity_ok[]
        @test d.E_end_gas[] > 0
        @test d.deposit_cells_ok[]
        @test d.producer_ok[]
        @test d.max_split_error[] <= 1e-12
        @test d.max_R[] <= 1e-6*d.initial[]
        @test d.E_final[] <= 1e-12*d.initial[]
    end
    @info "Surfactant deposit column, L = $L" first_deposit_s = d.first_deposit[] deposit_end_gas_kg = d.E_end_gas[] deposit_final_kg = d.E_final[] newton = "$(d.newton)" accepted = "$(d.accepted)" failed = "$(d.failed)" failed_at_kink = "$(d.failed_at_kink)" steps_with_kink_crossing = "$(d.kink_steps)" max_release_margin_kg = d.max_release_margin[] max_balance_residual_kg = d.max_R[] relative_balance_error = d.max_balance_error[]/d.initial[]
end

