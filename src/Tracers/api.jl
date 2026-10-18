function number_of_tracers(model::MultiModel)
    return number_of_tracers(reservoir_model(model))
end

function number_of_tracers(model::SimulationModel)
    t = get(model.equations, :tracers, nothing)
    if isnothing(t)
        n = 0
    else
        n = number_of_tracers(t.flux_type)
    end
    return n
end

number_of_tracers(t::TracerFluxType) = length(t.tracers)

"""
    tracer_total_mass(tracer::AbstractTracer, model, state, concentration, resident_mass, cell, index)

# Arguments:
    - `tracer::AbstractTracer`: Tracer in question
    - `model`: Model object.
    - `state`: Current state
    - `concentration`: Tracer concentration.
    - `resident_mass_density`: Resident mass density (i.e. total mass density of all associated phases where the tracer is present)
    - `volume`: Fluid volume in cell.
    - `cell`: Cell index.
    - `index`: Index of tracer in the list of tracers
"""
function tracer_total_mass(tracer::AbstractTracer, model, state, concentration, resident_mass_density, volume, cell, index)
    return concentration*resident_mass_density*volume
end

function tracer_total_mass_outer(tracer::AbstractTracer, model, state, concentration, resident_mass_density, vol, cell, index)
    if resident_mass_density <= TRACER_TOL
        v = vol*Jutul.replace_value(concentration, 0.0)
    else
        v = tracer_total_mass(tracer, model, state, concentration, resident_mass_density, vol, cell, index)
    end
    return v
end

"""
    tracer_total_mass_flux(tracer, model, state, phase_mass_fluxes, C, index, upw, T = eltype(phase_mass_fluxes))

# Arguments:
    - `tracer`: Tracer in question
    - `model`: Model object.
    - `state`: Current state
    - `phase_mass_fluxes`: Mass fluxes for each phase
    - `C`: Tracer concentrations (as a matrix with the tracers in each cell as the columns)
    - `index`: Index of tracer in the list of tracers
    - `upw`: Upwind scheme
    - `T`: Type of the tracer concentrations
"""
function tracer_total_mass_flux(tracer, model, state, phase_mass_fluxes, index, upw, C = state.TracerConcentrations, T = eltype(C))
    v = zero(T)
    for phase in tracer_phase_indices(tracer)
        q_ph = phase_mass_fluxes[phase]
        C_iface = phase_upwind(upw, C, index, q_ph)
        v += C_iface*q_ph
    end
    return v
end

"""
    tracer_face_flux(tracer, model, state, phase_mass_fluxes, index, upw, kgrad, face)

Mass flux of tracer `index` through a face, for tracers whose flux needs the
face geometry: `kgrad` holds the cell pair and `face` is the face index. The
default forwards to [`tracer_total_mass_flux`](@ref), which only upwinds the
cell concentrations.
"""
function tracer_face_flux(tracer, model, state, phase_mass_fluxes, index, upw, kgrad, face)
    return tracer_total_mass_flux(tracer, model, state, phase_mass_fluxes, index, upw)
end

"""
    add_tracers_to_model!(model, tracers; names=missing, half_life=missing, kwarg...)

Install tracer transport equations in the reservoir and its simple wells.
`half_life` is a positive time in seconds for a single tracer, or an iterable
with one entry per tracer. `Inf` or `missing` disables decay; zero is invalid.
The rate is `log(2)/half_life`. A supplied `TracerFluxType` keeps its rates when
`half_life` is omitted; specifying it again for a type with nonzero rates throws
an `ArgumentError`. Rates and the products `k*dt` and `k*M` must be finite.

Decay adds `k*M` implicitly to the reservoir conservation residual, where `M`
is the total tracer inventory, including surfactant adsorption and deposit.
ECLIPSE `TRDCY` also decays the adsorbed inventory at the tracer half-life;
applying the same rate to the surfactant deposit is this implementation's
extension, not documented behavior of that keyword. `TRDCY` half-lives are in
days for METRIC and FIELD and in hours for LAB, so convert them to seconds.
Wells do not decay, and injection concentrations are unchanged.

Nonzero decay requires a fully implicit CPU reservoir with
`TwoPointPotentialFlowHardCoded` or `PotentialFlow{:fvm}` discretization and a
trivial global map. Generic AD reservoirs (`general_ad=true`), partitioned
models, non-CPU contexts and backend adaptation of nonzero rates are rejected.
The installer accepts simple wells only. Full adjoint/sensitivity and NLDD runs
have not been verified; an adjoint matrix layout alone is not rejected.

Foam can remain saturated while the aqueous concentration exceeds `fmsurf`.
For a closed cell at fixed other physical conditions, the threshold time is
`log(M(Csw_initial)/M(fmsurf))/k`. It reduces to
`half_life*log2(Csw_initial/fmsurf)` only when inventory is proportional to
concentration. The input half-life is not a calibrated foam lifetime.

The existing tracer convergence scaling is unchanged. In a closed nonlinear
adsorption example the default nonlinear tolerance of `1e-3` left a 1.6 percent
per-step inventory error; use an explicitly tighter tolerance for quantitative
inventory comparisons (`1e-10` for discrete analytical checks) and check
sensitivity in flowing cases. No cumulative decay state variable is added.
Postprocess `sum(k*M_end*dt)` over accepted reservoir ministeps only, excluding
failed attempts and wells. Enable `output_substates=true` or collect ministeps
in a driver hook. Using report-step states gives only an approximate integral.
"""
function add_tracers_to_model!(model::MultiModel, tracers; names = missing, half_life = missing, kwarg...)
    if !isnothing(tracers)
        tracers = TracerFluxType(tracers; names, half_life)
        rmodel = reservoir_model(model)
        add_tracers_to_model!(rmodel, tracers; kwarg...)
        for (k, m) in pairs(model.models)
            if JutulDarcy.model_or_domain_is_well(m)
                # TODO: Add support for tracer for multisegment wells by
                # specializing the equation update.
                physical_representation(m.domain)::JutulDarcy.SimpleWell
                add_tracers_to_model!(m, tracers; kwarg...)
            end
        end
        # Add cross-terms here.
        new_ctp = []
        function add_next!(ctp, ctt)
            if ctp.target_equation == :mass_conservation
                te = :tracers
            else
                te = ctp.target_equation
            end
            if ctp.source_equation == :mass_conservation
                se = :tracers
            else
                se = ctp.source_equation
            end
            v = Jutul.CrossTermPair(ctp.target, ctp.source, te, se, ctt)
            push!(new_ctp, v)
        end
        for ctp in model.cross_terms
            ct = ctp.cross_term
            if ctp.cross_term isa JutulDarcy.ReservoirFromWellFlowCT
                ctt = ReservoirFromWellTracerCT(ct.reservoir_cells, ct.well_cells)
                add_next!(ctp, ctt)
            end
            if ct isa JutulDarcy.WellFromFacilityFlowCT
                add_next!(ctp, WellFromFacilityTracerCT(
                    ct.wells, ct.facility_cells, ct.well_cells))
            end
        end
        for ctp in new_ctp
            add_cross_term!(model.cross_terms, ctp)
        end
    end
    return model
end

function add_tracers_to_model!(model::SimulationModel, tracers; names = missing, half_life = missing, discretization = missing)
    if !isnothing(tracers)
        if ismissing(discretization)
            discretization = model.domain.discretizations.mass_flow
        end
        tracers = TracerFluxType(tracers; names, half_life)
        N = number_of_tracers(tracers)
        !haskey(model.equations, :tracers) || error("Tracers already set in model")
        if model_applies_tracer_decay(model, tracers)
            check_tracer_decay(model, discretization, tracers)
        end
        model.equations[:tracers] = ConservationLaw(discretization, :TracerMasses, N, flux = tracers)
        model.secondary_variables[:TracerMasses] = TracerMasses(tracers)
        model.primary_variables[:TracerConcentrations] = TracerConcentrations(tracers)

        push!(model.output_variables, :TracerMasses)
        push!(model.output_variables, :TracerConcentrations)
        unique!(model.output_variables)
    end
    return model
end
