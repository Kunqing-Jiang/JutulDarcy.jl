"""
    set_surfactant_model!(model; foam=FmFactor(), shear=nothing)

Install the surfactant diffusion face parameter when `D !== nothing`, the
surfactant partition, the volumetric liquid concentration, the
foam factor `FmFactor` and the foam-adjusted gas viscosity. The model must
hold exactly one [`SurfactantTracer`](@ref), added with
`add_tracers_to_model!`; other tracers may be present. Call this installer
before `setup_parameters` so that the diffusion parameter is initialized.
`foam` must be an [`FmFactor`](@ref); its vector length is checked before
changing the model. Call the installer once. To replace an installed foam
configuration, use `replace_variables!(reservoir_model(model), FmFactor=foam)`.
The low-level replacement checks vector length during default initialization;
explicit `FmFactor` state values bypass that initialization check.
`shear` optionally installs [`FoamShearThinning`](@ref); an exponent of zero
keeps the original flux and Jacobian paths. Install before creating states or
simulators, and rebuild them to change the shear configuration.
When adsorption is enabled, it also installs the static `SurfactantRockMass`
parameter on the reservoir.

For a `MultiModel` only the reservoir is modified. Wells keep plain
`TracerConcentrations`. A producing perforation reads the phase
concentrations of the reservoir cell from `SurfactantPartition`; an
injecting perforation and the well-facility coupling use the well-node
concentration. The simple wells that `add_tracers_to_model!` accepts have
no internal faces that would need the phase partition.
"""
function set_surfactant_model!(outer_model::MultiModel; foam=FmFactor(), shear=nothing)
    set_surfactant_model!(reservoir_model(outer_model); foam, shear)
    return outer_model
end

function set_surfactant_model!(model::SimulationModel; foam=FmFactor(), shear=nothing)
    tracers = model.equations[:tracers].flux_type.tracers
    n_surf = count(x -> isa(x, SurfactantTracer), tracers)
    if n_surf != 1
        throw(ArgumentError("set_surfactant_model! needs exactly one SurfactantTracer, found $n_surf"))
    end
    foam isa FmFactor || throw(ArgumentError("foam must be an FmFactor"))
    check_foam_length(model, foam)
    phases = get_phases(model.system)
    if length(phases) != 2 || count(==(VaporPhase()), phases) != 1
        throw(ArgumentError("set_surfactant_model! needs a two-phase system with one VaporPhase, got $phases"))
    end
    prms = Jutul.get_variables_by_type(model, :parameters)
    svars = Jutul.get_variables_by_type(model, :secondary)
    if get(svars, :PhaseViscosities, nothing) isa SurfactantAdjustedViscosities
        throw(ArgumentError("Surfactant foam is already installed; use replace_variables! to change FmFactor"))
    end
    # All checks precede the first modification, so a rejected call leaves the
    # model unchanged.
    viscosities_in = haskey(prms, :PhaseViscosities) ? prms : svars
    haskey(viscosities_in, :PhaseViscosities) ||
        throw(ArgumentError("set_surfactant_model! needs PhaseViscosities as a parameter or secondary variable of the reservoir model"))
    tracer_ix = findfirst(x -> isa(x, SurfactantTracer), tracers)
    tracer = tracers[tracer_ix]
    gas_index = findfirst(==(VaporPhase()), phases)
    tracer.gas_index == gas_index ||
        throw(ArgumentError("SurfactantTracer was built for a system with the gas phase at position $(tracer.gas_index), but this model has it at position $gas_index"))
    shear === nothing || shear isa FoamShearThinning || throw(ArgumentError("shear must be nothing or FoamShearThinning"))
    shear_gradient = foam_shear_enabled(shear) ? FoamShearGradient(model, shear) : nothing
    set_secondary_variables!(model,
        SurfactantPartition = SurfactantPartition(tracer_ix, tracer),
        ClVolumetricConcentration = ClVolumetricConcentration(tracer.liquid_index),
        FmFactor = foam
    )
    if shear_gradient !== nothing
        svars[:FoamShearGradient] = shear_gradient
    end
    mu = viscosities_in[:PhaseViscosities]
    viscosities_in[:BasePhaseViscosities] = mu
    delete!(viscosities_in, :PhaseViscosities)
    svars[:PhaseViscosities] = SurfactantAdjustedViscosities()
    if tracer.D !== nothing
        set_parameters!(model, SurfactantDiffusivities = SurfactantDiffusivities(tracer_ix))
    end
    if tracer.adsorption !== nothing && !model_or_domain_is_well(model)
        set_parameters!(model, SurfactantRockMass = SurfactantRockMass())
    end
    push!(model.output_variables, :SurfactantPartition)
    return model
end

"""
    check_surfactant_injection(model::MultiModel, forces)

Check static facility injection streams after assembling all forces and before
simulation. Accept a forces dictionary/NamedTuple or a vector of stage mappings;
`nothing` and an empty vector have no static controls. Dynamic functions are
unsupported. Each actual `WellGroup` controller is visited. Producers and
disabled controls are skipped, as are systems other than two-phase immiscible
liquid/gas systems. Reservoir sources/boundaries, callbacks and multisegment
wells are outside this check's scope.

Mixtures are phase mass fractions in reservoir phase order. Tracer concentration
`c` is kg surfactant/kg injected fluid, read at the surfactant's tracer index;
missing tracers mean zero. For mass ratio `k_d`, the equivalent liquid fraction
is `c/(w_l + k_d*w_g)`. Positive injection at zero capacity or reaching a finite
`L` throws `ArgumentError` with stage and well. Zero injection is allowed even
at zero capacity. For volume ratio `K`, reference densities only give an
estimate: unavailable estimates or estimated violations warn, never impose a
physical compliance error. Runtime densities can change. This check neither
modifies controls nor guarantees runtime absence of deposit or a maximum
principle for reservoir/initial states.
"""
function check_surfactant_injection(model::MultiModel, forces)
    rmodel = reservoir_model(model)
    haskey(rmodel.equations, :tracers) || return nothing
    tracers = rmodel.equations[:tracers].flux_type.tracers
    n_surf = count(t -> t isa SurfactantTracer, tracers)
    n_surf == 0 && return nothing
    n_surf == 1 || throw(ArgumentError("Injection check needs exactly one SurfactantTracer, found $n_surf"))
    sys = rmodel.system
    phases = get_phases(sys)
    sys isa ImmiscibleSystem && length(phases) == 2 && count(==(VaporPhase()), phases) == 1 || return nothing
    any(m -> m.domain isa WellGroup, values(model.models)) || return nothing
    forces === nothing && return nothing
    stages = forces isa AbstractVector ? forces : (forces,)
    tracer_ix = findfirst(t -> t isa SurfactantTracer, tracers)
    tracer = tracers[tracer_ix]
    gas = tracer.gas_index
    liquid = tracer.liquid_index
    for (stage, stage_forces) in enumerate(stages)
        stage_forces isa Union{AbstractDict, NamedTuple} ||
            throw(ArgumentError("Stage $stage: forces must be a static mapping, got $(typeof(stage_forces))"))
        for (name, controller) in pairs(model.models)
            controller.domain isa WellGroup || continue
            haskey(stage_forces, name) || throw(ArgumentError("Stage $stage: missing forces for controller $name"))
            cf = stage_forces[name]
            cf isa Union{AbstractDict, NamedTuple} && haskey(cf, :control) ||
                throw(ArgumentError("Stage $stage: controller $name forces must contain control"))
            controls = cf[:control]
            controls isa Union{AbstractDict, NamedTuple} ||
                throw(ArgumentError("Stage $stage: controller $name control must be a mapping"))
            for well in controller.domain.well_symbols
                haskey(controls, well) || throw(ArgumentError("Stage $stage, well $well: missing controller control"))
            end
            for (well, ctrl) in pairs(controls)
                well in controller.domain.well_symbols ||
                    throw(ArgumentError("Stage $stage: well $well does not belong to controller $name"))
                ctrl isa Union{ProducerControl, DisabledControl} && continue
                ctrl isa InjectorControl || throw(ArgumentError("Stage $stage, well $well: unsupported control $(typeof(ctrl))"))
                location = "Stage $stage, well $well"
                mix = ctrl.injection_mixture
                length(mix) == 2 && all(x -> x isa Real && isfinite(x) && x >= 0, mix) && isapprox(sum(mix), 1.0) ||
                    throw(ArgumentError("$location: injection_mixture must contain two finite non-negative phase mass fractions summing to one"))
                if ctrl.tracers === missing
                    c = 0.0
                else
                    ctrl.tracers isa Union{AbstractVector, Tuple} && length(ctrl.tracers) == length(tracers) ||
                        throw(ArgumentError("$location: tracers must match the reservoir tracer count $(length(tracers))"))
                    c = ctrl.tracers[tracer_ix]
                    c isa Real && isfinite(c) && c >= 0 ||
                        throw(ArgumentError("$location: surfactant injection concentration must be finite and non-negative"))
                end
                c == 0 && continue
                if tracer.partition isa MassPartitionRatio
                    kd = tracer.partition.k_d
                else
                    K = tracer.partition.K
                    rho = JutulDarcy.reference_densities(sys)
                    if K == 0
                        kd = 0.0
                    elseif rho === nothing || length(rho) != 2 || !all(x -> x isa Real && isfinite(x) && x > 0, rho)
                        @warn "$location: K reference-density estimate unavailable; runtime densities vary and the static estimate does not guarantee compliance"
                        continue
                    else
                        kd = K*(rho[liquid]/rho[gas])
                        if !isfinite(kd)
                            @warn "$location: K reference-density estimate overflow; runtime densities vary and the static estimate does not guarantee compliance"
                            continue
                        end
                    end
                end
                capacity = mix[liquid] + kd*mix[gas]
                reason = capacity == 0 ? "positive surfactant injection has zero dissolved carrying capacity and would cause reservoir deposit" :
                    isfinite(tracer.L) && c >= tracer.L*capacity ? "surfactant injection reaches the liquid limit L=$(tracer.L), leaving no dissolved capacity" : nothing
                if reason !== nothing
                    if tracer.partition isa MassPartitionRatio
                        throw(ArgumentError("$location: $reason"))
                    elseif tracer.partition.K == 0 || mix[gas] == 0
                        @warn "$location: $reason; this static injection violation is independent of phase densities (K checks warn only)"
                    else
                        @warn "$location: K reference-density estimate: $reason; runtime densities vary and the static estimate does not guarantee compliance" kd_ref=kd
                    end
                end
            end
        end
    end
    return nothing
end
