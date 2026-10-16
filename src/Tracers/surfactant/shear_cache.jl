# Installation and refresh of the cell gradients used by foam shear thinning.

# Reject configurations that would give wrong results without an error. The
# check runs at installation, when storage is allocated and before each
# gradient refresh, which covers sequential, partitioned and adjoint models
# derived after installation, including ones that reuse existing storage.
function check_foam_shear_model(model)
    # Other systems, such as black oil, assemble component fluxes without
    # face_potential_difference, so tracer and component fluxes would disagree.
    model.system isa Union{ImmiscibleSystem, JutulDarcy.LVCompositional2PhaseSystem} ||
        throw(ArgumentError("Foam shear requires an immiscible or CO2-brine two-phase system, got $(typeof(model.system))"))
    model.formulation isa Jutul.FullyImplicitFormulation ||
        throw(ArgumentError("Foam shear requires FullyImplicitFormulation, got $(typeof(model.formulation))"))
    model.domain.global_map isa Jutul.TrivialGlobalMap ||
        throw(ArgumentError("Foam shear does not support partitioned models"))
    if model.context isa Jutul.CPUJutulContext && Jutul.represented_as_adjoint(Jutul.matrix_layout(model.context))
        throw(ArgumentError("Foam shear does not support adjoint or sensitivity models"))
    end
    for key in (:mass_conservation, :tracers)
        eq = get(model.equations, key, nothing)
        eq isa ConservationLaw && eq.flow_discretization isa TwoPointPotentialFlowHardCoded ||
            throw(ArgumentError("Foam shear requires TwoPointPotentialFlowHardCoded for equation $key"))
    end
    # The multiplier scales the normal transmissibility, but the tangential
    # part uses the permeability fixed at installation.
    any(v -> haskey(v, :PermeabilityMultiplier), (model.parameters, model.secondary_variables)) &&
        throw(ArgumentError("Foam shear does not support PermeabilityMultiplier"))
    return nothing
end

function foam_shear_permeability(k, dim)
    if ndims(k) == 1
        return repeat(reshape(k, 1, :), dim, 1)
    elseif size(k, 1) == 1
        return repeat(k, dim, 1)
    elseif size(k, 1) == dim
        return Matrix{Float64}(k)
    end
    throw(ArgumentError("Foam shear requires scalar or diagonal permeability"))
end

function FoamShearGradient(model, config::FoamShearThinning)
    check_foam_shear_model(model)
    d = model.data_domain
    dim = size(d[:cell_centroids], 1)
    normals = copy(d[:normals])
    for f in axes(normals, 2)
        normals[:, f] ./= norm(normals[:, f])
    end
    kd = foam_shear_permeability(d[:permeability], dim)
    ls = least_squares_gradient_weights(d)
    nbad = count(ls.bad)
    if nbad > 0.1*length(ls.bad)
        @warn "Foam shear uses the face-normal fallback next to rank-deficient cells" fallback_cells=nbad total_cells=length(ls.bad)
    end
    gas = findfirst(==(VaporPhase()), get_phases(model.system))
    return FoamShearGradient(config, gas, Matrix{Int}(d[:neighbors]),
        Vector{Float64}(d[:areas]), normals, kd, ls.weights, ls.bad)
end

function Jutul.values_per_entity(model, v::FoamShearGradient)
    check_foam_shear_model(model)
    return size(v.kd, 1)
end
# The gradient is refreshed at the equation entry point (update_equations! below).
Jutul.update_secondary_variable!(G, ::FoamShearGradient, model, state, ix) = G
function Jutul.initialize_secondary_variable_ad!(state, model, v::FoamShearGradient, symb, npartials; kwarg...)
    # The gradient is frozen in the local Jacobian, so Float64 storage needs no
    # AD partials. Preserve Jutul's conversion behavior for other input types.
    if eltype(state[symb]) === Float64
        return state
    end
    return invoke(Jutul.initialize_secondary_variable_ad!, Tuple{Any, Any, Any, Any, Any},
        state, model, v, symb, npartials; kwarg...)
end

function update_foam_shear_gradients!(state, model, v::FoamShearGradient)
    check_foam_shear_model(model)
    pv = primal_potential_values(state)
    G = state.FoamShearGradient
    fill_cells(cells) = fill_least_squares_gradients!(G, v.weights, v.bad, v.neighbors, pv, model, v.gas, cells)
    nb = clamp(size(G, 2) ÷ Jutul.minbatch(model.context), 1, Jutul.nthreads(model.context))
    if nb == 1
        fill_cells(axes(G, 2))
    else
        Threads.@threads for batch in 1:nb
            fill_cells(Jutul.entity_eachindex(G, batch, nb))
        end
    end
    return G
end

# Neighbor properties are not synchronized between secondary-variable batches,
# so the gradients are refreshed once all batches finish, before the equations.
function Jutul.update_equations!(storage, model::JutulDarcy.DarcyFlowModel, dt)
    v = get(model.secondary_variables, :FoamShearGradient, nothing)
    if v isa FoamShearGradient
        update_foam_shear_gradients!(Jutul.evaluation_state(storage), model, v)
    end
    Jutul.update_equations!(storage, storage.equations, model.equations, model, dt)
    return nothing
end
