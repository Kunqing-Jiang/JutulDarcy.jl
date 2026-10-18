"""
    tracer_decay_rates(flux_type)

Return the per-tracer first-order decay rates in inverse seconds.
"""
tracer_decay_rates(ft::TracerFluxType) = ft.decay
tracer_has_decay(ft::TracerFluxType) = any(!iszero, ft.decay)
model_applies_tracer_decay(model, ft) =
    tracer_has_decay(ft) && !JutulDarcy.model_or_domain_is_well(model)

function check_tracer_decay_rates(rates)
    all(k -> isfinite(k) && k >= 0, rates) ||
        throw(ArgumentError("Tracer decay rates must be finite and nonnegative (1/s)"))
    return nothing
end

function tracer_decay_from_half_life(half_life, N)
    ismissing(half_life) && return ntuple(_ -> 0.0, N)
    if half_life isa Real
        N == 1 || throw(ArgumentError("A scalar half_life requires exactly one tracer"))
        values = (half_life,)
    else
        applicable(iterate, half_life) || throw(ArgumentError("half_life must be a positive time in seconds or an iterable of times"))
        values = Tuple(half_life)
        length(values) == N || throw(ArgumentError("half_life length must match the number of tracers ($N)"))
    end
    rates = ntuple(N) do i
        t = values[i]
        ismissing(t) && return 0.0
        t isa Real && t > 0 || throw(ArgumentError("half_life[$i] must be positive in seconds; use Inf or missing to disable decay"))
        k = Float64(log(2)/t)
        isfinite(k) || throw(ArgumentError("half_life[$i] produces a nonfinite tracer decay rate"))
        return k
    end
    check_tracer_decay_rates(rates)
    return rates
end

const DecayDisc = Union{TwoPointPotentialFlowHardCoded, PotentialFlow{:fvm}}

function check_tracer_decay(model, disc = model.equations[:tracers].flow_discretization,
        ft = model.equations[:tracers].flux_type)
    disc isa DecayDisc || throw(ArgumentError("Tracer decay requires TwoPointPotentialFlowHardCoded or PotentialFlow{:fvm}; general_ad=true reservoirs are unsupported"))
    model.formulation isa Jutul.FullyImplicitFormulation ||
        throw(ArgumentError("Tracer decay requires FullyImplicitFormulation"))
    model.context isa Jutul.CPUJutulContext ||
        throw(ArgumentError("Tracer decay requires a CPU context"))
    model.domain.global_map isa Jutul.TrivialGlobalMap ||
        throw(ArgumentError("Tracer decay does not support partitioned models"))
    check_tracer_decay_rates(ft.decay)
    return nothing
end

function check_tracer_decay_timestep(rates, dt)
    for k in rates
        iszero(k) && continue
        isfinite(k*dt) || throw(ArgumentError("Tracer decay k*dt must be finite"))
    end
    return nothing
end

@inline function add_tracer_decay!(buf, k, M, cell)
    for i in eachindex(buf)
        ki = k[i]
        if !iszero(ki)
            loss = ki*M[i, cell]
            isfinite(Jutul.value(loss)) || throw(ArgumentError("Tracer decay k*M must be finite (tracer $i, cell $cell)"))
            buf[i] += loss
        end
    end
    return buf
end

function Jutul.update_accumulation!(eq_s, law::ConservationLaw{:TracerMasses, <:Any, <:TracerFluxType},
        storage, model, dt)
    ft = law.flux_type
    if !model_applies_tracer_decay(model, ft)
        return invoke(Jutul.update_accumulation!, Tuple{Any, Any, Any, Any, Any},
            eq_s, law, storage, model, dt)
    end
    check_tracer_decay(model)
    k = tracer_decay_rates(ft)
    check_tracer_decay_timestep(k, dt)
    acc = invoke(Jutul.update_accumulation!, Tuple{Any, Any, Any, Any, Any},
        eq_s, law, storage, model, dt)
    m0, m = Jutul.state_pair(storage, eq_s.accumulation_symbol, model)
    for cell in axes(m, 2)
        add_tracer_decay!(view(acc, :, cell), k, m, cell)
    end
    return acc
end

# Widen the conserved quantity and flux slots so invoke selects the original
# discretization method rather than the tracer specialization.
function stock_equation_in_entity!(eq_buf::AbstractVector{T_e}, self_cell, state, state0,
        eq::ConservationLaw{<:Any, <:TwoPointPotentialFlowHardCoded, <:Any}, model, dt, ldisc) where {T_e}
    return invoke(Jutul.update_equation_in_entity!,
        Tuple{AbstractVector{T_e}, Any, Any, Any,
            ConservationLaw{<:Any, <:TwoPointPotentialFlowHardCoded, <:Jutul.FluxType}, Any, Any, Any},
        eq_buf, self_cell, state, state0, eq, model, dt, ldisc)
end

function stock_equation_in_entity!(eq_buf::AbstractVector{T_e}, self_cell, state, state0,
        eq::ConservationLaw{<:Any, <:PotentialFlow{:fvm}, <:Any}, model, dt, ldisc) where {T_e}
    return invoke(Jutul.update_equation_in_entity!,
        Tuple{AbstractVector{T_e}, Any, Any, Any,
            ConservationLaw{<:Any, <:PotentialFlow{:fvm}, <:Jutul.FluxType}, Any, Any, Any},
        eq_buf, self_cell, state, state0, eq, model, dt, ldisc)
end

function Jutul.update_equation_in_entity!(eq_buf::AbstractVector{T_e}, self_cell, state, state0,
        eq::ConservationLaw{:TracerMasses, <:DecayDisc, <:TracerFluxType},
        model, dt, ldisc = Jutul.local_discretization(eq, self_cell)) where {T_e}
    ft = eq.flux_type
    if !model_applies_tracer_decay(model, ft)
        stock_equation_in_entity!(eq_buf, self_cell, state, state0, eq, model, dt, ldisc)
        return nothing
    end
    check_tracer_decay(model)
    check_tracer_decay_timestep(tracer_decay_rates(ft), dt)
    stock_equation_in_entity!(eq_buf, self_cell, state, state0, eq, model, dt, ldisc)
    add_tracer_decay!(eq_buf, tracer_decay_rates(ft), state[Jutul.conserved_symbol(eq)], self_cell)
    return nothing
end

# These methods resolve the intersection of the tracer equation and well model
# specializations. Wells retain their original accumulation equation.
for WM in (:SimpleWellFlowModel, :SimpleWellModel)
    @eval function Jutul.update_equation_in_entity!(eq_buf::AbstractVector{T_e}, self_cell, state, state0,
            eq::ConservationLaw{:TracerMasses, <:DecayDisc, <:TracerFluxType},
            model::JutulDarcy.$WM, dt, ldisc = Jutul.local_discretization(eq, self_cell)) where {T_e}
        return invoke(Jutul.update_equation_in_entity!,
            Tuple{AbstractVector{T_e}, Any, Any, Any, ConservationLaw, JutulDarcy.$WM, Any, Any},
            eq_buf, self_cell, state, state0, eq, model, dt, ldisc)
    end
end
