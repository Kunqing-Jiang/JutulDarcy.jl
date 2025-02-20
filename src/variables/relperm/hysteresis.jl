abstract type AbstractHysteresis end

"""
    KilloughHysteresis(tol = 0.1, s_min = 0.0)

Killough hysteresis model. `tol` is a parameter for numerical stability and
`s_min` a minimum threshold for when hysteresis is activated. Larger values for
both of these parameters reduce numerical difficulties.

Reference: https://doi.org/10.2118/5106-PA
"""
Base.@kwdef struct KilloughHysteresis <: AbstractHysteresis
    tol::Float64 = 0.1
    s_min::Float64 = 0.0
end

"""
    CarlsonHysteresis()

Carlson's hysteresis model.

Note that this model requires an intersection between drainage and imbibition
relative permeability that comes at some cost during simulation.
"""
struct CarlsonHysteresis <: AbstractHysteresis end

"""
    JargonHysteresis()

Jargon's hystersis model.
"""
struct JargonHysteresis <: AbstractHysteresis end

"""
    NoHysteresis()

Type to indicate that no hysteresis is active, and the drainage curve will
always be used.
"""
struct NoHysteresis <: AbstractHysteresis end

"""
    ImbibitionOnlyHysteresis()

Type to indicate that the hysteresis does not make use of the drainage curve,
and the imbibition curve will always be used.
"""
struct ImbibitionOnlyHysteresis <: AbstractHysteresis end

struct MaxSaturations <: PhaseVariables end

function Jutul.update_parameter_before_step!(s_max, ::MaxSaturations, storage, model, dt, forces)
    s = storage.state.Saturations
    update_max_hysteresis_value!(s_max, s)
    return s_max
end

struct MaxPressure <: ScalarVariable end

function Jutul.update_parameter_before_step!(p_max, ::MaxPressure, storage, model, dt, forces)
    p = storage.state.Pressure
    update_max_hysteresis_value!(p_max, p)
    return p_max
end

function update_max_hysteresis_value!(v_max, v)
    for i in eachindex(v_max, v)
        v_prev = v_max[i]
        v_now = value(v[i])
        if v_now > v_prev
            v_max[i] = replace_value(v_prev, v_now)
        end
    end
end

struct MinPressure <: ScalarVariable end

Jutul.default_value(model, ::MinPressure) = 1e100 

function Jutul.update_parameter_before_step!(p_min, ::MinPressure, storage, model, dt, forces)
    p = storage.state.Pressure
    update_min_hysteresis_value!(p_min, p)
    return p_min
end

function update_min_hysteresis_value!(v_min, v)
    for i in eachindex(v_min, v)
        v_prev = v_min[i]
        v_now = value(v[i])
        if v_now < v_prev
            v_min[i] = replace_value(v_prev, v_now)
        end
    end
end

function hysteresis_is_active(x::AbstractRelativePermeabilities)
    return false
end

function kr_hysteresis(t::NoHysteresis, drain, imb, s, s_max, ϵ = 1e-10, s_min = 0.0)
    return drain(s)
end

function kr_hysteresis(t::ImbibitionOnlyHysteresis, drain, imb, s, s_max, ϵ = 1e-10, s_min = 0.0)
    return imb(s)
end

function kr_hysteresis(t, drain, imb, s, s_max, ϵ = 1e-10, s_min = 0.0)
    if s >= s_max - ϵ || s <= s_min
        kr = drain(s)
    elseif s_max >= imb.s_max - ϵ
        kr = imb(s)
    else
        kr = hysteresis_impl(t, drain, imb, s, s_max)
    end
    return kr
end

# TODO: Avoid this module hack by integrating Roots in package.
import JutulDarcy.MultiComponentFlash.Roots: find_zero, A42

function hysteresis_impl(t::CarlsonHysteresis, drain, imb, s, s_max)
    kr_at_max = drain(s_max)
    if imb isa PhaseRelativePermeability
        s_meet = Jutul.linear_interp(imb.k.F, imb.k.X, kr_at_max)
    else
        ϵ = 1e-4
        F = s -> imb(s) - kr_at_max
        s_meet = find_zero(
            F,
            (value(imb.connate), value(imb.s_max)),
            A42(),
            xatol = ϵ,
            rtol = ϵ
        )
    end
    s_shifted = s + s_meet - s_max
    return imb(s_shifted)
end

function hysteresis_impl(h_model::KilloughHysteresis, drain, imb, S, S_max)
    if S < h_model.s_min
        kr = drain(S)
    else
        S_crit_imbibition = imb.critical
        S_crit_drainage = drain.critical
        # TODO: Check that this matches that of imbibition?
        kr_s_max = drain.s_max
        K = 1.0/(S_crit_imbibition - S_crit_drainage) - 1.0/(kr_s_max - S_crit_drainage)
        M = 1.0 + h_model.tol*(kr_s_max - S_max)
        S_crit = S_crit_drainage + (S_max - S_crit_drainage)/(M + K*(S_max - S_crit_drainage))
        S_norm = S_crit_imbibition + (S - S_crit)*(kr_s_max - S_crit_imbibition)/(S_max - S_crit)
        kr = imb(S_norm)*drain(S_max)/drain(kr_s_max)
    end
    return kr
end

function hysteresis_impl(h_model::JargonHysteresis, drain, imb, S, S_max)
    S_crit_imbibition = imb.critical
    S_crit_drainage = drain.critical
    # TODO: Check that this matches that of imbibition?
    kr_s_max = drain.s_max
    S_norm = S_crit_drainage + (S - S_crit_drainage)*(kr_s_max - S_crit_drainage)/(S_max - S_crit_drainage)
    kr = (imb(S_norm)/drain(S_norm))*drain(S)
    return kr
end

function add_hysteresis_parameters!(model::MultiModel)
    add_hysteresis_parameters!(reservoir_model(model))
    return model
end

function add_hysteresis_parameters!(model::SimulationModel)
    add_hysteresis_parameters!(model.parameters, model[:RelativePermeabilities])
    return model
end

function add_hysteresis_parameters!(param, kr::AbstractRelativePermeabilities)
    if hysteresis_is_active(kr)
        param[:MaxSaturations] = MaxSaturations()
    end
    return param
end
