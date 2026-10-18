abstract type AbstractTracer

end

struct SinglePhaseTracer <: AbstractTracer
    phase_index::Int64
end

tracer_phase_indices(t::SinglePhaseTracer) = (t.phase_index, )

function SinglePhaseTracer(model::Union{SimulationModel, MultiModel}, phase::AbstractPhase)
    sys = reservoir_model(model).system
    return SinglePhaseTracer(sys, phase)
end

function SinglePhaseTracer(system::MultiPhaseSystem, phase::Int)
    nph = number_of_phases(system)
    phase <= nph || error("Phase index $phase out of bounds")
    return SinglePhaseTracer(phase)
end

function SinglePhaseTracer(system::MultiPhaseSystem, phase::AbstractPhase)
    for (i, ph) in enumerate(get_phases(system))
        if ph == phase
            return SinglePhaseTracer(i)
        end
    end
    error("Phase $phase not found in system")
end


struct MultiPhaseTracer{N} <: AbstractTracer
    phase_indices::NTuple{N, Int64}
end

function MultiPhaseTracer(system::MultiPhaseSystem, target_phases::AbstractArray)
    nph = number_of_phases(system)
    N = length(target_phases)
    vals = Int[]
    for (p, target) in enumerate(target_phases)
        if target isa Int
            target <= nph || error("Phase index $target at position $p out of bounds")
            push!(vals, target)
        else
            ix = 0
            for (i, ph) in enumerate(get_phases(system))
                if ph == target
                    ix = i
                    break
                end
            end
            ix > 0 || error("Phase $target at position $p not found in system")
            push!(vals, ix)
        end
    end
    return MultiPhaseTracer{N}(tuple(vals...))
end

function MultiPhaseTracer(system::MultiPhaseSystem)
    N = number_of_phases(system)
    return MultiPhaseTracer{N}(tuple((1:N)...))
end

tracer_phase_indices(t::MultiPhaseTracer) = t.phase_indices

struct TracerFluxType{T, N, Names} <: Jutul.FluxType
    tracers::T
    names::Names
    decay::NTuple{N, Float64}
    function TracerFluxType(tracers::Tuple; names = missing, half_life = missing)
        N = length(tracers)
        if ismissing(names)
            names = ["Tracer $i" for i in 1:N]
        else
            length(names) == N || error("Number of names does not match number of tracers")
        end
        T = typeof(tracers)
        names = tuple(names...)
        return new{T, N, typeof(names)}(tracers, names, tracer_decay_from_half_life(half_life, N))
    end
    function TracerFluxType(tracers::T, names::Names,
            ::Val{N}, decay::NTuple{N, Float64} = ntuple(_ -> 0.0, N)) where {T, Names, N}
        check_tracer_decay_rates(decay)
        return new{T, N, Names}(tracers, names, decay)
    end
end

function Adapt.adapt_structure(to, flux::TracerFluxType{T, N}) where {T, N}
    tracer_has_decay(flux) && throw(ArgumentError("Tracer decay does not support backend adaptation of nonzero rates"))
    # Names are reporting metadata and strings/symbols are not valid kernel
    # arguments. The host model retains the authoritative labels.
    return TracerFluxType(Adapt.adapt(to, flux.tracers), nothing, Val(N), flux.decay)
end

function TracerFluxType(tracers::AbstractTracer; kwarg...)
    return TracerFluxType((tracers, ); kwarg...)
end

function TracerFluxType(tracers::AbstractArray; kwarg...)
    return TracerFluxType(tuple(tracers...); kwarg...)
end

function TracerFluxType(t::TracerFluxType; names = missing, half_life = missing)
    ismissing(half_life) && return t
    tracer_has_decay(t) && throw(ArgumentError("half_life cannot override a TracerFluxType with nonzero decay rates"))
    return TracerFluxType(t.tracers, t.names, Val(length(t.tracers)),
        tracer_decay_from_half_life(half_life, length(t.tracers)))
end

