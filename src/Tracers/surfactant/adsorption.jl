function Jutul.get_dependencies(t::SurfactantTracer, model)
    if model_or_domain_is_well(model) || t.adsorption === nothing
        return ()
    end
    return (:SurfactantPartition, :PhaseMassDensities, :SurfactantRockMass)
end

function tracer_total_mass_outer(t::SurfactantTracer, model, state, c, rho, Vp, cell, index)
    v = invoke(tracer_total_mass_outer, Tuple{AbstractTracer, Any, Any, Any, Any, Any, Any, Any},
        t, model, state, c, rho, Vp, cell, index)
    if model_or_domain_is_well(model) || t.adsorption === nothing
        return v
    end
    Csw = state.SurfactantPartition[1, cell]*state.PhaseMassDensities[t.liquid_index, cell]
    v += state.SurfactantRockMass[cell]*t.adsorption(Csw)
    isfinite(Jutul.value(v)) || throw(DomainError(Jutul.value(v), "Surfactant inventory overflow"))
    return v
end
