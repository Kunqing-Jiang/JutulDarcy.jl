"""
    SurfactantDiffusivities(tracer_ix)

Reservoir diffusion face coefficients (m³/s), with rows in model phase order.
Default values are `compute_face_trans(data_domain, porosity .* D_α)`, using
[`SurfactantTracer`](@ref)'s physical diffusivities (m²/s). Half-face
coefficients are combined in series: `1 / (1/T_left + 1/T_right)`.
NNCs are unsupported. Negative or non-finite default coefficients throw
`ArgumentError`. On a uniform orthogonal grid the coefficient equals
`φ D_α A / Δx`. The diffusive mass flux through a face is
`-T_α min(S_α,L, S_α,R) ρ̄_α (C_α,R - C_α,L)` (kg/s), using the dissolved
phase concentrations from `SurfactantPartition` and excluding the deposit.
The TPFA discretization is consistent on orthogonal grids; on
non-orthogonal grids it remains a two-point approximation.

Install with `set_surfactant_model!` before calling `setup_parameters`.
Explicit coefficients can be supplied as a phase-by-face matrix `T` in m³/s:
`setup_parameters(model, Reservoir = Dict(:SurfactantDiffusivities => T))`
for a `MultiModel`, or `setup_parameters(rmodel, SurfactantDiffusivities = T)`
for a reservoir `SimulationModel`. This bypasses default geometry and its
checks. Callers must ensure finite, non-negative values, including after
editing parameters; Jutul clamps negative explicit initial values to zero.

The diffusion coefficient includes porosity only. Consistency with the
continuous diffusion equation assumes `FluidVolume = φ V`; NTG, `pore_volume`,
`pore_volume_multiplier` and `pore_volume_override` can break that assumption.
"""
struct SurfactantDiffusivities <: Jutul.VectorVariables
    tracer_ix::Int
end

Jutul.associated_entity(::SurfactantDiffusivities) = Jutul.Faces()
Jutul.values_per_entity(model, ::SurfactantDiffusivities) = JutulDarcy.number_of_phases(model.system)
Jutul.minimum_value(::SurfactantDiffusivities) = 0.0

function Jutul.default_parameter_values(data_domain, model, param::SurfactantDiffusivities, symb)
    haskey(data_domain, :nnc) && throw(ArgumentError("SurfactantDiffusivities: NNCs are unsupported by the diffusion geometry"))
    tracer = model.equations[:tracers].flux_type.tracers[param.tracer_ix]
    D = tracer.D
    gas_index = tracer.gas_index
    nf = Jutul.number_of_faces(model.domain)
    T = zeros(JutulDarcy.number_of_phases(model.system), nf)
    for phase in axes(T, 1)
        D_phase = phase == gas_index ? D.gas : D.liquid
        T[phase, :] .= Jutul.compute_face_trans(data_domain, data_domain[:porosity] .* D_phase)
    end
    nbad = count(x -> !isfinite(x) || x < 0, T)
    nbad == 0 || throw(ArgumentError("SurfactantDiffusivities: $nbad negative or non-finite face coefficients"))
    return T
end

# The dissolved concentrations of a surfactant live in SurfactantPartition,
# so the upwind-only flux of a generic tracer must not be used for it.
function tracer_total_mass_flux(tracer::SurfactantTracer, model, state, phase_mass_fluxes, index, upw, C = state.TracerConcentrations, T = eltype(C))
    throw(ArgumentError("SurfactantTracer flux needs the face geometry; call tracer_face_flux"))
end

function tracer_face_flux(tracer::SurfactantTracer, model, state, phase_mass_fluxes, index, upw, kgrad, face)
    D = tracer.D
    P = state.SurfactantPartition
    gas_index = tracer.gas_index
    gas_upstream = JutulDarcy.foam_shear_upstream(JutulDarcy.foam_shear_definition(model),
        face, gas_index, state, model, model.equations[:tracers].flux_type, kgrad, upw)
    v = zero(eltype(P))
    for phase in tracer_phase_indices(tracer)
        q_ph = phase_mass_fluxes[phase]
        row = phase == gas_index ? 2 : 1
        F = cell -> P[row, cell]
        C_iface = phase == gas_index && gas_upstream !== nothing ? F(gas_upstream) : upwind(upw, F, q_ph)
        v += C_iface * q_ph
        v = add_diffusive_tracer_flux(v, state, D, face, kgrad, phase, view(P, row, :), gas_index)
    end
    return v
end

# A producing perforation carries each phase at its own concentration. The
# deposit is immobile and stays in the reservoir cell.
function perforation_production_concentration(tracer::SurfactantTracer, model, state, i, cell, phase)
    row = phase == tracer.gas_index ? 2 : 1
    return state.SurfactantPartition[row, cell]
end

function add_diffusive_tracer_flux(v, state, D::Nothing, face, kgrad, phase, C, gas_index)
    return v
end

function add_diffusive_tracer_flux(v, state, D, face, kgrad, phase, C, gas_index)
    D_phase = state.SurfactantDiffusivities[phase, face]
    diff_mass = phase_diffused_mass(D_phase, state, face, kgrad, phase)
    dC = gradient(C, kgrad)
    v += diff_mass*dC
    return v
end

function phase_diffused_mass(D_phase, state, face, kgrad, phase)
    S = state.Saturations
    ρ = state.PhaseMassDensities
    den = cell -> @inbounds ρ[phase, cell]
    left, right = Jutul.cell_pair(kgrad)
    S_left, S_right = S[phase, left], S[phase, right]
    Sa = primal_value(S_right) < primal_value(S_left) ? S_right : S_left  # min, compared by value
    return -D_phase*Sa*face_average(den, kgrad)
end
