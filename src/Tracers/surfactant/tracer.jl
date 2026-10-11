import JutulDarcy: get_phases, model_or_domain_is_well, upwind, face_average, gradient
import JutulDarcy: PhaseVariables

"""
Mass-based partition coefficient: `C_g = k_d * C_l`, with `C_l`, `C_g` the
surfactant mass fractions in the liquid and gas phases (kg/kg).
"""
struct MassPartitionRatio
    k_d::Float64
end

"""
Volume-based partition coefficient: `ρ_g C_g = K ρ_l C_l`. Converted to the
mass ratio `k_d = K ρ_l / ρ_g` with the current phase densities, so `k_d`
changes with pressure when the densities do.
"""
struct VolumePartitionRatio
    K::Float64
end

partition_mass_ratio(p::MassPartitionRatio, ρl, ρg) = p.k_d
partition_mass_ratio(p::VolumePartitionRatio, ρl, ρg) = p.K*ρl/ρg

partition_is_zero(p::MassPartitionRatio) = p.k_d == 0
partition_is_zero(p::VolumePartitionRatio) = p.K == 0

struct SurfactantTracer{P, A} <: AbstractTracer
    gas_index::Int
    liquid_index::Int
    partition::P
    L::Float64
    D::Union{NamedTuple{(:gas, :liquid), Tuple{Float64, Float64}}, Nothing}  # Physical diffusivity (m^2/s)
    adsorption::A
end

"""
    SurfactantTracer(system; k_d = missing, K = missing, L = missing, D = nothing, adsorption = nothing)

Surfactant carried by the liquid and gas phases of a two-phase gas-liquid
`system`, for use with the STARS-type foam model installed by
[`set_surfactant_model!`](@ref).

The primary variable `TracerConcentrations` is the surfactant mass per fluid
mass in the cell, `c`, counting the liquid, the gas and an immobile deposit.
With `m_l = S_l ρ_l`, `m_g = S_g ρ_g` and `M = c (m_l + m_g)` (kg per m³ of
pore volume), each cell is closed locally by

    m_l C_l + m_g C_g + E = M,   C_g = k_d C_l,   0 ≤ C_l ≤ L,
    E ≥ 0,   E (L - C_l) = 0,

so that surfactant beyond the liquid-phase cap `L` stays in the cell as the
deposit `E` (kg/m³ pore) instead of being moved to the gas phase. With
`k_d = 0` a nearly dry cell is the exception: there the denominator floor of
[`surfactant_partition`](@ref) can give `E > 0` with `C_l < L`, at most
`1e-8 (m_l + m_g) L`. The deposit
dissolves again when the liquid capacity recovers. It does not change porosity
or permeability. The partition result is the secondary variable
`SurfactantPartition` with rows `C_l`, `C_g` and `E`.

Give exactly one partition coefficient, both finite and non-negative:

- `k_d`: mass ratio `C_g / C_l`;
- `K`: volume ratio `ρ_g C_g / (ρ_l C_l)`, converted with the current densities.
  A mole-fraction ratio `K_x` converts as `k_d = K_x M̄_l / M̄_g` with the
  phase molar masses under a dilute-surfactant assumption; it is not a
  constant conversion for arbitrary compositional flash states.

`L` is the liquid-phase cap in kg/kg, positive and possibly `Inf`. It
defaults to `Inf` for a positive partition coefficient and to 0.1 for a zero
one; a zero partition coefficient needs a finite `L`, since a dry cell then
has nowhere else to hold surfactant. A small positive mass ratio
`0 < k_d < 1e-6` with infinite `L` warns about large dry-cell concentrations.

`D = nothing` disables diffusion. Otherwise give `(gas = D_g, liquid = D_l)`
with one finite, non-negative real scalar per phase in m²/s. These effective
pore-fluid coefficients include tortuosity and any constant mechanical
dispersion. Both coefficients may be zero.

The reservoir face parameter [`SurfactantDiffusivities`](@ref) combines
`D_α` with the grid geometry; its docstring gives the discretization, the
explicit-coefficient input and the grid and pore-volume assumptions.
Callers that previously passed the pre-multiplied face coefficient
`φ A / Δx * D_α` must now pass the physical `D_α` directly.

`adsorption` accepts a linear, Langmuir or tabulated equilibrium isotherm.
Reservoir mass is `c*(S_l*ρ_l + S_g*ρ_g)*V_p + M_R*adsorption(ρ_l*C_l)`.
The primary `c` excludes adsorption and already counts the deposit. Wells
retain only the fluid term. Zero isotherms normalize to `nothing`.
"""
function SurfactantTracer(system::MultiPhaseSystem; k_d = missing, K = missing, L = missing, D = nothing, adsorption = nothing, kwarg...)
    removed = intersect(keys(kwarg), (:Concentration_limit, :C_max))
    if !isempty(removed)
        throw(ArgumentError("SurfactantTracer: $(join(removed, " and ")) have been removed. " *
            "Use L to cap the liquid-phase surfactant mass fraction; surfactant above L " *
            "becomes an immobile deposit and is no longer moved to the gas phase."))
    end
    isempty(kwarg) || throw(ArgumentError("SurfactantTracer: unknown keyword arguments $(collect(keys(kwarg)))"))

    phases = get_phases(system)
    if length(phases) != 2 || count(ph -> ph == VaporPhase(), phases) != 1
        throw(ArgumentError("SurfactantTracer needs a two-phase system with one VaporPhase, got $phases"))
    end
    gas_index = findfirst(==(VaporPhase()), phases)
    liquid_index = 3 - gas_index
    if ismissing(k_d) == ismissing(K)
        throw(ArgumentError("SurfactantTracer: give exactly one of k_d (mass ratio) and K (volume ratio)"))
    end
    if ismissing(K)
        value = Float64(k_d)
        partition = MassPartitionRatio(value)
        name = "k_d"
    else
        value = Float64(K)
        partition = VolumePartitionRatio(value)
        name = "K"
    end
    if !(isfinite(value) && value >= 0)
        throw(ArgumentError("SurfactantTracer: $name must be finite and non-negative, got $value"))
    end
    if ismissing(L)
        L = partition_is_zero(partition) ? 0.1 : Inf
    end
    L = Float64(L)
    if !(L > 0)
        throw(ArgumentError("SurfactantTracer: L must be positive, got $L"))
    end
    if partition_is_zero(partition) && !isfinite(L)
        throw(ArgumentError("SurfactantTracer: $name = 0 needs a finite L, since surfactant in a dry cell can only be held as deposit"))
    end
    if D !== nothing
        if !(D isa NamedTuple && length(D) == 2 && haskey(D, :gas) && haskey(D, :liquid))
            throw(ArgumentError("SurfactantTracer: D must be nothing or (gas = D_g, liquid = D_l) in m^2/s, got $D"))
        end
        D = (gas = checked_float(D.gas, "SurfactantTracer: D.gas"),
            liquid = checked_float(D.liquid, "SurfactantTracer: D.liquid"))
    end
    if adsorption !== nothing
        adsorption isa Union{LinearSurfactantAdsorption, LangmuirSurfactantAdsorption, TabulatedSurfactantAdsorption} ||
            throw(ArgumentError("adsorption must be a linear, Langmuir or tabulated surfactant isotherm, or nothing"))
        adsorption_is_zero(adsorption) && (adsorption = nothing)
    end
    if partition isa MassPartitionRatio && 0 < value < 1e-6 && !isfinite(L)
        @warn "Small positive k_d with infinite L can produce large liquid concentrations in dry cells; supply a finite L" k_d=value
    end
    return SurfactantTracer{typeof(partition), typeof(adsorption)}(gas_index, liquid_index, partition, L, D, adsorption)
end

function SurfactantTracer(system::MultiPhaseSystem, target_phases; kwarg...)
    throw(ArgumentError("SurfactantTracer always covers both the liquid and the gas phase, " *
        "since the partition between them needs both; target_phases is not supported"))
end

# The surfactant is carried by both phases of the two-phase system.
tracer_phase_indices(t::SurfactantTracer) = (1, 2)
