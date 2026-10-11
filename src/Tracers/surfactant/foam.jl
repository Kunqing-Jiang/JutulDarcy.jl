"""
    FmFactor(; epdry = 500.0, fmdry = 0.25, fmsurf = 2.5, epsurf = 1.0, fommb = 500.0,
        dry_correction = false, residual_water = nothing)

Gas mobility multiplier `1/(1 + fommb*Fdry*Fsurf)` of the CMG STARS foam
model (Ma, Ren, Mateen, Morel and Cordelier, "Modeling techniques for foam
flow in porous media", SPE Journal 2015, section "STARS Foam Model"). STARS
multiplies the gas relative permeability by this factor; here the gas
viscosity is divided by it, which changes the gas mobility `k_rg/μ_g` in the
Darcy flux identically. The default parameter values have no recorded
calibration source and should be replaced by values fitted to the case.
`fommb` retains the API spelling of the literature's fmmob. It is a non-negative scalar or a nonempty vector with one value per
reservoir cell. Vectors are copied to `Vector{Float64}`; treat the stored
configuration as read-only. Subdomains select values in their cell order.

The scalar shape parameters must be finite: `epdry ≥ 0`, `0 ≤ fmdry ≤ 1`,
`fmsurf > 0` (kg/m³ liquid = g/L), and `epsurf > 0`. CMG STARS papers
usually give fmsurf as an aqueous mole fraction `x`; with the liquid
density `ρ_l` and the surfactant and water molar masses `M_s`, `M_w` it
converts to `ρ_l x M_s / (x M_s + (1 - x) M_w)`. The five positional
arguments remain supported. The default dry factor is
`D(Sw) = 0.5 + atan(epdry*(clamp(Sw, 0, 1) - fmdry))/pi`.
With `dry_correction=true`, explicitly supply `residual_water` in [0, 1]:
the entire curve becomes zero for `Sw ≤ residual_water`, and
`D(Sw) - D(residual_water)` above it, without normalization. The derivative
at equality is zero. Residual water is never inferred from relative permeability.
"""
struct FmFactor{M} <: Jutul.ScalarVariable
    epdry::Float64
    fmdry::Float64
    fmsurf::Float64
    epsurf::Float64
    fommb::M
    dry_correction::Bool
    residual_water::Union{Nothing, Float64}
    function FmFactor(; epdry = 500.0, fmdry = 0.25, fmsurf = 2.5, epsurf = 1.0,
            fommb = 500.0, dry_correction = false, residual_water = nothing)
        epdry = checked_float(epdry, :epdry)
        fmdry = checked_float(fmdry, :fmdry; upper = 1.0)
        fmsurf = checked_float(fmsurf, :fmsurf; positive = true)
        epsurf = checked_float(epsurf, :epsurf; positive = true)
        if fommb isa AbstractVector
            Base.require_one_based_indexing(fommb)
            isempty(fommb) && throw(ArgumentError("fommb must be nonempty"))
            fommb = [checked_float(v, :fommb) for v in fommb]
        else
            fommb = checked_float(fommb, :fommb)
        end
        dry_correction isa Bool || throw(ArgumentError("dry_correction must be Bool"))
        if residual_water !== nothing
            residual_water = checked_float(residual_water, :residual_water; upper = 1.0)
        elseif dry_correction
            throw(ArgumentError("dry_correction requires explicit residual_water"))
        end
        return new{typeof(fommb)}(epdry, fmdry, fmsurf, epsurf, fommb, dry_correction, residual_water)
    end
end

FmFactor(epdry, fmdry, fmsurf, epsurf, fommb) =
    FmFactor(; epdry, fmdry, fmsurf, epsurf, fommb)

function check_foam_length(model, v::FmFactor)
    if v.fommb isa AbstractVector
        n = Jutul.number_of_cells(model.domain)
        length(v.fommb) == n || throw(ArgumentError("fommb must have $n reservoir cell values, got $(length(v.fommb))"))
    end
    return nothing
end

function Jutul.default_value(model, v::FmFactor)
    check_foam_length(model, v)
    return 1.0
end

function Jutul.subvariable(v::FmFactor, map::Jutul.FiniteVolumeGlobalMap)
    v.fommb isa AbstractVector || return v
    return FmFactor(; epdry = v.epdry, fmdry = v.fmdry, fmsurf = v.fmsurf,
        epsurf = v.epsurf, fommb = Jutul.partition_variable_slice(v.fommb, map.cells),
        dry_correction = v.dry_correction, residual_water = v.residual_water)
end

foam_strength(v::Real, cell) = v
foam_strength(v::AbstractVector, cell) = v[cell]

@jutul_secondary function update_foam_factor!(fm, def::FmFactor, model, ClVolumetricConcentration, Saturations, cells)
    epdry, fmdry, fmsurf, epsurf, fommb = def.epdry, def.fmdry, def.fmsurf, def.epsurf, def.fommb
    gas_index = findfirst(ph -> ph == VaporPhase(), get_phases(model.system))
    liquid_index = 3 - gas_index
    dry_offset = def.dry_correction ? foam_dry_factor(def.residual_water, epdry, fmdry) : 0.0

    for i in cells
        Csw = ClVolumetricConcentration[i]
        sl = Saturations[liquid_index, i]
        F_dry = foam_dry_factor(sl, epdry, fmdry)
        if def.dry_correction
            F_dry = primal_value(sl) <= def.residual_water ? zero(sl) : F_dry - dry_offset
        end
        F_surf = foam_surfactant_factor(Csw, fmsurf, epsurf)
        fm[i] = 1 / (1 + foam_strength(fommb, i) * F_dry * F_surf)
    end

    return fm
end

function foam_dry_factor(sl::T, epdry::Real, fmdry::Real) where T
    # clamp(sl, 0, 1) with the branch chosen by value, see primal_value.
    v = primal_value(sl)
    sl = v > 1 ? one(T) : (v < 0 ? zero(T) : sl)
    return 0.5+atan((epdry*(sl-fmdry)))/pi
end

# Start of the linear segment of the surfactant factor, as a fraction of fmsurf.
const FM_SURFACTANT_LINEAR_FRACTION = 1e-6

"""
    foam_surfactant_factor(C_sw, fmsurf, epsurf)

STARS surfactant factor `(C_sw/fmsurf)^epsurf`, equal to 1 above `fmsurf`.

For `epsurf < 1` the power law has an infinite slope at `C_sw = 0`, which
puts infinite entries into the Jacobian of every cell without surfactant.
Below `C_eps = 1e-6 fmsurf` it is therefore replaced by the chord from the
origin to the power law at `C_eps`: the value is continuous, the slope is
finite and jumps at `C_eps`. For `epsurf >= 1` the factor is unchanged.
"""
function foam_surfactant_factor(C_sw::T, fmsurf::Real, epsurf::Real) where T
    if primal_value(C_sw) > fmsurf
        return one(T)
    end
    C_eps = FM_SURFACTANT_LINEAR_FRACTION*fmsurf
    if epsurf < 1 && primal_value(C_sw) < C_eps
        return C_sw*(C_eps/fmsurf)^epsurf/C_eps
    end
    return (C_sw/fmsurf)^epsurf
end

"""
Phase viscosities with the foam factor applied: the gas viscosity is
`BasePhaseViscosities` divided by [`FmFactor`](@ref), the other phases are
unchanged. Dividing the viscosity instead of multiplying the gas relative
permeability, as STARS does, gives the same gas mobility in the Darcy flux.
"""
struct SurfactantAdjustedViscosities <: PhaseVariables
end

Jutul.@jutul_secondary function update_gas_viscosity!(vals, def::SurfactantAdjustedViscosities, model, BasePhaseViscosities, FmFactor, ix)

    phases = get_phases(model.system)
    gas_index = findfirst(ph -> ph == VaporPhase(), phases)
    nph = size(vals, 1)

    for cell in ix
        for ph in 1:nph
            val_phase = BasePhaseViscosities[ph, cell]
            if ph == gas_index
                val_phase /= FmFactor[cell]
            end
            vals[ph, cell] = val_phase
        end
    end
    return vals
end
