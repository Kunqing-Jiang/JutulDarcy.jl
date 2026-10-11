# Only used when k_d == 0: the partition denominator m_l + k_d m_g is then the
# liquid mass, which vanishes in a dry cell. It is floored at this fraction of
# the fluid mass, so that C_l rises linearly to L as c goes from 0 to
# SURFACTANT_PARTITION_FLOOR * L and stays continuous, while the surfactant
# mass is still conserved exactly (the excess is counted as deposit). For
# k_d > 0 the denominator is at least min(1, k_d) times the fluid mass and no
# floor is applied.
const SURFACTANT_PARTITION_FLOOR = 1e-8

"""
    surfactant_partition(c, ml, mg, kd, L)

Partition the surfactant of a cell between liquid, gas and deposit. `c` is the
surfactant mass per fluid mass, `ml`, `mg` the liquid and gas masses per pore
volume and `kd` the mass partition coefficient. Returns `(C_l, C_g, E)`, see
[`SurfactantTracer`](@ref).

Branches are chosen on the values; derivatives follow the selected formula.
Outside the capped branch `E` is computed as `(B̃ - B) C_l`, so it is exactly
zero unless the dry-cell floor is active.
"""
function surfactant_partition(c, ml, mg, kd, L)
    mf = ml + mg
    M = c*mf
    B = ml + kd*mg
    if primal_value(kd) == 0
        B_min = SURFACTANT_PARTITION_FLOOR*mf
        B_floor = ifelse(primal_value(B_min) < primal_value(B), promote(B, B_min)...)
    else
        B_floor = B
    end
    T = promote_type(typeof(M), typeof(B_floor))
    if isfinite(L) && primal_value(M) >= primal_value(B_floor)*L
        Cl = convert(T, L)
        E = M - B*L
    else
        Cl = M/B_floor
        E = (B_floor - B)*Cl
    end
    Cg = kd*Cl
    return (Cl, Cg, E)
end

"""
Phase partition of the foaming surfactant per cell: rows are the liquid and
gas mass fractions `C_l`, `C_g` (kg/kg) and the immobile deposit `E` (kg/m³ of
pore volume). See [`SurfactantTracer`](@ref).
"""
struct SurfactantPartition{P} <: Jutul.VectorVariables
    tracer_ix::Int
    partition::P
    L::Float64
    gas_index::Int
    liquid_index::Int
end

"""
    SurfactantPartition(tracer_ix, tracer::SurfactantTracer)

Partition variable for the tracer at `tracer_ix` in the model's tracer list.
The partition coefficient, liquid cap and phase indices are copied from the
tracer so that the per-cell update reads only this variable and the state;
the installer checks that the tracer's phase order matches the model.
"""
function SurfactantPartition(tracer_ix::Int, tracer::SurfactantTracer)
    return SurfactantPartition(tracer_ix, tracer.partition, tracer.L, tracer.gas_index, tracer.liquid_index)
end

Jutul.values_per_entity(model, ::SurfactantPartition) = 3

Jutul.@jutul_secondary function update_surfactant_partition!(vals, def::SurfactantPartition, model, TracerConcentrations, Saturations, PhaseMassDensities, ix)
    gas_index = def.gas_index
    liquid_index = def.liquid_index
    for cell in ix
        ρl = PhaseMassDensities[liquid_index, cell]
        ρg = PhaseMassDensities[gas_index, cell]
        ml = Saturations[liquid_index, cell]*ρl
        mg = Saturations[gas_index, cell]*ρg
        kd = partition_mass_ratio(def.partition, ρl, ρg)
        c = TracerConcentrations[def.tracer_ix, cell]
        Cl, Cg, E = surfactant_partition(c, ml, mg, kd, def.L)
        vals[1, cell] = Cl
        vals[2, cell] = Cg
        vals[3, cell] = E
    end
    return vals
end

"""
    ClVolumetricConcentration(liquid_index)

Liquid-phase surfactant concentration `C_sw = ρ_l C_l` (kg/m³ liquid, equal
to g/L), the input of the foam factor. `liquid_index` is the liquid phase's
position in the model's phase order.
"""
struct ClVolumetricConcentration <: Jutul.ScalarVariable
    liquid_index::Int
end

Jutul.minimum_value(t::ClVolumetricConcentration) = 0.0

# Converts the liquid-phase surfactant mass fraction (kg/kg) to a volumetric
# concentration C_sw = ρ_l C_l (kg/m³ = g/L), the input of the foam factor.
Jutul.@jutul_secondary function update_liquid_volumetric_concentration!(cl_vol, def::ClVolumetricConcentration, model, SurfactantPartition, PhaseMassDensities, ix)
    liquid_index = def.liquid_index
    for cell in ix
        cl_vol[cell] = SurfactantPartition[1, cell]*PhaseMassDensities[liquid_index, cell]
    end
    return cl_vol
end
