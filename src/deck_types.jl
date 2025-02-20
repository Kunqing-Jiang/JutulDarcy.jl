abstract type DeckPhaseVariables <: PhaseVariables end
abstract type AbstractReservoirDeckTable end
abstract type AbstractTablePVT <: AbstractReservoirDeckTable end

"""
    DeckPhaseViscosities(pvt, regions = nothing)

Secondary variable used to evaluate viscosities when a case is generated from a
input file. Typically not instantiated in user scripts.
"""
struct DeckPhaseViscosities{T, M, R} <: DeckPhaseVariables
    pvt::T
    thermal::M
    regions::R
    function DeckPhaseViscosities(pvt; regions = nothing, thermal = nothing)
        check_regions(regions)
        pvt_t = Tuple(pvt)
        thermal::Union{Nothing, DeckThermalViscosityTable}
        new{typeof(pvt_t), typeof(thermal), typeof(regions)}(pvt_t, thermal, regions)
    end
end

function Jutul.subvariable(p::DeckPhaseViscosities, map::FiniteVolumeGlobalMap)
    c = map.cells
    regions = Jutul.partition_variable_slice(p.regions, c)
    return DeckPhaseViscosities(p.pvt, regions = regions, thermal = p.thermal)
end

"""
    DeckPhaseMassDensities(pvt, regions = nothing)

Secondary variable used to evaluate densities when a case is generated from a
input file. Typically not instantiated in user scripts.
"""
struct DeckPhaseMassDensities{T, W, R} <: DeckPhaseVariables
    pvt::T
    watdent::W
    regions::R
    function DeckPhaseMassDensities(pvt; regions = nothing, watdent = nothing)
        check_regions(regions)
        pvt_t = Tuple(pvt)
        watdent::Union{Nothing, WATDENT}
        new{typeof(pvt_t), typeof(watdent), typeof(regions)}(pvt_t, watdent, regions)
    end
end

function Jutul.subvariable(p::DeckPhaseMassDensities, map::FiniteVolumeGlobalMap)
    c = map.cells
    regions = Jutul.partition_variable_slice(p.regions, c)
    return DeckPhaseMassDensities(p.pvt, regions = regions, watdent = p.watdent)
end

"""
DeckShrinkageFactors(pvt, regions = nothing)

Secondary variable used to evaluate shrinkage factors when a case is generated
from a input file. Typically not instantiated in user scripts.
"""
struct DeckShrinkageFactors{T, W, R} <: DeckPhaseVariables
    pvt::T
    watdent::W
    regions::R
    function DeckShrinkageFactors(pvt; watdent = nothing, regions = nothing)
        check_regions(regions)
        pvt_t = Tuple(pvt)
        watdent_t = region_wrap(watdent, regions)
        new{typeof(pvt_t), typeof(watdent_t), typeof(regions)}(pvt_t, watdent, regions)
    end
end

function Jutul.subvariable(p::DeckShrinkageFactors, map::FiniteVolumeGlobalMap)
    c = map.cells
    regions = Jutul.partition_variable_slice(p.regions, c)
    return DeckShrinkageFactors(p.pvt, regions = regions, watdent = p.watdent)
end

"""
    MuBTable(pvt, regions = nothing)

Table used to evaluate viscosities and shrinkage factors when a case is
generated from a input file. Typically used to wrap tables (e.g. PVDG, PVDO) for
use in simulation.
"""
struct MuBTable{V, I}
    pressure::V
    shrinkage::V
    shrinkage_interp::I
    viscosity::V
    viscosity_interp::I
    function MuBTable(p::T, b::T, mu::T; extrapolate = true, fix = true, kwarg...) where T<:AbstractVector
        @assert length(p) == length(b) == length(mu)
        I_b = get_1d_interpolator(p, b; cap_endpoints = !extrapolate, kwarg...)
        I_mu = get_1d_interpolator(p, mu; cap_endpoints = !extrapolate, kwarg...)
        all(extrema(b) .> 0) || throw(ArgumentError("b must be positive at at all pressures"))
        all(extrema(mu) .> 0) || throw(ArgumentError("mu must be positive at at all pressures"))
        if fix
            # Define a minimum b factor. Should really not be less than 1 at 1 atm pressure.
            ϵ = 0.01
            lowest_possible_b = min(0.99*b[1], 1.0 + ϵ)
            p0 = DEFAULT_MINIMUM_PRESSURE
            interval = (p0, p[1])
            if I_b(p0) <= 0
                p_intersect = MultiComponentFlash.Roots.find_zero(p -> I_b(p) - lowest_possible_b, interval)
                if p_intersect == p0
                    p0 = 0.999*p0
                end
                @assert p_intersect > p0
                jutul_message("PVT", "Fixing table for low pressure conditions.")
                p = copy(p)
                b = copy(b)
                mu = copy(mu)
                # Next extend the tables with more points
                pushfirst!(p, p_intersect)
                pushfirst!(p, p0)

                pushfirst!(b, lowest_possible_b)
                pushfirst!(b, lowest_possible_b - ϵ)

                mu0 = mu[1]
                pushfirst!(mu, mu0)
                pushfirst!(mu, mu0)

                I_b = get_1d_interpolator(p, b; cap_endpoints = !extrapolate, kwarg...)
                I_mu = get_1d_interpolator(p, mu; cap_endpoints = !extrapolate, kwarg...)
            end
        end
        new{T, typeof(I_b)}(p, b, I_b, mu, I_mu)
    end
end

function MuBTable(pvtx::T; kwarg...) where T<:AbstractMatrix
    N = size(pvtx, 1)
    p = vec(pvtx[:, 1])
    B = vec(pvtx[:, 2])
    b = 1.0./B
    mu = vec(pvtx[:, 3])

    # V = SVector{N, eltype(mu)}
    V = Vector{eltype(mu)}
    MuBTable(V(p), V(b), V(mu); kwarg...)
end

function viscosity(tbl::MuBTable, p)
    return tbl.viscosity_interp(p)
end

function shrinkage(tbl::MuBTable, p)
    return tbl.shrinkage_interp(p)
end

struct ConstMuBTable{R}
    p_ref::R
    b_ref::R
    b_c::R
    mu_ref::R
    mu_c::R
end

"""
    ConstMuBTable(pvtw::M) where M<:AbstractVector

Create a constant viscosity and formation-volume-factor table from a vector.
Typical usage is to wrap a PVTW type table generated from external software.
"""
function ConstMuBTable(pvtw::M) where M<:AbstractVector
    pvtw = flat_region_expand(pvtw)
    # Only one region supported atm
    pvtw = first(pvtw)
    return ConstMuBTable(pvtw[1], 1.0/pvtw[2], pvtw[3], pvtw[4], pvtw[5])
end

function viscosity(pvt::AbstractTablePVT, reg, p, cell)
    tbl = table_by_region(pvt.tab, region(reg, cell))
    return viscosity(tbl, p)
end


function viscosity(tbl::ConstMuBTable, p::T) where T
    p_r = tbl.p_ref
    μ_r = tbl.mu_ref
    c = tbl.mu_c

    F = -c*(p - p_r)
    μ = μ_r/(one(T) + F + 0.5*F^2)
    return μ::T
end

# 
function shrinkage(pvt::AbstractTablePVT, reg, p::T, cell) where T
    tbl = table_by_region(pvt.tab, region(reg, cell))
    return shrinkage(tbl, p)::T
end

function shrinkage(tbl::ConstMuBTable, p::T) where T
    p_r = tbl.p_ref
    b_r = tbl.b_ref
    c = tbl.b_c

    F = c*(p - p_r)
    b = b_r*(one(T) + F + 0.5*F^2)
    return b::T
end

struct PVTO{T} <: AbstractTablePVT
    tab::T
    function PVTO(tab::T) where T<:Tuple
        for (i, t) in enumerate(tab)
            t isa PVTOTable || throw(ArgumentError("Table $i was a $(typeof(t)), should be a PVTOTable."))
        end
        return new{T}(tab)
    end
end

struct PVTOTable{T,V}
    pos::T
    rs::V
    pressure::V
    sat_pressure::V
    shrinkage::V
    viscosity::V
end

function PVTO(pvto::Vector)
    c = map(PVTOTable, pvto)
    ct = Tuple(c)
    return PVTO(ct)
end

function PVTO(pvto::PVTOTable)
    ct = (pvto, )
    return PVTO(ct)
end

function PVTO(pvto::Dict)
    return PVTO(PVTOTable(pvto))
end

function PVTOTable(d::Dict; fix = true)
    rs = vec(copy(d["key"]))
    pos = vec(Int64.(d["pos"]))
    data = d["data"]
    # data, pos, rs = add_lower_pvto(data, pos, rs)
    p = vec(data[:, 1])
    B = vec(data[:, 2])
    b = 1.0 ./ B
    mu = vec(data[:, 3])
    p_sat = vec(p[pos[1:end-1]])
    T = typeof(pos)
    V = typeof(mu)
    @assert length(p) == length(b) == length(mu)
    @assert pos[end] == length(p) + 1
    @assert pos[1] == 1
    @assert length(p_sat) == length(rs) == length(pos)-1
    tab = PVTOTable{T, V}(pos, rs, p, p_sat, b, mu)
    if fix
        tab = extend_pvt_table_for_safe_extrapolation(tab)
    end
    return tab
end

function extend_pvt_table_line(p, mu, b, pos, i, max_p, min_b, min_mu)
    subs = pos[i]:(pos[i+1]-1)
    p_i = p[subs]
    mu_i = mu[subs]
    b_i = b[subs]
    if ismissing(min_mu)
        min_mu = 0.5*minimum(mu_i)
    end

    F_b = Jutul.get_1d_interpolator(p_i, b_i, cap_endpoints = false)
    F_mu = Jutul.get_1d_interpolator(p_i, mu_i, cap_endpoints = false)
    F_b_div_mu = Jutul.get_1d_interpolator(p_i, b_i./mu_i, cap_endpoints = false)

    min_b_div_mu = min_b*min_mu
    B_div_mu_at_p_max = F_b_div_mu(max_p)
    if B_div_mu_at_p_max < min_b_div_mu
        F = p -> F_b_div_mu(p) - min_b_div_mu
        # TODO: Ugly hack to get root finding available, could really add this
        # directly to JutulDarcy.
        p_intersect = MultiComponentFlash.Roots.find_zero(F, (p_i[1], max_p))

        # First point at intersection
        p_just_before = p_intersect - 0.01*si_unit(:bar)
        mu_just_before = F_mu(p_just_before)
        b_just_before = F_b(p_just_before)

        push!(p_i, p_just_before)
        push!(mu_i, mu_just_before)
        push!(b_i, b_just_before)
        # Second point - weak slope
        mu_intersect = F_mu(p_intersect)
        b_intersect = F_b(p_intersect)
        push!(p_i, max_p)
        push!(mu_i, mu_intersect)
        push!(b_i, b_intersect)

        @assert issorted(p_i)
    end
    return (p_i, mu_i, b_i)
end

function extend_pvt_table_for_safe_extrapolation(tab)
    b = tab.shrinkage
    mu = tab.viscosity
    p = tab.pressure
    pos = tab.pos
    min_b = minimum(b)/10.0
    min_mu = minimum(mu)
    max_p = 10*maximum(tab.pressure)
    N = length(tab.pos) - 1

    new_viscosity = Float64[]
    new_shrinkage = Float64[]
    new_pressure = Float64[]
    new_pos = Int[1]
    for i in 1:N
        p_i, mu_i, b_i = extend_pvt_table_line(p, mu, b, pos, i, max_p, min_b, min_mu)

        for j in eachindex(p_i)
            push!(new_pressure, p_i[j])
            push!(new_viscosity, mu_i[j])
            push!(new_shrinkage, b_i[j])
        end
        push!(new_pos, new_pos[i] + length(p_i))
    end
    if tab isa JutulDarcy.PVTOTable
        new_tab = JutulDarcy.PVTOTable(
            new_pos,
            copy(tab.rs),
            new_pressure,
            copy(tab.sat_pressure),
            new_shrinkage,
            new_viscosity
        )
    else
        error("$(typeof(tab)) not yet implemented")
    end
    return new_tab
end

function as_printed_table(tab::PVTO, u)
    n = tab.pos[end]-1
    P = copy(tab.pressure)
    B = 1.0./tab.shrinkage
    M = copy(tab.viscosity)

    Rs = copy(tab.rs)
    # Deal with units
    InputParser.swap_unit_system!(M, u, :viscosity)
    InputParser.swap_unit_system!(Rs, u, :u_rs)
    InputParser.swap_unit_system!(P, u, :pressure)
    InputParser.swap_unit_system!(B, u, :liquid_formation_volume_factor)

    mat = Matrix{Union{Nothing, Float64}}(undef, n, 4)
    end_records = Int[]
    for i in eachindex(Rs)
        start = tab.pos[i]
        stop = (tab.pos[i+1]-1)
        pos = start:stop
        mat[start, 1] = Rs[i]
        for j in start:stop
            mat[j, 2] = P[j]
            mat[j, 3] = B[j]
            mat[j, 4] = M[j]
        end
        push!(end_records, stop)
    end
    return ("PVTO", mat, ["Rs", "Pressure", "B_o", "mu_u"], end_records)
end

function saturated_table(t::PVTOTable)
    return saturated_table(t.sat_pressure, t.rs)
end

pvt_table_vectors(pvt::PVTOTable) = (pvt.pressure, pvt.rs, pvt.sat_pressure, pvt.pos)

function shrinkage(pvt::PVTO, reg, p, rs, cell)
    T = Base.promote_type(typeof(p), typeof(rs))
    tbl = table_by_region(pvt.tab, region(reg, cell))
    return interp_pvt(tbl, p, rs, tbl.shrinkage)::T
end

function viscosity(pvt::PVTO, reg, p, rs, cell)
    T = Base.promote_type(typeof(p), typeof(rs))
    tbl = table_by_region(pvt.tab, region(reg, cell))
    return interp_pvt(tbl, p, rs, tbl.viscosity)::T
end

struct PVTG{T} <: AbstractTablePVT
    tab::T
    function PVTG(tab::T) where T<:Tuple
        for (i, t) in enumerate(tab)
            t isa PVTGTable || throw(ArgumentError("Table $i was a $(typeof(t)), should be a PVTGTable."))
        end
        return new{T}(tab)
    end
end

function PVTG(pvtg::Vector)
    c = map(PVTGTable, pvtg)
    ct = Tuple(c)
    return PVTG(ct)
end
# PVTG - vaporized oil
struct PVTGTable{T,V}
    pos::T
    pressure::V
    rv::V
    sat_rv::V
    shrinkage::V
    viscosity::V
end

function PVTG(pvtg::PVTGTable)
    ct = (pvtg, )
    return PVTG(ct)
end

function PVTG(pvtg::Dict; kwarg...)
    return PVTG(PVTGTable(pvtg; kwarg...))
end

function PVTGTable(d::Dict; fix = true)
    pos = vec(Int64.(d["pos"]))
    data = copy(d["data"])
    for i in 1:length(pos)-1
        start = pos[i]
        stop = pos[i+1]-1
        if stop - start > 0
            if data[start, 1] > data[start+1, 1]
                # Reverse table 
                data[start:stop, :] = data[stop:-1:start, :]
            end
        end
    end
    pressure = vec(copy(d["key"]))
    if fix
        data, pos, pressure = add_lower_pvtg(data, pos, pressure)
    end
    rv = vec(data[:, 1])
    B = vec(data[:, 2])
    b = 1.0 ./ B
    mu = vec(data[:, 3])
    rv_sat = vec(rv[pos[2:end] .- 1])
    T = typeof(pos)
    V = typeof(mu)

    @assert length(rv) == length(b) == length(mu)
    @assert pos[end] == length(rv) + 1
    @assert pos[1] == 1
    @assert length(pressure) == length(rv_sat) == length(pos)-1
    return PVTGTable{T, V}(pos, pressure, rv, rv_sat, b, mu)
end

function saturated_table(t::PVTGTable)
    return saturated_table(t.pressure, t.sat_rv)
end

function saturated_table(p, r)
    if r[1] > 0
        @assert p[1] > 0.0
        p = vcat([-1.0, 0.0], p)
        r = vcat([0.0, 0.0], r)
    end
    # TODO: This is a bit unclear if it is a good idea, but it is required for
    # the SPE1 test case.
    return get_1d_interpolator(p, r, cap_end = false)
end

pvt_table_vectors(pvt::PVTGTable) = (pvt.rv, pvt.pressure, pvt.sat_rv, pvt.pos)

function shrinkage(pvt::PVTG, reg, p, rv, cell)
    p, rv = Base.promote(p, rv)
    tbl = table_by_region(pvt.tab, region(reg, cell))
    # Note: Reordered arguments!
    return interp_pvt(tbl, rv, p, tbl.shrinkage)::typeof(p)
end

function viscosity(pvt::PVTG, reg, p, rv, cell)
    p, rv = Base.promote(p, rv)
    tbl = table_by_region(pvt.tab, region(reg, cell))
    # Note: Reordered arguments!
    return interp_pvt(tbl, rv, p, tbl.viscosity)::typeof(p)
end

struct PVDO{T} <: AbstractTablePVT
    tab::T
end

function PVDO(pvdo::AbstractArray)
    c = map(MuBTable, pvdo)
    ct = Tuple(c)
    PVDO{typeof(ct)}(ct)
end

struct PVDG{T} <: AbstractTablePVT
    tab::T
end

function PVDG(pvdo::AbstractArray)
    c = map(MuBTable, pvdo)
    ct = Tuple(c)
    PVDG{typeof(ct)}(ct)
end

struct PVTW_EXTENDED{T} <: AbstractTablePVT
    tab::T
end

function PVTW_EXTENDED(pvtw_extended::AbstractArray)
    c = map(MuBTable, pvtw_extended)
    ct = Tuple(c)
    PVTW_EXTENDED{typeof(ct)}(ct)
end

struct PVTW{N, T} <: AbstractTablePVT
    tab::NTuple{N, T}
end

function PVTW(pvtw::AbstractArray)
    c = map(i -> ConstMuBTable(vec(pvtw[i, :])), axes(pvtw, 1))
    ct = Tuple(c)
    N = length(ct)
    T = typeof(ct[1])
    PVTW{N, T}(ct)
end

struct DeckThermalViscosityTable{T, V}
    visc_tab::T
    p_ref::V
    rs_ref::V
end

function DeckThermalViscosityTable(props::AbstractDict, pvt, water, oil, gas)
    visc_tab = []
    function tab_to_interp(tab)
        return map(x -> get_1d_interpolator(x[:, 1] .+ 273.15, x[:, 2]), tab)
    end
    if water
        push!(visc_tab, tab_to_interp(props["WATVISCT"]))
    end
    if oil
        push!(visc_tab, tab_to_interp(props["OILVISCT"]))
    end
    if gas
        push!(visc_tab, tab_to_interp(props["GASVISCT"]))
    end
    visc_tab = Tuple(visc_tab)
    if haskey(props, "VISCREF")
        vref = props["VISCREF"]
        rs_ref = map(x -> x[2], vref)
        p_ref = map(x -> x[1], vref)
    else
        nreg = length(first(visc_tab))
        rs_ref = fill(NaN, nreg)
        p_ref = fill(NaN, nreg)
    end
    return DeckThermalViscosityTable(visc_tab, p_ref, rs_ref)
end


struct WATDENT{N, T} <: AbstractTablePVT
    tab::NTuple{N, T}
end

function WATDENT(watdent::AbstractArray)
    c = map(rec ->  (T = rec[1], c1 = rec[2], c2 = rec[3]), watdent)
    ct = Tuple(c)
    N = length(ct)
    T = typeof(ct[1])
    return WATDENT{N, T}(ct)
end

struct PVCDO{N, T} <: AbstractTablePVT
    tab::NTuple{N, T}
end

function PVCDO(pvcdo::AbstractArray)
    if eltype(pvcdo)<:AbstractFloat
        pvcdo = [pvcdo]
    end
    c = map(x -> ConstMuBTable(vec(x)), pvcdo)
    ct = Tuple(c)
    N = length(c)
    N = length(ct)
    T = typeof(ct[1])
    PVCDO{N, T}(ct)
end


struct LinearlyCompressiblePoreVolume{V, R} <: ScalarVariable
    reference_pressure::V
    expansion::V
    regions::R
    function LinearlyCompressiblePoreVolume(; reference_pressure = 101325.0, expansion = 1e-10, regions = nothing)
        check_regions(regions, length(reference_pressure))
        reference_pressure = region_wrap(reference_pressure, regions)
        expansion = region_wrap(expansion, regions)
        new{typeof(reference_pressure), typeof(regions)}(reference_pressure, expansion, regions)
    end
end

function Jutul.subvariable(p::LinearlyCompressiblePoreVolume, map::FiniteVolumeGlobalMap)
    c = map.cells
    regions = Jutul.partition_variable_slice(p.regions, c)
    return LinearlyCompressiblePoreVolume(
        reference_pressure = p.reference_pressure,
        expansion = p.expansion,
        regions = regions
    )
end

struct TableCompressiblePoreVolume{V, R} <: ScalarVariable
    tab::V
    regions::R
    function TableCompressiblePoreVolume(tab; regions = nothing)
        check_regions(regions, length(tab))
        tab = region_wrap(tab, regions)
        new{typeof(tab), typeof(regions)}(tab, regions)
    end
end

function Jutul.subvariable(p::TableCompressiblePoreVolume, map::FiniteVolumeGlobalMap)
    c = map.cells
    regions = Jutul.partition_variable_slice(p.regions, c)
    return TableCompressiblePoreVolume(
        p.tab,
        regions = regions
    )
end

struct HystereticTableCompressiblePoreVolume{V, R} <: ScalarVariable
    tab::V
    regions::R
    function HystereticTableCompressiblePoreVolume(tab; regions = nothing)
        check_regions(regions, length(tab))
        tab = region_wrap(tab, regions)
        new{typeof(tab), typeof(regions)}(tab, regions)
    end
end

function Jutul.subvariable(p::HystereticTableCompressiblePoreVolume, map::FiniteVolumeGlobalMap)
    c = map.cells
    regions = Jutul.partition_variable_slice(p.regions, c)
    return HystereticTableCompressiblePoreVolume(
        p.tab,
        regions = regions
    )
end

struct ScalarPressureTable{V, R} <: ScalarVariable
    tab::V
    regions::R
    function ScalarPressureTable(tab; regions = nothing)
        check_regions(regions, length(tab))
        tab = region_wrap(tab, regions)
        new{typeof(tab), typeof(regions)}(tab, regions)
    end
end

function Jutul.subvariable(p::ScalarPressureTable, map::FiniteVolumeGlobalMap)
    c = map.cells
    regions = Jutul.partition_variable_slice(p.regions, c)
    return ScalarPressureTable(
        p.tab,
        regions = regions
    )
end

@jutul_secondary function update_variable!(pv, Φ::ScalarPressureTable, model, Pressure, ix)
    @inbounds for i in ix
        reg = region(Φ.regions, i)
        F = table_by_region(Φ.tab, reg)
        p = Pressure[i]
        pv[i] = F(p)
    end
end

struct HystereticScalarPressureTable{V, R} <: ScalarVariable
    tab::V
    regions::R
    function HystereticScalarPressureTable(tab; regions = nothing)
        check_regions(regions, length(tab))
        tab = region_wrap(tab, regions)
        new{typeof(tab), typeof(regions)}(tab, regions)
    end
end

function Jutul.subvariable(p::HystereticScalarPressureTable, map::FiniteVolumeGlobalMap)
    c = map.cells
    regions = Jutul.partition_variable_slice(p.regions, c)
    return HystereticScalarPressureTable(
        p.tab,
        regions = regions
    )
end

@jutul_secondary function update_variable!(pv, Φ::HystereticScalarPressureTable, model, MaxPressure, Pressure, ix)
    @inbounds for i in ix
        reg = region(Φ.regions, i)
        F = table_by_region(Φ.tab, reg)
        p = max(Pressure[i], MaxPressure[i])
        pv[i] = F(p)
    end
end

struct HystereticScalarPressureTableMin{V, R} <: ScalarVariable
    tab::V
    regions::R
    function HystereticScalarPressureTableMin(tab; regions = nothing)
        check_regions(regions, length(tab))
        tab = region_wrap(tab, regions)
        new{typeof(tab), typeof(regions)}(tab, regions)
    end
end

function Jutul.subvariable(p::HystereticScalarPressureTableMin, map::FiniteVolumeGlobalMap)
    c = map.cells
    regions = Jutul.partition_variable_slice(p.regions, c)
    return HystereticScalarPressureTableMin(
        p.tab,
        regions = regions
    )
end

@jutul_secondary function update_variable!(pv, Φ::HystereticScalarPressureTableMin, model, MinPressure, Pressure, ix)
    @inbounds for i in ix
        reg = region(Φ.regions, i)
        F = table_by_region(Φ.tab, reg)
        p = min(Pressure[i], MinPressure[i])
        pv[i] = F(p)
    end
end

function add_lower_pvto(data, pos, rs)
    ref_p = 101325.0
    first_offset = pos[2]-1
    start = 1:first_offset
    new_start = data[start, :]

    dp = ref_p - new_start[1, 1]
    for i in axes(new_start, 1)
        new_start[i, 1] += dp
        # new_start[i, 2] *= 0.99
    end
    @assert pos[1] == 1
    data = vcat(new_start, data)
    pos = vcat([1, first_offset+1], pos[2:end] .+ first_offset)
    rs = vcat(rs[1], rs)
    return (data, pos, rs)
end

function add_lower_pvtg(data, pos, pressure)
    ref_p = 101325.0
    if pressure[1] > ref_p
        first_offset = pos[2]-1
        start = 1:first_offset
        new_start = data[start, :]
        @assert pos[1] == 1
        @. new_start[:, 3] *= 1.01
        data = vcat(new_start, data)
        pos = vcat([1, first_offset+1], pos[2:end] .+ first_offset)
        pressure = vcat(ref_p, pressure)
    end
    return (data, pos, pressure)
end

function print_deck_table!(io, tab; units = :si, self = :si, print_keyword = true)
    u_target = InputParser.DeckUnitSystem(units)
    u_self = InputParser.DeckUnitSystem(self)
    u = (from = u_self, to = u_target)
    start_label, tab_as_mat, header, end_records = as_printed_table(tab, u)

    header = copy(header)
    header[1] = "-- $(header[1])"
    function fmt(x, i, j)
        if isnothing(x)
            return ""
        else
            return Jutul.Printf.@sprintf "%4.10g" x
        end
    end
    if print_keyword && !isnothing(start_label)
        println(io, start_label)
    end
    Jutul.PrettyTables.pretty_table(
        io,
        tab_as_mat,
        tf = Jutul.PrettyTables.tf_borderless,
        hlines = :none,
        formatters = fmt,
        body_hlines_format = (' ', ' ', '/', ' '),
        body_hlines = end_records,
        alignment = :l,
        header = header
    )
    if length(end_records) > 0 && last(end_records) == size(tab_as_mat, 1)
        println(io, "    /")
    end
    println(io, "/")
end

function print_deck_table!(tab; kwarg...)
    print_deck_table!(stdout, tab; kwarg...)
end
