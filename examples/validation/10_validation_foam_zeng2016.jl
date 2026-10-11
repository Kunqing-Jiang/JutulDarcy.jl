# # 10. Surfactant foam (2ph CO₂-water)
# <tags: Immiscible, Tracers, Validation>
# One-dimensional CO₂ foam flood with a partitioning surfactant, Case I of
# [Zeng, Ma, Farajzadeh, Puerto, Biswal and Hirasaki (2016), "Effect of
# Surfactant Partitioning Between Gaseous Phase and Aqueous Phase on CO₂ Foam
# Transport for Enhanced Oil Recovery", Transport in Porous Media 114,
# 777–793](https://doi.org/10.1007/s11242-016-0743-6). Dry CO₂ with
# dissolved surfactant is injected into a water-saturated core; the surfactant
# partitions between the gas and the water and the water-phase concentration
# controls the foam strength through the STARS-type foam model.
#
# The model uses the immiscible two-phase system with the CO₂-brine densities
# at the experimental pressure and temperature, a `SurfactantTracer` whose
# volumetric partition coefficient K = 1 is converted to the mass ratio with
# those densities, dispersion matching the paper's Péclet number of 500, and
# the foam parameters of the paper. After 0.5 injected pore volumes the gas
# saturation, aqueous surfactant concentration and apparent viscosity are
# compared with the curves digitized from the paper's Fig. 5, which are
# stored next to this example (`data/foam_zeng2016/README.md` describes the
# digitization).
using Jutul, JutulDarcy, GLMakie
# ## Grid, rock and fluids
# The core is 1 ft long; the cross-section is chosen so that the superficial
# velocity of 20 ft/day gives the paper's injection rate.
L = 0.3048
area = (0.01L)^2
porosity = 0.25
pressure = 27.579e6
temperature = 393.15
velocity = 20L/86400
viscosities = [0.24e-3, 0.045e-3] # water, CO₂ (Pa s)
nx = 100
mesh = UnstructuredMesh(CartesianMesh((nx, 1, 1), (L, sqrt(area), sqrt(area))))
domain = reservoir_domain(mesh; permeability = si_unit(:darcy), porosity = porosity, temperature = temperature)
injector = setup_well(domain, 1; name = :Injector, radius = 0.0005, simple_well = true)
producer = setup_well(domain, nx; name = :Producer, radius = 0.0005, simple_well = true)
# The phase densities at the experimental conditions come from the CO₂-brine
# property tables; the flow model itself is immiscible with constant densities.
pvt_model = setup_reservoir_model(domain, :co2brine; wells = [injector, producer], co2_physics = :kvalue)
rho = collect(reservoir_model(pvt_model).secondary_variables[:PhaseMassDensities].tab(pressure, temperature))
sys = ImmiscibleSystem(JutulDarcy.get_phases(:co2brine); reference_densities = rho)
model = setup_reservoir_model(domain, sys; wells = [injector, producer])
replace_variables!(model, PhaseMassDensities = ConstantCompressibilityDensities(sys, pressure, rho, [0.0, 0.0]))
rmodel = reservoir_model(model)
# ## Surfactant tracer and foam model
# K is the volumetric partition coefficient of the paper; the tracer takes the
# mass ratio k_d = K ρ_l/ρ_g. The physical dispersion coefficient follows from
# Pe = L u/(φ D) = 500.
K = 1.0
kd = K*rho[1]/rho[2]
D_paper = L*(velocity/porosity)/500
tracer = SurfactantTracer(rmodel.system; k_d = kd, D = (gas = D_paper, liquid = D_paper))
add_tracers_to_model!(model, tracer)
# Brooks-Corey relative permeabilities with the paper's exponents and end
# points; the CO₂-brine phases carry the liquid/gas labels.
s = range(0, 1; length = 1001)
krw = JutulDarcy.brooks_corey_relperm.(s; n = 4.2, residual = 0.05, residual_total = 0.1, kr_max = 0.2)
krg = JutulDarcy.brooks_corey_relperm.(s; n = 1.3, residual = 0.05, residual_total = 0.1, kr_max = 0.94)
relperm = JutulDarcy.ReservoirRelativePermeabilities(
    og = PhaseRelativePermeability(s, krw; label = :og),
    g = PhaseRelativePermeability(s, krg; label = :g))
replace_variables!(model, RelativePermeabilities = relperm)
JutulDarcy.add_relperm_parameters!(model)
# The foam factor divides the gas viscosity by 1 + fmmob F_dry F_surf with the
# paper's parameters (fmsurf in g/L of the water phase).
set_surfactant_model!(model; foam = FmFactor(epdry = 500.0, fmdry = 0.25, fmsurf = 2.0, epsurf = 1.0, fommb = 500.0))
append!(rmodel.output_variables, [:PhaseViscosities, :FmFactor, :ClVolumetricConcentration, :RelativePermeabilities])
unique!(rmodel.output_variables)
# Constant viscosities are written into the base viscosity parameter that the
# foam installer created; the wells keep their own viscosities.
parameters = setup_parameters(model)
parameters[:Reservoir][:BasePhaseViscosities] .= reshape(viscosities, 2, 1)
for name in (:Injector, :Producer)
    parameters[name][:PhaseViscosities] .= reshape(viscosities, 2, 1)
end
# ## Injection of 0.5 pore volumes of CO₂ with 2.5 g/L surfactant
rate = velocity*area
total_time = 0.5sum(pore_volume(model, parameters))/rate
controls = Dict(
    :Injector => InjectorControl(TotalRateTarget(rate), [0.0, 1.0]; density = rho[2], tracers = [2.5/rho[2]]),
    :Producer => ProducerControl(BottomHolePressureTarget(pressure)))
forces = setup_reservoir_forces(model; control = controls)
check_surfactant_injection(model, forces)
state0 = setup_reservoir_state(model; Pressure = pressure, Saturations = [1.0, 0.0], TracerConcentrations = 0.0)
ws, states = simulate_reservoir(state0, model, [total_time];
    parameters = parameters, forces = forces, max_timestep = 0.05, timesteps = :none,
    linear_solver = nothing, info_level = -1, output_path = nothing,
    nonlinear_tolerance = 1e-7, tol_cnv = 1e-6, tol_mb = 1e-10, tol_cnv_well = 1e-6)
state = states[end]
# ## Profiles at 0.5 PV
# The apparent viscosity follows from the total mobility: with zero capillary
# pressure both phases see the same pressure gradient, so
# μ_app = 1/(k_rw/μ_w + k_rg/μ_g) in cP.
x = vec(domain[:cell_centroids][1, :])/L
sg = vec(state[:Saturations][2, :])
csw = vec(state[:ClVolumetricConcentration])
mu_app = 1000 ./ vec(sum(state[:RelativePermeabilities] ./ state[:PhaseViscosities]; dims = 1))
# ## Digitized reference curves
# `reference.csv` holds (quantity, K, x/L, value) points read off Fig. 5. Points
# with 0.05 ≤ x/L ≤ 0.95 enter the error measures; the model is interpolated
# at the digitized positions.
data_dir = pkgdir(JutulDarcy, "examples", "validation", "data", "foam_zeng2016")
reference = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()
for line in Iterators.drop(eachline(joinpath(data_dir, "reference.csv")), 1)
    quantity, k, xr, value = split(line, ',')
    parse(Float64, k) == K || continue
    rx, ry = get!(reference, String(quantity), (Float64[], Float64[]))
    push!(rx, parse(Float64, xr)); push!(ry, parse(Float64, value))
end
function interpolate_model(xq, xm, ym)
    xq <= xm[1] && return ym[1]
    xq >= xm[end] && return ym[end]
    j = searchsortedlast(xm, xq)
    t = (xq - xm[j])/(xm[j+1] - xm[j])
    return ym[j] + t*(ym[j+1] - ym[j])
end
function rmse(quantity, ym)
    rx, ry = reference[quantity]
    inside = (rx .>= 0.05) .& (rx .<= 0.95)
    err = [interpolate_model(xi, x, ym) for xi in rx[inside]] .- ry[inside]
    return sqrt(sum(abs2, err)/length(err))
end
println("RMSE against the digitized Fig. 5 for K = $K:")
for (quantity, ym, unit) in (("gas_saturation", sg, ""), ("csw_g_L", csw, " g/L"), ("mu_app_cP", mu_app, " cP"))
    println("  ", quantity, " = ", round(rmse(quantity, ym); sigdigits = 4), unit)
end
# ## Plot
fig = Figure(size = (900, 900))
for (i, (quantity, ym, label, ymax)) in enumerate((
        ("gas_saturation", sg, "Gas saturation", 1.0),
        ("csw_g_L", csw, "Aqueous surfactant (g/L)", 6.0),
        ("mu_app_cP", mu_app, "Apparent viscosity (cP)", 30.0)))
    ax = Axis(fig[i, 1]; ylabel = label, limits = (0, 1, 0, ymax), xlabel = i == 3 ? "x / L" : "")
    lines!(ax, x, ym; linewidth = 2, label = "JutulDarcy, K = $K")
    rx, ry = reference[quantity]
    scatter!(ax, rx, ry; color = :transparent, strokecolor = :black, strokewidth = 1, markersize = 7, label = "Zeng et al. (2016), Fig. 5")
    i == 1 && axislegend(ax; position = :rt)
end
fig
