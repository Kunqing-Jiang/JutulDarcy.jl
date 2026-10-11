abstract type AbstractSurfactantAdsorption end

"""
    checked_float(x, name; lower = 0.0, upper = Inf, positive = false)

Validate a scalar input and return it as `Float64`. `x` must be a finite
real number with `lower <= x <= upper` (and `x > 0` when `positive`),
judged on the value as given; it must then convert to a finite `Float64`.
Rejected inputs raise `ArgumentError` naming the parameter. Inputs that
only round to the range after conversion, such as a tiny negative
`BigFloat`, are rejected.
"""
function checked_float(x, name; lower = 0.0, upper = Inf, positive = false)
    range = positive ? "(0, Inf)" : "[$lower, $upper]"
    if !(x isa Real && isfinite(x) && lower <= x <= upper && (!positive || x > 0))
        throw(ArgumentError("$name must be a finite real scalar in $range"))
    end
    y = try
        Float64(x)
    catch
        throw(ArgumentError("$name must be representable as Float64"))
    end
    isfinite(y) || throw(ArgumentError("$name must be finite in Float64"))
    return y
end

"""
    primal_value(x)

The plain number behind `x`, with every `ForwardDiff.Dual` layer removed.
Branch conditions in the surfactant code compare these values, so a kink
selects its branch, and hence its one-sided derivative, by value alone.
ForwardDiff 1 orders Dual numbers with equal values by their partials, which
would make that choice depend on the direction of differentiation.
"""
primal_value(x) = x
primal_value(x::Jutul.ForwardDiff.Dual) = primal_value(Jutul.ForwardDiff.value(x))

function check_adsorption_concentration(C)
    v = primal_value(C)
    if !(isfinite(v) && v >= 0)
        throw(DomainError(v, "Surfactant concentration must be finite and non-negative (kg/m³ liquid)"))
    end
end

"""
    LinearSurfactantAdsorption(kappa)

Reversible equilibrium adsorption `a(C) = kappa*C`, in kg/kg rock, for
liquid concentration `C` in kg/m³. `kappa` is finite and non-negative (m³/kg).
"""
struct LinearSurfactantAdsorption <: AbstractSurfactantAdsorption
    kappa::Float64
    LinearSurfactantAdsorption(kappa) = new(checked_float(kappa, "kappa"))
end

function (a::LinearSurfactantAdsorption)(C)
    check_adsorption_concentration(C)
    v = a.kappa*C
    isfinite(Jutul.value(v)) || throw(DomainError(Jutul.value(C), "Linear adsorption overflow"))
    return v
end

"""
    LangmuirSurfactantAdsorption(Q, b)

Reversible adsorption `Q*b*C/(1+b*C)` in kg/kg rock. `Q` is the capacity
(kg/kg rock), `b` is in m³/kg, and `C` is in kg/m³ liquid. Constants and
their product must be finite and non-negative.
"""
struct LangmuirSurfactantAdsorption <: AbstractSurfactantAdsorption
    Q::Float64
    b::Float64
    function LangmuirSurfactantAdsorption(Q, b)
        Q = checked_float(Q, "Q")
        b = checked_float(b, "b")
        isfinite(Q*b) || throw(ArgumentError("Q*b must be finite in Float64"))
        return new(Q, b)
    end
end

function (a::LangmuirSurfactantAdsorption)(C)
    check_adsorption_concentration(C)
    if a.Q == 0 || a.b == 0
        return zero(C)
    elseif primal_value(C) <= 1/a.b
        z = a.b*C
        return a.Q*(z/(1 + z))
    else
        r = (1/a.b)/C
        return a.Q/(1 + r)
    end
end

"""
    TabulatedSurfactantAdsorption(C, a)

Monotone piecewise linear adsorption in kg/kg rock versus kg/m³ liquid.
The copied one-dimensional arrays must start at `(0,0)` and contain at
least two points, strictly increasing concentrations and non-decreasing
adsorption. At zero the right slope is used; at other knots the left slope
is used. Above the last concentration adsorption is constant.
"""
struct TabulatedSurfactantAdsorption{I} <: AbstractSurfactantAdsorption
    interpolator::I
    function TabulatedSurfactantAdsorption(C, a)
        if !(C isa AbstractVector && a isa AbstractVector && length(C) == length(a) && length(C) >= 2)
            throw(ArgumentError("Adsorption table needs equally sized vectors with at least two points"))
        end
        X = [checked_float(x, "Table concentration") for x in C]
        Y = [checked_float(y, "Table adsorption") for y in a]
        if !(X[1] == Y[1] == 0 && all(diff(X) .> 0) && all(diff(Y) .>= 0) && all(isfinite, diff(Y)./diff(X)))
            throw(ArgumentError("Adsorption table must start at (0,0), increase in concentration, and have finite non-negative slopes"))
        end
        I = Jutul.get_1d_interpolator(X, Y; cap_endpoints = false, constant_dx = false)
        return new{typeof(I)}(I)
    end
end

function (a::TabulatedSurfactantAdsorption)(C)
    check_adsorption_concentration(C)
    X, F = a.interpolator.X, a.interpolator.F
    v = primal_value(C)
    if v > X[end]
        return zero(C) + F[end]
    end
    # Linear interpolation on the segment chosen by value: the first segment
    # at zero, the segment to the left at every other knot.
    ix = clamp(searchsortedfirst(X, v) - 1, 1, length(X) - 1)
    X_0 = X[ix]
    F_0 = F[ix]
    return F_0 + (F[ix + 1] - F_0)/(X[ix + 1] - X_0)*(C - X_0)
end

adsorption_is_zero(a::LinearSurfactantAdsorption) = a.kappa == 0
adsorption_is_zero(a::LangmuirSurfactantAdsorption) = a.Q == 0 || a.b == 0
adsorption_is_zero(a::TabulatedSurfactantAdsorption) = all(iszero, a.interpolator.F)

"""
    SurfactantRockMass()

Static rock mass per reservoir cell (kg). The default is
`rock_density*(1-porosity)*bulk_volume` from the initial Cells data, without
net-to-gross or pore-volume overrides. Install with `set_surfactant_model!`
before `setup_parameters`. Override with a finite non-negative scalar or
one value per cell, e.g. `setup_parameters(rmodel; SurfactantRockMass = MR)`.
Use an explicit override to represent only net rock or a domain without
bulk geometry. Initialized masses stay fixed when pressure changes.
"""
struct SurfactantRockMass <: Jutul.ScalarVariable end

function rock_mass_values(x, n, name)
    if x isa Real
        return fill(checked_float(x, name), n)
    elseif x isa AbstractVector && length(x) == n
        return [checked_float(v, name) for v in x]
    else
        throw(ArgumentError("$name must be a real scalar or a vector of $n values"))
    end
end

function Jutul.default_parameter_values(data_domain, model, param::SurfactantRockMass, symb)
    n = Jutul.number_of_entities(model, param)
    fields = (:rock_density, :porosity, :volumes)
    for k in fields
        haskey(data_domain, k, Cells()) || throw(ArgumentError("Default rock mass requires Cells data $k; provide SurfactantRockMass explicitly"))
    end
    rho, phi, V = map(fields) do k
        rock_mass_values(data_domain[k, Cells()], n, string(k))
    end
    all(>(0), rho) || throw(ArgumentError("rock_density must be positive"))
    all(<=(1), phi) || throw(ArgumentError("porosity must be at most one"))
    all(>(0), V) || throw(ArgumentError("bulk volumes must be positive"))
    return rock_mass_values(rho.*(1 .- phi).*V, n, "SurfactantRockMass")
end

function Jutul.initialize_parameter_value!(parameters, data_domain, model, param::SurfactantRockMass, symb,
        initializer::Union{Jutul.AbstractJutulStorage, AbstractDict}; kwarg...)
    if haskey(initializer, symb)
        initializer = Dict{Symbol, Any}(pairs(initializer))
        initializer[symb] = rock_mass_values(initializer[symb], Jutul.number_of_entities(model, param), string(symb))
    end
    return invoke(Jutul.initialize_parameter_value!,
        Tuple{Any, Any, Any, Any, Any, Union{Jutul.AbstractJutulStorage, AbstractDict}},
        parameters, data_domain, model, param, symb, initializer; kwarg...)
end
