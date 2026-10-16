@doc raw"""
    FoamShearThinning(; fmcap, epcap, sigma)

Least-squares gas shear thinning with the TPFA normal potential retained.
`fmcap > 0` and `epcap >= 0` define the capillary-number response; surface
tension `sigma > 0` is in N/m and must match the calibration of `fmcap`.
An exponent of zero disables shear entirely.

For an internal face, let `Q` be the gas permeability-potential difference
before multiplying by mobility, `A` its area, and `n̂` its unit normal. Bars
denote primal values. The cell least-squares fit `g` uses oriented `Q/T`
drops, with inverse squared distance weights in the active coordinates.
With diagonal permeability `K`,
```math
\mathbf t=(I-\hat{\mathbf n}\hat{\mathbf n}^{T})
    (K_l\mathbf g_l+K_r\mathbf g_r)/2,\qquad
N=\frac{\sqrt{(\bar Q/A)^2+|\mathbf t|^2}}{\sigma},\qquad
\eta=\frac{(\bar Q/A)^2}{(\bar Q/A)^2+|\mathbf t|^2},
```
with `η=0` when both components vanish. `N` is a dimensionless
pressure-gradient capillary number (`Q/A` and `K*g` are in Pa m), not a
velocity divided by surface tension. Writing `b=fmcap`, `e=epcap` and `FM`
for the zero-shear foam factor of the cell upstream of `Q`,
```math
F_{cap}(N)=\begin{cases}1 & N\le b\\(b/N)^e & N>b,\end{cases}\qquad
m=\frac{1}{FM+(1-FM)F_{cap}},
```
```math
s=\begin{cases}0 & N\le b\\
e(1-\overline{FM})F_{cap}/[\overline{FM}+(1-\overline{FM})F_{cap}] & N>b,
\end{cases}\qquad
\widehat Q=m(FM^{AD},\bar N)Q+\bar m\,s\,\frac{1+\eta}{2}(Q-\bar Q).
```
The gas flux is the upstream mobility times `Q̂`. The second term is zero in
value. It approximates the Jacobian contribution of the frozen tangential
gradient: the log-sensitivity of `N` to `Q` lies in `[η, 1]`, and the
midpoint minimizes the worst-case error. If either cell stencil is rank
deficient (`rtol=1e-12`), `N=abs(Q)/(A*sigma)` keeps its AD dependence and
`Q̂=m*Q`. Coordinate spans at most `1e-8` times the largest span are inactive
unless they dominate an internal neighbor displacement.

Install through `set_surfactant_model!` before creating states or simulators,
and rebuild the model and simulator to change the configuration. Requires a
two-phase immiscible or CO2-brine system, fully implicit and unpartitioned
forward simulation with `TwoPointPotentialFlowHardCoded`, and no
`PermeabilityMultiplier`; other configurations are rejected. Tested for scalar
or diagonal permeability. Thermal models, NNCs, other gas-phase tracers and
number types other than Float64 and ForwardDiff are not supported and not
checked. Boundary sources and perforations keep their cell foam factors.
"""
struct FoamShearThinning
    fmcap::Float64
    epcap::Float64
    sigma::Float64
    function FoamShearThinning(; fmcap, epcap, sigma)
        b, e, sig = Float64(fmcap), Float64(epcap), Float64(sigma)
        all(isfinite, (b, e, sig)) && b > 0 && e >= 0 && sig > 0 ||
            throw(ArgumentError("FoamShearThinning requires finite fmcap > 0, epcap >= 0 and sigma > 0"))
        return new(b, e, sig)
    end
end

foam_shear_enabled(::Nothing) = false
foam_shear_enabled(c::FoamShearThinning) = c.epcap > 0

# Installed shear thinning: configuration, gas phase and the geometry computed
# once at installation. As a secondary variable it holds the cell gradients
# `g` (dim × cells, Float64) in the state; installation and refresh are in
# shear_cache.jl, the least-squares fit in cell_gradient.jl.
struct FoamShearGradient <: Jutul.VectorVariables
    config::FoamShearThinning
    gas::Int
    neighbors::Matrix{Int}
    areas::Vector{Float64}
    normals::Matrix{Float64}
    kd::Matrix{Float64}
    # Per cell: (face, least-squares weight) for each connection.
    weights::Vector{Vector{Tuple{Int, Vector{Float64}}}}
    bad::BitVector
end

foam_fcap(n, b, e) = primal_value(n) <= b ? one(n) : (b/n)^e
foam_shear_multiplier(fm, n, c) = inv(fm + (1 - fm)*foam_fcap(n, c.fmcap, c.epcap))

# Capillary number N and normal share η from primal Q and the cell gradients.
function foam_ls_magnitude(f, Q, G, v::FoamShearGradient)
    l, r = v.neighbors[1, f], v.neighbors[2, f]
    Kg(j) = (v.kd[j, l]*primal_value(G[j, l]) + v.kd[j, r]*primal_value(G[j, r]))/2
    vn = 0.0
    for j in axes(v.kd, 1)
        vn += Kg(j)*v.normals[j, f]
    end
    t2 = 0.0
    for j in axes(v.kd, 1)
        t2 += (Kg(j) - vn*v.normals[j, f])^2
    end
    un = primal_value(Q)/v.areas[f]
    mag2 = un^2 + t2
    return sqrt(mag2)/v.config.sigma, mag2 > 0 ? un^2/mag2 : 0.0
end

function foam_corrected_potential(Q, fm, n, eta, c::FoamShearThinning)
    m = foam_shear_multiplier(fm, n, c)
    F = foam_fcap(n, c.fmcap, c.epcap)
    fm0 = primal_value(fm)
    s = n <= c.fmcap ? 0.0 : c.epcap*(1 - fm0)*F/(fm0 + (1 - fm0)*F)
    # This zero-valued term supplies a local approximate Jacobian, without
    # adding neighbor-of-neighbor entries to the TPFA sparsity pattern.
    return m*Q + primal_value(m)*s*((1 + eta)/2)*(Q - primal_value(Q))
end

function JutulDarcy.face_potential_difference(v::FoamShearGradient, face, phase, state, model, upw, Q)
    phase == v.gas || return Q
    fm = state.FmFactor[upwind(upw, identity, Q)]
    if v.bad[v.neighbors[1, face]] || v.bad[v.neighbors[2, face]]
        # Keep the potential derivative on a rank-deficient stencil.
        n = abs(Q)/(v.areas[face]*v.config.sigma)
        return foam_shear_multiplier(fm, n, v.config)*Q
    end
    n, eta = foam_ls_magnitude(face, Q, state.FoamShearGradient, v)
    return foam_corrected_potential(Q, fm, n, eta, v.config)
end
