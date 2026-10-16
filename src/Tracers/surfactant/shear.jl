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
# `g` (dim × cells, Float64) in the state.
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

# Reject configurations that would give wrong results without an error. The
# check runs at installation, when storage is allocated and before each
# gradient refresh, which covers sequential, partitioned and adjoint models
# derived after installation, including ones that reuse existing storage.
function check_foam_shear_model(model)
    # Other systems, such as black oil, assemble component fluxes without
    # face_potential_difference, so tracer and component fluxes would disagree.
    model.system isa Union{ImmiscibleSystem, JutulDarcy.LVCompositional2PhaseSystem} ||
        throw(ArgumentError("Foam shear requires an immiscible or CO2-brine two-phase system, got $(typeof(model.system))"))
    model.formulation isa Jutul.FullyImplicitFormulation ||
        throw(ArgumentError("Foam shear requires FullyImplicitFormulation, got $(typeof(model.formulation))"))
    model.domain.global_map isa Jutul.TrivialGlobalMap ||
        throw(ArgumentError("Foam shear does not support partitioned models"))
    if model.context isa Jutul.CPUJutulContext && Jutul.represented_as_adjoint(Jutul.matrix_layout(model.context))
        throw(ArgumentError("Foam shear does not support adjoint or sensitivity models"))
    end
    for key in (:mass_conservation, :tracers)
        eq = get(model.equations, key, nothing)
        eq isa ConservationLaw && eq.flow_discretization isa TwoPointPotentialFlowHardCoded ||
            throw(ArgumentError("Foam shear requires TwoPointPotentialFlowHardCoded for equation $key"))
    end
    # The multiplier scales the normal transmissibility, but the tangential
    # part uses the permeability fixed at installation.
    any(v -> haskey(v, :PermeabilityMultiplier), (model.parameters, model.secondary_variables)) &&
        throw(ArgumentError("Foam shear does not support PermeabilityMultiplier"))
    return nothing
end

function foam_shear_geometry(d)
    cc = d[:cell_centroids]
    N = d[:neighbors]
    dim, nc = size(cc)
    normals = copy(d[:normals])
    for f in axes(normals, 2)
        normals[:, f] ./= norm(normals[:, f])
    end
    k = d[:permeability]
    kd = if ndims(k) == 1
        repeat(reshape(k, 1, :), dim, 1)
    elseif size(k, 1) == 1
        repeat(k, dim, 1)
    elseif size(k, 1) == dim
        Matrix{Float64}(k)
    else
        throw(ArgumentError("Foam shear requires scalar or diagonal permeability"))
    end
    spans = vec(maximum(cc; dims = 2) - minimum(cc; dims = 2))
    # Ignore roundoff in nominally constant coordinates on single-layer grids,
    # but keep a direction that dominates a connection: it is a thin layer.
    active = findall(>(1e-8*maximum(spans)), spans)
    connections = [Tuple{Int, Int}[] for _ in 1:nc]
    for f in axes(N, 2)
        l, r = N[1, f], N[2, f]
        push!(connections[l], (f, r))
        push!(connections[r], (f, l))
        j = argmax(abs.(cc[:, r] - cc[:, l]))
        spans[j] > 0 && !(j in active) && push!(active, j)
    end
    sort!(active)
    weights = [Tuple{Int, Vector{Float64}}[] for _ in 1:nc]
    bad = falses(nc)
    for c in 1:nc
        cs = connections[c]
        D = hcat([cc[active, j] - cc[active, c] for (_, j) in cs]...)
        if isempty(cs) || rank(D; rtol = 1e-12) < length(active)
            bad[c] = true
            continue
        end
        w = 1.0 ./ vec(sum(abs2, D; dims = 1))
        M = (D .* reshape(w, 1, :))*D'
        B = M \ (D .* reshape(w, 1, :))
        for (i, (f, j)) in enumerate(cs)
            v = zeros(dim)
            v[active] .= B[:, i] .* (N[1, f] == c ? 1 : -1)
            push!(weights[c], (f, v))
        end
    end
    return (; weights, bad, normals, kd, active)
end

function FoamShearGradient(model, config::FoamShearThinning)
    check_foam_shear_model(model)
    d = model.data_domain
    g = foam_shear_geometry(d)
    nbad = count(g.bad)
    if nbad > 0.1*length(g.bad)
        @warn "Foam shear uses the face-normal fallback next to rank-deficient cells" fallback_cells=nbad total_cells=length(g.bad)
    end
    gas = findfirst(==(VaporPhase()), get_phases(model.system))
    return FoamShearGradient(config, gas, Matrix{Int}(d[:neighbors]),
        Vector{Float64}(d[:areas]), g.normals, g.kd, g.weights, g.bad)
end

function Jutul.values_per_entity(model, v::FoamShearGradient)
    check_foam_shear_model(model)
    return size(v.kd, 1)
end
# The gradient is refreshed at the equation entry point (update_equations! below).
Jutul.update_secondary_variable!(G, ::FoamShearGradient, model, state, ix) = G
function Jutul.initialize_secondary_variable_ad!(state, model, v::FoamShearGradient, symb, npartials; kwarg...)
    # The gradient is frozen in the local Jacobian, so Float64 storage needs no
    # AD partials. Preserve Jutul's conversion behavior for other input types.
    if eltype(state[symb]) === Float64
        return state
    end
    return invoke(Jutul.initialize_secondary_variable_ad!, Tuple{Any, Any, Any, Any, Any},
        state, model, v, symb, npartials; kwarg...)
end

# Strip all Dual tags without copying the underlying arrays.
foam_primal_view(A::AbstractArray{Float64}) = A
foam_primal_view(A::AbstractArray) = Jutul.mappedarray(primal_value, A)
function foam_potential_values(state)
    fields = (:Pressure, :PhaseMassDensities, :Saturations, :CapillaryPressure,
              :Transmissibilities, :TwoPointGravityDifference)
    return (; (k => foam_primal_view(state[k]) for k in fields if haskey(state, k))...)
end

function fill_foam_shear_gradients!(G, v::FoamShearGradient, pv, model, cells)
    g = zeros(size(G, 1))
    for c in cells
        fill!(g, 0.0)
        if !v.bad[c]
            for (f, w) in v.weights[c]
                l, r = v.neighbors[1, f], v.neighbors[2, f]
                q = JutulDarcy.darcy_permeability_potential_difference(f, pv, model, nothing, TPFA(l, r, 1), SPU(l, r), v.gas)
                g .+= w .* (q/pv.Transmissibilities[f])
            end
        end
        G[:, c] .= g
    end
    return G
end

function update_foam_shear_gradients!(state, model, v::FoamShearGradient)
    check_foam_shear_model(model)
    pv = foam_potential_values(state)
    G = state.FoamShearGradient
    nb = clamp(size(G, 2) ÷ Jutul.minbatch(model.context), 1, Jutul.nthreads(model.context))
    if nb == 1
        fill_foam_shear_gradients!(G, v, pv, model, axes(G, 2))
    else
        Threads.@threads for batch in 1:nb
            fill_foam_shear_gradients!(G, v, pv, model, Jutul.entity_eachindex(G, batch, nb))
        end
    end
    return G
end

# Neighbor properties are not synchronized between secondary-variable batches,
# so the gradients are refreshed once all batches finish, before the equations.
function Jutul.update_equations!(storage, model::JutulDarcy.DarcyFlowModel, dt)
    v = get(model.secondary_variables, :FoamShearGradient, nothing)
    if v isa FoamShearGradient
        update_foam_shear_gradients!(Jutul.evaluation_state(storage), model, v)
    end
    Jutul.update_equations!(storage, storage.equations, model.equations, model, dt)
    return nothing
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
