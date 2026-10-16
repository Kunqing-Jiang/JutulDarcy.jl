# Least-squares cell gradients of a phase potential. Each cell fits `g` to the
# oriented `Q/T` drops to its neighbors, with inverse squared distance weights
# in the active coordinates. Only the mesh and the potential are used.

# Per cell, the (face, weight) pairs whose sum of weight times drop gives `g`.
# `bad` marks cells whose stencil does not span the active coordinates
# (`rtol=1e-12`); their gradient is zero.
function least_squares_gradient_weights(d)
    cc = d[:cell_centroids]
    N = d[:neighbors]
    dim, nc = size(cc)
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
    return (; weights, bad, active)
end

# Strip all Dual tags without copying the underlying arrays.
primal_view(A::AbstractArray{Float64}) = A
primal_view(A::AbstractArray) = Jutul.mappedarray(primal_value, A)
function primal_potential_values(state)
    fields = (:Pressure, :PhaseMassDensities, :Saturations, :CapillaryPressure,
              :Transmissibilities, :TwoPointGravityDifference)
    return (; (k => primal_view(state[k]) for k in fields if haskey(state, k))...)
end

function fill_least_squares_gradients!(G, weights, bad, neighbors, pv, model, phase, cells)
    g = zeros(size(G, 1))
    for c in cells
        fill!(g, 0.0)
        if !bad[c]
            for (f, w) in weights[c]
                l, r = neighbors[1, f], neighbors[2, f]
                q = JutulDarcy.darcy_permeability_potential_difference(f, pv, model, nothing, TPFA(l, r, 1), SPU(l, r), phase)
                g .+= w .* (q/pv.Transmissibilities[f])
            end
        end
        G[:, c] .= g
    end
    return G
end
