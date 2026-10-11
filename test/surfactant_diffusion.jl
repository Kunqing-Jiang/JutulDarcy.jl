using JutulDarcy, Jutul, Test, LinearAlgebra, Printf
import JutulDarcy.Tracers: SurfactantDiffusivities, tracer_face_flux, tracer_total_mass_flux, surfactant_partition

# High precision avoids cancellation in the erf series for |x| <= 6.
function diffusion_erfc(x::Float64)
    setprecision(BigFloat, 256) do
        X = BigFloat(x); s = zero(BigFloat); term = X; n = 0
        while true
            add = term/(2n + 1); s += add
            abs(add) < big(1e-40) && break
            n += 1; term *= -X^2/n
        end
        Float64(1 - 2/sqrt(BigFloat(pi))*s)
    end
end
const RHO = (liquid = 1000.0, gas = 500.0)
graded_widths(nx, a) = diff([(i/nx) - a*sin(2pi*i/nx)/(2pi) for i in 0:nx])

# Build a model; `order` is the phase tuple, densities given per physical phase.
function diffusion_model(dims, widths, phi; sl, kd, Dl, Dg, c, order = (LiquidPhase(), VaporPhase()), general_ad = false)
    mesh = UnstructuredMesh(CartesianMesh(dims, widths))
    domain = reservoir_domain(mesh; permeability = 1e-12, porosity = phi)
    rho = [ph isa VaporPhase ? RHO.gas : RHO.liquid for ph in order]
    sys = ImmiscibleSystem(order; reference_densities = rho)
    model = setup_reservoir_model(domain, sys; general_ad = general_ad)
    replace_variables!(model, PhaseMassDensities = ConstantCompressibilityDensities(sys, 1e7, rho, [1e-10, 1e-10]))
    add_tracers_to_model!(model, SurfactantTracer(sys; k_d = kd, D = (gas = Dg, liquid = Dl)))
    set_surfactant_model!(model)
    prm = setup_parameters(model)
    S = [ph isa VaporPhase ? 1 - sl : sl for ph in order]
    state0 = setup_reservoir_state(model; Pressure = 1e7, Saturations = S, TracerConcentrations = reshape(c, 1, :))
    return (; model, prm, state0)
end
function diffusion_simulate(case, dt, nt)
    res = simulate_reservoir(case.state0, case.model, fill(dt, nt); parameters = case.prm,
        timesteps = :none, max_timestep = dt, linear_solver = nothing, info_level = -1, output_path = nothing)
    @test length(res.states) == nt
    isempty(res.states) && error("Diffusion simulation returned no states")
    p = res.states[end][:Pressure]
    @test all(isfinite, p)
    @test maximum(abs.(p .- 1e7)) <= 1e-6
    return vec(res.states[end][:TracerConcentrations])
end
# Hand coefficient per face: list of (i, j, k) with k the c-coefficient.
function hand_faces_1d(w, phi, A, coef)
    [(i, i + 1, coef/((w[i]/2)/(phi[i]*A) + (w[i+1]/2)/(phi[i+1]*A))) for i in 1:length(w)-1]
end
function phase_coef(sl, kd, Dl, Dg)
    sg = 1 - sl
    ml, mg = sl*RHO.liquid, sg*RHO.gas
    B = ml + kd*mg; mf = ml + mg
    # sum_a D_a S_a rho_a C_a / c ; times the geometric factor phi A / d per half
    return (Dl*sl*RHO.liquid*mf/B + Dg*sg*RHO.gas*kd*mf/B), mf
end
function reference(faces, acc, c0, dt, nt)
    n = length(acc); Mx = Matrix(Diagonal(acc))
    for (i, j, k) in faces
        Mx[i, i] += dt*k; Mx[j, j] += dt*k; Mx[i, j] -= dt*k; Mx[j, i] -= dt*k
    end
    F = lu(Mx); c = copy(c0)
    for _ in 1:nt
        c = F \ (acc .* c)
    end
    return c
end

@testset "Diffusion face geometry" begin
    @testset "Uniform orthogonal grid" begin
        b = diffusion_model((3, 1, 1), ([1.0, 1.0, 1.0], [1.0], [1.0]), 0.25;
            sl = 0.7, kd = 2.0, Dl = 1e-7, Dg = 5e-7, c = zeros(3))
        T = b.prm[:Reservoir][:SurfactantDiffusivities]
        @test all(isapprox.(T[1, :], 0.25e-7; rtol = 1e-12, atol = 0))
        @test all(isapprox.(T[2, :], 1.25e-7; rtol = 1e-12, atol = 0))
        zero_case = diffusion_model((3, 1, 1), ([1.0, 1.0, 1.0], [1.0], [1.0]), 0.25;
            sl = 0.7, kd = 2.0, Dl = 0.0, Dg = 0.0, c = zeros(3))
        @test all(iszero, zero_case.prm[:Reservoir][:SurfactantDiffusivities])
    end
    @testset "Graded widths and cell porosities in both phase orders" begin
        w = graded_widths(40, 0.5)
        phi = [0.1 + 0.3*(i % 3)/2 for i in 1:40]
        for order in ((LiquidPhase(), VaporPhase()), (VaporPhase(), LiquidPhase()))
            b = diffusion_model((40, 1, 1), (w, [0.01], [0.01]), phi;
                sl = 0.7, kd = 2.0, Dl = 1e-7, Dg = 5e-7, c = zeros(40), order = order)
            T = b.prm[:Reservoir][:SurfactantDiffusivities]
            for (phase, ph) in enumerate(order)
                D = ph isa VaporPhase ? 5e-7 : 1e-7
                hand = [1/((w[i]/2)/(phi[i]*D*1e-4) + (w[i+1]/2)/(phi[i+1]*D*1e-4)) for i in 1:39]
                @test maximum(abs.(T[phase, :] .- hand) ./ hand) <= 1e-12
            end
        end
    end
    @testset "Different face areas and distances in two dimensions" begin
        dx, dy, dz, phi = 0.2, 0.3, 0.05, 0.25
        b = diffusion_model((3, 2, 1), (fill(dx, 3), fill(dy, 2), [dz]), phi;
            sl = 0.7, kd = 2.0, Dl = 1e-7, Dg = 5e-7, c = zeros(6))
        rm = reservoir_model(b.model)
        N = rm.data_domain[:neighbors]
        T = b.prm[:Reservoir][:SurfactantDiffusivities]
        for f in axes(N, 2), (phase, D) in enumerate((1e-7, 5e-7))
            left, right = N[:, f]
            hand = right - left == 1 ? phi*D*dy*dz/dx : phi*D*dx*dz/dy
            @test isapprox(T[phase, f], hand; rtol = 1e-12, atol = 0)
        end
        domain = deepcopy(rm.data_domain)
        domain[:nnc, nothing] = []
        @test_throws ArgumentError Jutul.default_parameter_values(domain, rm, SurfactantDiffusivities(1), :SurfactantDiffusivities)
        # Invalid geometry must be rejected before Jutul's parameter clamping.
        for factor in (-1.0, NaN, Inf)
            bad = deepcopy(rm.data_domain)
            bad[:areas] .*= factor
            @test_throws ArgumentError Jutul.default_parameter_values(bad, rm, SurfactantDiffusivities(1), :SurfactantDiffusivities)
        end
    end
end

@testset "Dissolved diffusion flux" begin
    b = diffusion_model((3, 1, 1), ([1.0, 1.0, 1.0], [1.0], [1.0]), 0.25;
        sl = 0.7, kd = 2.0, Dl = 1e-7, Dg = 5e-7, c = zeros(3))
    rm = reservoir_model(b.model)
    tracer = rm.equations[:tracers].flux_type.tracers[1]
    T = b.prm[:Reservoir][:SurfactantDiffusivities]
    S = [0.7 0.4; 0.3 0.6]
    rho = [1000.0 900.0; 500.0 400.0]
    P = [1e-3 2e-3; 2e-3 4e-3; 0.0 0.0]
    state = (SurfactantPartition = P, Saturations = S, PhaseMassDensities = rho,
        SurfactantDiffusivities = T, TracerConcentrations = [0.01 0.03])
    flux(st; left = 1, right = 2, t = tracer) = tracer_face_flux(t, rm, st,
        (0.0, 0.0), 1, SPU(left, right), TPFA(left, right, 1), 1)
    uniform = (state..., SurfactantPartition = repeat(P[:, 1:1], 1, 2))
    @test flux(uniform) == 0.0
    @test flux(state; left = 2, right = 1) == -flux(state)
    # Independently use φ D A/Δx, the minimum saturation and arithmetic density.
    hand = -0.25e-7*0.4*950*1e-3 - 1.25e-7*0.3*450*2e-3
    @test isapprox(flux(state), hand; rtol = 1e-12, atol = 0)
    absent_gas = (state..., Saturations = [1.0 0.4; 0.0 0.6])
    liquid_only = -0.25e-7*0.4*950*1e-3
    @test flux(absent_gas) == flux((absent_gas..., SurfactantPartition = [1e-3 2e-3; 0.0 0.0; 0.0 0.0]))
    @test isapprox(flux(absent_gas), liquid_only; rtol = 1e-12, atol = 0)
    @test liquid_only != 0
    @test flux((absent_gas..., SurfactantPartition = [1e-3 1e-3; 2e-3 4e-3; 0.0 0.0])) == 0
    nondiffusive = SurfactantTracer(rm.system; k_d = 2.0)
    @test flux(state; t = nondiffusive) == 0
    # The upwind-only entry point is refused for a surfactant, while ordinary
    # and polymer tracers reach it through the face flux hook unchanged.
    @test_throws ArgumentError tracer_total_mass_flux(tracer, rm, state, (1.0, 2.0), 1, SPU(1, 2))
    plain = MultiPhaseTracer(rm.system, [1, 2])
    @test tracer_face_flux(plain, rm, state, (1.0, 2.0), 1, SPU(1, 2), TPFA(1, 2, 1), 1) ==
        tracer_total_mass_flux(plain, rm, state, (1.0, 2.0), 1, SPU(1, 2))
    polymer = JutulDarcy.Tracers.PolymerTracer(1)
    state_polymer = (state..., EffectivePolymerViscosityMultipliers = [1.0 1.0; 2.0 4.0])
    @test tracer_face_flux(polymer, rm, state_polymer, (1.0, 2.0), 1, SPU(1, 2), TPFA(1, 2, 1), 1) ==
        tracer_total_mass_flux(polymer, rm, state_polymer, (1.0, 2.0), 1, SPU(1, 2))
    @test tracer_total_mass_flux(plain, rm, state, (1.0, 2.0), 1, SPU(1, 2)) != 0
    # Two capped cells have the same dissolved concentration and different deposits.
    capped = SurfactantTracer(rm.system; k_d = 0.0, L = 0.1, D = tracer.D)
    columns = [collect(surfactant_partition(c, 700.0, 150.0, 0.0, 0.1)) for c in (0.2, 0.4)]
    Pc = hcat(columns...)
    @test Pc[1, :] == [0.1, 0.1]
    @test all(>(0), Pc[3, :]) && Pc[3, 1] != Pc[3, 2]
    @test flux((state..., SurfactantPartition = Pc, TracerConcentrations = [0.2 0.4]); t = capped) == 0
end

@testset "One-dimensional diffusion and refinement" begin
    W = 0.01; A = W^2; D = 1e-7; tend = 0.0025/D; c0 = 1e-3
    cases = (W = (sl = 1.0, kd = 1.0, Dl = D, Dg = 10D), G = (sl = 0.7, kd = 2.0, Dl = D, Dg = 5D))
    for (cname, p) in pairs(cases), grid in (:uniform, :graded)
        errs = Float64[]; refs = Float64[]
        coef, mf = phase_coef(p.sl, p.kd, p.Dl, p.Dg)
        sg = 1 - p.sl
        Deff = (p.sl*RHO.liquid*p.Dl + sg*RHO.gas*p.kd*p.Dg)/(p.sl*RHO.liquid + sg*RHO.gas*p.kd)
        for (nx, nt) in ((50, 25), (100, 100), (200, 400), (400, 1600))
            w = grid == :uniform ? fill(1.0/nx, nx) : graded_widths(nx, 0.5)
            xc = cumsum(w) .- w ./ 2
            ci = [x < 0.5 ? c0 : 0.0 for x in xc]
            phi = fill(0.25, nx)
            case = diffusion_model((nx, 1, 1), (w, [W], [W]), phi; p..., c = ci)
            dt = tend/nt
            t0 = time(); cs = diffusion_simulate(case, dt, nt); wall = time() - t0
            cref = reference(hand_faces_1d(w, phi, A, coef), [mf*phi[i]*w[i]*A for i in 1:nx], ci, dt, nt)
            cex = [0.5c0*diffusion_erfc((x - 0.5)/(2sqrt(Deff*tend))) for x in xc]
            @test maximum(abs.(cs .- cref))/c0 <= 1e-8
            push!(errs, maximum(abs.(cs .- cex))/c0); push!(refs, maximum(abs.(cs .- cref))/c0)
            println(@sprintf("Diffusion %s %s nx=%d nt=%d err_erfc=%.4e err_ref=%.3e Deff=%.6e wall=%.1fs", cname, grid, nx, nt, errs[end], refs[end], Deff, wall))
        end
        @test all(>=(1.8), [log2(errs[k]/errs[k+1]) for k in 1:3])
        @test errs[end] <= 1e-3
        println("Diffusion $cname $grid orders = ", round.([log2(errs[k]/errs[k+1]) for k in 1:3], digits = 3), " max ref err = ", maximum(refs))
    end
    # Graded grid with heterogeneous porosity and both phases present.
    let p = cases.G, nx = 50, nt = 25
        coef, mf = phase_coef(p.sl, p.kd, p.Dl, p.Dg)
        w = graded_widths(nx, 0.5); xc = cumsum(w) .- w ./ 2
        phi = [0.1 + 0.3*((7i) % 11)/10 for i in 1:nx]
        ci = [x < 0.5 ? c0 : 0.0 for x in xc]
        case = diffusion_model((nx, 1, 1), (w, [W], [W]), phi; p..., c = ci)
        dt = tend/nt
        cs = diffusion_simulate(case, dt, nt)
        cref = reference(hand_faces_1d(w, phi, A, coef), [mf*phi[i]*w[i]*A for i in 1:nx], ci, dt, nt)
        @test maximum(abs.(cs .- cref))/c0 <= 1e-8
        println(@sprintf("Diffusion heterogeneous porosity G graded: err_ref=%.3e  (phi %.2f..%.2f)", maximum(abs.(cs .- cref))/c0, extrema(phi)...))
    end
end

@testset "Two-dimensional assembled diffusion residual" begin
    p = (sl = 0.7, kd = 2.0, Dl = 1e-7, Dg = 5e-7)
    dx = [0.1, 0.2, 0.15, 0.25]; dy = [0.3, 0.1, 0.2]; dz = 0.05
    nx, ny = length(dx), length(dy)
    idx(i, j) = i + (j - 1)*nx
    phi = [0.1 + 0.05*((3i + 5j) % 7) for j in 1:ny for i in 1:nx]
    ci = [1e-3*(1 + sin(1.3i + 0.7j))/2 for j in 1:ny for i in 1:nx]
    sg = 1 - p.sl
    ml, mg = p.sl*RHO.liquid, sg*RHO.gas
    B = ml + p.kd*mg; mf = ml + mg
    coef = p.Dl*p.sl*RHO.liquid*mf/B + p.Dg*sg*RHO.gas*p.kd*mf/B
    faces = Tuple{Int, Int, Float64}[]
    for j in 1:ny, i in 1:nx-1
        a, b = idx(i, j), idx(i + 1, j); Af = dy[j]*dz
        push!(faces, (a, b, coef/((dx[i]/2)/(phi[a]*Af) + (dx[i+1]/2)/(phi[b]*Af))))
    end
    for j in 1:ny-1, i in 1:nx
        a, b = idx(i, j), idx(i, j + 1); Af = dx[i]*dz
        push!(faces, (a, b, coef/((dy[j]/2)/(phi[a]*Af) + (dy[j+1]/2)/(phi[b]*Af))))
    end
    acc = [mf*phi[idx(i, j)]*dx[i]*dy[j]*dz for j in 1:ny for i in 1:nx]
    dt = 2000.0; nt = 5
    Mx = Matrix(Diagonal(acc))
    for (i, j, k) in faces
        Mx[i, i] += dt*k; Mx[j, j] += dt*k; Mx[i, j] -= dt*k; Mx[j, i] -= dt*k
    end
    F = lu(Mx)
    hand = [ci]
    for n in 1:nt
        push!(hand, F \ (acc .* hand[end]))
    end

    for gad in (false, true), order in ((LiquidPhase(), VaporPhase()), (VaporPhase(), LiquidPhase()))
        b = diffusion_model((nx, ny, 1), (dx, dy, [dz]), phi; p..., c = ci, order = order, general_ad = gad)
        rmodel = reservoir_model(b.model)
        prm = setup_parameters(rmodel)
        S = [ph isa VaporPhase ? 1 - p.sl : p.sl for ph in order]
        mkstate(c) = setup_state(rmodel; Pressure = 1e7, Saturations = S, TracerConcentrations = reshape(c, 1, :))
        sim = HelperSimulator(rmodel; parameters = prm, state0 = mkstate(ci))
        worst = 0.0
        for n in 1:nt
            # residual with no change is zero for the accumulation; use the flux part at c^n instead
            rflux = copy(model_residual(mkstate(hand[n]), mkstate(hand[n]), sim; dt = dt))
            rh = copy(model_residual(mkstate(hand[n + 1]), mkstate(hand[n]), sim; dt = dt))
            @test maximum(abs.(rflux)) > 0
            worst = max(worst, maximum(abs.(rh))/maximum(abs.(rflux)))
        end
        disc = typeof(rmodel.domain.discretizations.mass_flow).name.name
        @test worst <= 1e-10
        println(@sprintf("Assembled diffusion residual general_ad=%s order=%s disc=%s: max|r(hand)|/max|r(c^n)| = %.3e", gad, join(string.(typeof.(order)), ","), disc, worst))
    end

end
