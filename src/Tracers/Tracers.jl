module Tracers
    using Jutul, JutulDarcy
    using LinearAlgebra
    import Adapt
    import JutulDarcy: AbstractPhase, get_phases, darcy_phase_mass_flux, phase_upwind, MultiPhaseSystem
    import StaticArrays: setindex

    export SinglePhaseTracer, MultiPhaseTracer, add_tracers_to_model!, number_of_tracers, SurfactantTracer, FmFactor , set_surfactant_model!, check_surfactant_injection
    export FoamShearThinning, foam_shear_diagnostics
    export LinearSurfactantAdsorption, LangmuirSurfactantAdsorption, TabulatedSurfactantAdsorption, SurfactantRockMass

    include("tracer_variants.jl")
    include("variables.jl")
    include("equations.jl")
    include("facility.jl")
    include("api.jl")
    include("polymer.jl")
    include("surfactant/isotherms.jl")
    include("surfactant/tracer.jl")
    include("surfactant/partition.jl")
    include("surfactant/adsorption.jl")
    include("surfactant/diffusion.jl")
    include("surfactant/foam.jl")
    include("surfactant/shear.jl")
    include("surfactant/setup.jl")
end
