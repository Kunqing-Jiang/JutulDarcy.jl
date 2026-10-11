using JutulDarcy, Jutul, Test
import JutulDarcy.Tracers: surfactant_partition

include(joinpath(@__DIR__, "surfactant_deposit_column.jl"))

@testset "Surfactant deposit: evaporation and rewetting" begin
    d = run_column(0.1)
    deposit_column_checks(0.1, d)
    # A larger cap never creates more deposit from the same state.
    s = d.state_end_gas[]
    E(L) = [surfactant_partition(s.c[i], s.ml[i], s.mg[i], 0.0, L)[3] for i in eachindex(s.c)]
    @test all(E(0.05) .>= E(0.1) .>= E(1.0))
end
