# Heavier variants of the registered surfactant tests: the same checks on
# more cases. Selected with tags = [:slow]; the default test run skips them.
using JutulDarcy, Jutul, Test

include(joinpath(@__DIR__, "surfactant_deposit_column.jl"))

@testset "Surfactant deposit column, other caps" begin
    for L in (0.05, 1.0)
        deposit_column_checks(L, run_column(L))
    end
end
