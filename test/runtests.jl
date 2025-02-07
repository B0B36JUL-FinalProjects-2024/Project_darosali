using Test

@testset "clustering.jl and vae.jl Tests" begin
    include("test_vae.jl")
    include("test_clustering.jl")
end