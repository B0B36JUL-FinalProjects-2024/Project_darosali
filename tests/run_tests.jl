using Test
include("../src/clustering.jl")
include("../src/vae.jl")

@testset "clustering.jl and vae.jl Tests" begin
    include("test_vae.jl")
    include("test_clustering.jl")
end