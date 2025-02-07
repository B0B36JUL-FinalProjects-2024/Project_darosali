using LatentSpaceClustering
using Test
# using Flux

@testset "VAE Encoder and Decoder" begin
    # Test encoder
    encoder = build_encoder(784, 256, 2)  # Example for MNIST-like input
    X = rand(Float32, 784, 10) 
    μ, logσ = encoder(X)
    @test size(μ) == (2, 10)  #Latent dimension should be 2
    @test size(logσ) == (2, 10) #Latent dimension should be 2

    # Test decoder
    decoder = build_decoder(784, 256, 2)
    z = randn(Float32, 2, 10)
    x̂ = decoder(z)
    @test size(x̂) == (784, 10)  # Reconstructed image size should match input

    # Test VAE loss
    loss = vae_loss(encoder, decoder, X)
    @test loss > 0 # VAE loss should be positive
end
