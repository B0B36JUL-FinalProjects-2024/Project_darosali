using Flux

function build_encoder(input_dim::Int, hidden_dim::Int, latent_dim::Int)
    return Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, 2 * latent_dim)  # Outputs μ and logσ
    )
end

function build_decoder(latent_dim::Int, hidden_dim::Int, output_dim::Int)
    return Chain(
        Dense(latent_dim, hidden_dim, relu),
        Dense(hidden_dim, output_dim, sigmoid)  # Outputs reconstructed data
    )
end

function reparameterize(μ, logσ)
    σ = exp.(logσ)
    ϵ = randn(Float32, size(σ))
    println(ϵ)  # Sample from standard normal
    return μ + σ .* ϵ
end

function vae_loss(x, encoder, decoder, β=1.0f0)
    μ_logσ = encoder(x)
    μ, logσ = μ_logσ[1:end÷2], μ_logσ[end÷2+1:end]
    
    # Sample from latent distribution
    z = reparameterize(μ, logσ)
    x̂ = decoder(z)
    
    # Reconstruction loss
    reconstruction_loss = Flux.mse(x̂, x)
    
    # KL divergence
    kl_divergence = -0.5f0 * sum(1 .+ logσ .- μ.^2 .- exp.(logσ))
    
    return reconstruction_loss + β * kl_divergence
end
