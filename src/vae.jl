module vae
export build_encoder, build_decoder, train_vae, vae_loss, reparameterize, reconsrtuct

using Flux
using Statistics
using Flux: DataLoader, params, logitbinarycrossentropy


function build_encoder(input_dim::Int, hidden_dim::Int, latent_dim::Int)
    fcn = Dense(input_dim, hidden_dim, tanh)
    μ_layer = Dense(hidden_dim, latent_dim)
    logσ_layer = Dense(hidden_dim, latent_dim)

    function encoder(x)
        h = fcn(x)
        return μ_layer(h), logσ_layer(h)
    end

    return encoder
end

function build_decoder(input_dim::Int, hidden_dim::Int, latent_dim::Int)
    return Chain(
        Dense(latent_dim, hidden_dim, tanh),
        Dense(hidden_dim, input_dim)
    )
end

function reconstruct(encoder, decoder, x)
    μ, logσ = encoder(x)
    z = reparameterize(μ, logσ)
    return μ, logσ, decoder(z)
end

function reparameterize(μ, logσ)
    σ = exp.(logσ)
    ϵ = randn(Float32, size(σ))  # Sample from standard normal
    return μ + σ .* ϵ
end

function vae_loss(encoder, decoder, x, β=1.0f0)
    batch_size = size(x)[end]
    μ, logσ = encoder(x)
    # Sample from latent distribution
    z = reparameterize(μ, logσ)
    x̂ = decoder(z)
    
    # Reconstruction loss
    reconstruction_loss = -logitbinarycrossentropy(x̂, x, agg=sum) / batch_size
    # KL divergence
    kl_divergence = -0.5f0 * sum(1 .+ 2*logσ .- μ.^2 .- exp.(2*logσ)) / batch_size
    
    return -reconstruction_loss + β * kl_divergence
end

function train_vae(X_train, input_dim::Int, hidden_dim::Int, latent_dim::Int; epochs::Int=10, batchsize::Int=32, β::Float32=1.0f0)
    
    # Build encoder and decoder
    encoder = build_encoder(input_dim, hidden_dim, latent_dim)
    decoder = build_decoder(input_dim, hidden_dim, latent_dim)
    # Define optimizer
    opt_enc = Flux.setup(AdamW(eta=0.001, lambda=0.001), encoder)
    opt_dec = Flux.setup(AdamW(eta=0.001, lambda=0.001), decoder)
    # Create DataLoader
    train_loader = DataLoader(X_train, batchsize=batchsize, shuffle=true)
    # Training loop
    for epoch in 1:epochs
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader
            loss, (grad_enc, grad_dec) = Flux.withgradient(encoder, decoder) do enc, dec
                vae_loss(enc, dec, batch, β)
            end
            Flux.update!(opt_enc, encoder, grad_enc)
            Flux.update!(opt_dec, decoder, grad_dec)
            total_loss += loss
            num_batches += 1
        end
        
        avg_loss = total_loss / num_batches
        println("Epoch $epoch: Average Loss = $avg_loss")
    end

    return encoder, decoder
end
end