export build_encoder, build_decoder, train_vae, vae_loss, reparameterize, reconsrtuct
using Flux, Statistics
using Flux: DataLoader, logitbinarycrossentropy

"""
Creates the encoder network for the VAE. The encoder maps input data to a latent space representation, 
producing both the mean (`μ`) and log variance (`logσ`) of the latent distribution.

# Arguments
- `input_dim`: Dimension of input data.
- `hidden_dim`: Number of hidden units.
- `latent_dim`: Dimension of the latent space.

# Returns
- A function that takes an input and outputs (`μ`, `logσ`).
"""
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

"""
Creates the decoder network for the VAE. The decoder reconstructs the input from a latent space representation.

# Arguments
- `input_dim`: Dimension of the output data.
- `hidden_dim`: Number of hidden units.
- `latent_dim`: Dimension of the latent space.

# Returns
- A Flux `Chain` that maps latent vectors back to the data space.
"""
function build_decoder(input_dim::Int, hidden_dim::Int, latent_dim::Int)
    return Chain(
        Dense(latent_dim, hidden_dim, tanh),
        Dense(hidden_dim, input_dim)
    )
end

"""
Performs a forward pass through the encoder and decoder to reconstruct the input.

# Arguments
- `encoder`: The VAE encoder.
- `decoder`: The VAE decoder.
- `x`: Input data.

# Returns
- `μ`: The mean of the latent distribution.
- `logσ`: The log variance of the latent distribution.
- `x̂`: The reconstructed input.
"""
function reconstruct(encoder, decoder, x)
    μ, logσ = encoder(x)
    z = reparameterize(μ, logσ)
    return μ, logσ, decoder(z)
end

"""
Performs the reparameterization trick to sample from the latent distribution 
in a way that allows backpropagation.

# Arguments
- `μ`: Mean of the latent distribution.
- `logσ`: Log variance of the latent distribution.

# Returns
- `z`: A sampled latent vector.
"""
function reparameterize(μ, logσ)
    σ = exp.(logσ)
    ϵ = randn(Float32, size(σ))  # Sample from standard normal
    return μ + σ .* ϵ
end

"""
Computes the loss function for the VAE, consisting of:
1. **Reconstruction Loss**: Measures how well the model reconstructs the input.
2. **KL Divergence**: Regularizes the latent space to follow a normal distribution.

# Arguments
- `encoder`: The VAE encoder.
- `decoder`: The VAE decoder.
- `x`: Input data.
- `β`: Weighting factor for the KL divergence term (used in β-VAE for disentanglement).

# Returns
- The total loss (reconstruction loss + β * KL divergence).
"""
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

"""
Trains the VAE model.

# Arguments
- `X_train`: Training dataset.
- `input_dim`: Input data dimension.
- `hidden_dim`: Number of hidden units.
- `latent_dim`: Dimension of the latent space.
- `epochs`: Number of training iterations.
- `batchsize`: Batch size for training.
- `β`: Weighting factor for KL divergence (for β-VAE).

# Returns
- `encoder`: Trained encoder network.
- `decoder`: Trained decoder network.
"""
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
