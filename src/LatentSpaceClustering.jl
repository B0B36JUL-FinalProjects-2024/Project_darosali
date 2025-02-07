module LatentSpaceClustering

include("vae.jl")
export build_encoder, build_decoder, vae_loss, reparameterize, reconsrtuct

include("clustering.jl")
export k_meanspp, k_means, random_sample

include("data_preprocessing.jl")
export load_and_preprocess_mnist

include("utils.jl")
export sample_data, plot_samples

end


