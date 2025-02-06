module LatentSpaceClustering

include("vae.jl")
using .vae
export build_encoder, build_decoder, vae_loss, reparameterize, reconsrtuct

include("clustering.jl")
using .clustering
export k_meanspp, k_means, random_sample

include("data_preprocessing.jl")
using .data_preprocessing
export load_and_preprocess_mnist

include("utils.jl")
using .utils
export sample_data, plot_samples

end


