export sample_data, plot_samples
using Random, Plots

"""
Randomly selects a subset of data points from the dataset, along with their corresponding latent representations 
and cluster labels.

# Arguments
- `X`: Data matrix of shape `(features, samples)`, where each column is a data point.
- `z`: Corresponding latent space representations of shape `(latent_dim, samples)`.
- `cluster_labels`: Vector of cluster labels associated with the data points.
- `num_samples`: The number of samples to randomly select (default: 3000).

# Returns
- `X_sampled`: A subset of `X` with `num_samples` selected columns.
- `z_sampled`: A subset of `z` with `num_samples` selected columns.
- `cluster_labels_sampled`: A subset of `cluster_labels` with `num_samples` selected elements.
"""
function sample_data(X::Matrix{<:Real}, z::Matrix{<:Real}, cluster_labels::Vector{<:Real}; num_samples::Int=3000)
    
    # Get the indices of the random samples
    Random.seed!(0)
    indices = randperm(size(X, 2))[1:num_samples]

    # Select the sampled data
    X_sampled = X[:, indices]
    z_sampled = z[:, indices]
    cluster_labels_sampled = cluster_labels[indices]

    return X_sampled, z_sampled, cluster_labels_sampled
end

"""
Plots a selection of images from the dataset.

# Arguments
- `X`: Data matrix of shape `(features, samples)`, where each column represents a flattened image.
- `samples_to_plot`: The number of images to display.
- `img_size`: The dimensions `(height, width)` of each image.
- `layout`: The layout `(rows, cols)` for displaying multiple images.

# Behavior
- Extracts and reshapes the first `samples_to_plot` columns of `X` into `img_size`-shaped images.
- Displays them in a grid layout using grayscale heatmaps.

# Returns
- A plot of the selected images.
"""
function plot_samples(X::Matrix{<:Real}, samples_to_plot::Int, img_size::Tuple{Int, Int}, layout::Tuple{Int, Int})
    heatmaps = []

    for i in 1:samples_to_plot
        x̂ = reshape(X[:, i], img_size)
        push!(heatmaps, heatmap(x̂', color=:greys, title="Sample $i", size=(100, 100), yflip=true, colorbar=false))
    end

    # Display in a grid layout
    plot(heatmaps..., layout=layout, size=(700, 700))
end
