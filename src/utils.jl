module utils
export sample_data, plot_samples

using Random
using Plots

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

using Plots

function plot_samples(X::Matrix{<:Real}, samples_to_plot::Int, img_size::Tuple{Int, Int}, layout::Tuple{Int, Int})
    heatmaps = []

    for i in 1:samples_to_plot
        x̂ = reshape(X[:, i], img_size)
        push!(heatmaps, heatmap(x̂', color=:greys, title="Sample $i", size=(100, 100), yflip=true, colorbar=false))
    end

    # Display in a grid layout
    plot(heatmaps..., layout=layout, size=(700, 700))
end

end