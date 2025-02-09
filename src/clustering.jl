export init_centroids, k_means, random_sample, Rand, Kmeanspp

using Statistics, Random

abstract type InitStrategy end

struct Rand <: InitStrategy end
struct Kmeanspp <: InitStrategy end


"""
Randomly selects `k` initial centroids from the dataset for k-means clustering.

## Arguments
- `x`: Feature matrix of size (dimensions, number_of_samples).
- `k`: Number of clusters (centroids) to initialize.

## Returns
- `Matrix{<:Real}`: A matrix of size (dimensions, k) containing the selected centroids.
"""
function init_centroids(::Rand, x::Matrix{<:Real}, k::Int)
    n_vectors = size(x, 2)
    return x[:, randperm(n_vectors)[1:k]]
end

"""
Performs k-means++ initialization for k-means clustering.

## Arguments
- `x`: Feature vectors, Matrix (dim, number_of_vectors)
- `k`: Number of clusters

## Returns
- `centroids`: Proposed centroids for k-means initialization
"""
function init_centroids(::Kmeanspp, x::Matrix{<:Real}, k::Int)
    n_vectors = size(x, 2)
    if k > n_vectors
        throw(BoundsError("k ($k) cannot be greater than the number of data points ($n_vectors)"))
    end
    
    # Initialize centroids with the first random sample
    idx = [random_sample(ones(n_vectors))] 
    centroids = x[:, idx]
    
    while size(centroids, 2) < k
        # Compute squared distances to the closest centroid
        distances = sum((x .- reshape(centroids, size(centroids,1), 1, :)).^2, dims=1)
        distances = dropdims(distances, dims=1)
        distances, _ = findmin(distances, dims=2)
        
        # Select the next centroid based on the distances
        idx = [random_sample(vec(distances))]
        
        # Add the new centroid
        centroids = hcat(centroids, x[:, idx])
    end
    
    return centroids
end

"""
Picks a random sample based on the given sample weights.

## Arguments
- `weights`: Vector of sample weights

## Returns
- `idx`: Index of the chosen sample
"""
function random_sample(weights::Vector{<:Real})
    weights_norm = cumsum(weights / sum(weights))
    rand_value = rand()
    idx = first(searchsorted(weights_norm, rand_value))
    return idx
end

"""
Implementation of the k-means clustering algorithm.

## Arguments
- `strategy`: Initialization strategy, an object of type `InitStrategy` (e.g., `Rand`, `Kmeanspp`).
- `x`: Feature vectors, Matrix (dim, number_of_vectors).
- `k`: Required number of clusters, scalar.
- `max_iter`: Maximum number of iterations for convergence.

## Returns
- `cluster_labels`: Vector{Int}, cluster index for each feature vector.
- `centroids`: Matrix (dim, k), cluster centroids.
- `sq_dists`: Vector{Float64}, squared distances to the nearest centroid for each feature vector.
"""
function k_means(strategy::InitStrategy, x::Matrix{<:Real}, k::Int, max_iter::Int)
    n_vectors = size(x, 2)
    cluster_labels = zeros(Int, n_vectors)
    sq_dists = zeros(Float64, n_vectors)


    # Initialize centroids
    centroids = init_centroids(strategy, x, k)

    for i_iter in 1:max_iter
        # Compute distances between points and centroids
        distances = sum((x .- reshape(centroids, size(centroids,1), 1, :)).^2, dims=1)
        distances = dropdims(distances, dims=1)

        # Assign clusters based on minimum distance
        sq_dists, cluster_labels = findmin(distances, dims=2)
        cluster_labels = [idx[2] for idx in cluster_labels]

        # Compute new centroids
        new_centroids = zeros(size(centroids))
        for i in 1:k
            cluster_points = x[:, findall(cluster_labels .== i)]
            new_centroids[:, i] = size(cluster_points, 2) > 0 ? mean(cluster_points, dims=2)[:] : x[:, rand(1:n_vectors)]
        end
        # Check for convergence
        if new_centroids ≈ centroids
            break
        else
            centroids = new_centroids
        end

    end

    return cluster_labels[:], centroids, sq_dists[:]
end
