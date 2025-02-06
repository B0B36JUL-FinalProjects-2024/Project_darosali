module clustering
export k_meanspp, k_means, random_sample

using Random, Statistics, Test

function random_sample(weights::Vector{<:Real})
    """
    Picks randomly a sample based on the sample weights.

    :param weights: Vector of sample weights
    :return idx:    index of chosen sample
    """
    weights_norm = cumsum(weights / sum(weights))
    rand_value = rand()
    idx = first(searchsorted(weights_norm, rand_value))
    return idx
end

# k-means++ initialization
function k_meanspp(x::Matrix{<:Real}, k::Int)
    """
    Performs k-means++ initialization for k-means clustering.

    :param x:       Feature vectors, Matrix (dim, number_of_vectors)
    :param k:       Number of clusters
    
    :return centroids: Proposed centroids for k-means initialization
    """
    Random.seed!(0)
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

function k_means(x::Matrix{<:Real}, k::Int, max_iter::Int; init_means::Union{Matrix{<:Real}, Nothing}=nothing)
    """
    Implementation of the k-means clustering algorithm.

    :param x:          feature vectors, Matrix (dim, number_of_vectors)
    :param k:          required number of clusters, scalar
    :param max_iter:   stopping criterion: max. number of iterations
    :param show:       (optional) boolean switch to turn on/off visualization of partial results
    :param init_means: (optional) initial cluster prototypes, Matrix (dim, k)

    :return cluster_labels: Vector{Int}, cluster index for each feature vector
    :return centroids:      Matrix (dim, k), cluster centroids
    :return sq_dists:       Vector{Float64}, squared distances to the nearest centroid for each feature vector
    """
    Random.seed!(0)
    n_vectors = size(x, 2)
    cluster_labels = zeros(Int, n_vectors)
    sq_dists = zeros(Float64, n_vectors)

    # Initialize centroids
    centroids = if init_means === nothing
        x[:, randperm(n_vectors)[1:k]]  # Randomly select k points from x
    else
        init_means
    end

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
end