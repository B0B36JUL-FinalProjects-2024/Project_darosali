module clustering
export k_meanspp, k_means

using Random, Statistics, Test

function random_sample(weights::Vector{<:Real})
    """
    Picks randomly a sample based on the sample weights.

    :param weights: Vector of sample weights
    :return idx:    index of chosen sample
    """
    weights_norm = cumsum(weights / sum(weights))
    #println(weights_norm)
    rand_value = rand()
    #println(rand_value)  # Random number between 0 and 1
    idx = first(searchsorted(weights_norm, rand_value))
    #println(idx)
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
    
    # Initialize centroids with the first random sample
    idx = [random_sample(ones(n_vectors))]  # Randomly select the first centroid
    centroids = x[:, idx]
    
    # Add centroids until we have k
    while size(centroids, 2) < k
        # Compute squared distances to the closest centroid
        distances = sum((x .- reshape(centroids, size(centroids,1), 1, :)).^2, dims=1)
        distances = dropdims(distances, dims=1)
        distances, _ = findmin(distances, dims=2)
        #display(distances)
        
        # Select the next centroid based on the distances
        idx = [random_sample(vec(distances))]
        
        # Add the new centroid
        centroids = hcat(centroids, x[:, idx])
    end
    
    return centroids
end

function k_means(x::Matrix{<:Real}, k::Int, max_iter::Int; show::Bool=false, init_means::Union{Matrix{<:Real}, Nothing}=nothing)
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
    # Number of vectors
    n_vectors = size(x, 2)
    cluster_labels = zeros(Int, n_vectors)
    sq_dists = zeros(Float64, n_vectors)

    # Initialize centroids
    centroids = if init_means === nothing
        x[:, randperm(n_vectors)[1:k]]  # Randomly select k points from x
    else
        init_means
    end
    #print(centroids)
    #print("centroids")
    #display(centroids)

    for i_iter in 1:max_iter
        # Compute distances between points and centroids
        #distances = [sum((x[:, j] .- centroids[:, i]).^2) for j in 1:n_vectors, i in 1:k]
        #distances = [sum((x[:, j] .- centroids[:, i]).^2) for j in 1:n_vectors, i in 1:k]
        distances = sum((x .- reshape(centroids, size(centroids,1), 1, :)).^2, dims=1)
        distances = dropdims(distances, dims=1)

        #display(distances)

        # Assign clusters based on minimum distance
        sq_dists, cluster_labels = findmin(distances, dims=2)
        #display(sq_dists)
        cluster_labels = [idx[2] for idx in cluster_labels]
        #display(cluster_labels)

        # Compute new centroids
        new_centroids = zeros(size(centroids))
        for i in 1:k
            cluster_points = x[:, findall(cluster_labels .== i)]
            #display(size(cluster_points))
            new_centroids[:, i] = size(cluster_points, 2) > 0 ? mean(cluster_points, dims=2)[:] : x[:, rand(1:n_vectors)]
        end
        #println("new_centroids")
        #display(new_centroids)
        # Check for convergence
        if new_centroids ≈ centroids
            break
        else
            centroids = new_centroids
        end

        # Optional: Show clustering process
        if show
            println("Iteration: $i_iter")
        end
    end

    if show
        println("Done.")
    end

    return cluster_labels[:], centroids, sq_dists[:]
end

# Set the seed for reproducibility

# Test data
# x = [16 12 50 96 34 59 22 75 26 51]  # (1, 10) shape, similar to np.array([[...]])
# x = reshape(x, 1, :)  # Ensure it's a 1-row matrix
# x = Float64.(x)
# x = [1.0 1.5 3.0 5.0 6.0 7.0 8.0 9.0 9.5 10.0;  # 2D data (10 points)
#      1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 8.5 9.0]
# x = x';
# init_means=[50.0 26.0 34.0]
# init_means=[3.0 9.5 6; 3.0 8.5 5.0]

# # Run k-means with k=3, max_iter=Inf
# print(x)
# init_means = k_meanspp(x, 3)
# cluster_labels, centroids, sq_dists = k_means(x, 3, typemax(Int), init_means = init_means)  # Julia equivalent of np.inf
# print(centroids)

# # Expected results
# expected_labels = [1, 1, 0, 0, 2, 0, 1, 0, 1, 0]
# expected_centroids = [66 19 34]
# expected_sq_dists = [9.0, 49.0, 256.0, 900.0, 0.0, 49.0, 9.0, 81.0, 49.0, 225.0]
# # Assertions
# @test cluster_labels == expected_labels
# @test centroids == expected_centroids
# @test sq_dists ≈ expected_sq_dists  # Using ≈ to allow floating-point precision differences

# println("All tests passed!")

# random_sample(vec([1. 1. 1. 1. 1.]))
# k_meanspp(x, 3)

end