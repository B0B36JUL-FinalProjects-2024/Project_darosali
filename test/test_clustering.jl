using LatentSpaceClustering
using Random

@testset "Random Sampling with Known Seed" begin
    # Set random seed to ensure deterministic behavior
    Random.seed!(0)
    
    weights = [0.1, 0.3, 0.6]
    
    # Test that it returns an index (integer)
    idx = random_sample(weights)
    @test typeof(idx) == Int

    # Test if it returns one of the valid indices (1, 2, or 3)
    @test 1 <= idx <= 3
    
    # Test the exact expected index for the given input and random seed
    # With weights [0.1, 0.3, 0.6] and seed 0, the random sampling should pick index 3
    @test idx == 3

end

@testset "K-means++ Initialization" begin
    strategy = Kmeanspp()
    x = randn(2, 10)  # 10 2-dimensional points
    k = 3
    centroids = init_centroids(strategy, x, k)

    # Test the correct number of centroids
    @test size(centroids, 2) == k
    
    # Test that the centroids are data points
    for c in 1:k
        @test centroids[:, c] in eachcol(x)
    end

    k_large = 15
    @test_throws BoundsError init_centroids(strategy, x, k_large)  # Ensure failure when k > n_vectors
end

@testset "k_means tests" begin
    strategy = Rand()
    x = rand(2, 10)  # 2D points, 10 samples
    k = 3
    max_iter = 100
    cluster_labels, centroids, sq_dists = k_means(strategy, x, k, max_iter)

    @test length(cluster_labels) == size(x, 2)  # Correct number of cluster labels
    @test all(1 <= lbl <= k for lbl in cluster_labels)  # Valid cluster assignments
    @test size(centroids) == (2, k)  # Correct centroid dimensions
    @test length(sq_dists) == size(x, 2)  # Correct distance vector size

    # Test: Single point dataset
    x_single = ones(2, 1)  # One 2D point
    k_single = 1
    cluster_labels_single, centroids_single, sq_dists_single = k_means(strategy, x_single, k_single, max_iter)
    
    @test cluster_labels_single == [1]  # Should assign the only point to cluster 1
    @test centroids_single == x_single  # Centroid should be the point itself
    @test sq_dists_single == [0.0]  # Distance should be 0

end


