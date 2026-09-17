# GPU-accelerated Local Spatial Autocorrelation tests
using Test
using SpatialDependence
using KernelAbstractions
using Statistics
using StableRNGs

# Try loading Metal on macOS Apple Silicon or CUDA if available
const HAS_GPU = try
    if Sys.isapple()
        import Metal
        Metal.functional()
    else
        import CUDA
        CUDA.functional()
    end
catch
    false
end

@testset "GPU Permutation Acceleration" begin
    if !HAS_GPU
        @info "No functional GPU backend found. Skipping GPU execution tests."
        return
    end

    gpu_backend = Sys.isapple() ? Metal.MetalBackend() : CUDA.CUDABackend()
    @info "Running GPU tests on: $gpu_backend"

    # Synthetic lattice dataset
    geom = reggeomlattice(10, 10)
    W = polyneigh(geom)
    n = nobs(W)
    rng = StableRNG(42)
    x = randn(rng, n)

    @testset "Local Moran on GPU" begin
        # CPU reference
        lm_cpu = localmoran(x, W, permutations=4999, rng=rng)
        # GPU with explicit backend
        lm_gpu = localmoran(x, W, permutations=4999, backend=gpu_backend, seed=12345)
        # GPU with backend=:gpu
        lm_gpu_auto = localmoran(x, W, permutations=4999, backend=:gpu, seed=12345)

        @test length(pvalue(lm_gpu)) == n
        @test length(zscore(lm_gpu)) == n
        @test size(scoreperms(lm_gpu)) == (n, 4999)

        # Verify statistical correlation between CPU and GPU p-values
        corr_p = cor(pvalue(lm_cpu), pvalue(lm_gpu))
        @info "Local Moran p-value correlation (CPU vs GPU): $corr_p"
        @test corr_p > 0.85

        # Memory optimization: return_perms=false
        lm_gpu_noperms = localmoran(x, W, permutations=4999, backend=:gpu, return_perms=false, seed=12345)
        @test size(scoreperms(lm_gpu_noperms)) == (0, 0)
        @test length(pvalue(lm_gpu_noperms)) == n
        @test isapprox(pvalue(lm_gpu), pvalue(lm_gpu_noperms), atol=1e-5)
    end

    @testset "Local Geary on GPU" begin
        lg_cpu = localgeary(x, W, permutations=4999, rng=rng)
        lg_gpu = localgeary(x, W, permutations=4999, backend=:gpu, seed=12345)

        @test length(pvalue(lg_gpu)) == n
        @test length(zscore(lg_gpu)) == n
        @test size(scoreperms(lg_gpu)) == (n, 4999)

        corr_p = cor(pvalue(lg_cpu), pvalue(lg_gpu))
        @info "Local Geary p-value correlation (CPU vs GPU): $corr_p"
        @test corr_p > 0.80
    end

    @testset "Getis-Ord on GPU" begin
        # Values must be positive for Getis-Ord
        x_pos = abs.(x) .+ 1.0
        go_cpu = getisord(x_pos, W, permutations=4999, star=true, rng=rng)
        go_gpu = getisord(x_pos, W, permutations=4999, star=true, backend=:gpu, seed=12345)

        @test length(pvalue(go_gpu)) == n
        @test length(zscore(go_gpu)) == n

        corr_p = cor(pvalue(go_cpu), pvalue(go_gpu))
        @info "Getis-Ord p-value correlation (CPU vs GPU): $corr_p"
        @test corr_p > 0.80
    end
end
