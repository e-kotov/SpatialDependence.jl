using SpatialDependence
using KernelAbstractions
using Metal
using Statistics
using Printf
using StableRNGs

println("================================================================================")
println("High-Density Scaling Benchmark: N up to 50,000, P = 99,999")
println("================================================================================")

function run_scaling_benchmark(dim::Int, P::Int)
    n = dim * dim
    total_perms = Int64(n) * Int64(P)
    println("\n--------------------------------------------------------------------------------")
    @printf("Lattice: %d x %d (N = %d observations) | Permutations: %d\n", dim, dim, n, P)
    @printf("Total Permutations to Calculate: %d (%.1f Billion Permutations)\n", total_perms, total_perms / 1e9)
    println("--------------------------------------------------------------------------------")

    print("Generating polygon lattice and Queen weights... ")
    t0 = time()
    geom = reggeomlattice(dim, dim)
    W = polyneigh(geom)
    rng = StableRNG(42)
    x = randn(rng, n)
    @printf("Done in %.2fs\n", time() - t0)

    # Time CPU (for large N and P, only run small sample if too slow, or timed run)
    t_cpu = if n <= 10000 && P <= 9999
        print("Running CPU (Multi-threaded)... ")
        t0 = time()
        _ = localmoran(x, W, permutations=P)
        t = time() - t0
        @printf("%.3fs\n", t)
        t
    else
        # Extrapolate from smaller run to avoid stalling minutes on CPU
        print("Benchmarking CPU sample for projection... ")
        sample_P = 1000
        t0 = time()
        _ = localmoran(x, W, permutations=sample_P)
        t_sample = time() - t0
        est = (t_sample / sample_P) * P
        @printf("Projected: ~%.1fs (based on %d perms in %.2fs)\n", est, sample_P, t_sample)
        est
    end

    # Time GPU (Apple M4 Metal)
    print("Running GPU (Apple M4 Metal)... ")
    t0 = time()
    res_gpu = localmoran(x, W, permutations=P, backend=:gpu, return_perms=false, seed=12345)
    t_gpu = time() - t0
    @printf("%.3f seconds\n", t_gpu)

    speedup = t_cpu / t_gpu
    throughput_gpu = total_perms / t_gpu / 1e6 # Million perms / sec
    @printf(">>> GPU Speedup: %.1fx faster\n", speedup)
    @printf(">>> GPU Throughput: %.1f million permutations/sec\n", throughput_gpu)

    return (n=n, P=P, t_cpu=t_cpu, t_gpu=t_gpu, speedup=speedup, throughput=throughput_gpu)
end

# Warmup
println("Warming up...")
geom_w = reggeomlattice(10, 10)
W_w = polyneigh(geom_w)
_ = localmoran(randn(100), W_w, permutations=100, backend=:gpu, return_perms=false)

# Run Scaling Tests
r1 = run_scaling_benchmark(70, 9999)    # N = 4,900, P = 9,999 (~49M perms)
r2 = run_scaling_benchmark(100, 9999)   # N = 10,000, P = 9,999 (~100M perms)
r3 = run_scaling_benchmark(100, 99999)  # N = 10,000, P = 99,999 (~1 Billion perms)
r4 = run_scaling_benchmark(150, 99999)  # N = 22,500, P = 99,999 (~2.25 Billion perms)

println("\n================================================================================")
println("Scaling Summary: Apple M4 Metal GPU")
println("================================================================================")
@printf("%-10s | %-10s | %-15s | %-12s | %-12s | %-10s\n", "Obs (N)", "Perms (P)", "Total Perms", "CPU (Time)", "GPU Time (M4)", "Speedup")
println("-----------+------------+-----------------+--------------+--------------+-----------")
for r in [r1, r2, r3, r4]
    total_str = @sprintf("%.1fM", (r.n * r.P) / 1e6)
    if (r.n * r.P) >= 1e9
        total_str = @sprintf("%.2fB", (r.n * r.P) / 1e9)
    end
    @printf("%-10d | %-10d | %-15s | %10.2fs | %10.3fs | %8.1fx\n", r.n, r.P, total_str, r.t_cpu, r.t_gpu, r.speedup)
end
println("================================================================================")
