using SpatialDependence
using KernelAbstractions
using Metal
using Statistics
using Printf
using StableRNGs

println("================================================================================")
println("Fast Benchmark: 50k Grid (N = 50,176, P = 999 Permutations)")
println("================================================================================")

dim = 224
n = dim * dim
P = 999
total_perms = Int64(n) * Int64(P)

@printf("Lattice: %d x %d (N = %d cells) | Permutations: %d\n", dim, dim, n, P)
@printf("Total Permutations to Calculate: %d (%.2f Million Permutations)\n", total_perms, total_perms / 1e6)
println("--------------------------------------------------------------------------------")

print("Generating regular lattice and Queen weights... ")
t0 = time()
geom = reggeomlattice(dim, dim)
W = polyneigh(geom)
rng = StableRNG(42)
x = randn(rng, n)
@printf("Done in %.2fs\n", time() - t0)

# Warmup GPU
print("Warming up GPU JIT... ")
_ = localmoran(x[1:100], polyneigh(reggeomlattice(10, 10)), permutations=10, backend=:gpu, return_perms=false)
println("Done.")

# Run CPU
println("\n1. Running CPU (Multi-threaded, $(Threads.nthreads()) threads)...")
GC.gc()
t0 = time()
res_cpu = localmoran(x, W, permutations=P, seed=12345)
t_cpu = time() - t0
@printf("   CPU Time: %.3f seconds (%.2f M perms/sec)\n", t_cpu, total_perms / t_cpu / 1e6)

# Run GPU
println("\n2. Running GPU (Apple M4 Metal)...")
GC.gc()
t0 = time()
res_gpu = localmoran(x, W, permutations=P, backend=:gpu, return_perms=false, seed=12345)
t_gpu = time() - t0
@printf("   GPU Time: %.3f seconds (%.2f M perms/sec)\n", t_gpu, total_perms / t_gpu / 1e6)

# Correlation and Verification
valid = .!isnan.(res_cpu.p) .& .!isnan.(res_gpu.p)
p_corr = cor(res_cpu.p[valid], res_gpu.p[valid])
z_corr = cor(res_cpu.z[valid], res_gpu.z[valid])

println("\n================================================================================")
println("RESULTS SUMMARY")
println("================================================================================")
@printf("Observations (N)   : %d\n", n)
@printf("Permutations (P)   : %d\n", P)
@printf("Total Permutations : %.2f Million\n", total_perms / 1e6)
@printf("CPU Time           : %.3f s\n", t_cpu)
@printf("GPU Time           : %.3f s\n", t_gpu)
@printf("GPU Speedup        : %.2fx faster\n", t_cpu / t_gpu)
@printf("GPU Throughput     : %.1f Million perms/sec\n", total_perms / t_gpu / 1e6)
@printf("p-value Correlation: r = %.5f\n", p_corr)
@printf("z-score Correlation: r = %.5f\n", z_corr)
println("================================================================================")
