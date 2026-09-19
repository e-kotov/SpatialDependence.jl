using SpatialDependence, KernelAbstractions, Metal, StableRNGs

geom = reggeomlattice(50, 50) # 2,500 observations
W = polyneigh(geom)
n = nobs(W)
rng = StableRNG(42)
x = randn(rng, n)
permutations = 9999

println("=========================================================")
println("Benchmark: N = $n observations, P = $permutations permutations")
println("Total permutations calculated: ", n * permutations, " (25,000,000 permutations)")
println("=========================================================")

# Warmup
_ = localmoran(x, W, permutations=100)
_ = localmoran(x, W, permutations=100, backend=:gpu)

# Time CPU
t0 = time()
res_cpu = localmoran(x, W, permutations=permutations)
t_cpu = time() - t0
println("CPU time: ", round(t_cpu, digits=3), " seconds")

# Time GPU
t0 = time()
res_gpu = localmoran(x, W, permutations=permutations, backend=:gpu, return_perms=false)
t_gpu = time() - t0
println("GPU time (Apple M4 Metal): ", round(t_gpu, digits=3), " seconds")

speedup = t_cpu / t_gpu
println("Speedup: ", round(speedup, digits=1), "x faster on Apple M4 GPU!")
println("=========================================================")
