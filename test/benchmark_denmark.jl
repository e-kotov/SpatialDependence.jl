using SpatialDependence
using KernelAbstractions
using Metal
using Statistics
using Printf
using StableRNGs
using DelimitedFiles

println("================================================================================")
println("Real-World Benchmark: All of Denmark 500m Grid (N = 179,674)")
println("================================================================================")

# 1. Load Denmark grid coordinates
csv_path = joinpath(@__DIR__, "data", "dk_grid_coords.csv")
if !isfile(csv_path)
    error("CSV file not found at $(csv_path)")
end

print("Loading Denmark 500m grid coordinates ($(csv_path))... ")
t0 = time()
data = readdlm(csv_path, ',', Float64)
n = size(data, 1)
@printf("Loaded %d cells in %.2fs\n", n, time() - t0)

# 2. Build coordinate index and Queen contiguity weights
print("Indexing grid & building Queen contiguity matrix (500m spacing)... ")
t0 = time()
grid_map = Dict{Tuple{Int,Int}, Int}()
sizehint!(grid_map, n)
for i in 1:n
    gx = round(Int, data[i, 2] / 500.0)
    gy = round(Int, data[i, 3] / 500.0)
    grid_map[(gx, gy)] = i
end

neighs = Vector{Vector{Int64}}(undef, n)
weights = Vector{Vector{Float64}}(undef, n)
nneighs = Vector{Int64}(undef, n)

for i in 1:n
    gx = round(Int, data[i, 2] / 500.0)
    gy = round(Int, data[i, 3] / 500.0)
    nb = Int64[]
    for dx in -1:1, dy in -1:1
        (dx == 0 && dy == 0) && continue
        cand = get(grid_map, (gx + dx, gy + dy), 0)
        if cand > 0
            push!(nb, cand)
        end
    end
    k = length(nb)
    neighs[i] = nb
    nneighs[i] = k
    weights[i] = k > 0 ? fill(1.0 / k, k) : Float64[]
end
W = SpatialWeights(n, neighs, weights, nneighs, :row)
@printf("Done in %.2fs\n", time() - t0)
@printf("Spatial Weights: N = %d, Mean neighbors: %.2f (max: %d, isolated islands: %d)\n",
        nobs(W), sum(nneighs)/n, maximum(nneighs), count(==(0), nneighs))

# 3. Simulate realistic spatially autocorrelated synthetic variable on Denmark
println("\nGenerating spatially correlated variable on Denmark terrain...")
rng = StableRNG(42)
# Spatial trend based on coordinates + local autoregressive component
x_norm = (data[:, 2] .- mean(data[:, 2])) ./ std(data[:, 2])
y_norm = (data[:, 3] .- mean(data[:, 3])) ./ std(data[:, 3])
x = 0.5 .* x_norm .+ 0.3 .* y_norm .+ 0.2 .* (x_norm .* y_norm) .+ randn(rng, n)
# Global Moran to confirm spatial autocorrelation
g_moran = moran(x, W)
@printf("Global Moran's I: %.4f (p-value: %.4f)\n", g_moran.I, g_moran.p)

P = 999
total_perms = Int64(n) * Int64(P)
println("--------------------------------------------------------------------------------")
@printf("Benchmark Configuration: N = %d | Permutations P = %d\n", n, P)
@printf("Total Permutations to Calculate: %d (%.2f Million Permutations)\n", total_perms, total_perms / 1e6)
println("--------------------------------------------------------------------------------")

# Warmup GPU
print("Warming up GPU pipeline... ")
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
println("RESULTS SUMMARY: ALL OF DENMARK (N = 179,674, P = 999)")
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
