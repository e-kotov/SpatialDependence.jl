using SpatialDependence
using KernelAbstractions
using Metal
using Shapefile
using Statistics
using Printf

const DATA_DIR = joinpath(@__DIR__, "data")
const SHP_PATH = joinpath(DATA_DIR, "natregimes.shp")

if !isfile(SHP_PATH)
    error("natregimes.shp not found at $SHP_PATH")
end

println("================================================================================")
println("Real-World Benchmark: US Counties (NAT / natregimes, N = 3,085)")
println("================================================================================")

# 1. Load Data
println("Loading $SHP_PATH...")
table = Shapefile.Table(SHP_PATH)
n = length(table)
println("Loaded $n counties.")

# 2. Build Spatial Weights (Queen Contiguity)
print("Building Queen contiguity weights matrix (polyneigh)... ")
t0 = time()
W = polyneigh(table)
t_w = time() - t0
@printf("Done in %.2fs (mean neighbors: %.2f, max: %d)\n", t_w, mean(W), maximum(W.nneighs))

# Extract variables: hr90 (Homicide rate 1990) and rd90 (Resource deprivation 1990)
hr90 = Float64.(table.hr90)
rd90 = Float64.(table.rd90)

# 3. Warmup
println("\nWarming up CPU and Metal GPU pipelines...")
_ = localmoran(hr90, W, permutations=100)
_ = localmoran(hr90, W, permutations=100, backend=:gpu, return_perms=false)

# 4. Benchmark Function
function run_benchmark(var_name::String, x::Vector{Float64}, P::Int)
    total_perms = Int64(n) * Int64(P)
    println("\n--------------------------------------------------------------------------------")
    @printf("Testing Variable: %s | Permutations: %d | Total: %d permutations\n", var_name, P, total_perms)
    println("--------------------------------------------------------------------------------")

    # Time CPU
    print("Running CPU (Multi-threaded)... ")
    t0 = time()
    res_cpu = localmoran(x, W, permutations=P)
    t_cpu = time() - t0
    @printf("%.3f seconds\n", t_cpu)

    # Time GPU (Apple M4 Metal)
    print("Running GPU (Apple M4 Metal)... ")
    t0 = time()
    res_gpu = localmoran(x, W, permutations=P, backend=:gpu, return_perms=false, seed=12345)
    t_gpu = time() - t0
    @printf("%.3f seconds\n", t_gpu)

    speedup = t_cpu / t_gpu
    throughput_gpu = total_perms / t_gpu / 1e6 # Million perms / sec
    @printf(">>> Speedup: %.1fx faster on Apple M4 GPU\n", speedup)
    @printf(">>> GPU Throughput: %.1f million permutations/sec\n", throughput_gpu)

    # Validate statistical equivalence
    corr_p = cor(pvalue(res_cpu), pvalue(res_gpu))
    corr_z = cor(zscore(res_cpu), zscore(res_gpu))
    @printf(">>> Statistical Consistency: p-value r = %.5f, z-score r = %.5f\n", corr_p, corr_z)

    return (P=P, t_cpu=t_cpu, t_gpu=t_gpu, speedup=speedup, throughput=throughput_gpu, corr_p=corr_p)
end

# Run benchmarks across permutation counts
results_hr90_999 = run_benchmark("hr90 (Homicide Rates)", hr90, 999)
results_hr90_9999 = run_benchmark("hr90 (Homicide Rates)", hr90, 9999)
results_hr90_99999 = run_benchmark("hr90 (Homicide Rates)", hr90, 99999)

results_rd90_9999 = run_benchmark("rd90 (Resource Deprivation)", rd90, 9999)

println("\n================================================================================")
println("Summary Table: US Counties (N = 3,085)")
println("================================================================================")
@printf("%-10s | %-12s | %-12s | %-10s | %-12s\n", "Perms (P)", "CPU Time", "GPU Time (M4)", "Speedup", "Correlation")
println("-----------+--------------+--------------+------------+-------------")
for r in [results_hr90_999, results_hr90_9999, results_hr90_99999, results_rd90_9999]
    @printf("%-10d | %10.3fs | %10.3fs | %8.1fx | %11.5f\n", r.P, r.t_cpu, r.t_gpu, r.speedup, r.corr_p)
end
println("================================================================================")
