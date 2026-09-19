using SpatialDependence
using KernelAbstractions
using Metal
using Statistics
using Printf
using StableRNGs
using DelimitedFiles

# Fast 32-bit Thomas-Wang Hash PRNG mapping to [0, max_rand) via float multiply
@inline function wang_hash(seed::UInt32)
    s = seed
    s = (s ⊻ UInt32(61)) ⊻ (s >> UInt32(16))
    s = s * UInt32(9)
    s = s ⊻ (s >> UInt32(4))
    s = s * UInt32(0x27d4eb2d)
    s = s ⊻ (s >> UInt32(15))
    return s
end

@inline function fast_rand_index(seed::UInt32, n_minus_1::Float32, n::Int32, exclude::Int32)
    s = wang_hash(seed)
    rng_val = (Float32(s) * 2.3283061f-10) * n_minus_1
    idx = min(Int32(trunc(rng_val)) + Int32(1), n - Int32(1))
    if idx >= exclude
        idx += Int32(1)
    end
    return idx
end

# 1D Optimized Kernel
@kernel function fast_lisa_1d_kernel!(
    p_out,
    @Const(z), @Const(obs_stat), @Const(cardinalities),
    n::Int32, total_permutations::Int32, m2::Float32, base_seed::UInt32,
    ::Val{MAX_K}
) where {MAX_K}
    idx_global = @index(Global, Linear)
    i = Int32(idx_global)
    if i <= n
        k = cardinalities[i]
        if k > Int32(0)
            zi = z[i]
            s_obs = obs_stat[i]
            n_minus_1 = Float32(n - Int32(1))
            inv_k = 1.0f0 / Float32(k)

            larger_count = Int32(0)
            chosen = @private Int32 (MAX_K,)
            seed_thread = base_seed + reinterpret(UInt32, i) * 0x0019660d

            for perm in Int32(1):total_permutations
                num_sampled = Int32(0)
                sum_zj = 0.0f0

                while num_sampled < k
                    seed_thread += UInt32(1)
                    idx = fast_rand_index(seed_thread, n_minus_1, n, i)
                    
                    # Register collision check
                    is_dup = false
                    for prev in Int32(1):num_sampled
                        if chosen[prev] == idx
                            is_dup = true
                            break
                        end
                    end
                    if !is_dup
                        num_sampled += Int32(1)
                        chosen[num_sampled] = idx
                        sum_zj += z[idx]
                    end
                end

                # Hoisted row-standardized Local Moran
                s_perm = (zi / m2) * (sum_zj * inv_k)

                if s_perm >= s_obs
                    larger_count += Int32(1)
                end
            end

            low = total_permutations - larger_count
            c_larger = (low < larger_count) ? low : larger_count
            p_out[i] = Float32(c_larger + Int32(1)) / Float32(total_permutations + Int32(1))
        else
            p_out[i] = 1.0f0
        end
    end
end

# Load Denmark dataset
csv_path = joinpath(@__DIR__, "data", "dk_grid_coords.csv")
data = readdlm(csv_path, ',', Float64)
n = size(data, 1)

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

# Spatially correlated variable
rng = StableRNG(42)
x_norm = (data[:, 2] .- mean(data[:, 2])) ./ std(data[:, 2])
y_norm = (data[:, 3] .- mean(data[:, 3])) ./ std(data[:, 3])
x = 0.5 .* x_norm .+ 0.3 .* y_norm .+ 0.2 .* (x_norm .* y_norm) .+ randn(rng, n)

# Prepare arrays
z = Float32.(x .- mean(x))
m2 = Float32(sum(z.^2) / n)
lz = slag(W, z)
obs_stat = Float32.((z ./ m2) .* lz)

backend = Metal.MetalBackend()
P = Int32(999)
val_k = Val(16)

z_gpu = KernelAbstractions.allocate(backend, Float32, n)
KernelAbstractions.copyto!(backend, z_gpu, z)

obs_gpu = KernelAbstractions.allocate(backend, Float32, n)
KernelAbstractions.copyto!(backend, obs_gpu, obs_stat)

cards_gpu = KernelAbstractions.allocate(backend, Int32, n)
KernelAbstractions.copyto!(backend, cards_gpu, Int32.(nneighs))

p_out_gpu = KernelAbstractions.zeros(backend, Float32, n)

kernel! = fast_lisa_1d_kernel!(backend)

# Warmup
kernel!(p_out_gpu, z_gpu, obs_gpu, cards_gpu, Int32(100), Int32(10), m2, UInt32(123), val_k, ndrange=100)
KernelAbstractions.synchronize(backend)

# Benchmark Optimized Kernel
println("Benchmarking Optimized 1D Julia Metal Kernel on Denmark (N = $n, P = $P)...")
t0 = time()
kernel!(p_out_gpu, z_gpu, obs_gpu, cards_gpu, Int32(n), P, m2, UInt32(12345), val_k, ndrange=n)
KernelAbstractions.synchronize(backend)
t_opt = time() - t0
p_res = Array(p_out_gpu)

total_perms = Int64(n) * Int64(P)
throughput = total_perms / t_opt / 1e6

@printf("Optimized Julia GPU Time (Denmark N=179k): %.3f seconds (%.2f ms)\n", t_opt, t_opt * 1000)
@printf("Optimized GPU Throughput: %.1f Million perms/sec\n", throughput)

# 50k Grid Test
dim50 = 224
n50 = dim50 * dim50
geom50 = reggeomlattice(dim50, dim50)
W50 = polyneigh(geom50)
cards50_gpu = KernelAbstractions.allocate(backend, Int32, n50)
KernelAbstractions.copyto!(backend, cards50_gpu, Int32.(W50.nneighs))
z50_gpu = KernelAbstractions.allocate(backend, Float32, n50)
KernelAbstractions.copyto!(backend, z50_gpu, randn(Float32, n50))
obs50_gpu = KernelAbstractions.allocate(backend, Float32, n50)
KernelAbstractions.copyto!(backend, obs50_gpu, randn(Float32, n50))
p_out50 = KernelAbstractions.zeros(backend, Float32, n50)

t0 = time()
kernel!(p_out50, z50_gpu, obs50_gpu, cards50_gpu, Int32(n50), P, 1.0f0, UInt32(12345), val_k, ndrange=n50)
KernelAbstractions.synchronize(backend)
t_50k = time() - t0
@printf("Optimized Julia GPU Time (50k Grid N=50k): %.3f seconds (%.2f ms)\n", t_50k, t_50k * 1000)
@printf("Optimized 50k Throughput: %.1f Million perms/sec\n", (Int64(n50) * Int64(P)) / t_50k / 1e6)
