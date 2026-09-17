module SpatialDependenceKernelAbstractionsExt

using SpatialDependence
using KernelAbstractions
using Statistics: mean

# Fast 64-bit SplitMix PRNG for device threads
@inline function splitmix64(state::UInt64)
    state += 0x9e3779b97f4a7c15
    z = state
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return (z ⊻ (z >> 31)), state
end

# Draw random index in 1:n excluding the focal observation
@inline function rand_index(state::UInt64, n::Int32, exclude::Int32)
    val, new_state = splitmix64(state)
    idx = Int32(val % UInt64(n - Int32(1))) + Int32(1)
    if idx >= exclude
        idx += Int32(1)
    end
    return idx, new_state
end

# Pack ragged spatial weights into dense/padded device buffers
function prepare_gpu_weights(backend, W::SpatialWeights, ::Type{T}, stat_code::Int32) where T
    n = W.n
    max_k = maximum(W.nneighs)
    wt = wtransformation(W)
    
    padded_weights_cpu = zeros(T, max_k, n)
    cardinalities_cpu = Int32.(W.nneighs)

    for i in 1:n
        k = W.nneighs[i]
        for (k_idx, w) in enumerate(W.weights[i])
            if stat_code == Int32(4)
                wistar = (wt == :row) ? T(1.0 / (k + 1)) : T(1.0)
                padded_weights_cpu[k_idx, i] = wistar
            else
                padded_weights_cpu[k_idx, i] = T(w)
            end
        end
    end

    padded_weights_gpu = KernelAbstractions.allocate(backend, T, max_k, n)
    KernelAbstractions.copyto!(backend, padded_weights_gpu, padded_weights_cpu)

    cardinalities_gpu = KernelAbstractions.allocate(backend, Int32, n)
    KernelAbstractions.copyto!(backend, cardinalities_gpu, cardinalities_cpu)

    return padded_weights_gpu, cardinalities_gpu, max_k
end

# General permutation kernel for Local Spatial Autocorrelation
@kernel function local_perm_chunk_kernel!(
    partial_larger, partial_sum, partial_sum_sq, full_perms,
    @Const(z), @Const(obs_stat), @Const(padded_weights), @Const(cardinalities),
    n::Int32, chunk_size::Int32, total_permutations::Int32, scale::T, base_seed::UInt64,
    stat_code::Int32, return_perms::Bool, ::Val{MAX_K}
) where {T, MAX_K}
    i, c = @index(Global, NTuple)
    if i <= n
        k = cardinalities[i]
        p_start = (c - Int32(1)) * chunk_size + Int32(1)
        p_end = min(total_permutations, c * chunk_size)

        if k > 0 && p_start <= p_end
            zi = z[i]
            s_obs = obs_stat[i]

            larger_count = Int32(0)
            sum_val = T(0)
            sum_sq_val = T(0)

            chosen = @private Int32 (MAX_K,)
            state = base_seed + UInt64(i) * 0x9e3779b97f4a7c15 + UInt64(c) * 0x517cc1b727220a95

            for p in p_start:p_end
                num_sampled = Int32(0)
                lag = T(0)
                geary_sum = T(0)

                while num_sampled < k
                    idx, state = rand_index(state, n, Int32(i))
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
                        w = padded_weights[num_sampled, i]
                        zj = z[idx]
                        if stat_code == Int32(1) || stat_code == Int32(3) || stat_code == Int32(4)
                            lag += w * zj
                        elseif stat_code == Int32(2)
                            diff = zi - zj
                            geary_sum += w * diff * diff
                        end
                    end
                end

                # Calculate permutation statistic based on stat_code:
                # 1: Local Moran: (zi / m2) * lag
                # 2: Local Geary: (1 / m2) * geary_sum
                # 3: Getis-Ord Gi: lag / (denom - xi)
                # 4: Getis-Ord Gi*: (wistar * xi + lag) / denom
                s_perm = T(0)
                if stat_code == Int32(1)
                    s_perm = (zi / scale) * lag
                elseif stat_code == Int32(2)
                    s_perm = (T(1) / scale) * geary_sum
                elseif stat_code == Int32(3)
                    denom_minus_xi = scale - zi
                    s_perm = lag / denom_minus_xi
                elseif stat_code == Int32(4)
                    # scale = denom, padded_weights contains wistar
                    s_perm = (padded_weights[1, i] * zi + lag) / scale
                end

                if s_perm >= s_obs
                    larger_count += Int32(1)
                end
                sum_val += s_perm
                sum_sq_val += s_perm * s_perm

                if return_perms
                    full_perms[i, p] = s_perm
                end
            end

            partial_larger[i, c] = larger_count
            partial_sum[i, c] = sum_val
            partial_sum_sq[i, c] = sum_sq_val
        end
    end
end

function SpatialDependence.crand_local_gpu(
    backend,
    stat_type::Symbol,
    permutations::Int,
    data::AbstractVector,
    W::SpatialWeights,
    obs_stat::AbstractVector,
    scale_param::Number;
    return_perms::Bool = true,
    seed::Union{Integer, Nothing} = nothing
)
    # Automatically resolve backend = :gpu if a GPU package is loaded
    actual_backend = if backend === :gpu
        found = nothing
        for (modkey, mod) in Base.loaded_modules
            if modkey.name == "Metal" && isdefined(mod, :MetalBackend)
                found = mod.MetalBackend()
                break
            elseif modkey.name == "CUDA" && isdefined(mod, :CUDABackend)
                found = mod.CUDABackend()
                break
            elseif modkey.name == "AMDGPU" && isdefined(mod, :ROCBackend)
                found = mod.ROCBackend()
                break
            end
        end
        if found === nothing
            throw(ArgumentError("backend=:gpu specified, but no loaded GPU backend was found. Please run `using Metal` (for Mac) or `using CUDA` (for NVIDIA) before calling."))
        end
        found
    else
        backend
    end

    n = length(data)
    T = Float32
    
    # Map stat_type to code
    stat_code = if stat_type == :moran
        Int32(1)
    elseif stat_type == :geary
        Int32(2)
    elseif stat_type == :getisord
        Int32(3)
    elseif stat_type == :getisord_star
        Int32(4)
    else
        throw(ArgumentError("Unknown stat_type $stat_type"))
    end

    # Dynamic tuning for 2D chunking
    num_chunks = Int(min(64, cld(permutations, 64)))
    chunk_size = Int(cld(permutations, num_chunks))

    # Prepare GPU data
    padded_weights_gpu, cardinalities_gpu, max_k = prepare_gpu_weights(actual_backend, W, T, stat_code)
    
    data_gpu = KernelAbstractions.allocate(actual_backend, T, n)
    KernelAbstractions.copyto!(actual_backend, data_gpu, T.(data))

    obs_stat_gpu = KernelAbstractions.allocate(actual_backend, T, n)
    KernelAbstractions.copyto!(actual_backend, obs_stat_gpu, T.(obs_stat))

    partial_larger_gpu = KernelAbstractions.zeros(actual_backend, Int32, n, num_chunks)
    partial_sum_gpu = KernelAbstractions.zeros(actual_backend, T, n, num_chunks)
    partial_sum_sq_gpu = KernelAbstractions.zeros(actual_backend, T, n, num_chunks)

    full_perms_gpu = if return_perms
        KernelAbstractions.zeros(actual_backend, T, n, permutations)
    else
        KernelAbstractions.zeros(actual_backend, T, 0, 0)
    end

    base_seed = seed === nothing ? UInt64(time_ns()) : UInt64(seed)

    # Dispatch compile-time MAX_K capacity
    val_k = if max_k <= 16
        Val(16)
    elseif max_k <= 32
        Val(32)
    elseif max_k <= 64
        Val(64)
    elseif max_k <= 128
        Val(128)
    else
        Val(256)
    end

    kernel! = local_perm_chunk_kernel!(actual_backend)
    kernel!(
        partial_larger_gpu, partial_sum_gpu, partial_sum_sq_gpu, full_perms_gpu,
        data_gpu, obs_stat_gpu, padded_weights_gpu, cardinalities_gpu,
        Int32(n), Int32(chunk_size), Int32(permutations), T(scale_param), base_seed,
        stat_code, return_perms, val_k,
        ndrange=(n, num_chunks)
    )
    KernelAbstractions.synchronize(actual_backend)

    # Download summary vectors to host
    h_larger = Array(partial_larger_gpu)
    h_sum = Array(partial_sum_gpu)
    h_sum_sq = Array(partial_sum_sq_gpu)

    total_larger = vec(sum(h_larger, dims=2))
    total_sum = vec(sum(h_sum, dims=2))
    total_sum_sq = vec(sum(h_sum_sq, dims=2))

    # Two-sided pseudo p-value
    larger_twosided = copy(total_larger)
    for i in 1:n
        low = (permutations - larger_twosided[i])
        if low < larger_twosided[i]
            larger_twosided[i] = low
        end
    end
    p_values = (Float64.(larger_twosided) .+ 1.0) ./ (permutations + 1)

    perms_mean = Float64.(total_sum) ./ permutations
    perms_var = max.(0.0, (Float64.(total_sum_sq) ./ permutations) .- perms_mean.^2)
    perms_std = sqrt.(perms_var)
    zval = (Float64.(obs_stat) .- perms_mean) ./ perms_std

    Iperms = if return_perms
        Float64.(Array(full_perms_gpu))
    else
        Matrix{Float64}(undef, 0, 0)
    end

    return Iperms, p_values, perms_mean, perms_std, zval
end

end # module
