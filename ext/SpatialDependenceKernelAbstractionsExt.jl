module SpatialDependenceKernelAbstractionsExt

using SpatialDependence
using KernelAbstractions
using Random: AbstractRNG, default_rng, rand
using Statistics: mean

include("SpatialDependenceKernelAbstractionsExact.jl")

const _GPU_PRIVATE_CROSSOVER = 256
const _GPU_DEFAULT_SCRATCH_BUDGET = 64 * 1024 * 1024
const _GPU_SCRATCH_BUDGET = Ref{Int}(_GPU_DEFAULT_SCRATCH_BUDGET)

"""Internal test hook; deliberately not part of the public keyword API."""
function _set_gpu_scratch_budget!(bytes::Integer)
    bytes > 0 || throw(ArgumentError("GPU scratch budget must be positive"))
    old = _GPU_SCRATCH_BUDGET[]
    _GPU_SCRATCH_BUDGET[] = Int(bytes)
    return old
end

const _GPU_CHUNK_COUNT_OVERRIDE = Ref{Int}(0)

"""
Internal test hook; deliberately not part of the public keyword API.

`0` restores the default rule.  Streams are keyed by (observation, permutation)
and moments are accumulated in blocks of the global permutation index, so the
chunk count only decides how permutations are spread over workers: it cannot
change any returned value.  Chunk sizes are rounded down to whole moment blocks
and the count renormalised, so the effective count may differ from (and is
usually at least) the one passed here.
"""
function _set_gpu_chunk_count!(chunks::Integer)
    chunks >= 0 || throw(ArgumentError("GPU chunk count override must be nonnegative"))
    old = _GPU_CHUNK_COUNT_OVERRIDE[]
    _GPU_CHUNK_COUNT_OVERRIDE[] = Int(chunks)
    return old
end

"""
Length of one on-device moment block, as a function of `permutations` alone.

MOMENT_BLOCK_GLOBAL_INDEX: Welford moments are accumulated in fixed blocks of
the *global* permutation index rather than per chunk.  Chunk sizes are whole
numbers of blocks (`crand_local_gpu` rounds the blocks per chunk down), so every
block is produced start to finish by exactly one worker and the host always
merges the same blocks in the same order.  Together with the (observation,
permutation) RNG keying that makes the whole accelerated result — draws,
p-values, mean, standard deviation and z-score — a function of the seed only,
independent of the chunk count, of the row/chunk batching and of the scratch
budget.

The block *length* is therefore free, and is purely a cost trade-off.  Short
blocks mean more partial moments to copy back and merge on the host
(`n × permutations ÷ block` of them), which dominates `return_perms = false`
runs at large `n`; long blocks mean fewer, longer Welford runs in device
precision.  Growing the block with `permutations` caps the number of blocks per
row at 128 for any `permutations`, so the merge cost stays proportional to `n`.

That cap is also the only thing bounding the moment buffers.  `_run_bucket!`
allocates `partial_mean`, `partial_m2` and `partial_anchor` with one column per
block of the launched *span*, so a row now carries up to 128 partials where the
earlier per-chunk layout carried one per chunk in the batch (at most 64 under
the default chunk rule).  All three are allocated on the device and copied whole
to the host, i.e. `rows in the batch × blocks in the span × 3 × sizeof(T)` bytes
on each side, at most `rows × 128 × 3 × sizeof(T)`.  Nothing clamps that:
`_GPU_SCRATCH_BUDGET` sizes only the per-worker hash scratch, and the moment
buffers are allocated outside its accounting.
"""
_moment_block(permutations::Int) = 64 * nextpow(2, max(1, cld(permutations, 8192)))

@inline function splitmix64(state::UInt64)
    state += 0x9e3779b97f4a7c15
    z = state
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return (z ⊻ (z >> 31)), state
end

@inline function rand_index(state::UInt64, n::Int32, exclude::Int32)
    value, new_state = splitmix64(state)
    idx = unsafe_trunc(Int32, value % unsafe_trunc(UInt64, n - Int32(1))) + Int32(1)
    idx >= exclude && (idx += Int32(1))
    return idx, new_state
end

@inline function _checked_i32(value::Integer, label::AbstractString)
    (0 <= value <= typemax(Int32)) ||
        throw(ArgumentError("$label does not fit in the device Int32 domain"))
    return Int32(value)
end

@inline function _chunk_start_i64(chunk_first::Int64, chunk_size::Int64)
    return (chunk_first - Int64(1)) * chunk_size + Int64(1)
end

@inline function _chunk_end_i64(total_permutations::Int64,
                                chunk_first::Int64,
                                chunk_size::Int64)
    return min(total_permutations, chunk_first * chunk_size)
end

function _checked_edges(degrees::Vector{Int})
    total = Int64(0)
    offsets = Vector{Int64}(undef, length(degrees) + 1)
    offsets[1] = Int64(1)
    for i in eachindex(degrees)
        degree = degrees[i]
        degree >= 0 || throw(ArgumentError("neighbor cardinalities must be nonnegative"))
        degree <= typemax(Int64) - total || throw(ArgumentError("CSR edge count overflows Int64"))
        total += Int64(degree)
        offsets[i + 1] = total + Int64(1)
    end
    total <= typemax(Int) || throw(ArgumentError("CSR edge count exceeds host addressable storage"))
    return offsets, Int(total)
end

"""
Reject a structurally malformed `SpatialWeights` before any payload is built.

`crand_local_gpu` constructs the exact-tail payload before `_checked_edges` and
`_gpu_csr_host` ever look at `W`, so without this a malformed graph reaches the
host encoder first.  That mattered: with `precision=Float32` a negative degree
made the reference encoder's CSR edge total under-count while its row offsets
kept advancing, and the packer wrote past the end of its own buffer, while
`precision=Float64` (which skips the payload) correctly reported
`ArgumentError`.  Running this first makes both precisions reject the same
graphs with the same messages, before any RNG draw or device allocation.

The classification comes from `_exact_weights_defect`, the same predicate the
exact-tail fast path uses as its `@inbounds` precondition, so there is one
definition of "malformed".  Degrees above `n` are not checked here: the caller's
`max_k <= n - 1` test already reports those with its own message.
"""
function _validate_weights_structure(W::SpatialWeights, n::Int)
    defect = _exact_weights_defect(W, n)
    defect === :ok && return nothing
    defect === :negative_degree &&
        throw(ArgumentError("neighbor cardinalities must be nonnegative"))
    defect === :neighbor_domain &&
        throw(ArgumentError("neighbor index is outside the weights domain"))
    throw(ArgumentError("spatial weights are structurally inconsistent: nneighs[i], neighs[i] and weights[i] must agree"))
end

# STORAGE_RAGGED_CSR: the production representation is flat CSR plus Int64 offsets.
"""Build ragged CSR buffers; no n × max-degree storage is allocated."""
function _gpu_csr_host(W::SpatialWeights, ::Type{T}, stat_code::Int32) where T
    n = W.n
    _checked_i32(n, "number of observations")
    degrees = Int[Int(W.nneighs[i]) + (stat_code == Int32(4) ? 1 : 0) for i in 1:n]
    offsets, total_edges = _checked_edges(degrees)
    neighbors = Vector{Int32}(undef, total_edges)
    weights = Vector{T}(undef, total_edges)
    wt = wtransformation(W)
    edge = 1
    for i in 1:n
        k = Int(W.nneighs[i])
        _checked_i32(k, "neighbor cardinality")
        if stat_code == Int32(4)
            wistar = wt == :row ? T(1.0 / (k + 1)) : one(T)
            neighbors[edge] = Int32(i)
            weights[edge] = wistar
            edge += 1
            for slot in 1:k
                neighbor = W.neighs[i][slot]
                1 <= neighbor <= n || throw(ArgumentError("neighbor index is outside the weights domain"))
                neighbors[edge] = _checked_i32(neighbor, "neighbor index")
                weights[edge] = wistar
                edge += 1
            end
        else
            for slot in 1:k
                neighbor = W.neighs[i][slot]
                1 <= neighbor <= n || throw(ArgumentError("neighbor index is outside the weights domain"))
                neighbors[edge] = _checked_i32(neighbor, "neighbor index")
                weights[edge] = T(W.weights[i][slot])
                edge += 1
            end
        end
    end
    edge == total_edges + 1 || throw(ArgumentError("internal CSR edge-count mismatch"))
    return weights, neighbors, offsets
end

"""Upload ragged weights and return `(weights, neighbors, offsets, max_degree)`."""
function prepare_gpu_weights(backend, W::SpatialWeights, ::Type{T}, stat_code::Int32) where T
    weights, neighbors, offsets = _gpu_csr_host(W, T, stat_code)
    weights_gpu = KernelAbstractions.allocate(backend, T, max(1, length(weights)))
    neighbors_gpu = KernelAbstractions.allocate(backend, Int32, max(1, length(neighbors)))
    offsets_gpu = KernelAbstractions.allocate(backend, Int64, length(offsets))
    !isempty(weights) && KernelAbstractions.copyto!(backend, weights_gpu, weights)
    !isempty(neighbors) && KernelAbstractions.copyto!(backend, neighbors_gpu, neighbors)
    KernelAbstractions.copyto!(backend, offsets_gpu, offsets)
    return weights_gpu, neighbors_gpu, offsets_gpu, maximum(Int.(W.nneighs); init = 0)
end

@inline function _edge_at(start::Int64, slot::Int32, stat_code::Int32)
    return start + Int64(slot - (stat_code == Int32(4) ? Int32(0) : Int32(1)))
end

@kernel function _local_perm_worker_kernel!(
    partial_upper, partial_lower, partial_mean, partial_m2, partial_anchor, full_perms,
    @Const(z), @Const(weights), @Const(neighbors), @Const(row_offsets),
    @Const(exact_values), @Const(exact_weight_magnitudes), @Const(exact_weight_signs),
    @Const(exact_observed), @Const(exact_defined), @Const(row_ids), global_scratch,
    n::Int32, row_count::Int32, chunk_count::Int32, chunk_size::Int32, total_permutations::Int32,
    chunk_first::Int32, scale::T, base_seed::UInt64, stat_code::Int32,
    return_perms::Bool, use_global_scratch::Bool, strides::NTuple{2, Int64},
    ::Val{PRIVATE_K}, ::Val{EXACT_TAILS}
) where {T, PRIVATE_K, EXACT_TAILS}
    local_i, local_c = @index(Global, NTuple)
    # METAL_ARGUMENT_LIMIT: Metal binds at most 31 kernel arguments and this
    # kernel sits exactly at that limit, so the two Int64 lengths travel as one
    # tuple.  Pack further scalars the same way instead of adding arguments.
    scratch_stride = strides[1]
    moment_block = strides[2]
    chosen = @private Int32 (PRIVATE_K,)
    private_hash = @private Int32 (2 * PRIVATE_K,)
    exact_acc = @private UInt32 (EXACT_TAILS ? 8 : 1,)
    exact_term = @private UInt32 (EXACT_TAILS ? 8 : 1,)
    exact_left = @private UInt32 (EXACT_TAILS ? 8 : 1,)
    exact_right = @private UInt32 (EXACT_TAILS ? 8 : 1,)
    exact_square = @private UInt32 (EXACT_TAILS ? 8 : 1,)
    exact_observed_local = @private UInt32 (EXACT_TAILS ? 8 : 1,)
    # KernelAbstractions already masks every ndrange dimension through
    # __validindex; both guards are kept explicit so the two indices read
    # symmetrically.
    if local_i <= row_count && local_c <= chunk_count
        i = row_ids[local_i]
        k = unsafe_trunc(Int32, row_offsets[i + Int64(1)] - row_offsets[i])
        stat_code == Int32(4) && (k -= Int32(1))
        global_c = Int64(chunk_first) + Int64(local_c) - Int64(1)
        p_start = _chunk_start_i64(global_c, Int64(chunk_size))
        p_end = _chunk_end_i64(Int64(total_permutations), global_c, Int64(chunk_size))
        row_start = row_offsets[i]
        if p_start <= p_end
            zi = z[i]
            obs_lag = T(0); obs_abs_lag = T(0)
            obs_geary_sum = T(0); obs_abs_geary_sum = T(0)
            for slot in Int32(1):k
                edge = _edge_at(row_start, slot, stat_code)
                idx = neighbors[edge]; w = weights[edge]; zj = z[idx]
                if stat_code == Int32(1) || stat_code == Int32(3) || stat_code == Int32(4)
                    contribution = w * zj
                    obs_lag += contribution; obs_abs_lag += abs(contribution)
                elseif stat_code == Int32(2)
                    diff = zi - zj; contribution = w * diff * diff
                    obs_geary_sum += contribution; obs_abs_geary_sum += abs(contribution)
                end
            end
            focal_weight = stat_code == Int32(4) ? weights[row_start] : T(0)
            obs_cmp = stat_code == Int32(4) ? focal_weight * zi + obs_lag :
                      stat_code == Int32(3) ? obs_lag :
                      stat_code == Int32(1) ? (zi / scale) * obs_lag :
                      stat_code == Int32(2) ? (T(1) / scale) * obs_geary_sum : T(0)
            if EXACT_TAILS
                _exact_zero!(exact_observed_local)
                for limb in 1:8
                    exact_observed_local[limb] = exact_observed[limb, i]
                end
            end
            mean_val = T(0); m2_val = T(0); anchor = T(0)
            upper_count = Int32(0); lower_count = Int32(0)
            # RNG_KEY_OBSERVATION_PERMUTATION: every (observation, permutation)
            # pair owns its stream, so seeded draws never depend on how
            # permutations are chunked or batched across workers.
            row_key = base_seed ⊻ (unsafe_trunc(UInt64, i) * 0x9e3779b97f4a7c15)
            # First global permutation index this batch of chunks covers, minus
            # one: both the retained-draw column and the moment-block slot are
            # relative to it.  `chunk_size` is a whole multiple of
            # `moment_block` whenever more than one chunk exists, so this offset
            # is block-aligned and no block ever straddles two workers.
            permutation_offset = (Int64(chunk_first) - Int64(1)) * Int64(chunk_size)
            worker_linear = (Int64(local_i) - Int64(1)) * Int64(chunk_count) + Int64(local_c) - Int64(1)
            scratch_base = worker_linear * scratch_stride
            hash_capacity = use_global_scratch ? scratch_stride : Int64(2 * PRIVATE_K)
            for p in p_start:p_end
                state, _ = splitmix64(row_key ⊻ (unsafe_trunc(UInt64, p) * 0x517cc1b727220a95))
                num_sampled = Int32(0)
                lag = T(0); abs_lag = T(0); geary_sum = T(0); abs_geary_sum = T(0)
                EXACT_TAILS && _exact_zero!(exact_acc)
                if k > Int32(64)
                    for slot in Int64(1):hash_capacity
                        if use_global_scratch
                            global_scratch[scratch_base + slot] = Int32(0)
                        else
                            private_hash[slot] = Int32(0)
                        end
                    end
                end
                while num_sampled < k
                    idx, state = rand_index(state, n, i)
                    is_dup = false
                    if k <= Int32(64)
                        for prev in Int32(1):num_sampled
                            if chosen[prev] == idx
                                is_dup = true; break
                            end
                        end
                    else
                        hash_slot = unsafe_trunc(Int64,
                            (unsafe_trunc(UInt64, idx) * UInt64(0x9e3779b9)) &
                            unsafe_trunc(UInt64, hash_capacity - Int64(1))) + Int64(1)
                        while true
                            stored = use_global_scratch ?
                                global_scratch[scratch_base + hash_slot] : private_hash[hash_slot]
                            if stored == Int32(0)
                                if use_global_scratch
                                    global_scratch[scratch_base + hash_slot] = idx
                                else
                                    private_hash[hash_slot] = idx
                                    chosen[num_sampled + Int32(1)] = unsafe_trunc(Int32, hash_slot)
                                end
                                break
                            elseif stored == idx
                                is_dup = true; break
                            end
                            hash_slot = hash_slot == hash_capacity ? Int64(1) : hash_slot + Int64(1)
                        end
                    end
                    if !is_dup
                        num_sampled += Int32(1)
                        k <= Int32(64) && (chosen[num_sampled] = idx)
                        edge = _edge_at(row_start, num_sampled, stat_code)
                        w = weights[edge]; zj = z[idx]
                        if stat_code == Int32(1) || stat_code == Int32(3) || stat_code == Int32(4)
                            contribution = w * zj
                            lag += contribution; abs_lag += abs(contribution)
                        elseif stat_code == Int32(2)
                            diff = zi - zj; contribution = w * diff * diff
                            geary_sum += contribution; abs_geary_sum += abs(contribution)
                        end
                        if EXACT_TAILS
                            if stat_code == Int32(2)
                                _exact_load_value!(exact_left, exact_values, i)
                                _exact_load_value!(exact_right, exact_values, idx)
                                if _exact_unsigned_cmp96(exact_left, exact_right) < Int32(0)
                                    _exact_sub96!(exact_term, exact_right, exact_left)
                                else
                                    _exact_sub96!(exact_term, exact_left, exact_right)
                                end
                                _exact_mul_3x3!(exact_square, exact_term, exact_term)
                                _exact_load_weight!(exact_right, exact_weight_magnitudes, edge)
                                _exact_mul_6x2!(exact_term, exact_square, exact_right)
                            else
                                _exact_load_value!(exact_left, exact_values, idx)
                                _exact_load_weight!(exact_right, exact_weight_magnitudes, edge)
                                _exact_mul_3x2!(exact_term, exact_left, exact_right)
                            end
                            if exact_weight_signs[edge] < Int8(0)
                                _exact_negate!(exact_term)
                            end
                            _exact_add!(exact_acc, exact_acc, exact_term)
                        end
                    end
                end
                s_perm = stat_code == Int32(4) ? focal_weight * zi + lag :
                         stat_code == Int32(3) ? lag :
                         stat_code == Int32(1) ? (zi / scale) * lag :
                         stat_code == Int32(2) ? (T(1) / scale) * geary_sum : T(0)
                obs_bound = stat_code == Int32(1) ? abs(zi / scale) * obs_abs_lag :
                            stat_code == Int32(2) ? abs(T(1) / scale) * obs_abs_geary_sum :
                            stat_code == Int32(4) ? abs(focal_weight * zi) + obs_abs_lag : obs_abs_lag
                perm_bound = stat_code == Int32(1) ? abs(zi / scale) * abs_lag :
                             stat_code == Int32(2) ? abs(T(1) / scale) * abs_geary_sum :
                             stat_code == Int32(4) ? abs(focal_weight * zi) + abs_lag : abs_lag
                tol = T(4) * eps(T) * (T(k) + T(2)) * (obs_bound + perm_bound)
                if EXACT_TAILS && exact_defined[i]
                    cmp = _exact_signed_cmp(exact_acc, exact_observed_local)
                    cmp >= Int32(0) && (upper_count += Int32(1))
                    cmp <= Int32(0) && (lower_count += Int32(1))
                else
                    delta_cmp = s_perm - obs_cmp
                    delta_cmp >= -tol && (upper_count += Int32(1))
                    delta_cmp <= tol && (lower_count += Int32(1))
                end
                if return_perms
                    full_perms[local_i, p - permutation_offset] = s_perm
                end
                # Moment block boundaries follow the global permutation index.
                # The block restarts at its first permutation, which also
                # supplies the shift anchor, and is flushed at its last one
                # (or at `p_end`, which only cuts a block short on the final
                # permutation of the run).
                block_pos = (p - Int64(1)) % moment_block
                if block_pos == Int64(0)
                    anchor = s_perm; mean_val = T(0); m2_val = T(0)
                end
                centered = s_perm - anchor
                count = block_pos + Int64(1)
                delta = centered - mean_val
                mean_val += delta / T(count)
                delta2 = centered - mean_val
                m2_val += delta * delta2
                if block_pos == moment_block - Int64(1) || p == p_end
                    block_slot = (p - Int64(1) - permutation_offset) ÷ moment_block + Int64(1)
                    partial_mean[local_i, block_slot] = mean_val
                    partial_m2[local_i, block_slot] = m2_val
                    partial_anchor[local_i, block_slot] = anchor
                end
            end
            # Tail counts stay per chunk: they are exact integers, so the host
            # sum does not depend on the order they are accumulated in.
            partial_upper[local_i, local_c] = upper_count
            partial_lower[local_i, local_c] = lower_count
        end
    end
end

# Compatibility adapter for exact-tail helper tests. It invokes the same worker
# kernel in private mode; production uses the richer batched launcher.
#
# The adapter keeps the pre-moment-block contract: one moment block per chunk,
# so `partial_mean`/`partial_m2`/`partial_anchor` keep their `nrows × nchunks`
# shape and column `c` still holds the moments of chunk `c`.  Passing
# `moment_block = chunk_size` reproduces that exactly, because the adapter
# always launches from `chunk_first = 1`: block boundaries then coincide with
# chunk boundaries and the block slot equals the chunk index.
function local_perm_chunk_kernel!(backend, groupsize)
    worker_kernel = _local_perm_worker_kernel!(backend, groupsize)
    return function (partial_upper, partial_lower, partial_mean, partial_m2, partial_anchor,
                    full_perms, z, weights, neighbors, row_offsets, exact_values,
                    exact_weight_magnitudes, exact_weight_signs, exact_observed,
                    exact_defined, n, chunk_size, total_permutations, scale, base_seed,
                    stat_code, return_perms, private_val, exact_val; ndrange)
        nrows = ndrange isa Tuple ? Int(ndrange[1]) : Int(ndrange)
        nchunks = ndrange isa Tuple ? Int(ndrange[2]) : 1
        row_ids = collect(Int32, 1:nrows)
        dummy_scratch = KernelAbstractions.zeros(backend, Int32, 1)
        worker_kernel(partial_upper, partial_lower, partial_mean, partial_m2, partial_anchor,
                      full_perms, z, weights, neighbors, row_offsets,
                      exact_values, exact_weight_magnitudes, exact_weight_signs,
                      exact_observed, exact_defined, row_ids, dummy_scratch,
                      Int32(n), Int32(nrows), Int32(nchunks), Int32(chunk_size), Int32(total_permutations),
                      Int32(1), scale, base_seed, stat_code, return_perms, false,
                      (Int64(1), Int64(max(1, Int(chunk_size)))), private_val, exact_val;
                      ndrange = ndrange)
    end
end

local_perm_chunk_kernel!(backend) = local_perm_chunk_kernel!(backend, 1)

function _degree_bucket(k::Int)
    k <= _GPU_PRIVATE_CROSSOVER && return max(16, nextpow(2, max(k, 1)))
    bucket = 1
    while bucket < k
        bucket <= typemax(Int) ÷ 2 || throw(ArgumentError("neighbor degree bucket overflows host size arithmetic"))
        bucket *= 2
    end
    return bucket
end

# Chan-Golub-LeVeque merge of one partial (mean, m2, count) into a row's
# running moments.  Callers must merge blocks in increasing global permutation
# order; that order, and hence the result down to its last bit, is fixed by
# `_moment_block` alone.
@inline function _merge_moments!(mean_all, m2_all, count_all, row, block_mean, block_m2, block_count)
    if count_all[row] == 0
        mean_all[row] = block_mean; m2_all[row] = block_m2; count_all[row] = block_count
    else
        delta = block_mean - mean_all[row]
        total = count_all[row] + block_count
        mean_all[row] += delta * (block_count / total)
        m2_all[row] += block_m2 + delta * delta * (count_all[row] * block_count / total)
        count_all[row] = total
    end
end

function _upload_row_ids(backend, rows::Vector{Int})
    ids = Int32.(rows)
    gpu = KernelAbstractions.allocate(backend, Int32, length(ids))
    KernelAbstractions.copyto!(backend, gpu, ids)
    return gpu
end

function _run_bucket!(backend, worker_kernel, rows::Vector{Int}, bucket::Int,
                      data_gpu, weights_gpu, neighbors_gpu, offsets_gpu,
                      exact_values_gpu, exact_weight_magnitudes_gpu, exact_weight_signs_gpu,
                      exact_observed_gpu, exact_defined_gpu, n::Int, permutations::Int,
                      chunk_size::Int, num_chunks::Int, base_seed::UInt64, stat_code::Int32,
                      return_perms::Bool, exact_tails::Bool, kernel_scale, total_upper,
                      total_lower, means, m2s, counts, retained, scratch_budget::Int)
    use_global = bucket > _GPU_PRIVATE_CROSSOVER
    scratch_stride = use_global ? Int64(2) * Int64(bucket) : Int64(1)
    bytes_per_worker = use_global ? scratch_stride * Int64(sizeof(Int32)) : Int64(1)
    max_workers = max(Int64(1), Int64(scratch_budget) ÷ bytes_per_worker)
    # Every span this loop hands to the kernel starts at a moment-block
    # boundary, because its first permutation is `(chunk_first - 1) * chunk_size
    # + 1` and `chunk_size` is a whole number of blocks.  A single chunk covers
    # the whole run, so it needs no padding and is exempt.
    moment_block = _moment_block(permutations)
    (num_chunks <= 1 || chunk_size % moment_block == 0) ||
        throw(ArgumentError("internal GPU chunk size is not a whole number of moment blocks"))
    row_pos = 1; chunk_first = 1
    while chunk_first <= num_chunks
        chunk_count = Int(min(num_chunks - chunk_first + 1, max_workers))
        rows_per_batch = Int(max(Int64(1), max_workers ÷ Int64(chunk_count)))
        while row_pos <= length(rows)
            last_row = min(length(rows), row_pos + rows_per_batch - 1)
            row_batch = rows[row_pos:last_row]
            nrows = length(row_batch)
            workers = Int64(nrows) * Int64(chunk_count)
            workers <= max_workers || throw(ArgumentError("internal GPU scratch batch exceeds its checked budget"))
            ids_gpu = _upload_row_ids(backend, row_batch)
            worker_scratch = if use_global
                total_slots = workers * scratch_stride
                total_slots <= typemax(Int) || throw(ArgumentError("GPU scratch allocation exceeds host addressable storage"))
                KernelAbstractions.zeros(backend, Int32, Int(total_slots))
            else
                KernelAbstractions.zeros(backend, Int32, 1)
            end
            span_start = (chunk_first - 1) * chunk_size + 1
            span_end = min(permutations, (chunk_first + chunk_count - 1) * chunk_size)
            span = span_end - span_start + 1
            # One moment column per block of this span, not per chunk; at most
            # 128 columns for any `permutations` (see `_moment_block`).
            span_blocks = cld(span, moment_block)
            batch_upper = KernelAbstractions.zeros(backend, Int32, nrows, chunk_count)
            batch_lower = KernelAbstractions.zeros(backend, Int32, nrows, chunk_count)
            batch_mean = KernelAbstractions.zeros(backend, eltype(data_gpu), nrows, span_blocks)
            batch_m2 = KernelAbstractions.zeros(backend, eltype(data_gpu), nrows, span_blocks)
            batch_anchor = KernelAbstractions.zeros(backend, eltype(data_gpu), nrows, span_blocks)
            batch_perms = return_perms ? KernelAbstractions.zeros(backend, eltype(data_gpu), nrows, span) :
                         KernelAbstractions.zeros(backend, eltype(data_gpu), 0, 0)
            worker_kernel(batch_upper, batch_lower, batch_mean, batch_m2, batch_anchor,
                          batch_perms, data_gpu, weights_gpu, neighbors_gpu, offsets_gpu,
                      exact_values_gpu, exact_weight_magnitudes_gpu,
                          exact_weight_signs_gpu, exact_observed_gpu, exact_defined_gpu,
                          ids_gpu, worker_scratch, _checked_i32(n, "number of observations"),
                          _checked_i32(nrows, "logical worker row batch"),
                          _checked_i32(chunk_count, "logical worker chunk batch"),
                          _checked_i32(chunk_size, "permutation chunk size"),
                          _checked_i32(permutations, "permutation count"),
                          _checked_i32(chunk_first, "chunk identity"), kernel_scale, base_seed,
                          stat_code, return_perms, use_global,
                          (scratch_stride, Int64(moment_block)),
                          Val(use_global ? _GPU_PRIVATE_CROSSOVER : bucket), Val(exact_tails);
                          ndrange = (nrows, chunk_count))
            KernelAbstractions.synchronize(backend)
            hu = Array(batch_upper); hl = Array(batch_lower)
            hm = Array(batch_mean); hq = Array(batch_m2); ha = Array(batch_anchor)
            hp = return_perms ? Array(batch_perms) : nothing
            for (local_row, row) in enumerate(row_batch)
                for local_chunk in 1:chunk_count
                    total_upper[row] += Int64(hu[local_row, local_chunk])
                    total_lower[row] += Int64(hl[local_row, local_chunk])
                end
                # Blocks are merged in global permutation order; only the last
                # block of the last batch can be shorter than `moment_block`.
                for block in 1:span_blocks
                    block_n = min(moment_block, span - (block - 1) * moment_block)
                    block_mean = Float64(ha[local_row, block]) + Float64(hm[local_row, block])
                    _merge_moments!(means, m2s, counts, row, block_mean,
                                    Float64(hq[local_row, block]), block_n)
                end
                if return_perms
                    for local_p in 1:span
                        retained[row, span_start + local_p - 1] = Float64(hp[local_row, local_p])
                    end
                end
            end
            row_pos = last_row + 1
        end
        row_pos = 1
        chunk_first += chunk_count
    end
    return nothing
end

function _cpu_comparison_pvalues(data::AbstractVector, W::SpatialWeights,
                                 observed::AbstractVector, permutations::Int,
                                 base_seed::UInt64, local_calc_function::Function,
                                 local_tolerance)
    n = length(data)
    upper = zeros(Int, n)
    lower = zeros(Int, n)
    Threads.@threads for i in 1:n
        k = Int(W.nneighs[i])
        k == 0 && continue
        sampled_values = Vector{eltype(data)}(undef, k)
        seen = k <= 64 ? Vector{Int}(undef, k) : nothing
        seen_set = k > 64 ? Set{Int}() : nothing
        wi = weights(W, i)
        n32 = _checked_i32(n, "number of observations")
        i32 = _checked_i32(i, "observation index")
        tol = local_tolerance === nothing ? nothing : local_tolerance[i]
        for p in 1:permutations
            state, _ = splitmix64(base_seed ⊻
                (unsafe_trunc(UInt64, i) * 0x9e3779b97f4a7c15) ⊻
                (unsafe_trunc(UInt64, p) * 0x517cc1b727220a95))
            k > 64 && empty!(seen_set)
            accepted = 0
            while accepted < k
                index, state = rand_index(state, n32, i32)
                idx = Int(index)
                duplicate = if k <= 64
                    found = false
                    @inbounds for prior in 1:accepted
                        if seen[prior] == idx
                            found = true
                            break
                        end
                    end
                    found
                else
                    idx in seen_set
                end
                duplicate && continue
                accepted += 1
                if k <= 64
                    seen[accepted] = idx
                else
                    push!(seen_set, idx)
                end
                @inbounds sampled_values[accepted] = data[idx]
            end
            value = local_calc_function(data[i], wi, sampled_values)
            upper_i, lower_i = SpatialDependence._local_tail_counts(
                (value,), observed[i]; tolerance = tol)
            upper[i] += upper_i
            lower[i] += lower_i
        end
    end
    p = (Float64.(min.(upper, lower)) .+ 1.0) ./ (permutations + 1.0)
    p[W.nneighs .== 0] .= 1.0
    return p
end

function SpatialDependence.crand_local_gpu(
    backend, stat_type::Symbol, permutations::Int, data::AbstractVector,
    W::SpatialWeights, obs_stat::AbstractVector, scale_param::Number;
    return_perms::Bool = true, seed::Union{Integer, Nothing} = nothing,
    rng::AbstractRNG = default_rng(), precision = nothing,
    comparison_data::Union{Nothing, AbstractVector} = nothing,
    comparison = nothing, local_calc_function = nothing, local_tolerance = nothing)
    permutations >= 0 || throw(ArgumentError("permutations must be nonnegative"))
    length(data) == length(obs_stat) || throw(ArgumentError("data and observed statistic lengths must match"))
    SpatialDependence._validate_local_precision(backend, precision)
    SpatialDependence._validate_local_comparison(backend, comparison)
    comparison === :cpu && !(local_calc_function isa Function) &&
        throw(ArgumentError("comparison=:cpu requires a CPU statistic closure"))
    validated_seed = SpatialDependence._validate_local_seed(seed)
    actual_backend = if backend === :gpu
        candidates = Tuple{Symbol, Any}[]
        for (vendor, constructor) in ((:Metal, :MetalBackend), (:CUDA, :CUDABackend),
                                      (:AMDGPU, :ROCBackend), (:oneAPI, :oneAPIBackend))
            for (modkey, mod) in Base.loaded_modules
                modkey.name == String(vendor) || continue
                isdefined(mod, constructor) && isdefined(mod, :functional) || continue
                functional = try mod.functional() catch; false end
                functional || continue
                if vendor == :AMDGPU
                    isdefined(mod, :has_rocm_gpu) || continue
                    (try mod.has_rocm_gpu() catch; false end) || continue
                end
                candidate = try getfield(mod, constructor)() catch; nothing end
                candidate === nothing || push!(candidates, (vendor, candidate))
            end
        end
        isempty(candidates) && throw(ArgumentError("backend=:gpu specified, but no loaded GPU backend was found. Please run `using Metal` (for Mac) or `using CUDA` (for NVIDIA) before calling."))
        length(candidates) == 1 || throw(ArgumentError("backend=:gpu is ambiguous; loaded functional GPU backends: " * join(string.(first.(candidates)), ", ")))
        last(candidates)[2]
    else
        backend
    end
    n = length(data)
    W.n == n || throw(ArgumentError("data length must match the number of observations in W"))
    _checked_i32(n, "number of observations")
    has_float64_trait = isdefined(KernelAbstractions, :supports_float64)
    backend_supports_float64 = has_float64_trait ? (try KernelAbstractions.supports_float64(actual_backend) catch; false end) : false
    precision === Float64 && !backend_supports_float64 && throw(ArgumentError("precision=Float64 is not supported by backend $(typeof(actual_backend)); use precision=Float32 or a backend with Float64 support"))
    T = precision === Float64 ? Float64 : Float32
    stat_code = stat_type == :moran ? Int32(1) : stat_type == :geary ? Int32(2) :
                stat_type == :getisord ? Int32(3) : stat_type == :getisord_star ? Int32(4) :
                throw(ArgumentError("Unknown stat_type $stat_type"))
    if permutations == 0
        Iperms = return_perms ? Matrix{Float64}(undef, n, 0) : Matrix{Float64}(undef, 0, 0)
        return Iperms, ones(Float64, n), fill(NaN, n), fill(NaN, n), fill(NaN, n)
    end
    _checked_i32(permutations, "permutation count")
    max_k = maximum(Int.(W.nneighs); init = 0)
    max_k <= n - 1 || throw(ArgumentError("accelerated permutation sampling requires at most n-1 neighbors per observation"))
    # Hoisted above payload construction so Float32 and Float64 reject the same
    # graphs, and so nothing indexes a malformed W before it has been checked.
    _validate_weights_structure(W, n)
    exact_tails = T === Float32 && comparison !== :cpu
    comparison_source = comparison_data === nothing ? data : comparison_data
    exact_payload = exact_tails ? _exact_tail_payload(comparison_source, W; stat_code=Int(stat_code)) : nothing
    if exact_tails
        any_defined = any(exact_payload.defined)
        if stat_code == Int32(1) || stat_code == Int32(2)
            any_defined && (!isfinite(Float64(scale_param)) || Float64(scale_param) <= 0.0) && throw(ArgumentError("exact accelerated tails require a finite positive Moran/Geary moment scale for nonconstant data"))
            for i in 1:n
                exact_payload.defined[i] || continue
                isfinite(Float64(obs_stat[i])) || throw(ArgumentError("exact accelerated Moran/Geary tails require finite observed scores on defined rows"))
            end
        elseif any_defined
            isfinite(Float64(scale_param)) || throw(ArgumentError("exact accelerated Getis tails require a finite denominator"))
            for i in 1:n
                exact_payload.defined[i] || continue
                isfinite(Float64(obs_stat[i])) || throw(ArgumentError("exact accelerated Getis tails require finite observed scores on defined rows"))
            end
        end
    end
    num_chunks = _GPU_CHUNK_COUNT_OVERRIDE[] > 0 ?
                 min(_GPU_CHUNK_COUNT_OVERRIDE[], permutations) :
                 min(64, cld(permutations, 64))
    # Chunks are whole moment blocks so every block is produced by one worker.
    # The block count per chunk is rounded down, never up: rounding up would
    # leave fewer chunks than the rule asked for (40 instead of 64 at
    # P = 9,999), which starves small problems of device threads.  A chunk that
    # covers the whole run needs no alignment, and clamping it keeps
    # `chunk_size` inside the device Int32 domain.
    moment_block = _moment_block(permutations)
    chunk_size = min(permutations,
                     moment_block * max(1, fld(cld(permutations, moment_block), num_chunks)))
    num_chunks = cld(permutations, chunk_size)  # drop chunks the rounding left empty
    centered_getis = stat_code == Int32(3) || stat_code == Int32(4)
    data64 = Float64.(data)
    getis_center = centered_getis ? mean(data64) : 0.0
    getis_scale = centered_getis ? max(maximum(abs.(data64 .- getis_center)), 1.0) : 1.0
    kernel_data = centered_getis ? (data64 .- getis_center) ./ getis_scale : data64
    kernel_scale = Float64(scale_param)
    if !centered_getis && isfinite(kernel_scale) && kernel_scale > 0.0
        kernel_data ./= sqrt(kernel_scale); kernel_scale = 1.0
    elseif centered_getis
        kernel_scale = 1.0
    end
    weights_gpu, neighbors_gpu, offsets_gpu, _ = prepare_gpu_weights(actual_backend, W, T, stat_code)
    data_gpu = KernelAbstractions.allocate(actual_backend, T, n)
    KernelAbstractions.copyto!(actual_backend, data_gpu, T.(kernel_data))
    exact_values_gpu, exact_weight_magnitudes_gpu, exact_weight_signs_gpu,
    exact_observed_gpu, exact_defined_gpu = if exact_tails
        v = KernelAbstractions.allocate(actual_backend, UInt32, size(exact_payload.values.values)...)
        wm = KernelAbstractions.allocate(actual_backend, UInt32, size(exact_payload.weights.magnitudes)...)
        ws = KernelAbstractions.allocate(actual_backend, Int8, length(exact_payload.weights.signs))
        o = KernelAbstractions.allocate(actual_backend, UInt32, size(exact_payload.observed)...)
        d = KernelAbstractions.allocate(actual_backend, Bool, n)
        KernelAbstractions.copyto!(actual_backend, v, exact_payload.values.values)
        KernelAbstractions.copyto!(actual_backend, wm, exact_payload.weights.magnitudes)
        KernelAbstractions.copyto!(actual_backend, ws, exact_payload.weights.signs)
        KernelAbstractions.copyto!(actual_backend, o, exact_payload.observed)
        KernelAbstractions.copyto!(actual_backend, d, collect(exact_payload.defined))
        (v, wm, ws, o, d)
    else
        (KernelAbstractions.zeros(actual_backend, UInt32, 1, 1), KernelAbstractions.zeros(actual_backend, UInt32, 2, 1),
         KernelAbstractions.zeros(actual_backend, Int8, 1), KernelAbstractions.zeros(actual_backend, UInt32, 8, 1),
         KernelAbstractions.zeros(actual_backend, Bool, 1))
    end
    compare_sign_cpu = ones(Int32, n); getis_offset = zeros(Float64, n); getis_factor = zeros(Float64, n)
    if centered_getis
        wt = wtransformation(W); total = Float64(scale_param)
        for i in 1:n
            k = Int(W.nneighs[i])
            wistar = stat_code == Int32(4) && wt == :row ? 1.0 / (k + 1) : 1.0
            weight_sum = stat_code == Int32(4) ? (k + 1) * wistar : sum(W.weights[i])
            denominator = stat_code == Int32(4) ? total : total - data64[i]
            getis_offset[i] = getis_center * weight_sum / denominator
            getis_factor[i] = getis_scale / denominator
            compare_sign_cpu[i] = getis_factor[i] < 0.0 ? Int32(-1) : Int32(1)
        end
    end
    total_upper = zeros(Int64, n); total_lower = zeros(Int64, n)
    means = zeros(Float64, n); m2s = zeros(Float64, n); counts = zeros(Int, n)
    retained = return_perms ? Matrix{Float64}(undef, n, permutations) : Matrix{Float64}(undef, 0, permutations)
    base_seed = validated_seed === nothing ? rand(rng, UInt64) : validated_seed
    rows_by_bucket = Dict{Int, Vector{Int}}()
    for i in 1:n
        bucket = _degree_bucket(Int(W.nneighs[i]))
        push!(get!(rows_by_bucket, bucket, Int[]), i)
    end
    worker_kernel = _local_perm_worker_kernel!(actual_backend)
    for bucket in sort!(collect(keys(rows_by_bucket)))
        _run_bucket!(actual_backend, worker_kernel, rows_by_bucket[bucket], bucket,
                     data_gpu, weights_gpu, neighbors_gpu, offsets_gpu,
                     exact_values_gpu, exact_weight_magnitudes_gpu, exact_weight_signs_gpu,
                     exact_observed_gpu, exact_defined_gpu, n, permutations, chunk_size,
                     num_chunks, base_seed, stat_code, return_perms, exact_tails,
                     T(kernel_scale), total_upper, total_lower, means, m2s, counts,
                     retained, _GPU_SCRATCH_BUDGET[])
    end
    p_values = (Float64.(min.(total_upper, total_lower)) .+ 1.0) ./ (permutations + 1.0)
    p_values[W.nneighs .== 0] .= 1.0
    exact_tails && (p_values[.!exact_payload.defined] .= 1.0)
    varying_mean = means; varying_std = sqrt.(max.(0.0, m2s ./ permutations))
    perms_mean = copy(varying_mean); perms_std = copy(varying_std)
    if centered_getis
        perms_mean = getis_offset .+ getis_factor .* varying_mean
        perms_std = abs.(getis_factor) .* varying_std
    end
    invalid_summary = .!isfinite.(perms_mean) .| .!isfinite.(perms_std)
    if exact_tails
        p_values[invalid_summary .& .!exact_payload.defined] .= 1.0
    else
        p_values[invalid_summary] .= 1.0
    end
    if comparison === :cpu
        p_values = _cpu_comparison_pvalues(data, W, obs_stat, permutations,
                                           base_seed, local_calc_function, local_tolerance)
    end
    if centered_getis
        observed_varying = zeros(Float64, n); wt = wtransformation(W)
        for i in 1:n
            yi = kernel_data[i]; neigh = W.neighs[i]
            if stat_code == Int32(4)
                wistar = wt == :row ? 1.0 / (length(neigh) + 1) : 1.0
                observed_varying[i] = wistar * yi + sum((wistar * kernel_data[j] for j in neigh); init=0.0)
            else
                observed_varying[i] = sum((W.weights[i][j] * kernel_data[neigh[j]] for j in eachindex(neigh)); init=0.0)
            end
        end
        zval = compare_sign_cpu .* (observed_varying .- varying_mean) ./ varying_std
        zval[varying_std .== 0.0] .= NaN
    else
        zval = (Float64.(obs_stat) .- perms_mean) ./ perms_std
        zval[perms_std .== 0.0] .= NaN
    end
    if return_perms && centered_getis
        for i in 1:n
            retained[i, :] .= getis_offset[i] .+ getis_factor[i] .* retained[i, :]
        end
    end
    return retained, p_values, perms_mean, perms_std, zval
end

end # module
