# Abstract type for Local Spatial Autocorrelation
abstract type AbstractLocalSpatialAutocorrelation end

# Function hook for GPU / KernelAbstractions extension
crand_local_gpu(args...; kwargs...) =
    throw(ArgumentError("load KernelAbstractions and the desired backend package before requesting accelerated permutations"))

const _LOCAL_PERM_BATCH_SIZE = 256

# Explicit arithmetic precision is an accelerated-backend option.  Keep the
# default (`nothing`) on the historical CPU path, while rejecting requests
# that could otherwise be silently ignored by that path.
function _validate_local_precision(backend, precision)
    precision === nothing && return nothing
    (precision === Float32 || precision === Float64) ||
        throw(ArgumentError("precision must be nothing, Float32, or Float64"))
    backend === nothing &&
        throw(ArgumentError("precision is only available when a backend is specified"))
    return precision
end

function _validate_local_comparison(backend, comparison)
    comparison === nothing || comparison === :cpu ||
        throw(ArgumentError("comparison must be nothing or :cpu"))
    comparison === :cpu && backend === nothing &&
        throw(ArgumentError("comparison=:cpu requires an accelerated backend"))
    return comparison
end

function _validate_local_seed(seed::Union{Integer, Nothing})
    seed === nothing && return nothing
    seed isa Integer || throw(ArgumentError("seed must be a nonnegative integer in the UInt64 range"))
    seed < 0 && throw(ArgumentError("seed must be nonnegative and fit in UInt64"))
    BigInt(seed) <= BigInt(typemax(UInt64)) ||
        throw(ArgumentError("seed must be nonnegative and fit in UInt64"))
    return UInt64(seed)
end

# An explicit seed takes precedence over rng for CPU permutations. With no
# seed, the caller's RNG is used unchanged, preserving the historical draw
# order and reproducibility of StableRNG/MersenneTwister inputs.
function _local_rng(rng::AbstractRNG, seed::Union{Integer, Nothing})
    validated = _validate_local_seed(seed)
    validated === nothing ? rng : MersenneTwister(validated)
end

# Function to build the conditional randomization sample and calculate the local scores
function crand_local(permutations::Int, z::AbstractVector{T} where T, W::SpatialWeights, local_calc_function::Function, rng::AbstractRNG)::Matrix{Float64}
    # Build conditional permutations array
    ni = cardinalities(W)
    n = length(ni)
    maxni = maximum(ni)
    Cperms = zeros(Int, permutations, maxni)
    samplevec = 1:n-1
    for i in 1:permutations
        Cperms[i,:] = sample(rng, samplevec, maxni, replace = false)
    end
    
    # Calculate LISA for all the permutations
    lisaperms = zeros(n, permutations)

    Threads.@threads for i in 1:n
        bnoi = ones(Bool, n)
        bnoi[i] = false

        zi = z[i]
        znoi = z[bnoi]

        nni = ni[i]
        wi = weights(W, i)
        
        for p in 1:permutations
            zcrand = view(znoi, Cperms[p, 1:nni])
            lisaperms[i, p] = local_calc_function(zi, wi, zcrand)      
        end
    end

    return lisaperms
end

# Summarize conditional permutation values with the same tie and degenerate-case
# conventions used by the GPU implementation. In particular, an empty
# permutation sample has no defined moments, while a non-empty constant sample
# has a zero standard deviation and an undefined (NaN) z-score.
@inline function _local_tail_counts(values, observed; tolerance = nothing)
    upper = 0
    lower = 0
    for value in values
        # Undefined statistics are not evidence against the null. Count them
        # conservatively in both tails rather than producing 1/(P+1).
        if !isfinite(observed) || !isfinite(value)
            upper += 1
            lower += 1
            continue
        end
        bound = max(abs(Float64(observed)), abs(Float64(value)))
        # A few ulps cover different summation orders for mathematically tied
        # Float64 statistics without imposing an absolute tolerance on tiny Gi
        # values. Ties are included in both tails by construction.
        tol = tolerance === nothing ? 8 * eps(Float64) * bound : tolerance
        upper += value >= observed - tol
        lower += value <= observed + tol
    end
    return upper, lower
end

function _local_perm_summary(perms::AbstractMatrix, observed::AbstractVector, permutations::Int;
                            islands = nothing, tolerances = nothing)
    n = length(observed)
    if permutations == 0
        return ones(Float64, n), fill(NaN, n), fill(NaN, n), fill(NaN, n)
    end

    upper = zeros(Int, n)
    lower = zeros(Int, n)
    for i in 1:n
        tol = tolerances === nothing ? nothing : tolerances[i]
        upper[i], lower[i] = _local_tail_counts(view(perms, i, :), observed[i]; tolerance = tol)
    end
    p = (min.(upper, lower) .+ 1.0) ./ (permutations + 1.0)
    islands !== nothing && (p[islands] .= 1.0)

    # Use first-draw-centered Welford moments so retained and streaming paths
    # preserve tiny real variation without cancellation from a large offset.
    permsmean = zeros(Float64, n)
    permsstd = zeros(Float64, n)
    anchors = zeros(Float64, n)
    for i in axes(perms, 1)
        row = view(perms, i, :)
        anchor = row[1]
        anchors[i] = anchor
        mean_centered = 0.0
        m2 = 0.0
        for (count_i, value) in enumerate(row)
            centered = value - anchor
            delta = centered - mean_centered
            mean_centered += delta / count_i
            m2 += delta * (centered - mean_centered)
        end
        permsmean[i] = mean_centered
        permsstd[i] = sqrt(max(0.0, m2 / permutations))
        if isfinite(anchor) && all(==(anchor), row)
            permsmean[i] = 0.0
            permsstd[i] = 0.0
        end
    end
    zval = similar(permsmean)
    for i in 1:n
        zval[i] = ((observed[i] - anchors[i]) - (permsmean[i])) / permsstd[i]
    end
    permsmean .+= anchors
    # A z-score is undefined for a degenerate permutation distribution. Use a
    # single representation across CPU and GPU rather than exposing backend
    # rounding differences as signed infinities.
    zval[permsstd .== 0.0] .= NaN
    return p, permsmean, permsstd, zval
end

function _local_stream_summary(permutations::Int, z::AbstractVector{T} where T,
                               W::SpatialWeights, local_calc_function::Function,
                               observed::AbstractVector, rng::AbstractRNG;
                               tolerances = nothing)
    # Generate batches and reduce each batch immediately. The reducer is kept
    # here rather than exposing temporary matrices to callers.
    n = length(z)
    if permutations == 0
        return ones(Float64, n), fill(NaN, n), fill(NaN, n), fill(NaN, n)
    end
    ni = cardinalities(W)
    maxni = maximum(ni)
    batch = min(_LOCAL_PERM_BATCH_SIZE, permutations)
    Cperms = zeros(Int, batch, maxni)
    lisaperms = zeros(Float64, n, batch)
    samplevec = 1:n-1
    upper = zeros(Int, n)
    lower = zeros(Int, n)
    nseen = zeros(Int, n)
    anchors = zeros(Float64, n)
    permsmean = zeros(Float64, n)
    permsm2 = zeros(Float64, n)
    permsmin = fill(Inf, n)
    permsmax = fill(-Inf, n)

    for first_perm in 1:_LOCAL_PERM_BATCH_SIZE:permutations
        last_perm = min(permutations, first_perm + batch - 1)
        nb = last_perm - first_perm + 1
        for p in 1:nb
            Cperms[p, :] = sample(rng, samplevec, maxni, replace = false)
        end
        Threads.@threads for i in 1:n
            zi = z[i]
            nni = ni[i]
            wi = weights(W, i)
            sampled_values = Vector{eltype(z)}(undef, maxni)
            for p in 1:nb
                # Cperms indexes the compact vector with the focal entry
                # removed. Map back to the original vector directly, avoiding
                # a fresh O(n) znoi copy for every observation and batch.
                for j in 1:nni
                    compact_index = Cperms[p, j]
                    original_index = compact_index >= i ? compact_index + 1 : compact_index
                    sampled_values[j] = z[original_index]
                end
                lisaperms[i, p] = local_calc_function(zi, wi,
                    view(sampled_values, 1:nni))
            end
        end
        for i in 1:n
            for p in 1:nb
                value = lisaperms[i, p]
                tol = tolerances === nothing ? nothing : tolerances[i]
                upper_i, lower_i = _local_tail_counts((value,), observed[i]; tolerance = tol)
                upper[i] += upper_i
                lower[i] += lower_i
                nseen[i] += 1
                count_i = nseen[i]
                if count_i == 1
                    anchors[i] = value
                end
                centered = value - anchors[i]
                delta = centered - permsmean[i]
                permsmean[i] += delta / count_i
                permsm2[i] += delta * (centered - permsmean[i])
                permsmin[i] = min(permsmin[i], value)
                permsmax[i] = max(permsmax[i], value)
            end
        end
    end

    p = (min.(upper, lower) .+ 1.0) ./ (permutations + 1.0)
    p[ni .== 0] .= 1.0
    permsstd = sqrt.(max.(0.0, permsm2 ./ permutations))
    for i in 1:n
        if isfinite(anchors[i]) && isfinite(permsmin[i]) && isfinite(permsmax[i]) &&
           permsmax[i] == permsmin[i]
            permsmean[i] = 0.0
            permsstd[i] = 0.0
        end
    end
    zval = similar(permsmean)
    for i in 1:n
        zval[i] = ((observed[i] - anchors[i]) - permsmean[i]) / permsstd[i]
    end
    permsmean .+= anchors
    zval[permsstd .== 0.0] .= NaN
    return p, permsmean, permsstd, zval
end

"""
    issignificant(x, α, adjust = :none)
Return a vector of boolean values indicating if the local statistics are significant or not at the desired threshold ``α``.

p-values can be adjusted with the `adjust` parameter using the Bonferroni correction `:bonferroni` or controlling for the False Discovery Rate, `:fdr`
"""
function issignificant(x::AbstractLocalSpatialAutocorrelation, α::Float64; adjust::Symbol = :none)::Vector{Bool}
    p = pvalue(x)
    n = length(p)

    (adjust == :none || adjust == :bonferroni || adjust == :fdr) ||  
        throw(ArgumentError("unknown p-value adjustment $(adjust)"))

    if adjust == :none
        return p .< α
    elseif adjust == :bonferroni
        return p .< (α / n)
    elseif adjust == :fdr
        psort = sort(p)
        pfdr = (1:n) .* α ./ n
        threshold = findlast(psort .<= pfdr)
        threshold === nothing && return falses(n)
        return p .<= pfdr[threshold]
    end
end

"""
    assignments(x, α, adjust = :none)
Return a vector with the categories assigned by the local statistic with a significance threshold ``α``.

p-values can be adjusted with the `adjust` parameter using the Bonferroni correction `:bonferroni` or controlling for the False Discovery Rate, `:fdr`
"""
function assignments(x::AbstractLocalSpatialAutocorrelation, α::Float64; adjust::Symbol = :none)::Vector{Symbol}
    q = deepcopy(assignments(x))
    p = pvalue(x)
    
    q[.! issignificant(x, α, adjust = adjust)] .= :ns

    return q
end

function Base.show(io::IO, x::AbstractLocalSpatialAutocorrelation)

    println(io, testname(x) * " test of Spatial Autocorrelation")
    println(io, "--------------------------------------------")
    println(io, "")

    permutations = size(scoreperms(x), 2)
    println(io, "Randomization test with ", permutations, " permutations.")

    if permutations > 0       
        # Display interesting locations
        println(io, "Interesting locations at 0.05 significance level:")

        q = assignments(x)
        p = pvalue(x)

        labels = labelsorder(x)
        nl = length(labels)
        countcat = zeros(Int, nl)
        for i in 1:nl
            issig = issignificant(x, 0.05, adjust = :none)
            if labels[i] == :ns
                countcat[i] = count(.! issig)
            else
                countcat[i] = count(issig .& (q .== labels[i]))
            end
        end

        labelsstr = labelsnames(x)
        labelmaxlength = maximum(length.(labelsstr))
        for i in 1:nl
            if labels[i] != :ns
                println(io, " ", lpad(labelsstr[i], labelmaxlength), ": ", countcat[i])
            end
        end
    else
        println(io, "Interesting locations cannot be identified with 0 permutations.")
    end

end
