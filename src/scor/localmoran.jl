# Local Moran test of Spatial Autocorrelation
struct LocalMoran <: AbstractLocalSpatialAutocorrelation
    n::Int
    I::Vector{Float64}
    p::Vector{Float64}
    Iperms::Matrix{Float64}
    Ipermsmean::Vector{Float64}
    Ipermsstd::Vector{Float64}
    z::Vector{Float64}
    q::Vector{Symbol}
end

"""
    localmoran(x, W)
Compute the Local Moran test of spatial autocorrelation.

# Optional Arguments
- `permutations=9999`: number of permutations for the randomization test.
- `rng=default_rng()`: random number generator for CPU permutations; with a backend and no explicit `seed`, one `UInt64` seed is drawn from `rng`. An explicit `seed` takes precedence.
- `corrected=true`: divide the scaling factor by ``n-1`` instead of ``n``.
- `backend=nothing`: execution backend (e.g. `MetalBackend()`, `CUDABackend()`, or `:gpu`); accelerated permutations default to Float32 and support up to ``n-1`` neighbors per observation, subject to available device scratch memory. Set `precision=Float64` on a backend that supports Float64; Metal currently does not.
- `precision=nothing`: accelerated arithmetic precision (`Float32` or `Float64`); an explicit precision requires `backend`. The default Float32 path uses bounded exact integer tail comparisons only on its validated input domain; Float64 p-values use floating-point comparisons. Summaries and retained draws remain backend-derived and approximate (and can be nonfinite with Float32). `comparison=:cpu` replays the accelerated samples through the native CPU statistic and tolerance to replace only p-values. Explicit Float64 also uses shift-first host centering for large offsets; with `comparison=:cpu` it uses ordinary CPU centering instead, so Moran scores and summaries can differ from the default Float64 path. `nothing` preserves the default CPU behavior when no backend is supplied.
- `comparison=nothing`: optionally set to `:cpu` with a backend for full CPU replay of the accelerated samples. The replay redraws each observation's `k` distinct neighbors per permutation on the CPU by rejection sampling (a linear duplicate scan up to `k = 64`, quadratic in `k`, and a hash set above that, with more redraws as `k` approaches ``n-1``), so it can add substantial CPU work, and it does not reproduce the ordinary CPU run's random samples.
- `return_perms=true`: retain the full permutation matrix; otherwise return an empty `0 × permutations` matrix and use fixed-size CPU batches.
- `seed=nothing`: optional nonnegative seed in the UInt64 range. It overrides `rng` on CPU; on a backend it overrides the backend seed draw.
"""
function localmoran(x::AbstractVector{T} where T, W::SpatialWeights; permutations::Int = 9999,
    corrected::Bool = true, rng::AbstractRNG = default_rng(),
    backend = nothing, return_perms::Bool = true, seed::Union{Integer, Nothing} = nothing,
    precision = nothing, comparison = nothing)::LocalMoran

    _validate_local_precision(backend, precision)
    _validate_local_comparison(backend, comparison)

    n = length(x)
    z = if backend !== nothing && precision === Float64 && comparison !== :cpu && !isempty(x)
        # Shift before centering so a large location offset does not consume
        # Float64 mantissa bits before the small spatial signal is formed.
        x64 = Float64.(x)
        shifted = x64 .- x64[1]
        shifted .-= mean(shifted)
        shifted
    else
        x .- mean(x)
    end
    Wz = slag(W, z)
    
    m2::Float64 = sum(z.^2)
    if corrected
        m2 = m2 ./ (n - 1)
    else
        m2 = m2 ./ n
    end

    # Auxiliar function to calculate Local Moran
    function localmoran_calc(zi::Number, wi::AbstractVector, zneighi::AbstractVector)::Float64
        return (zi / m2) .* sum(wi .* zneighi)
    end

    # Bound summation roundoff independently of the sampled permutation. This
    # lets both retained and streaming paths recognize the same mathematical
    # ties, including cancellations near zero, without an absolute tolerance.
    local_tolerance = zeros(Float64, n)
    if isfinite(m2) && m2 != 0.0
        maxabsz = maximum(abs, z; init = 0.0)
        for i in 1:n
            local_tolerance[i] = 8 * eps(Float64) * (W.nneighs[i] + 2) *
                abs(z[i] / m2) * sum(abs, weights(W, i)) * maxabsz
        end
    end
    
    # Local Moran
    # I = (z / m2) .* Wz
    I = zeros(n)
    for i in 1:n
        I[i] = localmoran_calc(z[i], weights(W, i), z[neighbors(W, i)]) 
    end

    # Conditional randomization
    if backend !== nothing
        Iperms, p, Ipermsmean, Ipermsstd, zval = crand_local_gpu(
            backend, :moran, permutations, z, W, I, m2;
            return_perms=return_perms, seed=seed, rng=rng, precision=precision,
            comparison=comparison, local_calc_function=localmoran_calc,
            local_tolerance=local_tolerance, comparison_data=x
        )
    else
        local_rng = _local_rng(rng, seed)
        if return_perms
            Iperms = crand_local(permutations, z, W, localmoran_calc, local_rng)
            p, Ipermsmean, Ipermsstd, zval = _local_perm_summary(
                Iperms, I, permutations; islands = W.nneighs .== 0,
                tolerances = local_tolerance)
        else
            p, Ipermsmean, Ipermsstd, zval = _local_stream_summary(
                permutations, z, W, localmoran_calc, I, local_rng;
                tolerances = local_tolerance)
            Iperms = Matrix{Float64}(undef, 0, permutations)
        end
    end

    # Classification
    q = Array{Symbol}(undef, n)
    for i in 1:n
        if z[i] > 0
            if Wz[i] > 0
                q[i] = :HH
            else
                q[i] = :HL
            end
        else
            if Wz[i] > 0
                q[i] = :LH
            else
                q[i] = :LL
            end
        end
    end

    return LocalMoran(n, I, p, Iperms, Ipermsmean, Ipermsstd, zval, q)

end

score(x::LocalMoran) = x.I;

scoreperms(x::LocalMoran) = x.Iperms;

mean(x::LocalMoran) = x.Ipermsmean;

std(x::LocalMoran) = x.Ipermsstd;

zscore(x::LocalMoran) = x.z;

pvalue(x::LocalMoran) = x.p;

assignments(x::LocalMoran) = x.q;

testname(::LocalMoran) = "Local Moran";

labelsorder(::LocalMoran) = [:ns; :HH; :LL; :LH; :HL];

labelsnames(::LocalMoran) = ["Not Significant"; "High-High"; "Low-Low"; "Low-High"; "High-Low"]

function labelcolor(::LocalMoran, x::String)

    if x == "HH"
        catcolor = :red
        alpha = 1
    elseif x == "LL"
        catcolor = :blue
        alpha = 1
    elseif x == "LH"
        catcolor = :blue
        alpha = 0.4
    elseif x == "HL"
        catcolor = :red
        alpha = 0.4 
    elseif x == "ns"
        catcolor = :lightgrey
        alpha = 0.4
    end

    return (catcolor, alpha)
end
