# Getis-Ord statistic
struct GetisOrd <: AbstractLocalSpatialAutocorrelation
    n::Int
    G::Vector{Float64}
    p::Vector{Float64}
    Gperms::Matrix{Float64}
    Gpermsmean::Vector{Float64}
    Gpermsstd::Vector{Float64}
    z::Vector{Float64}
    q::Vector{Symbol}
    star::Bool
end

"""
    getisord(x, W)
Compute the Getis-Ord statistic.

# Optional Arguments
- `star=true`: compute the Gi* statistic, or the Gi if set to `false`.
- `permutations=9999`: number of permutations for the randomization test.
- `rng=default_rng()`: random number generator for CPU permutations; with a backend and no explicit `seed`, one `UInt64` seed is drawn from `rng`. An explicit `seed` takes precedence.
- `backend=nothing`: execution backend (e.g. `MetalBackend()`, `CUDABackend()`, or `:gpu`); accelerated permutations default to Float32 and support up to ``n-1`` neighbors per observation, subject to available device scratch memory. Set `precision=Float64` on a backend that supports Float64; Metal currently does not.
- `precision=nothing`: accelerated arithmetic precision (`Float32` or `Float64`); an explicit precision requires `backend`. The default Float32 path uses bounded exact integer tail comparisons only on its validated input domain; Float64 p-values use floating-point comparisons. Summaries and retained draws remain backend-derived and approximate. `comparison=:cpu` replays the accelerated samples through the native CPU statistic and tolerance to replace only p-values. `nothing` preserves the default CPU behavior when no backend is supplied.
- `comparison=nothing`: optionally set to `:cpu` with a backend for full CPU replay of the accelerated samples. Replay costs CPU statistic work proportional to the total sampled degree and does not reproduce the ordinary CPU run's random samples.
- `return_perms=true`: retain the full permutation matrix; otherwise return an empty `0 × permutations` matrix and use fixed-size CPU batches.
- `seed=nothing`: optional nonnegative seed in the UInt64 range. It overrides `rng` on CPU; on a backend it overrides the backend seed draw.
"""
function getisord(x::AbstractVector{T} where T, W::SpatialWeights; permutations::Int = 9999,
    star::Bool = true, rng::AbstractRNG = default_rng(),
    backend = nothing, return_perms::Bool = true, seed::Union{Integer, Nothing} = nothing,
    precision = nothing, comparison = nothing)::GetisOrd

    _validate_local_precision(backend, precision)
    _validate_local_comparison(backend, comparison)

    wt = wtransformation(W) 
    wt == :row || wt == :binary || throw(ArgumentError("W must be row standardized or binary"))

    n = length(x)
    
    denon::Float64 = sum(x)

    # Auxiliar function to calculate Getis Ord
    function getisord_calc(xi::Number, wi::AbstractVector, xneighi::AbstractVector)::Float64
        return sum( wi .* xneighi) ./ (denon - xi)
    end

    function getisord_calc_star(xi::Number, wi::AbstractVector, xneighi::AbstractVector)::Float64
        if wt == :row
            wistar = 1 / (length(wi) + 1)
        elseif wt == :binary
            wistar = 1
        end
        return (wistar .* xi + sum(wistar .* xneighi)) ./ (denon)
    end

    if star
        getisord_calc_fun = getisord_calc_star
    else
        getisord_calc_fun = getisord_calc
    end

    local_tolerance = zeros(Float64, n)
    if isfinite(denon)
        maxabsx = maximum(abs, x; init = 0.0)
        for i in 1:n
            wi = weights(W, i)
            denominator = star ? denon : denon - x[i]
            weight_sum = star ? ((wt == :row ? 1.0 / (W.nneighs[i] + 1) : 1.0) *
                                 (W.nneighs[i] + 1)) : sum(abs, wi)
            local_tolerance[i] = denominator == 0.0 ? 0.0 :
                8 * eps(Float64) * (W.nneighs[i] + 2) *
                weight_sum * maxabsx / abs(denominator)
        end
    end
    
    # Getis-Ord
    G = zeros(n)
    for i in 1:n
        xi = x[i]
        wi = weights(W, i)
        xneighi = x[neighbors(W, i)]
        G[i] = getisord_calc_fun(xi, wi, xneighi) 
    end

    # Conditional randomization
    if backend !== nothing
        stat_sym = star ? :getisord_star : :getisord
        Gperms, p, Gpermsmean, Gpermsstd, zval = crand_local_gpu(
            backend, stat_sym, permutations, x, W, G, denon;
            return_perms=return_perms, seed=seed, rng=rng, precision=precision,
            comparison=comparison, local_calc_function=getisord_calc_fun,
            local_tolerance=local_tolerance, comparison_data=x
        )
    else
        local_rng = _local_rng(rng, seed)
        if return_perms
            Gperms = crand_local(permutations, x, W, getisord_calc_fun, local_rng)
            p, Gpermsmean, Gpermsstd, zval = _local_perm_summary(
                Gperms, G, permutations; islands = W.nneighs .== 0,
                tolerances = local_tolerance)
        else
            p, Gpermsmean, Gpermsstd, zval = _local_stream_summary(
                permutations, x, W, getisord_calc_fun, G, local_rng;
                tolerances = local_tolerance)
            Gperms = Matrix{Float64}(undef, 0, permutations)
        end
    end

    # Classification
    q = Array{Symbol}(undef, n)
    for i in 1:n
        if zval[i] > 0
            q[i] = :H
        else
            q[i] = :L
        end
    end

    return GetisOrd(n, G, p, Gperms, Gpermsmean, Gpermsstd, zval, q, star)

end

score(x::GetisOrd) = x.G;

scoreperms(x::GetisOrd) = x.Gperms;

mean(x::GetisOrd) = x.Gpermsmean;

std(x::GetisOrd) = x.Gpermsstd;

zscore(x::GetisOrd) = x.z;

pvalue(x::GetisOrd) = x.p;

assignments(x::GetisOrd) = x.q;

testname(x::GetisOrd) = x.star ? "Getis-Ord Gi*" : "Getis-Ord Gi";

function labelsorder(::GetisOrd) 
    return [:ns, :H, :L]
end

function labelsnames(::GetisOrd) 
    return ["Not Significant"; "High"; "Low"]
end

function labelcolor(::GetisOrd, x::String)

    if x == "H"
        catcolor = :red
        alpha = 1.0
    elseif x == "L"
        catcolor =:blue
        alpha = 1.0
    # not significant
    elseif x == "ns"
        catcolor = :lightgrey
        alpha = 0.4
    end

    return (catcolor, alpha)
end
