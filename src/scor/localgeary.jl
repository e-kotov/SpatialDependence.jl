# Local Geary test of Spatial Autocorrelation
struct LocalGeary <: AbstractLocalSpatialAutocorrelation
    n::Int
    C::Vector{Float64}
    p::Vector{Float64}
    Cperms::Matrix{Float64}
    Cpermsmean::Vector{Float64}
    Cpermsstd::Vector{Float64}
    z::Vector{Float64}
    q::Vector{Symbol}
    categories::Symbol
end

"""
    localgeary(x, W)
Compute the Local Geary test of spatial autocorrelation.

# Optional Arguments
- `permutations=9999`: number of permutations for the randomization test.
- `rng=default_rng()`: random number generator for CPU permutations; with a backend and no explicit `seed`, one `UInt64` seed is drawn from `rng`. An explicit `seed` takes precedence.
- `corrected=true`: divide the scaling factor by ``n-1`` instead of ``n``.
- `categories=:positivenegative`: assing observations to positive or negative spatial autocorrelation, or in combination with the `:moran` scatterplot.
- `backend=nothing`: execution backend (e.g. `MetalBackend()`, `CUDABackend()`, or `:gpu`); accelerated permutations default to Float32 and support up to ``n-1`` neighbors per observation, subject to available device scratch memory. Set `precision=Float64` on a backend that supports Float64; Metal currently does not.
- `precision=nothing`: accelerated arithmetic precision (`Float32` or `Float64`); an explicit precision requires `backend`. The default Float32 path uses bounded exact integer tail comparisons only on its validated input domain; Float64 p-values use floating-point comparisons. Summaries and retained draws remain backend-derived and approximate (and can be nonfinite with Float32). `comparison=:cpu` replays the accelerated samples through the native CPU statistic and tolerance to replace only p-values. Explicit Float64 also uses shift-first host centering for large offsets; with `comparison=:cpu` it uses ordinary CPU centering instead, so Geary scores and summaries can differ from the default Float64 path. `nothing` preserves the default CPU behavior when no backend is supplied.
- `comparison=nothing`: optionally set to `:cpu` with a backend for full CPU replay of the accelerated samples. The replay redraws each observation's `k` distinct neighbors per permutation on the CPU by rejection sampling (a linear duplicate scan up to `k = 64`, quadratic in `k`, and a hash set above that, with more redraws as `k` approaches ``n-1``), so it can add substantial CPU work, and it does not reproduce the ordinary CPU run's random samples.
- `return_perms=true`: retain the full permutation matrix; otherwise return an empty `0 × permutations` matrix and use fixed-size CPU batches.
- `seed=nothing`: optional nonnegative seed in the UInt64 range. It overrides `rng` on CPU; on a backend it overrides the backend seed draw.
"""
function localgeary(x::AbstractVector{T} where T, W::SpatialWeights; permutations::Int = 9999,
    corrected::Bool = true, categories::Symbol = :positivenegative,
    rng::AbstractRNG = default_rng(),
    backend = nothing, return_perms::Bool = true, seed::Union{Integer, Nothing} = nothing,
    precision = nothing, comparison = nothing)::LocalGeary

    _validate_local_precision(backend, precision)
    _validate_local_comparison(backend, comparison)

    (categories == :positivenegative) || (categories == :moran) || throw(ArgumentError("`categories` must be :positivenegative or :moran"))

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
    
    m2::Float64 = sum(z.^2) 
    if corrected
        m2 = m2 ./ (n - 1)
    else
        m2 = m2 ./ n
    end

    # Auxiliar function to calculate Local Moran
    function localgeary_calc(zi::Number, wi::AbstractVector, zneighi::AbstractVector)::Float64
        return (1 ./ m2) .* sum(wi .* (zi .- zneighi).^2 )
    end

    local_tolerance = zeros(Float64, n)
    if isfinite(m2) && m2 != 0.0
        maxabsz = maximum(abs, z; init = 0.0)
        for i in 1:n
            local_tolerance[i] = 8 * eps(Float64) * (W.nneighs[i] + 2) *
                abs(1 / m2) * sum(abs, weights(W, i)) * (abs(z[i]) + maxabsz)^2
        end
    end
    
    # Local Geary
    C = zeros(n)
    for i in 1:n
        zi = z[i]
        wi = weights(W, i)
        zneighi = z[neighbors(W, i)]
        C[i] = localgeary_calc(zi, wi, zneighi) 
    end

    # Conditional randomization
    if backend !== nothing
        Cperms, p, Cpermsmean, Cpermsstd, zval = crand_local_gpu(
            backend, :geary, permutations, z, W, C, m2;
            return_perms=return_perms, seed=seed, rng=rng, precision=precision,
            comparison=comparison, local_calc_function=localgeary_calc,
            local_tolerance=local_tolerance, comparison_data=x
        )
    else
        local_rng = _local_rng(rng, seed)
        if return_perms
            Cperms = crand_local(permutations, z, W, localgeary_calc, local_rng)
            p, Cpermsmean, Cpermsstd, zval = _local_perm_summary(
                Cperms, C, permutations; islands = W.nneighs .== 0,
                tolerances = local_tolerance)
        else
            p, Cpermsmean, Cpermsstd, zval = _local_stream_summary(
                permutations, z, W, localgeary_calc, C, local_rng;
                tolerances = local_tolerance)
            Cperms = Matrix{Float64}(undef, 0, permutations)
        end
    end

    # Classification
    Cmean = mean(C)
    Wz = slag(W, z)

    q = Array{Symbol}(undef, n)
    if categories == :positivenegative
        for i in 1:n
            if zval[i] < 0
                q[i] = :P
            else
                q[i] = :N
            end
        end
    elseif categories == :moran
        for i in 1:n
            if zval[i] < 0
                if (z[i] > 0) & (Wz[i] > 0)
                    q[i] = :HH
                elseif (z[i] < 0) & (Wz[i] < 0)
                    q[i] = :LL
                else
                    q[i] = :OP
                end
            else
                q[i] = :NE
            end
        end
    end

    return LocalGeary(n, C, p, Cperms, Cpermsmean, Cpermsstd, zval, q, categories)

end

score(x::LocalGeary) = x.C;

scoreperms(x::LocalGeary) = x.Cperms;

mean(x::LocalGeary) = x.Cpermsmean;

std(x::LocalGeary) = x.Cpermsstd;

zscore(x::LocalGeary) = x.z;

pvalue(x::LocalGeary) = x.p;

assignments(x::LocalGeary) = x.q;

testname(::LocalGeary) = "Local Geary";

function labelsorder(x::LocalGeary) 
    if x.categories == :positivenegative
        return [:ns, :P, :N]
    elseif x.categories == :moran
        return [:ns; :HH; :LL; :OP; :NE];
    end
end

function labelsnames(x::LocalGeary) 
    if x.categories == :positivenegative
        return ["Not Significant"; "Positive"; "Negative"]
    elseif x.categories == :moran
        return ["Not Significant"; "High-High"; "Low-Low"; "Other Positive"; "Negative"]
    end
end

function labelcolor(::LocalGeary, x::String)

    # :positivenegative categories
    if x == "P"
        catcolor = :red
        alpha = 1.0
    elseif x == "N"
        catcolor =:blue
        alpha = 1.0
    # :moran categories
    elseif x == "HH"
        catcolor = "#B2182B" # (178, 24, 43)
        alpha = 1.0
    elseif x == "LL"
        catcolor = "#EF8A62" # (239, 138, 98)
        alpha = 1.0
    elseif x == "OP"
        catcolor = "#FDDBC7" # (253, 219, 199)
        alpha = 1.0
    elseif x == "NE"
        catcolor = "#67ADC7" # (103, 173, 199)
        alpha = 1.0
    # not significant
    elseif x == "ns"
        catcolor = :lightgrey
        alpha = 0.4
    end

    return (catcolor, alpha)
end
