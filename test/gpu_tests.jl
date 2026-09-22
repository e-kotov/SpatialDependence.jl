# KernelAbstractions permutation tests. CPU() is always exercised; vendor
# backends are optional and run only when their runtime reports functionality.
using Test
using SpatialDependence
using KernelAbstractions
using Statistics
using StableRNGs
using Random: rand, seed!

const GPU_EXTENSION = Base.get_extension(SpatialDependence,
                                         :SpatialDependenceKernelAbstractionsExt)

include(joinpath(@__DIR__, "gpu_backend.jl"))

const METAL_BACKEND = let
    candidate = nothing
    try
        import Metal
        if Metal.functional()
            candidate = Metal.MetalBackend()
        end
    catch
    end
    candidate
end

function supports_float64(backend)
    isdefined(KernelAbstractions, :supports_float64) || return false
    try
        KernelAbstractions.supports_float64(backend)
    catch
        false
    end
end

function degree_weights(n::Int, k::Int)
    neighs = Vector{Vector{Int64}}(undef, n)
    w = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        if i == 1
            neighs[i] = Int64.(2:(k + 1))
            w[i] = fill(1.0 / k, k)
        else
            j = i == n ? 1 : i + 1
            neighs[i] = Int64[j]
            w[i] = [1.0]
        end
    end
    SpatialWeights(Int64(n), neighs, w, Int64.(length.(neighs)), :row)
end

function mixed_degree_weights(n::Int, k::Int)
    neighs = Vector{Vector{Int64}}(undef, n)
    w = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        if i == 1
            neighs[i] = Int64.(2:(k + 1))
            raw = Float64.(1:k)
            w[i] = raw ./ sum(raw)
        else
            j = i == n ? 1 : i + 1
            neighs[i] = Int64[j]
            w[i] = [1.0]
        end
    end
    SpatialWeights(Int64(n), neighs, w, Int64.(length.(neighs)), :row)
end

function assert_summary_agrees(result)
    perms = scoreperms(result)
    @test size(perms, 1) == length(score(result))
    m = vec(Statistics.mean(perms, dims = 2))
    s = vec(Statistics.std(perms, dims = 2, corrected = false))
    # GPU values are calculated in Float32; host recomputation from the
    # returned Float64 view is therefore expected to differ by a few ulps.
    @test mean(result) ≈ m atol = 1e-7 rtol = 1e-5
    @test std(result) ≈ s atol = 1e-12 rtol = 1e-5
    expected_z = (score(result) .- m) ./ s
    expected_z[s .== 0.0] .= NaN
    @test all((isnan.(zscore(result)) .& isnan.(expected_z)) .|
              isapprox.(zscore(result), expected_z; atol = 1e-6, rtol = 1e-5))
end

function assert_summary_equal(a, b)
    @test pvalue(a) ≈ pvalue(b)
    @test mean(a) ≈ mean(b)
    @test std(a) ≈ std(b)
    @test all((isnan.(zscore(a)) .& isnan.(zscore(b))) .|
              isapprox.(zscore(a), zscore(b); atol = 1e-12, rtol = 1e-12))
end

# Independent Float64 replay of the device's SplitMix sampler.  This is
# intentionally kept separate from the implementation so a self-consistent
# kernel cannot make the test pass by reproducing its own arithmetic mistake.
function replay_splitmix(state::UInt64)
    state += 0x9e3779b97f4a7c15
    z = state
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return (z ⊻ (z >> 31)), state
end

function replay_index(state::UInt64, n::Int, exclude::Int)
    value, state = replay_splitmix(state)
    index = Int(value % UInt64(n - 1)) + 1
    index >= exclude && (index += 1)
    return index, state
end

function replay_oracle(stat::Symbol, x::Vector{Float64}, W::SpatialWeights,
                       permutations::Int, seed::Integer)
    n = length(x)
    z = x .- Statistics.mean(x)
    m2 = sum(z .^ 2) / (n - 1)
    wt = wtransformation(W)
    denominator = sum(x)
    observed = zeros(Float64, n)
    observed_bound = zeros(Float64, n)
    for i in 1:n
        neigh = neighbors(W, i)
        wi = weights(W, i)
        if stat == :moran
            observed[i] = (z[i] / m2) * sum(wi .* z[neigh])
            observed_bound[i] = abs(z[i] / m2) * sum(abs.(wi .* z[neigh]))
        elseif stat == :geary
            observed[i] = sum(wi .* (z[i] .- z[neigh]) .^ 2) / m2
            observed_bound[i] = abs(1 / m2) * sum(abs.(wi .* (z[i] .- z[neigh]) .^ 2))
        elseif stat == :getisord
            observed[i] = sum(wi .* x[neigh]) / (denominator - x[i])
            observed_bound[i] = sum(abs.(wi .* x[neigh])) / abs(denominator - x[i])
        else
            wistar = wt == :row ? 1 / (length(neigh) + 1) : 1.0
            observed[i] = (wistar * x[i] + sum(wistar .* x[neigh])) / denominator
            observed_bound[i] = (abs(wistar * x[i]) + sum(abs.(wistar .* x[neigh]))) / abs(denominator)
        end
    end

    draws = zeros(Float64, n, permutations)
    draw_bound = zeros(Float64, n, permutations)
    upper = zeros(Int, n)
    lower = zeros(Int, n)
    upper_strict = zeros(Int, n)
    lower_strict = zeros(Int, n)
    for i in 1:n
        k = W.nneighs[i]
        neigh = neighbors(W, i)
        wi = weights(W, i)
        wistar = wt == :row ? 1 / (k + 1) : 1.0
        # Streams are keyed by (observation, permutation); the replay never
        # needs to know how the device chunks its permutations.
        row_key = UInt64(seed) ⊻ (UInt64(i) * 0x9e3779b97f4a7c15)
        for p in 1:permutations
            state, _ = replay_splitmix(row_key ⊻ (UInt64(p) * 0x517cc1b727220a95))
            chosen = Int[]
            while length(chosen) < k
                index, state = replay_index(state, n, i)
                index in chosen || push!(chosen, index)
            end
            if stat == :moran
                draws[i, p] = (z[i] / m2) * sum(wi[j] * z[chosen[j]] for j in 1:k)
                draw_bound[i, p] = abs(z[i] / m2) * sum(abs(wi[j] * z[chosen[j]]) for j in 1:k)
            elseif stat == :geary
                draws[i, p] = sum(wi[j] * (z[i] - z[chosen[j]])^2 for j in 1:k) / m2
                draw_bound[i, p] = abs(1 / m2) * sum(abs(wi[j] * (z[i] - z[chosen[j]])^2) for j in 1:k)
            elseif stat == :getisord
                draws[i, p] = sum(wi[j] * x[chosen[j]] for j in 1:k) / (denominator - x[i])
                draw_bound[i, p] = sum(abs(wi[j] * x[chosen[j]]) for j in 1:k) / abs(denominator - x[i])
            else
                draws[i, p] = (wistar * x[i] + sum(wistar * x[chosen[j]] for j in 1:k)) / denominator
                draw_bound[i, p] = (abs(wistar * x[i]) + sum(abs(wistar * x[chosen[j]]) for j in 1:k)) / abs(denominator)
            end
            tol = 128eps(Float64) * (draw_bound[i, p] + observed_bound[i])
            upper[i] += draws[i, p] + tol >= observed[i]
            lower[i] += draws[i, p] - tol <= observed[i]
            # Draws inside the Float64 rounding envelope cannot be ordered
            # reliably; `p_strict` counts none of them, `p` counts all of them.
            upper_strict[i] += draws[i, p] - tol > observed[i]
            lower_strict[i] += draws[i, p] + tol < observed[i]
        end
    end
    p = (min.(upper, lower) .+ 1) ./ (permutations + 1)
    p_strict = (min.(upper_strict, lower_strict) .+ 1) ./ (permutations + 1)
    m = vec(Statistics.mean(draws, dims = 2))
    s = vec(Statistics.std(draws, dims = 2, corrected = false))
    zscore = (observed .- m) ./ s
    zscore[s .== 0] .= NaN
    (; observed, draws, p, p_strict, mean = m, std = s, zscore)
end

function exact_ordered_moran_p(x::Vector{Float64}, k::Int,
                               permutations::Int, seed::Integer)
    offsets = Int128.(x .- minimum(x))
    z = x .- Statistics.mean(x)
    m2 = sum(z .^ 2) / (length(x) - 1)
    scale = z[1] / m2
    observed_lag = sum(Int128(slot) * offsets[slot + 1] for slot in 1:k)
    upper = 0
    lower = 0
    row_key = UInt64(seed) ⊻ (UInt64(1) * 0x9e3779b97f4a7c15)
    for p in 1:permutations
        state, _ = replay_splitmix(row_key ⊻ (UInt64(p) * 0x517cc1b727220a95))
        chosen = Int[]
        while length(chosen) < k
            index, state = replay_index(state, length(x), 1)
            index in chosen || push!(chosen, index)
        end
        perm_lag = sum(Int128(slot) * offsets[chosen[slot]] for slot in 1:k)
        if scale >= 0
            upper += perm_lag >= observed_lag
            lower += perm_lag <= observed_lag
        else
            upper += perm_lag <= observed_lag
            lower += perm_lag >= observed_lag
        end
    end
    (min(upper, lower) + 1) / (permutations + 1)
end

function worker_chunk_boundary_probe(backend)
    upload(T, value) = begin
        device_value = KernelAbstractions.allocate(backend, T, size(value)...)
        KernelAbstractions.copyto!(backend, device_value, value)
        device_value
    end
    upper = KernelAbstractions.zeros(backend, Int32, 1, 1)
    lower = KernelAbstractions.zeros(backend, Int32, 1, 1)
    means = KernelAbstractions.zeros(backend, Float32, 1, 1)
    m2s = KernelAbstractions.zeros(backend, Float32, 1, 1)
    anchors = KernelAbstractions.zeros(backend, Float32, 1, 1)
    perms = KernelAbstractions.zeros(backend, Float32, 1, 9)
    z = upload(Float32, Float32[-1, 1])
    weights = upload(Float32, Float32[1])
    neighbors = upload(Int32, Int32[2])
    offsets = upload(Int64, Int64[1, 2, 2])
    exact_values = KernelAbstractions.zeros(backend, UInt32, 1, 1)
    exact_weight_magnitudes = KernelAbstractions.zeros(backend, UInt32, 2, 1)
    exact_weight_signs = KernelAbstractions.zeros(backend, Int8, 1)
    exact_observed = KernelAbstractions.zeros(backend, UInt32, 8, 1)
    exact_defined = KernelAbstractions.zeros(backend, Bool, 1)
    row_ids = upload(Int32, Int32[1])
    scratch = KernelAbstractions.zeros(backend, Int32, 1)
    total = typemax(Int32)
    chunk_size = total - Int32(9)
    kernel = GPU_EXTENSION._local_perm_worker_kernel!(backend, 1)
    kernel(upper, lower, means, m2s, anchors, perms, z, weights, neighbors, offsets,
           exact_values, exact_weight_magnitudes, exact_weight_signs, exact_observed,
           exact_defined, row_ids, scratch, Int32(2), Int32(1), Int32(1), chunk_size,
           total, Int32(2), Float32(1), UInt64(7), Int32(1), true, false,
           # One moment block per chunk, so the 1x1 moment buffers stay valid
           # while the probe pushes the permutation index to the Int32 ceiling.
           (Int64(1), Int64(chunk_size)),
           Val(16), Val(false); ndrange = (1, 1))
    KernelAbstractions.synchronize(backend)
    return Array(upper), Array(lower), Array(perms)
end

function worker_ir_types(exact_tails::Bool)
    base = (Matrix{Int32}, Matrix{Int32}, Matrix{Float32}, Matrix{Float32},
            Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Float32},
            Vector{Int32}, Vector{Int64}, Matrix{UInt32}, Matrix{UInt32},
            Vector{Int8}, Matrix{UInt32}, Vector{Bool}, Vector{Int32}, Vector{Int32},
            Int32, Int32, Int32, Int32, Int32, Int32, Float32, UInt64, Int32,
            Bool, Bool, NTuple{2, Int64}, Val{16})
    return exact_tails ? Tuple{base..., Val{true}} : Tuple{base..., Val{false}}
end

backends = VENDOR_BACKEND === nothing ? (nothing, CPU()) : (nothing, CPU(), VENDOR_BACKEND)

@testset "inclusive tails, seeds, and CPU streaming" begin
    @testset "64-bit chunk endpoint arithmetic" begin
        total = Int64(typemax(Int32))
        chunk_size = cld(total, Int64(64))
        @test GPU_EXTENSION._chunk_start_i64(Int64(64), chunk_size) ==
              (Int64(64) - 1) * chunk_size + 1
        @test GPU_EXTENSION._chunk_end_i64(total, Int64(64), chunk_size) == total
    end

    @testset "device IR has no checked conversion throws" begin
        helper_specs = ((GPU_EXTENSION.rand_index, Tuple{UInt64, Int32, Int32}),
                        (GPU_EXTENSION._chunk_start_i64, Tuple{Int64, Int64}),
                        (GPU_EXTENSION._chunk_end_i64, Tuple{Int64, Int64, Int64}),
                        (GPU_EXTENSION._exact_u32, Tuple{UInt64}))
        for (helper, signature) in helper_specs
            ir = sprint(show, first(code_typed(helper, signature; optimize = true)))
            @test !occursin("throw_inexacterror", ir)
            @test !occursin("throw_boundserror", ir)
        end
        kernel = GPU_EXTENSION._local_perm_worker_kernel!(CPU(), 1)
        for exact_tails in (false, true)
            ir = KernelAbstractions.ka_code_typed(
                kernel, worker_ir_types(exact_tails);
                ndrange = (1, 1), optimize = true)
            # A signature that no longer matches the kernel yields no IR at
            # all, which would make the check below pass vacuously.
            @test !isempty(ir)
            text = join(sprint(show, item) for item in ir)
            @test !occursin("throw_inexacterror", text)
            # `throw_boundserror` is deliberately *not* audited away for the
            # worker kernel: it carries no `@inbounds`, so the computed moment
            # column `(p - 1 - permutation_offset) ÷ moment_block + 1` traps on
            # a wrong block slot instead of writing past the buffer.  Pinned so
            # that blanket `@inbounds` has to be a deliberate change.
            @test occursin("throw_boundserror", text)
        end
    end

    perms = reshape([1.0, 1.0, 2.0, 3.0, 4.0], 1, :)
    @test SpatialDependence._local_perm_summary(perms, [2.0], 5)[1] == [4 / 6]
    @test SpatialDependence._local_perm_summary(reshape([2.0], 1, :), [1.0], 1)[1] == [0.5]
    @test SpatialDependence._local_perm_summary(reshape([2.0], 1, :), [NaN], 1)[1] == [1.0]

    Wseed = SpatialWeights([0.0 1 1 0; 1 0 1 0; 1 1 0 1; 0 0 1 0])
    xseed = [0.0, 1.0, 4.0, 10.0]
    seeded_a = localmoran(xseed, Wseed, permutations = 31, seed = 123,
                          rng = StableRNG(1))
    seeded_b = localmoran(xseed, Wseed, permutations = 31, seed = 123,
                          rng = StableRNG(2))
    @test scoreperms(seeded_a) == scoreperms(seeded_b)
    seeded_c = localmoran(xseed, Wseed, permutations = 31, seed = 124)
    @test scoreperms(seeded_a) != scoreperms(seeded_c)
    explicit_rng_a = StableRNG(1)
    explicit_rng_b = StableRNG(1)
    localmoran(xseed, Wseed, permutations = 7, seed = 123, rng = explicit_rng_a)
    @test rand(explicit_rng_a, UInt64) == rand(explicit_rng_b, UInt64)
    gpu_rng_a = StableRNG(7)
    gpu_rng_b = StableRNG(7)
    gpu_rng_reference = StableRNG(7)
    rand(gpu_rng_reference, UInt64)
    gpu_seeded_a = localmoran(xseed, Wseed, permutations = 7, backend = CPU(),
                              rng = gpu_rng_a)
    gpu_seeded_b = localmoran(xseed, Wseed, permutations = 7, backend = CPU(),
                              rng = gpu_rng_b)
    @test scoreperms(gpu_seeded_a) == scoreperms(gpu_seeded_b)
    expected_gpu_next = rand(gpu_rng_reference, UInt64)
    @test rand(gpu_rng_a, UInt64) == expected_gpu_next
    @test rand(gpu_rng_b, UInt64) == expected_gpu_next
    @test_throws ArgumentError localmoran(xseed, Wseed, permutations = 1, seed = -1)
    @test_throws ArgumentError localmoran(xseed, Wseed, permutations = 1,
                                          seed = big(typemax(UInt64)) + 1)

    seed!(11)
    random_seed_a = localmoran(xseed, Wseed, permutations = 9, backend = CPU())
    seed!(11)
    random_seed_b = localmoran(xseed, Wseed, permutations = 9, backend = CPU())
    seed!(22)
    random_seed_c = localmoran(xseed, Wseed, permutations = 9, backend = CPU())
    @test scoreperms(random_seed_a) == scoreperms(random_seed_b)
    @test scoreperms(random_seed_a) != scoreperms(random_seed_c)

    function worker_indices(seed, worker, permutation, n, draws)
        key = UInt64(seed) ⊻ (UInt64(worker) * 0x9e3779b97f4a7c15) ⊻
              (UInt64(permutation) * 0x517cc1b727220a95)
        state, _ = replay_splitmix(key)
        out = Int[]
        while length(out) < draws
            index, state = replay_index(state, n, worker)
            push!(out, index)
        end
        out
    end
    stream_a = worker_indices(42, 1, 1, 9, 16)
    stream_b = worker_indices(42, 2, 1, 9, 16)
    @test stream_a != stream_b
    @test stream_a[2:end] != stream_b[1:end-1]
    stream_c = worker_indices(42, 1, 2, 9, 16)
    @test stream_a != stream_c
    @test stream_a[2:end] != stream_c[1:end-1]
end

@testset "Permutation regression checks: $backend" for backend in backends
    A = [0.0 1 1; 1 0 1; 1 1 0]
    W = SpatialWeights(A)
    Wbin = wtransform(W, :binary)
    x = [1.0, 2.0, 3.0]

    @testset "hand-calculable complete graph" begin
        lm = localmoran(x, W, permutations = 5, backend = backend, seed = 1)
        lg = localgeary(x, W, permutations = 5, backend = backend, seed = 1)
        gi = getisord(x, W, star = false, permutations = 5, backend = backend, seed = 1)
        gis = getisord(x, W, star = true, permutations = 5, backend = backend, seed = 1)

        @test scoreperms(lm) ≈ [-.5 -.5 -.5 -.5 -.5; 0 0 0 0 0; -.5 -.5 -.5 -.5 -.5]
        @test scoreperms(lg) ≈ [2.5 2.5 2.5 2.5 2.5; 1 1 1 1 1; 2.5 2.5 2.5 2.5 2.5]
        @test scoreperms(gi) ≈ fill(.5, 3, 5)
        @test scoreperms(gis) ≈ fill(1 / 3, 3, 5) atol = 1f-6
        for result in (lm, lg, gi, gis)
            @test all(std(result) .== 0)
            @test all(isnan, zscore(result))
            @test mean(result) ≈ score(result) atol = 1e-7
        end

        @test scoreperms(localmoran(x, Wbin, permutations = 3, backend = backend, seed = 1)) ≈
              [-1 -1 -1; 0 0 0; -1 -1 -1]
        @test scoreperms(localgeary(x, Wbin, permutations = 3, backend = backend, seed = 1)) ≈
              [5 5 5; 2 2 2; 5 5 5]
        @test scoreperms(getisord(x, Wbin, star = false, permutations = 3, backend = backend, seed = 1)) ≈ fill(1.0, 3, 3)
        @test scoreperms(getisord(x, Wbin, star = true, permutations = 3, backend = backend, seed = 1)) ≈ fill(1.0, 3, 3)
    end

    @testset "islands and return_perms" begin
        Wi = SpatialWeights([0.0 0 0; 0 0 1; 0 1 0])
        gis = getisord(x, Wi, star = true, permutations = 99, backend = backend, seed = 1)
        @test scoreperms(gis)[1, :] ≈ fill(1 / 6, 99) atol = 1e-6
        @test mean(gis)[1] ≈ 1 / 6 atol = 1e-6
        @test std(gis)[1] == 0
        @test isnan(zscore(gis)[1])
        isolated = getisord(x, SpatialWeights(zeros(3, 3)); permutations=7, backend=backend)
        @test scoreperms(isolated) ≈ repeat(x ./ sum(x), 1, 7) atol=1e-6

        for f in ((y, W; kwargs...) -> localmoran(y, W; kwargs...),
                  (y, W; kwargs...) -> localgeary(y, W; kwargs...),
                  (y, W; kwargs...) -> getisord(y, W, star = false; kwargs...),
                  (y, W; kwargs...) -> getisord(y, W, star = true; kwargs...))
            full = f(x, Wi; permutations = 17, backend = backend, seed = 42, rng=StableRNG(42))
            omitted = f(x, Wi; permutations = 17, backend = backend, seed = 42, rng=StableRNG(42), return_perms = false)
            @test size(scoreperms(omitted)) == (0, 17)
            assert_summary_equal(full, omitted)
            io = IOBuffer(); show(io, omitted)
            @test occursin("17 permutations", String(take!(io)))
        end
    end

    @testset "streaming batch boundaries and RNG state" begin
        runners = ((y, Z; kwargs...) -> localmoran(y, Z; kwargs...),
                   (y, Z; kwargs...) -> localgeary(y, Z; kwargs...),
                   (y, Z; kwargs...) -> getisord(y, Z, star = false; kwargs...),
                   (y, Z; kwargs...) -> getisord(y, Z, star = true; kwargs...))
        for P in (255, 256, 257, 513), run in runners
            full_rng = StableRNG(700 + P)
            stream_rng = StableRNG(700 + P)
            full = run(x, W; permutations = P, backend = backend, rng = full_rng)
            streamed = run(x, W; permutations = P, backend = backend,
                           rng = stream_rng, return_perms = false)
            @test size(scoreperms(streamed)) == (0, P)
            @test pvalue(full) == pvalue(streamed)
            @test mean(full) ≈ mean(streamed) rtol = 1e-12 atol = 1e-14
            @test std(full) ≈ std(streamed) rtol = 1e-12 atol = 1e-14
            @test rand(full_rng, UInt64) == rand(stream_rng, UInt64)
        end
        tiny_W = polyneigh(reggeomlattice(4, 4))
        tiny_x = 1 .+ 1e-14 .* sin.(collect(1.0:16.0))
        tiny_full = getisord(tiny_x, tiny_W, star = true, permutations = 513,
                             backend = backend, rng = StableRNG(72))
        tiny_stream = getisord(tiny_x, tiny_W, star = true, permutations = 513,
                               backend = backend, rng = StableRNG(72),
                               return_perms = false)
        @test std(tiny_full) ≈ std(tiny_stream) rtol = 1e-12 atol = 1e-30
        @test any(std(tiny_full) .> 0)
    end

    @testset "zero permutations" begin
        result = localmoran(x, W, permutations = 0, backend = backend, seed = 1)
        @test pvalue(result) == ones(3)
        @test all(isnan, mean(result))
        @test all(isnan, std(result))
        @test all(isnan, zscore(result))
        @test size(scoreperms(result)) == (3, 0)
    end

    @testset "boundary seeds are repeatable within a backend" begin
        for seed in (UInt64(0), typemax(UInt64))
            a = localmoran(x, W; permutations = 17, backend = backend, seed = seed)
            b = localmoran(x, W; permutations = 17, backend = backend, seed = seed)
            @test scoreperms(a) == scoreperms(b)
            @test pvalue(a) == pvalue(b)
        end
    end

    if backend !== nothing
      @testset "private-buffer boundary and rejection" begin
        n = 257
        B = zeros(Float64, n, n)
        B[1, 2:n] .= 1
        wb = SpatialWeights(B)
        boundary = localmoran(randn(StableRNG(1), n), wb, permutations = 1, backend = backend, seed = 1)
        @test size(scoreperms(boundary)) == (n, 1)

        nlarge = 2050
        wlarge = degree_weights(nlarge, nlarge - 1)
        large = localmoran(randn(StableRNG(1), nlarge), wlarge,
                           permutations = 1, backend = backend, seed = 1)
        @test size(scoreperms(large)) == (nlarge, 1)
        @test_throws ArgumentError localmoran(x, SpatialWeights(ones(3, 3)); permutations=1, backend=backend)
      end

      @testset "worker final chunk uses Int64 endpoints" begin
        upper, lower, perms = worker_chunk_boundary_probe(backend)
        @test upper == fill(Int32(9), 1, 1)
        @test lower == fill(Int32(9), 1, 1)
        @test perms == fill(Float32(-1), 1, 9)
      end

      @testset "extended neighbor capacities" begin
        runners = ((y, Z; kwargs...) -> localmoran(y, Z; kwargs...),
                   (y, Z; kwargs...) -> localgeary(y, Z; kwargs...),
                   (y, Z; kwargs...) -> getisord(y, Z, star = false; kwargs...),
                   (y, Z; kwargs...) -> getisord(y, Z, star = true; kwargs...))
        for (k, n) in ((64, 65), (65, 66), (255, 256), (256, 257),
                       (257, 258), (2048, 2049), (2049, 2050))
            wk = degree_weights(n, k)
            y = sin.(collect(1.0:n))
            for run in runners
                result = run(y, wk; permutations = 1, backend = backend, seed = 7)
                @test size(scoreperms(result)) == (n, 1)
            end
        end
      end

      @testset "seeded results do not depend on the chunk count" begin
        @test GPU_EXTENSION !== nothing
        n = 70
        Wmixed = mixed_degree_weights(n, n - 1)
        y = Float64.(mod.(collect(1:n), 17))
        runners = ((y, Z; kwargs...) -> localmoran(y, Z; kwargs...),
                   (y, Z; kwargs...) -> localgeary(y, Z; kwargs...),
                   (y, Z; kwargs...) -> getisord(y, Z, star = false; kwargs...),
                   (y, Z; kwargs...) -> getisord(y, Z, star = true; kwargs...))
        # Streams are keyed by (observation, permutation) and moments are
        # accumulated in blocks of the global permutation index, so the chunk
        # count changes nothing at all: draws, p-values and the summaries are
        # bit-identical.  `isequal` rather than `==` because z-scores are NaN
        # on zero-variance rows.
        # `budget` additionally shrinks the scratch budget for the rechunked
        # runs only, which forces one worker per launch: the host then merges
        # blocks across many launches and row batches, so the merge order is
        # compared with the default run's single launch.
        assert_chunk_invariant = function (run, Z, y, P, chunk_counts; budget = nothing)
            GPU_EXTENSION._set_gpu_chunk_count!(0)
            default = run(y, Z; permutations = P, backend = backend, seed = 31)
            default_streamed = run(y, Z; permutations = P, backend = backend, seed = 31,
                                   return_perms = false)
            # Keeping the draws must not change the summaries either.
            @test isequal(mean(default_streamed), mean(default))
            @test isequal(std(default_streamed), std(default))
            @test isequal(zscore(default_streamed), zscore(default))
            old_budget = budget === nothing ? nothing :
                         GPU_EXTENSION._set_gpu_scratch_budget!(budget)
            try
            for chunks in chunk_counts
                GPU_EXTENSION._set_gpu_chunk_count!(chunks)
                rechunked = run(y, Z; permutations = P, backend = backend, seed = 31)
                @test scoreperms(rechunked) == scoreperms(default)
                @test pvalue(rechunked) == pvalue(default)
                @test isequal(mean(rechunked), mean(default))
                @test isequal(std(rechunked), std(default))
                @test isequal(zscore(rechunked), zscore(default))
                streamed = run(y, Z; permutations = P, backend = backend, seed = 31,
                               return_perms = false)
                @test pvalue(streamed) == pvalue(default)
                @test isequal(mean(streamed), mean(default))
                @test isequal(std(streamed), std(default))
                @test isequal(zscore(streamed), zscore(default))
            end
            finally
                old_budget === nothing || GPU_EXTENSION._set_gpu_scratch_budget!(old_budget)
            end
        end
        old_chunks = GPU_EXTENSION._set_gpu_chunk_count!(0)
        try
            # At P = 130 there are three 64-permutation blocks, the last one
            # partial (130 = 2 * 64 + 2).  Chunk sizes round down to whole
            # blocks, so the seven requests below collapse to two schedules: 1
            # gives the single unaligned chunk of 130 that spans the whole run
            # (the case exempt from block alignment), and 2, 7, 64, 129, 130 and
            # 1000 all give three chunks of one block, which is what the default
            # rule gives too.  For P <= 64 the override is inert altogether:
            # there is one block, so every request floors to one block per chunk
            # and `cld(P, chunk_size)` renormalises it to a single chunk.
            @test GPU_EXTENSION._moment_block(130) == 64
            for run in runners
                assert_chunk_invariant(run, Wmixed, y, 130, (1, 2, 7, 64, 129, 130, 1000))
            end
            # P above 8192 selects a moment block longer than 64, on a graph
            # small enough to keep the run cheap.  None of these chunk counts
            # divides P, and they straddle the rounding to whole blocks: with a
            # 256-permutation block (79 blocks) the requested 3, 7 and 97 become
            # three distinct schedules of 4, 8 and 79 chunks, of 26, 11 and 1
            # blocks; the default rule gives 79.
            ntiny = 8
            Wtiny = mixed_degree_weights(ntiny, ntiny - 1)
            ytiny = Float64.(mod.(collect(1:ntiny), 5))
            @test GPU_EXTENSION._moment_block(20000) == 256
            for run in runners
                assert_chunk_invariant(run, Wtiny, ytiny, 20000, (3, 7, 97))
            end
            # Two-block chunks (a request of 39) with one worker per launch:
            # merging the blocks of a span in any other order than the global
            # permutation order changes the last bits of the moments.
            for run in runners
                assert_chunk_invariant(run, Wtiny, ytiny, 20000, (39,); budget = 1)
            end
            # Everything above either varies the chunk count over just two
            # schedules (P = 130) or does so with the 256-permutation block
            # (P = 20000), leaving multi-block chunks at the default 64-length
            # block untested.  P = 1000 is 16 such blocks, and requests of 2, 3
            # and 7 give three distinct schedules there - 2, 4 and 8 chunks of
            # 8, 5 and 2 blocks, against a default rule of 16 chunks of one
            # block - on the same tiny graph, for a twentieth of the
            # permutations of the P = 20000 case.
            for run in runners
                assert_chunk_invariant(run, Wtiny, ytiny, 1000, (2, 3, 7))
            end
        finally
            GPU_EXTENSION._set_gpu_chunk_count!(old_chunks)
        end
      end

      @testset "ragged worker batches and scratch-budget invariance" begin
        @test GPU_EXTENSION !== nothing
        n = 258
        Wlarge = mixed_degree_weights(n, n - 1)
        y = Float64.(mod.(collect(1:n), 17))
        runners = ((y, Z; kwargs...) -> localmoran(y, Z; kwargs...),
                   (y, Z; kwargs...) -> localgeary(y, Z; kwargs...),
                   (y, Z; kwargs...) -> getisord(y, Z, star = false; kwargs...),
                   (y, Z; kwargs...) -> getisord(y, Z, star = true; kwargs...))
        old_budget = GPU_EXTENSION._set_gpu_scratch_budget!(2048)
        try
            for run in runners
                tiny = run(y, Wlarge; permutations = 130, backend = backend, seed = 31)
                @test size(scoreperms(tiny)) == (n, 130)
                GPU_EXTENSION._set_gpu_scratch_budget!(64 * 1024 * 1024)
                wide = run(y, Wlarge; permutations = 130, backend = backend, seed = 31)
                @test scoreperms(tiny) == scoreperms(wide)
                @test pvalue(tiny) == pvalue(wide)
                @test mean(tiny) == mean(wide)
                @test std(tiny) == std(wide)
                GPU_EXTENSION._set_gpu_scratch_budget!(2048)
                streamed = run(y, Wlarge; permutations = 130, backend = backend, seed = 31,
                               return_perms = false)
                @test size(scoreperms(streamed)) == (0, 130)
                @test pvalue(streamed) == pvalue(tiny)
                @test mean(streamed) == mean(tiny)
                @test std(streamed) == std(tiny)
            end
        finally
            GPU_EXTENSION._set_gpu_scratch_budget!(old_budget)
        end
      end

      @testset "hashed mixed-degree weighted replay" begin
        for k in (64, 65)
            Wmixed = mixed_degree_weights(k + 1, k)
            ymixed = sin.(collect(1.0:(k + 1.0)))
            for (stat, run) in ((:moran, (y, Z; kwargs...) -> localmoran(y, Z; kwargs...)),
                                (:geary, (y, Z; kwargs...) -> localgeary(y, Z; kwargs...)),
                                (:getisord, (y, Z; kwargs...) -> getisord(y, Z, star = false; kwargs...)),
                                (:getisord_star, (y, Z; kwargs...) -> getisord(y, Z, star = true; kwargs...)))
                oracle = replay_oracle(stat, Float64.(ymixed), Wmixed, 7, 91)
                result = run(ymixed, Wmixed; permutations = 7, backend = backend, seed = 91)
                streamed = run(ymixed, Wmixed; permutations = 7, backend = backend,
                               seed = 91, return_perms = false)
                @test scoreperms(result) ≈ oracle.draws atol = 4e-5 rtol = 4e-5
                @test pvalue(result) == oracle.p
                @test mean(result) ≈ oracle.mean atol = 4e-5 rtol = 4e-5
                @test std(result) ≈ oracle.std atol = 4e-5 rtol = 4e-5
                assert_summary_equal(result, streamed)
            end
        end
      end

      @testset "independent ordered-weight replay across global scratch" begin
        for (k, n, P) in ((257, 258, 7), (2049, 2050, 3))
            Wmixed = mixed_degree_weights(n, k)
            ymixed = sin.(collect(1.0:n)) .+ 0.003 .* cos.(collect(1.0:n))
            oracle = replay_oracle(:moran, Float64.(ymixed), Wmixed, P, 91)
            result = localmoran(ymixed, Wmixed; permutations = P,
                                backend = backend, seed = 91)
            @test scoreperms(result) ≈ oracle.draws atol = 4e-5 rtol = 4e-5
            @test pvalue(result) == oracle.p
            @test mean(result) ≈ oracle.mean atol = 4e-5 rtol = 4e-5
            @test std(result) ≈ oracle.std atol = 4e-5 rtol = 4e-5
        end
      end

      @testset "exact ordered-weight tail in global scratch" begin
        n = 2050
        k = 2049
        P = 3
        neighs = [i == 1 ? Int64.(2:n) : Int64[] for i in 1:n]
        weights = [i == 1 ? Float64.(1:k) : Float64[] for i in 1:n]
        Wexact = SpatialWeights(Int64(n), neighs, weights,
                                Int64.(length.(neighs)), :original)
        yexact = Float64.(mod.(collect(1:n), 17))
        result = localmoran(yexact, Wexact; permutations = P,
                            backend = backend, seed = 91)
        expected = exact_ordered_moran_p(yexact, k, P, 91)
        @test pvalue(result)[1] == expected
        @test all(pvalue(result)[2:end] .== 1.0)
      end

      @testset "exact tails survive invalid approximate summaries" begin
        neighs = [Int64[2], Int64[3], Int64[4], Int64[1]]
        weights = [fill(1e300, 1) for _ in 1:4]
        Woverflow = SpatialWeights(Int64(4), neighs, weights,
                                   Int64.(length.(neighs)), :original)
        yoverflow = [1.0, 2.0, 4.0, 8.0]
        result = localmoran(yoverflow, Woverflow; permutations = 31,
                            backend = backend, seed = 42)
        @test pvalue(result) == [0.34375, 0.5, 0.40625, 0.375]
      end
    end

    @testset "stable moments at variation $delta" for delta in (0.001, 0.0001)
        geom = reggeomlattice(5, 5)
        Wsmall = polyneigh(geom)
        nearly = 1 .+ delta .* sin.(collect(1.0:25.0))
        result = getisord(nearly, Wsmall, star = true, permutations = 999, backend = backend, seed = 42)
        assert_summary_agrees(result)
        @test all(std(result) .> 0)
    end

    if backend !== nothing
        @testset "independent replay oracle on discrete nondegenerate graph" begin
            A6 = [0.0 1 1 0 0 0;
                  1 0 1 0 0 0;
                  1 1 0 1 0 0;
                  0 0 1 0 1 1;
                  0 0 0 1 0 1;
                  0 0 0 1 1 0]
            for x6 in ([0.0, 1, 0, 1, 1, 0], [0.0, 1, 2, 3, 1, 0]),
                W6 in (SpatialWeights(A6), wtransform(SpatialWeights(A6), :binary))
                for (stat, run) in ((:moran, (x, W; kwargs...) -> localmoran(x, W; kwargs...)),
                                    (:geary, (x, W; kwargs...) -> localgeary(x, W; kwargs...)),
                                    (:getisord, (x, W; kwargs...) -> getisord(x, W, star = false; kwargs...)),
                                    (:getisord_star, (x, W; kwargs...) -> getisord(x, W, star = true; kwargs...)))
                    oracle = replay_oracle(stat, x6, W6, 31, 42)
                    result = run(x6, W6; permutations = 31, backend = backend, seed = 42)
                    @test score(result) ≈ oracle.observed atol = 1e-7 rtol = 1e-6
                    @test scoreperms(result) ≈ oracle.draws atol = 3e-5 rtol = 3e-5
                    @test pvalue(result) == oracle.p
                    @test mean(result) ≈ oracle.mean atol = 3e-5 rtol = 3e-5
                    @test std(result) ≈ oracle.std atol = 3e-5 rtol = 3e-5
                    @test zscore(result) ≈ oracle.zscore atol = 3e-4 rtol = 3e-4
                end
            end
        end

        @testset "comparison-boundary Geary tail" begin
            if supports_float64(backend)
                # This small public reproducer exercises the Float64
                # comparison boundary where forming s_perm ± tol rounds
                # differently from subtracting obs_cmp first. The expected
                # lower count is 21, hence min-tail p=(21+1)/(130+1).
                xboundary = -7 .- Float64.(mod.(7 .* (1:23), 5)) .-
                            3e-7 .* sin.(0.43 .* (1:23))
                Aboundary = zeros(Float64, 23, 23)
                Aboundary[12, 11] = 1
                Aboundary[12, 22] = 1
                Wboundary = SpatialWeights(Aboundary)
                result = localgeary(xboundary, Wboundary; corrected = false,
                                    permutations = 130, backend = backend,
                                    precision = Float64, seed = 1)
                @test pvalue(result)[12] == 22 / 131
            end
        end

        @testset "Getis offset and small variation" begin
            A6 = [0.0 1 1 0 0 0;
                  1 0 1 0 0 0;
                  1 1 0 1 0 0;
                  0 0 1 0 1 1;
                  0 0 0 1 0 1;
                  0 0 0 1 1 0]
            W6 = SpatialWeights(A6)
            for xoffset in (1e6 .+ randn(StableRNG(7), 6),
                            -1e6 .+ randn(StableRNG(7), 6),
                            1 .+ 1e-6 .* sin.(collect(1.0:6.0)))
                for star in (false, true)
                    result = getisord(xoffset, W6, star = star, permutations = 31,
                                      backend = backend, seed = 42)
                    oracle = replay_oracle(star ? :getisord_star : :getisord,
                                           Float64.(xoffset), W6, 31, 42)
                    @test maximum(abs.(scoreperms(result) .- oracle.draws) ./ oracle.std) < 1e-4
                    @test pvalue(result) == oracle.p
                    @test mean(result) ≈ oracle.mean atol = 3e-5 rtol = 3e-5
                    @test maximum(abs.(std(result) ./ oracle.std .- 1)) < 1e-4
                    @test zscore(result) ≈ oracle.zscore atol = 3e-3 rtol = 3e-3
                end
            end
        end

        @testset "Moran and Geary scale invariance" begin
            Wscale = polyneigh(reggeomlattice(3, 3))
            xscale = randn(StableRNG(7), 9)
            for (stat, run) in ((:moran, localmoran), (:geary, localgeary)), scale in (1e20, 1e-23)
                oracle = replay_oracle(stat, scale .* xscale, Wscale, 65, 42)
                result = run(scale .* xscale, Wscale; permutations=65, backend, seed=42)
                @test scoreperms(result) ≈ oracle.draws rtol=1e-5 atol=1e-6
                @test pvalue(result) == oracle.p
                @test std(result) ≈ oracle.std rtol=1e-5 atol=1e-6
            end
        end
    end

    if backend === VENDOR_BACKEND && backend !== nothing
        @testset "functional vendor backend" begin
            result = localmoran(x, W, permutations = 17, backend = VENDOR_BACKEND, seed = 42)
            assert_summary_agrees(result)
            automatic = localmoran(x, W, permutations = 17, backend = :gpu, seed = 42)
            assert_summary_equal(result, automatic)
        end
    end
end

@testset "explicit accelerated precision" begin
    A6 = [0.0 1 1 0 0 0;
          1 0 1 0 0 0;
          1 1 0 1 0 0;
          0 0 1 0 1 1;
          0 0 0 1 0 1;
          0 0 0 1 1 0]
    W6 = SpatialWeights(A6)
    x6 = [0.0, 1, 2, 3, 1, 0]
    runners = ((:moran, (x, W; kwargs...) -> localmoran(x, W; kwargs...)),
               (:geary, (x, W; kwargs...) -> localgeary(x, W; kwargs...)),
               (:getisord, (x, W; kwargs...) -> getisord(x, W, star = false; kwargs...)),
               (:getisord_star, (x, W; kwargs...) -> getisord(x, W, star = true; kwargs...)))

    @test_throws ArgumentError localmoran(x6, W6, permutations = 0, precision = Float64)
    @test_throws ArgumentError localmoran(x6, W6, permutations = 0,
                                          backend = CPU(), precision = Float16)
    @test_throws ArgumentError localmoran(x6, W6, permutations = 0,
                                          backend = CPU(), precision = :double)

    cpu = CPU()
    cpu_fp64 = supports_float64(cpu)
    if cpu_fp64
      for (stat, run) in runners
        oracle = replay_oracle(stat, Float64.(x6), W6, 31, 42)
        retained = run(x6, W6; permutations = 31, backend = cpu,
                       precision = Float64, seed = 42)
        streamed = run(x6, W6; permutations = 31, backend = cpu,
                       precision = Float64, seed = 42, return_perms = false)
        @test score(retained) ≈ oracle.observed atol = 1e-12 rtol = 1e-12
        @test scoreperms(retained) ≈ oracle.draws atol = 1e-12 rtol = 1e-12
        @test pvalue(retained) == oracle.p
        @test mean(retained) ≈ oracle.mean atol = 1e-12 rtol = 1e-12
        @test std(retained) ≈ oracle.std atol = 1e-12 rtol = 1e-12
        @test zscore(retained) ≈ oracle.zscore atol = 1e-11 rtol = 1e-11
        @test size(scoreperms(streamed)) == (0, 31)
        assert_summary_equal(retained, streamed)

        p0 = run(x6, W6; permutations = 0, backend = cpu, precision = Float64)
        p1 = run(x6, W6; permutations = 1, backend = cpu, precision = Float64, seed = 42)
        @test pvalue(p0) == ones(6)
        @test size(scoreperms(p0)) == (6, 0)
        @test size(scoreperms(p1)) == (6, 1)
      end
    else
        @test_throws ArgumentError localmoran(x6, W6, permutations = 0,
                                              backend = cpu, precision = Float64)
    end

    implicit = localmoran(x6, W6; permutations = 31, backend = cpu, seed = 73)
    explicit32 = localmoran(x6, W6; permutations = 31, backend = cpu,
                            precision = Float32, seed = 73)
    @test scoreperms(implicit) == scoreperms(explicit32)

    # A tiny perturbation of count data is below Float32's useful resolution
    # for the returned draws, but accelerated Float32 tails still use the
    # independent exact comparator and therefore retain the oracle p-values.
    Wnear = polyneigh(reggeomlattice(10, 10))
    xnear = Float64.(rand(StableRNG(3), 100) .> 0.5) .+
            1e-8 .* sin.(collect(1.0:100.0))
    for (stat, run) in runners
        if cpu_fp64
            oracle = replay_oracle(stat, xnear, Wnear, 999, 42)
            fp64 = run(xnear, Wnear; permutations = 999, backend = cpu,
                       precision = Float64, seed = 42)
            fp32 = run(xnear, Wnear; permutations = 999, backend = cpu, seed = 42)
            # Near-constant data puts draws inside the Float64 rounding
            # envelope, where tie classification is not determinable.
            @test all(oracle.p_strict .<= pvalue(fp64) .<= oracle.p)
            @test scoreperms(fp64) ≈ oracle.draws atol = 1e-12 rtol = 1e-12
            @test all(isfinite, pvalue(fp32))
            @test all((0.0 .<= pvalue(fp32)) .& (pvalue(fp32) .<= 1.0))
        else
            fp32 = run(xnear, Wnear; permutations = 31, backend = cpu, seed = 42)
            @test size(scoreperms(fp32)) == (100, 31)
        end
    end

    if cpu_fp64
        @testset "Float64 shift-first host centering" begin
            nlarge = 19
            Alarge = zeros(Float64, nlarge, nlarge)
            for i in 1:nlarge
                Alarge[i, i == 1 ? nlarge : i - 1] = 1
                Alarge[i, i == nlarge ? 1 : i + 1] = 2
            end
            Wlarge = SpatialWeights(Alarge)
            xlarge = 1e8 .+ sin.(collect(1.0:nlarge))
            zbig = setprecision(BigFloat, 256) do
                xb = BigFloat.(xlarge)
                μ = sum(xb) / BigFloat(nlarge)
                Float64.(xb .- μ)
            end
            m2big = sum(zbig .^ 2) / (nlarge - 1)
            expected_moran = [zbig[i] / m2big *
                sum(BigFloat(weights(Wlarge, i)[slot]) * zbig[j]
                    for (slot, j) in enumerate(neighbors(Wlarge, i)))
                for i in 1:nlarge]
            expected_geary = [sum(BigFloat(weights(Wlarge, i)[slot]) *
                (zbig[i] - zbig[j])^2
                for (slot, j) in enumerate(neighbors(Wlarge, i))) / m2big
                for i in 1:nlarge]
            moran_large = localmoran(xlarge, Wlarge; permutations = 31,
                                     seed = 42,
                                     backend = cpu, precision = Float64)
            geary_large = localgeary(xlarge, Wlarge; permutations = 31,
                                     seed = 42,
                                     backend = cpu, precision = Float64)
            @test maximum(abs.(score(moran_large) .- expected_moran)) < 1e-12
            @test maximum(abs.(score(geary_large) .- expected_geary)) < 1e-12

            # Replay the same device index stream from independently
            # BigFloat-centered Float64 values, using BigFloat weighted
            # accumulation before converting each expected draw to Float64.
            for (result, stat) in ((moran_large, :moran), (geary_large, :geary))
                expected_draws = zeros(Float64, nlarge, 31)
                for i in 1:nlarge
                    k = Wlarge.nneighs[i]
                    neigh = neighbors(Wlarge, i)
                    wi = weights(Wlarge, i)
                    row_key = UInt64(42) ⊻ (UInt64(i) * 0x9e3779b97f4a7c15)
                    for p in 1:31
                        state, _ = replay_splitmix(row_key ⊻ (UInt64(p) * 0x517cc1b727220a95))
                        chosen = Int[]
                        while length(chosen) < k
                            index, state = replay_index(state, nlarge, i)
                            index in chosen || push!(chosen, index)
                        end
                        value = if stat == :moran
                            zbig[i] / m2big * sum(BigFloat(wi[j]) * zbig[chosen[j]] for j in 1:k)
                        else
                            sum(BigFloat(wi[j]) * (zbig[i] - zbig[chosen[j]])^2 for j in 1:k) / m2big
                        end
                        expected_draws[i, p] = Float64(value)
                    end
                end
                @test scoreperms(result) ≈ expected_draws atol = 1e-12 rtol = 1e-12
            end
        end
    end

    # The ordinary CPU implementation accepts degree 257; the accelerated
    # the ragged/global path has no fixed production neighbor ceiling.
    nwide = 258
    Awide = zeros(Float64, nwide, nwide)
    Awide[1, 2:nwide] .= 1
    Wwide = SpatialWeights(Awide)
    xwide = randn(StableRNG(11), nwide)
    @test size(scoreperms(localmoran(xwide, Wwide, permutations = 1))) == (nwide, 1)
    @test size(scoreperms(localmoran(xwide, Wwide, permutations = 1,
                                     backend = cpu))) == (nwide, 1)

    if cpu_fp64
        xnegative = [-1.0, -2, -3, -4, -5, -6]
        for star in (false, true)
            retained = getisord(xnegative, W6; star = star, permutations = 31,
                                 backend = cpu, precision = Float64, seed = 42)
            streamed = getisord(xnegative, W6; star = star, permutations = 31,
                                backend = cpu, precision = Float64, seed = 42,
                                return_perms = false)
            @test all(isfinite, scoreperms(retained))
            assert_summary_equal(retained, streamed)
        end
    end

    if METAL_BACKEND !== nothing
        if supports_float64(METAL_BACKEND)
            @test localmoran(x6, W6; permutations = 0,
                             backend = METAL_BACKEND, precision = Float64) isa LocalMoran
        else
            @test_throws ArgumentError localmoran(x6, W6; permutations = 0,
                                                  backend = METAL_BACKEND,
                                                  precision = Float64)
        end
    end
    if VENDOR_BACKEND !== nothing
        if supports_float64(VENDOR_BACKEND)
            oracle = replay_oracle(:moran, xnear, Wnear, 31, 42)
            vendor_fp64 = localmoran(xnear, Wnear; permutations = 31,
                                     backend = VENDOR_BACKEND,
                                     precision = Float64, seed = 42)
            @test pvalue(vendor_fp64) == oracle.p
            @test scoreperms(vendor_fp64) ≈ oracle.draws atol = 1e-12 rtol = 1e-12
        else
            @test_throws ArgumentError localmoran(x6, W6; permutations = 0,
                                                  backend = VENDOR_BACKEND,
                                                  precision = Float64)
        end
    end
end
