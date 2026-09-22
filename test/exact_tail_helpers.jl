using SpatialDependence
using KernelAbstractions
using Test
using StableRNGs

if !isdefined(@__MODULE__, :VENDOR_BACKEND)
    include(joinpath(@__DIR__, "gpu_backend.jl"))
end

const _exact_tail_ext = Base.get_extension(SpatialDependence,
                                           :SpatialDependenceKernelAbstractionsExt)
const _exact_sub96! = _exact_tail_ext._exact_sub96!
const _exact_unsigned_cmp96 = _exact_tail_ext._exact_unsigned_cmp96
const _exact_mul! = _exact_tail_ext._exact_mul!
const _exact_mul_3x3! = _exact_tail_ext._exact_mul_3x3!
const _exact_mul_6x2! = _exact_tail_ext._exact_mul_6x2!

@kernel function _exact_limb_smoke!(out)
    i = @index(Global)
    a = @private UInt32 (8,)
    b = @private UInt32 (8,)
    d = @private UInt32 (8,)
    square = @private UInt32 (8,)
    weight = @private UInt32 (8,)
    product = @private UInt32 (8,)
    if i == 1
        for j in 1:8
            a[j] = UInt32(0)
            b[j] = UInt32(0)
        end
        a[1] = UInt32(0xffffffff)
        a[2] = UInt32(0xffffffff)
        a[3] = UInt32(1)
        b[1] = UInt32(1)
        b[3] = UInt32(2)
        _exact_sub96!(d, b, a)
        _exact_mul_3x3!(square, d, d)
        for j in 1:8
            weight[j] = UInt32(0)
        end
        weight[1] = UInt32(3)
        _exact_mul_6x2!(product, square, weight)
        for j in 1:8
            out[j] = product[j]
        end
        out[9] = _exact_unsigned_cmp96(a, b)
    end
end

@testset "exact-tail limb and packing helpers" begin
    exact_ext = Base.get_extension(SpatialDependence,
                                   :SpatialDependenceKernelAbstractionsExt)
    @test exact_ext !== nothing

    limbs_to_unsigned(a) = sum(BigInt(a[j]) << (32 * (j - 1)) for j in 1:8)
    limbs_to_signed(a) = begin
        u = limbs_to_unsigned(a)
        u >= (BigInt(1) << 255) ? u - (BigInt(1) << 256) : u
    end
    pack_unsigned(v) = [UInt32((v >> (32 * (j - 1))) & BigInt(0xffffffff)) for j in 1:8]

    a = UInt32[0xffffffff, 0x12345678, 0, 0, 0, 0, 0, 0]
    b = UInt32[1, 0, 0, 0, 0, 0, 0, 0]
    out = zeros(UInt32, 8)
    exact_ext._exact_add!(out, a, b)
    @test limbs_to_unsigned(out) == mod(limbs_to_unsigned(a) + limbs_to_unsigned(b), BigInt(1) << 256)

    neg = copy(a)
    exact_ext._exact_negate!(neg)
    @test limbs_to_signed(neg) == -limbs_to_signed(a)
    @test exact_ext._exact_signed_cmp(a, b) ==
          (limbs_to_signed(a) < limbs_to_signed(b) ? -1 : 1)

    mul_a = UInt32[0xffffffff, 0xffffffff, 0x12345678, 0, 0, 0, 0, 0]
    mul_b = UInt32[0x87654321, 0, 0xabcdef01, 0, 0, 0, 0, 0]
    exact_ext._exact_mul!(out, mul_a, mul_b)
    @test limbs_to_unsigned(out) == mod(limbs_to_unsigned(mul_a) * limbs_to_unsigned(mul_b), BigInt(1) << 256)

    for (a_limbs, b_limbs) in ((3, 2), (3, 3), (6, 2))
        aa = fill(UInt32(0), 8)
        bb = fill(UInt32(0), 8)
        aa[1:a_limbs] .= typemax(UInt32)
        bb[1:b_limbs] .= typemax(UInt32)
        exact_ext._exact_mul!(out, aa, bb, Val(a_limbs), Val(b_limbs))
        @test limbs_to_unsigned(out) ==
              mod(limbs_to_unsigned(aa) * limbs_to_unsigned(bb), BigInt(1) << 256)
    end

    left = zeros(UInt32, 8)
    right = zeros(UInt32, 8)
    left[1:3] .= UInt32[0xffffffff, 0xffffffff, 1]
    right[1:3] .= UInt32[1, 0, 2]
    diff = zeros(UInt32, 8)
    exact_ext._exact_sub96!(diff, right, left)
    @test [diff[j] for j in 1:3] == UInt32[2, 0, 0]

    smoke = KernelAbstractions.zeros(CPU(), Int32, 9)
    _exact_limb_smoke!(CPU())(smoke; ndrange = 1)
    synchronize(CPU())
    @test Array(smoke) == Int32[12, 0, 0, 0, 0, 0, 0, 0, -1]

    vendor_backend = VENDOR_BACKEND
    if vendor_backend !== nothing
        vendor_smoke = KernelAbstractions.zeros(vendor_backend, Int32, 9)
        _exact_limb_smoke!(vendor_backend)(vendor_smoke; ndrange = 1)
        synchronize(vendor_backend)
        @test Array(vendor_smoke) == Int32[12, 0, 0, 0, 0, 0, 0, 0, -1]

        # Weight reduction keeps the exact payload within its integer bound,
        # but the public observed Float64 scores are non-finite.  This must be
        # rejected before the backend seed is drawn or device buffers launch.
        bad_neighs = [Int64[2, 4], Int64[1, 3], Int64[2, 4], Int64[1, 3]]
        bad_weights = [fill(floatmax(Float64), 2) for _ in 1:4]
        bad_W = SpatialWeights(4, bad_neighs, bad_weights, fill(Int64(2), 4), :binary)
        bad_x = [0.0, 1.0, 4.0, 9.0]
        for f in (localmoran, localgeary)
            rng = StableRNG(918)
            reference = StableRNG(918)
            @test_throws ArgumentError f(bad_x, bad_W; permutations = 3,
                                         backend = vendor_backend, rng = rng)
            @test rand(rng, UInt64) == rand(reference, UInt64)
        end
    end

    encoded_low_precision = setprecision(BigFloat, 32) do
        exact_ext._exact_scaled_values([0.0, Float64(pi), 1e-4])[5]
    end
    encoded_default = exact_ext._exact_scaled_values([0.0, Float64(pi), 1e-4])[5]
    @test encoded_low_precision == encoded_default

    x = [1.0, 1.5, 2.0, 4.0]
    W = SpatialWeights([0 1 0 0; 1 0 1 0; 0 1 0 1; 0 0 1 0]; standardize = false)
    payload = exact_ext._exact_tail_payload(x, W; stat_code = 1,
                                             observed = BigInt[0, 0, 0, 0],
                                             defined = [true, false, true, true])
    @test size(payload.values.values) == (3, 4)
    @test payload.values.divisor == BigInt(2.0^51)
    @test payload.defined == BitVector([true, false, true, true])
    @test [limbs_to_signed(payload.observed[:, i]) for i in 1:4] == BigInt[0, 0, 0, 0]
    @test all(<(BigInt(1) << 255), payload.row_bounds)

    zero_weights = SpatialWeights(zeros(3, 3); standardize = false)
    zero_payload = exact_ext._exact_tail_payload([1.0, 1.0, 1.0], zero_weights)
    @test all(iszero, zero_payload.weights.signs)
    @test all(iszero, zero_payload.row_bounds)

    # Full exact-tail sampling regression on KA CPU.  The expected values use
    # an independent exact-Rational replay of the raw Float64 data and raw
    # weights; the production payload is used only to launch the device.
    xoracle = [0.0, 1.0, 4.0, 9.0]
    oracle_neighs = [Int64[2, 4], Int64[1, 3], Int64[2, 4], Int64[1, 3]]
    oracle_weights = [[2.0, -1.0], [-3.0, 1.0], [1.0, 4.0], [-2.0, 3.0]]
    Woracle = SpatialWeights(4, oracle_neighs, oracle_weights,
                             Int64.(length.(oracle_neighs)), :original)
    P = 7
    seed = UInt64(17)
    function raw_rational(v::Float64)
        bits = reinterpret(UInt64, v)
        sign = (bits >> 63) == UInt64(0) ? BigInt(1) : BigInt(-1)
        fraction = bits & UInt64(0x000fffffffffffff)
        exponent_bits = (bits >> 52) & UInt64(0x7ff)
        exponent_bits == UInt64(0) && return sign *
            (Rational{BigInt}(BigInt(fraction), BigInt(1) << 1074))
        mantissa = BigInt((UInt64(1) << 52) | fraction)
        exponent = Int(exponent_bits) - 1023 - 52
        exponent >= 0 ? sign * (mantissa << exponent) :
            sign * Rational{BigInt}(mantissa, BigInt(1) << (-exponent))
    end
    function rational_replay_counts(stat_code)
        xr = raw_rational.(xoracle)
        zr = xr .- (sum(xr; init = Rational{BigInt}(0)) / length(xr))
        m2r = sum(zr .^ 2; init = Rational{BigInt}(0)) / (length(xr) - 1)
        totalr = sum(xr; init = Rational{BigInt}(0))
        upper = zeros(Int, Woracle.n)
        lower = zeros(Int, Woracle.n)
        for i in 1:Woracle.n
            row_key = seed ⊻ (UInt64(i) * 0x9e3779b97f4a7c15)
            neigh = Woracle.neighs[i]
            raw_weights = raw_rational.(Woracle.weights[i])
            wistar = stat_code == 4 ? Rational{BigInt}(1) : nothing
            observed = if stat_code == 1
                (zr[i] / m2r) * sum(raw_weights[j] * zr[neigh[j]] for j in eachindex(neigh); init = Rational{BigInt}(0))
            elseif stat_code == 2
                sum(raw_weights[j] * (zr[i] - zr[neigh[j]])^2 for j in eachindex(neigh); init = Rational{BigInt}(0)) / m2r
            elseif stat_code == 3
                sum(raw_weights[j] * xr[neigh[j]] for j in eachindex(neigh); init = Rational{BigInt}(0)) / (totalr - xr[i])
            else
                (wistar * xr[i] + sum(wistar * xr[neigh[j]] for j in eachindex(neigh); init = Rational{BigInt}(0))) / totalr
            end
            for p in 1:P
                state, _ = _exact_tail_ext.splitmix64(row_key ⊻ (UInt64(p) * 0x517cc1b727220a95))
                chosen = Int[]
                while length(chosen) < Woracle.nneighs[i]
                    idx, state = _exact_tail_ext.rand_index(state, Int32(Woracle.n), Int32(i))
                    idx in chosen || push!(chosen, Int(idx))
                end
                perm = if stat_code == 1
                    (zr[i] / m2r) * sum(raw_weights[j] * zr[chosen[j]] for j in eachindex(chosen); init = Rational{BigInt}(0))
                elseif stat_code == 2
                    sum(raw_weights[j] * (zr[i] - zr[chosen[j]])^2 for j in eachindex(chosen); init = Rational{BigInt}(0)) / m2r
                elseif stat_code == 3
                    sum(raw_weights[j] * xr[chosen[j]] for j in eachindex(chosen); init = Rational{BigInt}(0)) / (totalr - xr[i])
                else
                    (wistar * xr[i] + sum(wistar * xr[chosen[j]] for j in eachindex(chosen); init = Rational{BigInt}(0))) / totalr
                end
                perm >= observed && (upper[i] += 1)
                perm <= observed && (lower[i] += 1)
            end
        end
        return upper, lower
    end

    for stat_code in 1:4
        payload = _exact_tail_ext._exact_tail_payload(xoracle, Woracle;
                                                       stat_code = stat_code)
        expected_upper, expected_lower = rational_replay_counts(stat_code)
        pw, pn, card, max_k = _exact_tail_ext.prepare_gpu_weights(
            CPU(), Woracle, Float32, Int32(stat_code))
        partial_upper = zeros(Int32, Woracle.n, 1)
        partial_lower = zeros(Int32, Woracle.n, 1)
        partial_mean = zeros(Float32, Woracle.n, 1)
        partial_m2 = zeros(Float32, Woracle.n, 1)
        partial_anchor = zeros(Float32, Woracle.n, 1)
        full_perms = zeros(Float32, Woracle.n, P)
        exact_z = Float32.(xoracle .- mean(xoracle))
        scale = stat_code == 2 ? Float32(sum(exact_z .^ 2) / 3) : Float32(1)
        _exact_tail_ext.local_perm_chunk_kernel!(CPU())(
            partial_upper, partial_lower, partial_mean, partial_m2, partial_anchor,
            full_perms, exact_z, pw, pn, card,
            payload.values.values, payload.weights.magnitudes, payload.weights.signs,
            payload.observed, payload.defined,
            Int32(Woracle.n), Int32(P), Int32(P), scale, seed,
            Int32(stat_code), true, Val(16), Val(true); ndrange=(Woracle.n, 1))
        synchronize(CPU())
        # Exact p-values use the smaller inclusive tail; Moran/Getis factors
        # can reverse the two directional counts without changing it.
        @test min.(vec(partial_upper), vec(partial_lower)) ==
              Int32.(min.(expected_upper, expected_lower))
    end

    # This is the previously failing Float32 Geary boundary: the third and
    # fourth values are distinct in Float64 but collapse in the device input.
    fixture_x = [-10.999999729554903, -8.999999700046521,
                 -10.999999989435572, -11.000000039363957]
    fixture_neighs = [Int64[2, 3], Int64[1], Int64[1], Int64[1]]
    fixture_weights = [[1.0, 1.0], [1.0], [1.0], [1.0]]
    fixture_W = SpatialWeights(4, fixture_neighs, fixture_weights,
                               Int64.(length.(fixture_neighs)), :binary)
    fixture_payload = _exact_tail_ext._exact_tail_payload(fixture_x, fixture_W;
                                                           stat_code = 2)
    fxr = raw_rational.(fixture_x)
    fzr = fxr .- (sum(fxr; init = Rational{BigInt}(0)) / 4)
    fm2r = sum(fzr .^ 2; init = Rational{BigInt}(0)) / 3
    fseed = UInt64(18)
    fstate, _ = _exact_tail_ext.splitmix64(
        fseed ⊻ (UInt64(1) * 0x9e3779b97f4a7c15) ⊻ 0x517cc1b727220a95)
    fchosen = Int[]
    while length(fchosen) < 2
        idx, fstate = _exact_tail_ext.rand_index(fstate, Int32(4), Int32(1))
        idx in fchosen || push!(fchosen, Int(idx))
    end
    @test fchosen == [2, 4]
    fobs = (fzr[1] - fzr[2])^2 + (fzr[1] - fzr[3])^2
    fperm = (fzr[1] - fzr[fchosen[1]])^2 + (fzr[1] - fzr[fchosen[2]])^2
    @test fperm > fobs
    fpw, fpn, fcard, _ = _exact_tail_ext.prepare_gpu_weights(
        CPU(), fixture_W, Float32, Int32(2))
    fu = zeros(Int32, 4, 1); fl = zeros(Int32, 4, 1)
    fm = zeros(Float32, 4, 1); fmm = zeros(Float32, 4, 1); fa = zeros(Float32, 4, 1)
    fperms = zeros(Float32, 4, 1)
    _exact_tail_ext.local_perm_chunk_kernel!(CPU())(
        fu, fl, fm, fmm, fa, fperms, Float32.(fixture_x .- mean(fixture_x)),
        fpw, fpn, fcard, fixture_payload.values.values,
        fixture_payload.weights.magnitudes, fixture_payload.weights.signs,
        fixture_payload.observed, fixture_payload.defined,
        Int32(4), Int32(1), Int32(1), Float32(sum((fixture_x .- mean(fixture_x)).^2) / 3),
        fseed, Int32(2), true, Val(16), Val(true); ndrange = (4, 1))
    synchronize(CPU())
    @test fu[1, 1] == 1
    @test fl[1, 1] == 0

    # Cross the public API: the exact p-tail remains 1/2 even though the old
    # Float32 tolerance path merges the fixture's distinct values and returns
    # one-sided p=1 for this P=1 draw.
    public_fixture = localgeary(fixture_x, fixture_W; permutations = 1,
                                backend = CPU(), seed = 18)
    @test pvalue(public_fixture)[1] == 0.5
end

# A minimal non-1-based vector, used to check that the fast path declines
# offset-indexed data instead of raising an exception of its own.
struct OffsetLike <: AbstractVector{Float64}
    v::Vector{Float64}
end
Base.size(a::OffsetLike) = size(a.v)
Base.axes(a::OffsetLike) = (Base.IdentityUnitRange(0:(length(a.v) - 1)),)
Base.getindex(a::OffsetLike, i::Int) = a.v[i + 1]

# The exact-tail payload is built by a machine-integer fast path that falls
# back to the original BigInt encoder whenever it cannot prove its fixed-width
# intermediates are exact.  These tests pin the fast path to the reference:
# identical arrays, identical exceptions, on deliberately hostile inputs.
@testset "exact-tail fast payload equals the BigInt reference" begin
    ext = _exact_tail_ext

    function ragged_W(n::Int, ks::Vector{Int}, wfun; style::Symbol = :original)
        neighs = Vector{Vector{Int64}}(undef, n)
        w = Vector{Vector{Float64}}(undef, n)
        for i in 1:n
            nb = Int64[]
            for d in 1:ks[i]
                j = mod1(i + d, n)
                j == i && continue
                push!(nb, Int64(j))
            end
            neighs[i] = nb
            w[i] = wfun(i, length(nb))
        end
        SpatialWeights(Int64(n), neighs, w, Int64.(length.(neighs)), style)
    end
    row_W(n, k) = ragged_W(n, fill(k, n), (i, m) -> fill(1.0 / m, m); style = :row)
    binary_W(n, k) = ragged_W(n, fill(k, n), (i, m) -> ones(Float64, m); style = :original)

    payload_fields(p) = Any[p.values.exponent, p.values.origin, p.values.divisor,
                            p.values.values, p.weights.exponents, p.weights.gcds,
                            p.weights.offsets, p.weights.magnitudes, p.weights.signs,
                            p.observed, p.defined, p.row_bounds]

    rng = StableRNG(20260918)
    cases = Tuple{String, Vector{Float64}, SpatialWeights}[]

    # plain, integer-valued and irrational-looking data, both weight styles
    push!(cases, ("row", randn(rng, 24), row_W(24, 3)))
    push!(cases, ("binary", randn(rng, 24), binary_W(24, 3)))
    push!(cases, ("integers", Float64.(1:20), row_W(20, 4)))
    push!(cases, ("irrational", Float64[pi, exp(1.0), sqrt(2.0), log(3.0),
                                        sqrt(5.0), pi^2, -exp(2.0), 1 / 7],
                  binary_W(8, 3)))
    # degenerate data
    push!(cases, ("constant", fill(2.5, 12), row_W(12, 2)))
    push!(cases, ("allzero", zeros(10), row_W(10, 2)))
    push!(cases, ("signed zeros", Float64[0.0, -0.0, 0.0, -0.0, 1.0, -1.0, 0.0, 2.0],
                  row_W(8, 2)))
    # subnormals, huge and tiny exponents, wide common units
    push!(cases, ("subnormals", Float64[5.0e-324, -5.0e-324, 1.0e-320, 0.0,
                                        3.0e-322, -2.5e-323, 7.0e-324, 1.5e-323],
                  row_W(8, 2)))
    push!(cases, ("all subnormal", Float64[m * 5.0e-324 for m in (3, 1, 7, 0, 12, 5, 2, 9)],
                  binary_W(8, 3)))
    push!(cases, ("huge data", Float64[2.0^90, -2.0^88, 2.0^80, 0.0, 2.0^60, -2.0^60],
                  row_W(6, 2)))
    push!(cases, ("tiny data", Float64[2.0^-60, -2.0^-62, 2.0^-70, 0.0, 2.0^-58, 2.0^-59],
                  row_W(6, 2)))
    push!(cases, ("mixed scale", Float64[2.0^-20, 2.0^20 + 1.0, -2.0^-20, 1.5, 0.0, 2.0^10],
                  row_W(6, 2)))
    push!(cases, ("adjacent floats",
                  Float64[1.0, 1.0 + 2.0^-52, 1.0 + 2.0^-51, 1.0 + 3 * 2.0^-52,
                          1.0, 1.0 + 2.0^-50], row_W(6, 2)))
    push!(cases, ("near 96 bits", Float64[0.0, 2.0^-40, 2.0^55, 0.0, 2.0^-40, 2.0^55],
                  row_W(6, 2)))
    # weight extremes: 2^990, 2^-1040, subnormal, negative, zero, mixed scale
    push!(cases, ("huge weights", randn(rng, 10),
                  ragged_W(10, fill(3, 10), (i, m) -> [2.0^990, 2.0^985, 2.0^980][1:m])))
    push!(cases, ("tiny weights", randn(rng, 10),
                  ragged_W(10, fill(3, 10), (i, m) -> [2.0^-1040, 2.0^-1042, 2.0^-1045][1:m])))
    push!(cases, ("subnormal weights", randn(rng, 8),
                  ragged_W(8, fill(2, 8), (i, m) -> [5.0e-324, 1.0e-323][1:m])))
    push!(cases, ("mixed weight scale", randn(rng, 8),
                  ragged_W(8, fill(3, 8), (i, m) -> [2.0^-30, 2.0^30, 1.0][1:m])))
    push!(cases, ("negative weights", randn(rng, 12),
                  ragged_W(12, fill(3, 12), (i, m) -> [-1.0, 2.0, -0.5][1:m])))
    push!(cases, ("zero weights", randn(rng, 12),
                  ragged_W(12, fill(3, 12), (i, m) -> [0.0, 1.0, -0.0][1:m])))
    push!(cases, ("all-zero weights", randn(rng, 10),
                  ragged_W(10, fill(2, 10), (i, m) -> zeros(m))))
    # islands and mixed degrees, including a row wider than 256 neighbors
    let ks = fill(2, 12)
        ks[3] = 0; ks[7] = 0
        push!(cases, ("islands", randn(rng, 12),
                      ragged_W(12, ks, (i, m) -> m == 0 ? Float64[] : fill(1.0 / m, m);
                               style = :row)))
    end
    let n = 300, ks = [i == 1 ? 260 : (i % 7) for i in 1:n]
        push!(cases, ("wide row", randn(rng, n),
                      ragged_W(n, ks, (i, m) -> m == 0 ? Float64[] : fill(1.0 / m, m);
                               style = :row)))
    end
    # inputs that must throw
    push!(cases, ("NaN data", Float64[1.0, NaN, 2.0, 3.0], binary_W(4, 1)))
    push!(cases, ("Inf data", Float64[1.0, Inf, 2.0, 3.0], binary_W(4, 1)))
    push!(cases, ("-Inf data", Float64[1.0, -Inf, 2.0, 3.0], binary_W(4, 1)))
    push!(cases, ("NaN weight", Float64[1.0, 2.0, 3.0, 4.0],
                  ragged_W(4, fill(1, 4), (i, m) -> [i == 1 ? NaN : 1.0])))
    push!(cases, ("Inf weight", Float64[1.0, 2.0, 3.0, 4.0],
                  ragged_W(4, fill(1, 4), (i, m) -> [i == 2 ? Inf : 1.0])))
    push!(cases, ("over 96 bits", Float64[0.0, 2.0^400, 1.0, 3.0], binary_W(4, 1)))
    push!(cases, ("over 96 bits (both ends)", Float64[2.0^-200, 2.0^200, 1.0, 3.0],
                  binary_W(4, 1)))
    push!(cases, ("over 64-bit weights", Float64[1.0, 2.0, 3.0, 4.0],
                  ragged_W(4, fill(2, 4), (i, m) -> [2.0^-200, 2.0^200])))
    # Geary squares the value range, so this row exceeds the signed 256-bit bound.
    push!(cases, ("over 256-bit accumulation",
                  Float64[(0.0, 2.0^-40, 2.0^55)[mod1(i, 3)] for i in 1:12],
                  ragged_W(12, fill(8, 12), (i, m) -> Float64[j == 1 ? 1.0 : 2.0^63 for j in 1:m])))

    # coverage for specific internal decisions (see the route assertions below)
    # 128-bit weight gate: [1.0, 2^128] needs 129 bits and must decline, while
    # the nextfloat pair needs 116 and must take the UInt128 weight path.
    push!(cases, ("weight gate 129 bits", randn(rng, 6),
                  ragged_W(6, fill(2, 6), (i, m) -> [1.0, 2.0^128][1:m])))
    push!(cases, ("weight gate 116 bits", randn(rng, 6),
                  ragged_W(6, fill(2, 6),
                           (i, m) -> [nextfloat(1.0), nextfloat(1.0) * 2.0^63][1:m])))
    # a zero slot next to a huge weight pins the row exponent (min over *all*
    # slots, where a zero contributes exponent 0) apart from the reduced unit
    push!(cases, ("zero slot with 2^990", randn(rng, 6),
                  ragged_W(6, fill(2, 6), (i, m) -> [0.0, 2.0^990][1:m])))
    # ... and with the zero *after* the huge weight, so a row exponent taken
    # over nonzero slots only would differ from the reference's
    push!(cases, ("zero slot after 2^990", randn(rng, 6),
                  ragged_W(6, fill(2, 6), (i, m) -> [2.0^990, 0.0][1:m])))
    # Getis denominator: exercises the remainder branch of the rewritten mask
    push!(cases, ("getis denominator", Float64[1.5, -0.5, 3.0, 2.0], binary_W(4, 1)))
    # origin -3, divisor 2^53: -(n-1)*origin leaves a nonzero remainder and its
    # quotient (4) equals sum(offsets) - offsets[3], so dropping the remainder
    # check would wrongly mark row 3 undefined
    push!(cases, ("getis nonzero remainder", Float64[-3.0, -1.0, 1.0, 3.0], binary_W(4, 1)))
    # Moran mask: row 3 has n * offset == sum(offsets) and must be undefined
    push!(cases, ("moran centered zero", Float64[0.0, 1.0, 2.0, 3.0, 4.0], binary_W(5, 1)))
    # offsets share a factor 3 after the power-of-two reduction
    push!(cases, ("post-reduction gcd 3", Float64[0.0, 3.0, 6.0, 9.0], binary_W(4, 1)))
    # reaches the fast path's own 96-bit offset guard (max_bits is only 97, so
    # the width gate lets this through and the guard is what rejects it)
    push!(cases, ("fast-path 96-bit guard", Float64[0.0, 1.0, 2.0^96, 4.0], binary_W(4, 1)))
    # max_bits is exactly 127: admitting it would overflow Int128 in the
    # origin subtraction, so the 126-bit gate must decline
    push!(cases, ("value gate 127 bits",
                  Float64[1.0, 3.0 * 2.0^125, -3.0 * 2.0^125], binary_W(3, 1)))
    # row bound exactly 2^255 under Geary: max offset 2^95, magnitudes summing
    # to 2^65, so 2^65 * (2^95)^2 == 2^255 and the strict `<` must reject it
    let n = 70
        neighs = Vector{Vector{Int64}}(undef, n)
        w = Vector{Vector{Float64}}(undef, n)
        neighs[1] = Int64.(2:68)
        w[1] = vcat([1.0, 1.0], [2.0^e for e in 1:63], [2.0^63, 2.0^63])
        for i in 2:n
            neighs[i] = Int64[mod1(i + 1, n)]
            w[i] = [1.0]
        end
        dat = zeros(n); dat[2] = 2.0^-40; dat[3] = 2.0^55
        push!(cases, ("row bound exactly 2^255", dat,
                      SpatialWeights(Int64(n), neighs, w, Int64.(length.(neighs)), :original)))
    end

    # randomized fuzz with a fixed seed: a tame pool that stays representable
    # and a wild pool that routinely hits the fallback and the guards
    fuzz = StableRNG(777)
    tame = Float64[0.0, -0.0, 1.0, -1.0, 0.5, 0.25, 2.0^-12, 2.0^12, 3.0, -2.25, 7.5]
    wild = Float64[0.0, -0.0, 1.0, -1.0, 2.0^-30, 2.0^30, 5.0e-324, 2.0^90, 2.0^-90,
                   nextfloat(0.0), prevfloat(1.0), -2.25]
    for (tag, pool, reps) in (("tame", tame, 10), ("wild", wild, 10))
        for t in 1:reps
            n = rand(fuzz, 5:18)
            kmax = max(1, min(n - 1, rand(fuzz, 1:4)))
            ks = [rand(fuzz, 0:kmax) for _ in 1:n]
            style = rand(fuzz, (:row, :original))
            dat = [rand(fuzz, Bool) ? randn(fuzz) : rand(fuzz, pool) for _ in 1:n]
            wfun = (i, m) -> begin
                m == 0 && return Float64[]
                style === :row ? fill(1.0 / m, m) :
                    [rand(fuzz, Bool) ? rand(fuzz, pool) : randn(fuzz) for _ in 1:m]
            end
            push!(cases, ("fuzz $tag $t", dat, ragged_W(n, ks, wfun; style = style)))
        end
    end

    # Which internal route an input takes.  The fast path never reports a
    # problem itself: it either builds a payload, returns nothing, or raises the
    # private decline signal, and every caller-visible exception comes from the
    # reference.  `:escaped` would mean that invariant is broken.
    function route(dat, W, stat_code)
        try
            ext._exact_tail_payload_fast(dat, W; stat_code = stat_code) === nothing ?
                :declined : :fast
        catch err
            err isa ext._ExactFastDecline ? :signalled : :escaped
        end
    end

    fast_built = 0
    fell_back = 0
    guarded = 0
    mismatches = String[]
    for (name, dat, W) in cases, stat_code in 1:4
        probe = route(dat, W, stat_code)
        probe === :escaped &&
            push!(mismatches, "$name / stat $stat_code: fast path raised a real exception")
        probe === :declined ? (fell_back += 1) :
            probe === :signalled ? (guarded += 1) : (fast_built += 1)

        fast = try ext._exact_tail_payload(dat, W; stat_code = stat_code)
        catch err; err end
        slow = try ext._exact_tail_payload_reference(dat, W; stat_code = stat_code)
        catch err; err end
        if slow isa Exception || fast isa Exception
            if !(slow isa Exception && fast isa Exception &&
                 typeof(fast) === typeof(slow) &&
                 sprint(showerror, fast) == sprint(showerror, slow))
                push!(mismatches, "$name / stat $stat_code: exception mismatch")
            end
            continue
        end
        a = payload_fields(slow)
        b = payload_fields(fast)
        for (idx, (av, bv)) in enumerate(zip(a, b))
            isequal(av, bv) || push!(mismatches, "$name / stat $stat_code: field $idx")
        end
    end
    @test isempty(mismatches)
    isempty(mismatches) || foreach(m -> @info(m), first(mismatches, 10))
    # The comparison would be vacuous if everything took one route, so pin the
    # fast path down as actually exercised and the fallback as reachable.
    @test fast_built >= 120
    @test fell_back >= 8
    @test guarded >= 8
    @test fast_built + fell_back + guarded == 4 * length(cases)

    # Pin the specific internal decisions the corpus above is there to cover,
    # so a future change cannot quietly route them somewhere else and keep the
    # differential comparison green by never running the fast code at all.
    lookup = Dict(name => (dat, W) for (name, dat, W) in cases)
    expected_routes = [("weight gate 129 bits", 1, :declined),
                       ("weight gate 129 bits", 4, :fast),
                       ("weight gate 116 bits", 1, :fast),
                       ("zero slot with 2^990", 1, :fast),
                       ("getis denominator", 3, :fast),
                       ("getis nonzero remainder", 3, :fast),
                       ("zero slot after 2^990", 1, :fast),
                       ("moran centered zero", 1, :fast),
                       ("post-reduction gcd 3", 1, :fast),
                       ("huge weights", 1, :fast),
                       ("tiny weights", 1, :fast),
                       ("fast-path 96-bit guard", 1, :signalled),
                       ("value gate 127 bits", 1, :declined),
                       ("row bound exactly 2^255", 1, :fast),
                       ("row bound exactly 2^255", 2, :signalled),
                       ("over 64-bit weights", 1, :declined),
                       ("NaN data", 1, :signalled)]
    for (name, stat_code, want) in expected_routes
        dat, W = lookup[name]
        @test route(dat, W, stat_code) === want
    end
    # the 2^255 case must be exactly at the boundary, not merely over it
    let (dat, W) = lookup["row bound exactly 2^255"]
        moran = ext._exact_tail_payload(dat, W; stat_code = 1)
        @test moran.row_bounds[1] == BigInt(1) << 160     # 2^65 * 2^95
        @test_throws ArgumentError ext._exact_tail_payload(dat, W; stat_code = 2)
    end

    # `crand_local_gpu` builds the payload before `prepare_gpu_weights`
    # validates neighbor indices, so a structurally inconsistent SpatialWeights
    # reaches the encoder.  The fast path uses @inbounds on W-derived indices,
    # so it must decline every such graph up front and let the reference decide
    # what to report -- crashing or reading out of bounds is not acceptable.
    malformed = Tuple{String, SpatialWeights}[]
    # nneighs[1] claims two neighbors but only one is listed (reviewer repro)
    push!(malformed, ("short neighbor row",
                      SpatialWeights(Int64(3), Vector{Int64}[[Int64(2)], [Int64(1)], [Int64(1)]],
                                     Vector{Float64}[[1.0, 1.0], [1.0], [1.0]],
                                     Int64[2, 1, 1], :binary)))
    # weights[1] is longer than nneighs[1]: the reference folds the whole weight
    # vector into the row gcd, the fast path would only see the first entry
    push!(malformed, ("long weight row",
                      SpatialWeights(Int64(3), Vector{Int64}[[Int64(2)], [Int64(3)], [Int64(1)]],
                                     Vector{Float64}[[2.0, 3.0], [1.0], [1.0]],
                                     Int64[1, 1, 1], :original)))
    for (nm, bad_index) in (("neighbor above n", Int64(9)), ("neighbor zero", Int64(0)),
                            ("neighbor negative", Int64(-2)))
        push!(malformed, (nm,
              SpatialWeights(Int64(3), Vector{Int64}[[bad_index], [Int64(1)], [Int64(1)]],
                             Vector{Float64}[[1.0], [1.0], [1.0]], Int64[1, 1, 1], :original)))
    end
    push!(malformed, ("short nneighs",
                      SpatialWeights(Int64(3), Vector{Int64}[[Int64(2)], [Int64(1)], [Int64(1)]],
                                     Vector{Float64}[[1.0], [1.0], [1.0]],
                                     Int64[1, 1], :original)))
    push!(malformed, ("short neighs vector",
                      SpatialWeights(Int64(3), Vector{Int64}[[Int64(2)], [Int64(1)]],
                                     Vector{Float64}[[1.0], [1.0], [1.0]],
                                     Int64[1, 1, 1], :original)))
    push!(malformed, ("k above n",
                      SpatialWeights(Int64(2), Vector{Int64}[[Int64(1), Int64(2), Int64(1)], [Int64(1)]],
                                     Vector{Float64}[[1.0, 1.0, 1.0], [1.0]],
                                     Int64[3, 1], :original)))
    malformed_x = [1.0, 2.0, 3.0]
    for (nm, W) in malformed, stat_code in 1:4
        x = fill(1.0, Int(W.n)); x[1:min(3, length(x))] .= malformed_x[1:min(3, length(x))]
        @test route(x, W, stat_code) === :declined
        fast = try ext._exact_tail_payload(x, W; stat_code = stat_code) catch err; err end
        slow = try ext._exact_tail_payload_reference(x, W; stat_code = stat_code) catch err; err end
        if slow isa Exception
            @test fast isa Exception && typeof(fast) === typeof(slow) &&
                  sprint(showerror, fast) == sprint(showerror, slow)
        else
            @test !(fast isa Exception) && isequal(payload_fields(fast), payload_fields(slow))
        end
    end
    # a defined=false mask on the offending row must not let it through either
    for (_, W) in malformed[1:5], stat_code in 1:4
        mask = falses(3)
        @test route(malformed_x, W, stat_code) === :declined
        fast = try ext._exact_tail_payload(malformed_x, W; stat_code = stat_code,
                                           defined = collect(mask)) catch err; err end
        slow = try ext._exact_tail_payload_reference(malformed_x, W; stat_code = stat_code,
                                                     defined = collect(mask)) catch err; err end
        @test typeof(fast) === typeof(slow)
    end
    # Negative nneighs decline too, and now compare cleanly against the
    # reference as well.
    let W = SpatialWeights(Int64(3), Vector{Int64}[Int64[], [Int64(1)], [Int64(1)]],
                           Vector{Float64}[Float64[], [1.0], [1.0]], Int64[-1, 1, 1], :original)
        for stat_code in 1:4
            @test route(malformed_x, W, stat_code) === :declined
            fast = try ext._exact_tail_payload(malformed_x, W; stat_code = stat_code)
                   catch err; err end
            slow = try ext._exact_tail_payload_reference(malformed_x, W; stat_code = stat_code)
                   catch err; err end
            @test typeof(fast) === typeof(slow)
        end
    end

    # `crand_local_gpu` validates W before building the payload, so the public
    # entry point reports the package's own message rather than crashing or
    # leaking a BoundsError from the encoder, and Float32 agrees with Float64.
    let W = malformed[1][2]
        @test typeof(try ext._exact_tail_payload_reference([1.0, 2.0, 3.0], W;
                                                           stat_code = 1)
                     catch err; err end) === BoundsError
        for precision in (Float32, Float64)
            err = try
                localmoran([1.0, 2.0, 3.0], W; permutations = 9, backend = CPU(),
                           precision = precision, seed = 1)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test sprint(showerror, err) ==
                  "ArgumentError: spatial weights are structurally inconsistent: " *
                  "nneighs[i], neighs[i] and weights[i] must agree"
        end
    end
    # Direct unit test of the hoisted validator, including the branch the public
    # statistics never reach because they index the data by neighbor first.
    @test ext._validate_weights_structure(binary_W(4, 1), 4) === nothing
    for (W, expected) in
        ((malformed[3][2], "ArgumentError: neighbor index is outside the weights domain"),
         (malformed[4][2], "ArgumentError: neighbor index is outside the weights domain"),
         (malformed[5][2], "ArgumentError: neighbor index is outside the weights domain"),
         (malformed[1][2], "ArgumentError: spatial weights are structurally inconsistent: " *
                           "nneighs[i], neighs[i] and weights[i] must agree"),
         (malformed[2][2], "ArgumentError: spatial weights are structurally inconsistent: " *
                           "nneighs[i], neighs[i] and weights[i] must agree"),
         (malformed[6][2], "ArgumentError: spatial weights are structurally inconsistent: " *
                           "nneighs[i], neighs[i] and weights[i] must agree"),
         (malformed[7][2], "ArgumentError: spatial weights are structurally inconsistent: " *
                           "nneighs[i], neighs[i] and weights[i] must agree"))
        err = try
            ext._validate_weights_structure(W, Int(W.n))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test sprint(showerror, err) == expected
    end
    let W = SpatialWeights(Int64(3), Vector{Int64}[Int64[], [Int64(1)], [Int64(1)]],
                           Vector{Float64}[Float64[], [1.0], [1.0]], Int64[-1, 1, 1], :original)
        err = try
            ext._validate_weights_structure(W, 3)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test sprint(showerror, err) == "ArgumentError: neighbor cardinalities must be nonnegative"
    end
    # Scan order must match the existing validators: `_checked_edges` inspects
    # every degree before `_gpu_csr_host` looks at any neighbor index, so a
    # graph that is malformed both ways reports the cardinality message.
    let W = SpatialWeights(Int64(3),
                           Vector{Int64}[[Int64(9)], [Int64(1)], Int64[]],
                           Vector{Float64}[[1.0], [1.0], Float64[]],
                           Int64[1, 1, -1], :original)
        err = try
            ext._validate_weights_structure(W, 3)
            nothing
        catch e
            e
        end
        @test sprint(showerror, err) ==
              "ArgumentError: neighbor cardinalities must be nonnegative"
    end
    # An out-of-domain neighbor is rejected upstream of the payload by the
    # statistic itself; what matters here is that Float32 and Float64 agree.
    let W = malformed[3][2]
        errors = map((Float32, Float64)) do precision
            try
                localmoran([1.0, 2.0, 3.0], W; permutations = 9, backend = CPU(),
                           precision = precision, seed = 1)
                nothing
            catch e
                (typeof(e), sprint(showerror, e))
            end
        end
        @test errors[1] !== nothing
        @test errors[1] == errors[2]
    end

    # Negative degrees used to make the reference encoder's CSR edge total
    # under-count while its row offsets kept advancing, so the packer wrote past
    # the end of its buffer into unrelated live objects -- reachable from the
    # public API with precision=Float32, because the payload was built before W
    # was validated.  Both graphs below triggered that; they must now raise the
    # package's own ArgumentError for every statistic and both precisions, and
    # the process must survive.
    wild_a = SpatialWeights(Int64(8),
        Vector{Int64}[Int64[], Int64[], Int64[8, 3, 5], Int64[5, 3], Int64[],
                      Int64[5], Int64[4, 6], Int64[]],
        Vector{Float64}[Float64[], Float64[], [1.1, -0.7, -1.2], [0.8, 0.1],
                        Float64[], [0.76], [-2.2, -0.5], Float64[]],
        Int64[0, 0, 3, 2, 0, 1, 2, -1], :binary)
    wild_b = let n = 2000
        neighs = Vector{Vector{Int64}}(undef, n)
        w = Vector{Vector{Float64}}(undef, n)
        for i in 1:n
            neighs[i] = Int64[mod1(i + 1, n), mod1(i + 2, n)]
            w[i] = [0.5, 0.5]
        end
        nn = fill(Int64(2), n)
        neighs[1] = Int64[]; w[1] = Float64[]; nn[1] = Int64(-5000)
        SpatialWeights(Int64(n), neighs, w, nn, :row)
    end
    for W in (wild_a, wild_b)
        x = randn(StableRNG(3), Int(W.n))
        for statistic in ((x, w; kw...) -> localmoran(x, w; kw...),
                          (x, w; kw...) -> localgeary(x, w; kw...),
                          (x, w; kw...) -> getisord(x, w; star = false, kw...),
                          (x, w; kw...) -> getisord(x, w; star = true, kw...)),
            precision in (Float32, Float64)
            err = try
                statistic(x, W; permutations = 9, backend = CPU(),
                          precision = precision, seed = 1)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test sprint(showerror, err) ==
                  "ArgumentError: neighbor cardinalities must be nonnegative"
        end
    end
    # the reference encoder itself is now bounds-checked on a direct call
    let x = randn(StableRNG(3), 8)
        for stat_code in 1:3
            @test_throws BoundsError ext._exact_tail_payload_reference(x, wild_a;
                                                                       stat_code = stat_code)
        end
        for stat_code in 1:4
            @test route(x, wild_a, stat_code) === :declined
        end
    end
    GC.gc()

    # Non-1-based data declines, so the reference decides what to report.
    let offset_x = OffsetLike([1.0, 2.0, 3.0, 4.0]), W = binary_W(4, 1)
        for stat_code in 1:4
            @test route(offset_x, W, stat_code) === :declined
            fast = try ext._exact_tail_payload(offset_x, W; stat_code = stat_code) catch e; e end
            slow = try ext._exact_tail_payload_reference(offset_x, W; stat_code = stat_code) catch e; e end
            @test typeof(fast) === typeof(slow)
        end
    end

    # Malformed-W differential fuzz.  Negative degrees are included: the
    # reference packer is bounds-checked now, so they raise BoundsError instead
    # of corrupting this process.
    mal_rng = StableRNG(5150)
    malformed_mismatches = 0
    for trial in 1:250
        n = rand(mal_rng, 1:6)
        ks = [rand(mal_rng, 0:3) for _ in 1:n]
        neighs = [Int64[rand(mal_rng, 1:n) for _ in 1:ks[i]] for i in 1:n]
        wts = [Float64[randn(mal_rng) for _ in 1:ks[i]] for i in 1:n]
        nn = Int64.(ks)
        i = rand(mal_rng, 1:n)
        what = rand(mal_rng, 1:6)
        if what == 1
            nn[i] += rand(mal_rng, (-1, -5, 1, 2))
        elseif what == 2
            push!(neighs[i], Int64(rand(mal_rng, (-3, 0, n + 1, n + 7))))
            push!(wts[i], 1.0)
            nn[i] = length(neighs[i])
        elseif what == 3
            push!(wts[i], 1.0)
        elseif what == 4
            push!(neighs[i], Int64(1))
        elseif what == 5
            pop!(nn)
        elseif !isempty(neighs[i])
            neighs[i][1] = Int64(rand(mal_rng, (-1, 0, n + 1)))
        end
        W = SpatialWeights(Int64(n), neighs, wts, nn, rand(mal_rng, (:row, :original)))
        dat = randn(mal_rng, n)
        for stat_code in 1:4
            fast = try ext._exact_tail_payload(dat, W; stat_code = stat_code) catch e; e end
            slow = try ext._exact_tail_payload_reference(dat, W; stat_code = stat_code) catch e; e end
            same = if slow isa Exception
                fast isa Exception && typeof(fast) === typeof(slow) &&
                    sprint(showerror, fast) == sprint(showerror, slow)
            else
                !(fast isa Exception) && isequal(payload_fields(fast), payload_fields(slow))
            end
            same || (malformed_mismatches += 1)
        end
    end
    @test malformed_mismatches == 0

    # The two length gates in the fast path's preamble: both decline, and the
    # reference supplies the message.
    let W = binary_W(3, 1)
        for (dat, defined, expected) in
            ((Float64.(1:64), nothing,
              "ArgumentError: exact-tail data and weights lengths must match"),
             ([1.0, 2.0, 3.0], trues(6),
              "ArgumentError: exact-tail defined mask must match the number of rows"))
            for stat_code in 1:4
                probe = try
                    ext._exact_tail_payload_fast(dat, W; stat_code = stat_code,
                                                 defined = defined === nothing ?
                                                           nothing : collect(defined))
                catch err
                    err isa ext._ExactFastDecline ? :signalled : :escaped
                end
                @test probe === nothing
                err = try
                    ext._exact_tail_payload(dat, W; stat_code = stat_code,
                                            defined = defined === nothing ?
                                                      nothing : collect(defined))
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test sprint(showerror, err) == expected
            end
        end
    end

    # Sensitivity guard: the comparison must notice a one-ulp data change, a
    # dropped sign and a one-step exponent shift.
    sens_x = Float64[0.5, 1.25, -2.5, 3.0, -0.75, 4.0]
    sens_W = ragged_W(6, fill(2, 6), (i, m) -> [1.5, -0.75][1:m])
    base = ext._exact_tail_payload(sens_x, sens_W; stat_code = 1)
    for (tag, mutated) in (("ulp", [nextfloat(sens_x[1]); sens_x[2:end]]),
                           ("sign", [-sens_x[1]; sens_x[2:end]]),
                           ("exponent", [2 * sens_x[1]; sens_x[2:end]]))
        other = ext._exact_tail_payload(mutated, sens_W; stat_code = 1)
        @test !isequal(base.values.values, other.values.values)
    end
    flipped_W = ragged_W(6, fill(2, 6), (i, m) -> [1.5, 0.75][1:m])
    @test !isequal(base.weights.signs,
                   ext._exact_tail_payload(sens_x, flipped_W; stat_code = 1).weights.signs)

    # Independent reconstruction: the payload is only correct if the encoded
    # integers rebuild the original Float64 data and weights exactly, and if
    # the packed observed limbs equal the statistic recomputed from them.
    half = Rational{BigInt}(1, 2)
    pow2(e::Int) = e >= 0 ? Rational{BigInt}(big(2)^e) : half^(-e)
    limbs_to_big(m, i, rows) = sum(BigInt(m[j, i]) << (32 * (j - 1)) for j in 1:rows)
    signed256(m, i) = begin
        u = limbs_to_big(m, i, 8)
        u >= (BigInt(1) << 255) ? u - (BigInt(1) << 256) : u
    end
    for (name, dat, W) in cases[1:20], stat_code in 1:4
        payload = try
            ext._exact_tail_payload(dat, W; stat_code = stat_code)
        catch
            continue
        end
        offs = [limbs_to_big(payload.values.values, i, 3) for i in 1:W.n]
        unit = pow2(payload.values.exponent)
        values_ok = all(Rational{BigInt}(dat[i]) ==
                        (payload.values.origin + payload.values.divisor * offs[i]) * unit
                        for i in 1:W.n)
        weights_ok = true
        observed_ok = true
        for i in 1:W.n
            first_edge = Int(payload.weights.offsets[i])
            wunit = pow2(payload.weights.exponents[i]) * payload.weights.gcds[i]
            base_edge = first_edge + (stat_code == 4 ? 1 : 0)
            expected = BigInt(0)
            for slot in 1:Int(W.nneighs[i])
                edge = base_edge + slot - 1
                magnitude = BigInt(payload.weights.magnitudes[1, edge]) |
                            (BigInt(payload.weights.magnitudes[2, edge]) << 32)
                if stat_code != 4
                    weights_ok &= Rational{BigInt}(W.weights[i][slot]) ==
                                  payload.weights.signs[edge] * magnitude * wunit
                end
                term = stat_code == 2 ? (offs[i] - offs[Int(W.neighs[i][slot])])^2 :
                       offs[Int(W.neighs[i][slot])]
                expected += BigInt(payload.weights.signs[edge]) * magnitude * term
            end
            observed_ok &= signed256(payload.observed, i) == expected
        end
        @test values_ok
        @test weights_ok
        @test observed_ok
    end
end
