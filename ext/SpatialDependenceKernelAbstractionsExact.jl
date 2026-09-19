# Bounded exact-tail support.  The host-side encoder uses BigInt only while
# validating and packing finite Float64 inputs; permutation work will consume
# the packed UInt32 arrays in the kernel.  In particular, this file must not
# become a host permutation fallback.

const _EXACT_U32_MASK = BigInt(0xffffffff)
const _EXACT_U96_LIMIT = BigInt(1) << 96
const _EXACT_U64_LIMIT = BigInt(1) << 64
const _EXACT_S256_LIMIT = BigInt(1) << 255
const _EXACT_U256_MODULUS = BigInt(1) << 256

@inline _exact_u32(value::UInt64) = unsafe_trunc(UInt32, value)

"""Set an eight-limb little-endian UInt256 value to zero."""
@inline function _exact_zero!(a::AbstractVector{UInt32})
    @inbounds for j in 1:8
        a[j] = UInt32(0)
    end
    return a
end

"""Add two UInt256 values modulo 2²⁵⁶, storing the result in `out`."""
@inline function _exact_add!(out::AbstractVector{UInt32}, a::AbstractVector{UInt32},
                             b::AbstractVector{UInt32})
    carry = UInt64(0)
    @inbounds for j in 1:8
        s = UInt64(a[j]) + UInt64(b[j]) + carry
        out[j] = _exact_u32(s & UInt64(0xffffffff))
        carry = s >> 32
    end
    return out
end

"""Negate a UInt256 value in place using two's complement."""
@inline function _exact_negate!(a::AbstractVector{UInt32})
    @inbounds for j in 1:8
        a[j] = ~a[j]
    end
    carry = UInt64(1)
    @inbounds for j in 1:8
        s = UInt64(a[j]) + carry
        a[j] = _exact_u32(s & UInt64(0xffffffff))
        carry = s >> 32
    end
    return a
end

"""Compare signed two's-complement UInt256 values without subtracting them."""
@inline function _exact_signed_cmp(a::AbstractVector{UInt32}, b::AbstractVector{UInt32})
    @inbounds begin
        aneg = (a[8] & UInt32(0x80000000)) != UInt32(0)
        bneg = (b[8] & UInt32(0x80000000)) != UInt32(0)
        aneg != bneg && return aneg ? Int32(-1) : Int32(1)
        for j in 8:-1:1
            a[j] != b[j] && return a[j] < b[j] ? Int32(-1) : Int32(1)
        end
    end
    return Int32(0)
end

"""Multiply two UInt256 values modulo 2²⁵⁶ using UInt64 32×32 products."""
@inline function _exact_mul!(out::AbstractVector{UInt32}, a::AbstractVector{UInt32},
                             b::AbstractVector{UInt32})
    _exact_zero!(out)
    # Terms above limb eight are intentionally discarded.  The exact-tail
    # host proof rejects inputs whose admitted products/sums need those bits.
    @inbounds for i in 1:8
        carry = UInt64(0)
        lastj = 9 - i
        for j in 1:lastj
            limb = i + j - 1
            product = UInt64(a[i]) * UInt64(b[j]) + UInt64(out[limb]) + carry
            out[limb] = _exact_u32(product & UInt64(0xffffffff))
            carry = product >> 32
        end
    end
    return out
end

"""Multiply only the low `A`×`B` limbs, retaining every in-range carry."""
@inline function _exact_mul!(out::AbstractVector{UInt32}, a::AbstractVector{UInt32},
                             b::AbstractVector{UInt32}, ::Val{A}, ::Val{B}) where {A, B}
    _exact_zero!(out)
    @inbounds for i in 1:A
        carry = UInt64(0)
        for j in 1:B
            limb = i + j - 1
            product = UInt64(a[i]) * UInt64(b[j]) + UInt64(out[limb]) + carry
            out[limb] = _exact_u32(product & UInt64(0xffffffff))
            carry = product >> 32
        end
        limb = i + B
        while carry != UInt64(0) && limb <= 8
            sum_limb = UInt64(out[limb]) + carry
            out[limb] = _exact_u32(sum_limb & UInt64(0xffffffff))
            carry = sum_limb >> 32
            limb += 1
        end
    end
    return out
end

@inline function _exact_accum_product!(out::AbstractVector{UInt32}, a::UInt32, b::UInt32,
                                       limb)
    product = UInt64(a) * UInt64(b) + UInt64(out[limb])
    @inbounds out[limb] = _exact_u32(product & UInt64(0xffffffff))
    carry = product >> 32
    if limb < 8
        for q in (limb + 1):8
            carry == UInt64(0) && break
            sum_limb = UInt64(out[q]) + carry
            @inbounds out[q] = _exact_u32(sum_limb & UInt64(0xffffffff))
            carry = sum_limb >> 32
        end
    end
    return out
end

@inline function _exact_mul_3x2!(out, a, b)
    _exact_zero!(out)
    _exact_accum_product!(out, a[1], b[1], 1)
    _exact_accum_product!(out, a[1], b[2], 2)
    _exact_accum_product!(out, a[2], b[1], 2)
    _exact_accum_product!(out, a[2], b[2], 3)
    _exact_accum_product!(out, a[3], b[1], 3)
    _exact_accum_product!(out, a[3], b[2], 4)
    return out
end

@inline function _exact_mul_3x3!(out, a, b)
    _exact_zero!(out)
    for i in 1:3
        _exact_accum_product!(out, a[i], b[1], i)
        _exact_accum_product!(out, a[i], b[2], i + 1)
        _exact_accum_product!(out, a[i], b[3], i + 2)
    end
    return out
end

@inline function _exact_mul_6x2!(out, a, b)
    _exact_zero!(out)
    for i in 1:6
        _exact_accum_product!(out, a[i], b[1], i)
        _exact_accum_product!(out, a[i], b[2], i + 1)
    end
    return out
end

@inline function _exact_load_value!(out::AbstractVector{UInt32}, values, i)
    _exact_zero!(out)
    @inbounds begin
        out[1] = values[1, Int(i)]
        out[2] = values[2, Int(i)]
        out[3] = values[3, Int(i)]
    end
    return out
end

@inline function _exact_load_weight!(out::AbstractVector{UInt32}, magnitudes, slot)
    _exact_zero!(out)
    @inbounds begin
        out[1] = magnitudes[1, Int(slot)]
        out[2] = magnitudes[2, Int(slot)]
    end
    return out
end

@inline function _exact_unsigned_cmp96(a::AbstractVector{UInt32}, b::AbstractVector{UInt32})
    @inbounds for j in 3:-1:1
        a[j] != b[j] && return a[j] < b[j] ? Int32(-1) : Int32(1)
    end
    return Int32(0)
end

@inline function _exact_sub96!(out::AbstractVector{UInt32}, a::AbstractVector{UInt32},
                               b::AbstractVector{UInt32})
    @inbounds for j in 4:8
        out[j] = UInt32(0)
    end
    borrow = UInt64(0)
    @inbounds for j in 1:3
        aj = UInt64(a[j])
        bj = UInt64(b[j]) + borrow
        out[j] = _exact_u32((aj - bj) & UInt64(0xffffffff))
        borrow = aj < bj ? UInt64(1) : UInt64(0)
    end
    return out
end

struct _ExactDyadic
    mantissa::BigInt
    exponent::Int
end

"""Decode a finite Float64 exactly as `mantissa * 2^exponent`."""
function _exact_decode(x::Float64)
    isfinite(x) || throw(ArgumentError("exact accelerated tails require finite Float64 data and weights"))
    bits = reinterpret(UInt64, x)
    sign = (bits >> 63) == UInt64(0) ? BigInt(1) : BigInt(-1)
    fraction = bits & UInt64(0x000fffffffffffff)
    exponent_bits = (bits >> 52) & UInt64(0x7ff)
    if exponent_bits == UInt64(0x7ff)
        throw(ArgumentError("exact accelerated tails require finite Float64 data and weights"))
    elseif exponent_bits == UInt64(0)
        fraction == UInt64(0) && return _ExactDyadic(BigInt(0), 0)
        return _ExactDyadic(sign * BigInt(fraction), -1074)
    else
        mantissa = (UInt64(1) << 52) | fraction
        return _ExactDyadic(sign * BigInt(mantissa), Int(exponent_bits) - 1023 - 52)
    end
end

function _exact_scaled_values(x::AbstractVector{<:Real})
    decoded = _ExactDyadic[_exact_decode(Float64(v)) for v in x]
    nonzero = findall(d -> !iszero(d.mantissa), decoded)
    isempty(nonzero) && return decoded, 0, BigInt(0), BigInt(1), fill(BigInt(0), length(x))
    common_exp = minimum(decoded[i].exponent for i in nonzero)
    scaled = BigInt[d.mantissa << (d.exponent - common_exp) for d in decoded]
    origin = minimum(scaled)
    offsets = BigInt[v - origin for v in scaled]
    # A positive common factor does not affect a linear comparison and only
    # contributes a positive square to Geary.  Removing it makes the bounded
    # 96-bit representation useful for values with shared trailing bits.
    divisor = foldl(gcd, (abs(v) for v in offsets); init = BigInt(0))
    divisor == 0 && (divisor = BigInt(1))
    offsets .÷= divisor
    return decoded, common_exp, origin, divisor, offsets
end

# `column` is derived from the caller's CSR offsets, which are derived from
# `W.nneighs`.  A structurally inconsistent `W` (for instance a negative
# degree, which makes the edge total under-count while the offsets keep
# advancing) can therefore land outside `out`.  This is host-only code and is
# not reachable from any device kernel, so it is bounds-checked: an out-of-range
# column raises `BoundsError` instead of writing over unrelated live objects.
function _exact_pack_unsigned!(out::AbstractMatrix{UInt32}, column::Int,
                               value::BigInt, limbs::Int)
    0 <= value < (BigInt(1) << (32 * limbs)) ||
        throw(ArgumentError("exact accelerated tail input exceeds the bounded UInt$(32 * limbs) representation"))
    for j in 1:limbs
        out[j, column] = UInt32((value >> (32 * (j - 1))) & _EXACT_U32_MASK)
    end
    return out
end

function _exact_pack_signed256(value::BigInt)
    -_EXACT_S256_LIMIT <= value < _EXACT_S256_LIMIT ||
        throw(ArgumentError("exact accelerated tail accumulator exceeds the signed 256-bit bound"))
    encoded = value < 0 ? value + _EXACT_U256_MODULUS : value
    out = zeros(UInt32, 8)
    @inbounds for j in 1:8
        out[j] = UInt32((encoded >> (32 * (j - 1))) & _EXACT_U32_MASK)
    end
    return out
end

struct ExactValueEncoding
    exponent::Int
    origin::BigInt
    divisor::BigInt
    values::Matrix{UInt32} # 3 × n, little-endian nonnegative offsets
end

"""Pack finite values as nonnegative, common-exponent 96-bit offsets."""
function _exact_encode_values(x::AbstractVector{<:Real})
    _, exponent, origin, divisor, offsets = _exact_scaled_values(x)
    values = zeros(UInt32, 3, length(x))
    for i in eachindex(offsets)
        _exact_pack_unsigned!(values, i, offsets[i], 3)
    end
    return ExactValueEncoding(exponent, origin, divisor, values)
end

struct ExactWeightEncoding
    exponents::Vector{Int}
    gcds::Vector{BigInt}
    offsets::Vector{Int64}        # CSR offsets, including Gi* focal slots
    magnitudes::Array{UInt32, 2} # 2 × nnz, little-endian
    signs::Vector{Int8}            # nnz; zero denotes a zero-weight slot
end

"""Encode row weights after positive row-local gcd reduction."""
function _exact_encode_weights(W; stat_code::Integer = 0)
    n = W.n
    degrees = Int[W.nneighs[i] + (stat_code == 4 ? 1 : 0) for i in 1:n]
    total_edges = sum(degrees; init = 0)
    offsets = Vector{Int64}(undef, n + 1)
    offsets[1] = Int64(1)
    for i in 1:n
        offsets[i + 1] = offsets[i] + Int64(degrees[i])
    end
    exponents = zeros(Int, n)
    gcds = fill(BigInt(1), n)
    magnitudes = zeros(UInt32, 2, total_edges)
    signs = zeros(Int8, total_edges)
    for i in 1:n
        k = Int(W.nneighs[i])
        first_edge = Int(offsets[i])
        if stat_code == 4
            # Carry the focal slot in the same ragged row as the neighbors.
            # The common positive factor is irrelevant to exact comparisons.
            exponents[i] = 0
            gcds[i] = BigInt(1)
            for slot in 0:k
                edge = first_edge + slot
                _exact_pack_unsigned!(magnitudes, edge, BigInt(1), 2)
                signs[edge] = Int8(1)
            end
            continue
        end
        k == 0 && continue
        decoded = _ExactDyadic[_exact_decode(Float64(w)) for w in W.weights[i]]
        common_exp = minimum(d.exponent for d in decoded)
        scaled = BigInt[d.mantissa << (d.exponent - common_exp) for d in decoded]
        divisor = foldl(gcd, (abs(v) for v in scaled); init = BigInt(0))
        # A zero-weight row is valid input.  Keep its zero slots zero and use
        # a neutral scale for the (unused) row-local exponent.
        divisor == 0 && (divisor = BigInt(1))
        exponents[i] = common_exp
        gcds[i] = divisor
        for slot in 1:k
            value = scaled[slot] ÷ divisor
            magnitude = abs(value)
            edge = first_edge + slot - 1
            _exact_pack_unsigned!(magnitudes, edge, magnitude, 2)
            signs[edge] = value == 0 ? Int8(0) : (value < 0 ? Int8(-1) : Int8(1))
        end
    end
    return ExactWeightEncoding(exponents, gcds, offsets, magnitudes, signs)
end

struct ExactTailPayload
    values::ExactValueEncoding
    weights::ExactWeightEncoding
    observed::Matrix{UInt32} # 8 × n signed two's-complement limbs
    defined::BitVector
    row_bounds::Vector{BigInt}
end

@inline function _exact_value_at(values::ExactValueEncoding, i::Int)
    return BigInt(values.values[1, i]) |
           (BigInt(values.values[2, i]) << 32) |
           (BigInt(values.values[3, i]) << 64)
end

function _exact_raw_scaled_values(x::AbstractVector{<:Real})
    decoded = _ExactDyadic[_exact_decode(Float64(v)) for v in x]
    nonzero = findall(d -> !iszero(d.mantissa), decoded)
    isempty(nonzero) && return fill(BigInt(0), length(x))
    common_exp = minimum(decoded[i].exponent for i in nonzero)
    return BigInt[d.mantissa << (d.exponent - common_exp) for d in decoded]
end

function _exact_observed_accumulators(values::ExactValueEncoding,
                                      weights::ExactWeightEncoding, W,
                                      stat_code::Integer)
    n = W.n
    observed = zeros(BigInt, n)
    for i in 1:n
    xi = _exact_value_at(values, i)
        first_edge = Int(weights.offsets[i])
        last_edge = Int(weights.offsets[i + 1]) - 1
        if stat_code == 2
            total = BigInt(0)
            for edge in first_edge:last_edge
                sign = Int(weights.signs[edge])
                sign == 0 && continue
                slot = edge - first_edge + 1
                # The Gi* focal slot is not used by Geary, but this branch is
                # never called for stat_code == 2 with such a payload.
                j = Int(W.neighs[i][slot])
                d = xi - _exact_value_at(values, j)
                magnitude = BigInt(weights.magnitudes[1, edge]) |
                            (BigInt(weights.magnitudes[2, edge]) << 32)
                total += sign * magnitude * d * d
            end
            observed[i] = total
        elseif stat_code == 1 || stat_code == 3 || stat_code == 4
            total = BigInt(0)
            first_neighbor_edge = first_edge + (stat_code == 4 ? 1 : 0)
            for edge in first_neighbor_edge:last_edge
                sign = Int(weights.signs[edge])
                sign == 0 && continue
                slot = edge - first_edge + 1
                j = Int(W.neighs[i][slot - (stat_code == 4 ? 1 : 0)])
                magnitude = BigInt(weights.magnitudes[1, edge]) |
                            (BigInt(weights.magnitudes[2, edge]) << 32)
                total += sign * magnitude * _exact_value_at(values, j)
            end
            observed[i] = total
        end
    end
    return observed
end

function _exact_check_row_bounds(values::ExactValueEncoding, weights::ExactWeightEncoding;
                                 geary::Bool = false,
                                 cardinalities = nothing)
    max_x = isempty(values.values) ? BigInt(0) :
        maximum(BigInt(values.values[1, i]) +
                (BigInt(values.values[2, i]) << 32) +
                (BigInt(values.values[3, i]) << 64) for i in axes(values.values, 2))
    bounds = zeros(BigInt, length(weights.offsets) - 1)
    for i in eachindex(bounds)
        first_edge = Int(weights.offsets[i])
        last_edge = Int(weights.offsets[i + 1]) - 1
        total = BigInt(0)
        for edge in first_edge:last_edge
            magnitude = BigInt(weights.magnitudes[1, edge]) |
                        (BigInt(weights.magnitudes[2, edge]) << 32)
            total += magnitude
        end
        range_term = geary ? max_x * max_x : max_x
        bounds[i] = total * range_term
        bounds[i] < _EXACT_S256_LIMIT ||
            throw(ArgumentError("exact accelerated tail row $i exceeds the signed 256-bit accumulation bound"))
    end
    return bounds
end

"""
Reference (BigInt) exact-tail payload builder.

This is the authoritative definition of the payload.  `_exact_tail_payload`
runs a machine-integer fast path and falls back to this function verbatim
whenever the fast path cannot prove that its fixed-width intermediates are
exact.  Keep this implementation unchanged; the fast path is validated
against it.
"""
function _exact_tail_payload_reference(data::AbstractVector{<:Real}, W;
                             stat_code::Integer = 1,
                             observed::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                             defined::Union{Nothing, AbstractVector{Bool}} = nothing)
    length(data) == W.n || throw(ArgumentError("exact-tail data and weights lengths must match"))
    values = _exact_encode_values(data)
    weights = _exact_encode_weights(W; stat_code=stat_code)
    row_bounds = _exact_check_row_bounds(values, weights;
                                         geary=stat_code == 2,
                                         cardinalities=W.nneighs)
    # The centered Moran factor and Getis denominator only affect comparison
    # direction.  The common mean/reference terms cancel between observed and
    # permuted linear sums, so the device can compare packed weighted sums.
    offsets = [_exact_value_at(values, i) for i in 1:W.n]
    centered_sum = sum(offsets; init=BigInt(0))
    raw_scaled = _exact_raw_scaled_values(data)
    raw_total = sum(raw_scaled; init=BigInt(0))
    defined_default = trues(W.n)
    for i in 1:W.n
        W.nneighs[i] == 0 && (defined_default[i] = false)
        if stat_code == 1
            centered = BigInt(W.n) * offsets[i] - centered_sum
            iszero(centered) && (defined_default[i] = false)
        elseif stat_code == 3
            denominator = raw_total - raw_scaled[i]
            iszero(denominator) && (defined_default[i] = false)
        elseif stat_code == 4
            iszero(raw_total) && (defined_default[i] = false)
        elseif stat_code == 2
        end
    end
    if stat_code == 2 && !isempty(offsets) && all(==(offsets[1]), offsets)
        defined_default .= false
    end
    obs = observed === nothing ? _exact_observed_accumulators(values, weights, W, stat_code) : BigInt.(observed)
    length(obs) == W.n || throw(ArgumentError("exact-tail observed values must match the number of rows"))
    packed_obs = zeros(UInt32, 8, W.n)
    for i in 1:W.n
        packed_obs[:, i] .= _exact_pack_signed256(obs[i])
    end
    mask = defined === nothing ? BitVector(defined_default) : BitVector(defined)
    length(mask) == W.n || throw(ArgumentError("exact-tail defined mask must match the number of rows"))
    for i in 1:W.n
        if mask[i] && abs(obs[i]) > row_bounds[i]
            throw(ArgumentError("exact-tail observed value exceeds the proven row bound at row $i"))
        end
    end
    return ExactTailPayload(values, weights, packed_obs, mask, row_bounds)
end

# ===========================================================================
# Machine-integer fast path
# ===========================================================================
#
# The reference builder above is exact but spends a BigInt allocation on every
# edge.  The fast path below reproduces its output bit for bit using fixed
# width machine integers.
#
# The fast path never reports a problem itself.  It *declines*: it returns
# `nothing`, or raises the private `_ExactFastDecline`, and `_exact_tail_payload`
# then reruns the reference from scratch.  That covers both "I cannot prove
# these fixed-width intermediates are exact" and "this input violates a bound",
# so every exception a caller can observe is raised by the reference, in the
# reference's order, with the reference's type and message, by construction.
# Every bound used here is stated next to the code that relies on it.

"""
Private signal that the fast path is giving up on this input.

It never escapes `_exact_tail_payload`, which converts it into a full rerun of
`_exact_tail_payload_reference`.  It is raised (rather than returned) only
where a plain return would have to be threaded through an inner loop.
"""
struct _ExactFastDecline <: Exception end

# A finite Float64 is decoded to (sign, odd mantissa, reduced exponent, raw
# exponent) with `value == sign * mantissa * 2^reduced`.  `raw` is the
# exponent the reference's `_exact_decode` reports (mantissa not reduced);
# `reduced` is `raw + trailing_zeros(mantissa)`.  Zero (either sign) decodes
# to `(0, 0, 0, 0)`, matching `_ExactDyadic(BigInt(0), 0)`.
@inline function _exact_split(x::Float64)
    bits = reinterpret(UInt64, x)
    exponent_bits = (bits >> 52) & UInt64(0x7ff)
    # Non-finite input is the reference's error to report, not ours.
    exponent_bits == UInt64(0x7ff) && throw(_ExactFastDecline())
    fraction = bits & UInt64(0x000fffffffffffff)
    local mantissa::UInt64, raw::Int
    if exponent_bits == UInt64(0)
        fraction == UInt64(0) && return (Int8(0), UInt64(0), 0, 0)
        mantissa = fraction
        raw = -1074
    else
        mantissa = (UInt64(1) << 52) | fraction
        raw = Int(exponent_bits) - 1075
    end
    tz = trailing_zeros(mantissa)
    sign = (bits >> 63) == UInt64(0) ? Int8(1) : Int8(-1)
    return (sign, mantissa >> tz, raw + tz, raw)
end

# Widths at which the fast path still has provable headroom.  Values are
# combined with a subtraction (origin removal), so they keep one extra bit.
const _EXACT_FAST_VALUE_BITS = 126
const _EXACT_FAST_WEIGHT_BITS = 128
# `n` is bounded so that `n * offset < 2^127` for every offset < 2^96, which
# keeps the defined-mask arithmetic inside UInt128.
const _EXACT_FAST_MAX_N = Int(typemax(Int32))

@inline _exact_bitlen(m::UInt64) = 64 - leading_zeros(m)

@inline _exact_lo64(v::UInt128) = UInt64(v & UInt128(typemax(UInt64)))
@inline _exact_hi64(v::UInt128) = UInt64(v >> 64)

"""Split a 96-bit unsigned offset into its two UInt64 words."""
@inline _exact_words96(v::UInt128) = (_exact_lo64(v), _exact_hi64(v))

# ---------------------------------------------------------------- UInt256 --

@inline function _exact256_add(a::NTuple{4, UInt64}, b::NTuple{4, UInt64})
    t = UInt128(a[1]) + UInt128(b[1]); r1 = _exact_lo64(t); c = _exact_hi64(t)
    t = UInt128(a[2]) + UInt128(b[2]) + UInt128(c); r2 = _exact_lo64(t); c = _exact_hi64(t)
    t = UInt128(a[3]) + UInt128(b[3]) + UInt128(c); r3 = _exact_lo64(t); c = _exact_hi64(t)
    t = UInt128(a[4]) + UInt128(b[4]) + UInt128(c); r4 = _exact_lo64(t)
    return (r1, r2, r3, r4)
end

@inline function _exact256_negate(a::NTuple{4, UInt64})
    t = UInt128(~a[1]) + UInt128(1); r1 = _exact_lo64(t); c = _exact_hi64(t)
    t = UInt128(~a[2]) + UInt128(c); r2 = _exact_lo64(t); c = _exact_hi64(t)
    t = UInt128(~a[3]) + UInt128(c); r3 = _exact_lo64(t); c = _exact_hi64(t)
    t = UInt128(~a[4]) + UInt128(c); r4 = _exact_lo64(t)
    return (r1, r2, r3, r4)
end

@inline _exact256_isneg(a::NTuple{4, UInt64}) = (a[4] & (UInt64(1) << 63)) != UInt64(0)

@inline function _exact256_gt(a::NTuple{4, UInt64}, b::NTuple{4, UInt64})
    @inbounds for j in 4:-1:1
        a[j] != b[j] && return a[j] > b[j]
    end
    return false
end

# (lo, hi) is a 96-bit unsigned value (hi < 2^32) and `m < 2^64`, so the
# product is < 2^160 and occupies three words exactly.
@inline function _exact256_mul_96x64(lo::UInt64, hi::UInt64, m::UInt64)
    p0 = UInt128(lo) * UInt128(m)
    r1 = _exact_lo64(p0)
    t = UInt128(hi) * UInt128(m) + UInt128(_exact_hi64(p0))
    r2 = _exact_lo64(t)
    r3 = _exact_hi64(t)
    return (r1, r2, r3, UInt64(0))
end

# Square of a 96-bit unsigned value: < 2^192, so three words suffice.
@inline function _exact256_square96(lo::UInt64, hi::UInt64)
    p00 = UInt128(lo) * UInt128(lo)
    p01 = UInt128(lo) * UInt128(hi)          # < 2^96
    p11 = UInt128(hi) * UInt128(hi)          # < 2^64
    r1 = _exact_lo64(p00)
    t = UInt128(_exact_hi64(p00)) + (p01 << 1)   # p01 << 1 < 2^97, no overflow
    r2 = _exact_lo64(t)
    t2 = p11 + UInt128(_exact_hi64(t))
    r3 = _exact_lo64(t2)
    r4 = _exact_lo64(t2 >> 64)                   # zero because the square < 2^192
    return (r1, r2, r3, r4)
end

# Full 256x64 product truncated to 256 bits.  Callers only use it where the
# mathematical product has been proven < 2^255, so truncation is a no-op.
@inline function _exact256_mul_small(a::NTuple{4, UInt64}, m::UInt64)
    t = UInt128(a[1]) * UInt128(m); r1 = _exact_lo64(t); c = _exact_hi64(t)
    t = UInt128(a[2]) * UInt128(m) + UInt128(c); r2 = _exact_lo64(t); c = _exact_hi64(t)
    t = UInt128(a[3]) * UInt128(m) + UInt128(c); r3 = _exact_lo64(t); c = _exact_hi64(t)
    t = UInt128(a[4]) * UInt128(m) + UInt128(c); r4 = _exact_lo64(t)
    return (r1, r2, r3, r4)
end

@inline function _exact256_store!(out::AbstractMatrix{UInt32}, column::Int,
                                  a::NTuple{4, UInt64})
    @inbounds for j in 1:4
        out[2 * j - 1, column] = UInt32(a[j] & UInt64(0xffffffff))
        out[2 * j, column] = UInt32(a[j] >> 32)
    end
    return out
end

"""Shift a UInt256 left by one 64-bit word, discarding the top word."""
@inline _exact256_shl64(a::NTuple{4, UInt64}) = (UInt64(0), a[1], a[2], a[3])

"""
One row bound, as the exact BigInt *and* as 256-bit words.

`total_big` must equal `total`; the BigInt product is the definition of the
bound and the machine-integer product is a truncating 256-bit multiply, which
is only exact because the BigInt value has just been proven below 2^255.  The
proof and the truncating multiply deliberately live in this one function so a
later edit cannot separate them: there is no way to obtain the word form
without going through the check.  A row that fails the check declines, and the
reference reruns and reports it.
"""
@inline function _exact_row_bound(total::UInt128, total_big::BigInt, range_big::BigInt,
                                  range_words::NTuple{4, UInt64})
    bound = total_big * range_big
    bound < _EXACT_S256_LIMIT || throw(_ExactFastDecline())
    low = _exact256_mul_small(range_words, _exact_lo64(total))
    high = _exact_hi64(total)
    limbs = high == UInt64(0) ? low :
        _exact256_add(low, _exact256_shl64(_exact256_mul_small(range_words, high)))
    return bound, limbs
end

# ---------------------------------------------------------------- values ---

"""
Fast value encoder.  Returns `(encoding, offsets, max_offset)`, or declines
(`nothing`, or `_ExactFastDecline` for an out-of-range offset) when the
fixed-width representation cannot be proven exact or the input is out of
bounds.

The reference forms `scaled_i = mantissa_i << (exp_i - common_exp)` with
`common_exp` the minimum *raw* exponent over nonzero entries, subtracts the
minimum, and divides by the gcd.  Here the same quantities are built over the
*reduced* unit `2^u`, `u = min reduced exponent over nonzero entries`.  Since
`u >= common_exp` and every `scaled_i` is divisible by `2^(u - common_exp)`,
the reference values are exactly `2^(u - common_exp)` times the ones computed
here; origin and divisor are shifted back at the end and the final offsets
(`(scaled - origin) / gcd`) are invariant under that common positive factor.
"""
function _exact_encode_values_fast(x::AbstractVector{<:Real})
    n = length(x)
    signs = Vector{Int8}(undef, n)
    mantissas = Vector{UInt64}(undef, n)
    reduced = Vector{Int}(undef, n)
    raw_min = 0
    reduced_min = 0
    any_nonzero = false
    @inbounds for i in 1:n
        s, m, er, ew = _exact_split(Float64(x[i]))
        signs[i] = s; mantissas[i] = m; reduced[i] = er
        s == Int8(0) && continue
        if !any_nonzero
            any_nonzero = true; raw_min = ew; reduced_min = er
        else
            ew < raw_min && (raw_min = ew)
            er < reduced_min && (reduced_min = er)
        end
    end
    if !any_nonzero
        # Matches the reference early return: exponent 0, origin 0, divisor 1.
        return (ExactValueEncoding(0, BigInt(0), BigInt(1), zeros(UInt32, 3, n)),
                zeros(UInt128, n), UInt128(0))
    end
    max_bits = 0
    @inbounds for i in 1:n
        signs[i] == Int8(0) && continue
        bits = (reduced[i] - reduced_min) + _exact_bitlen(mantissas[i])
        bits > max_bits && (max_bits = bits)
    end
    # |scaled| < 2^126 => scaled - origin fits Int128 (< 2^127) and offsets
    # fit UInt128 with room to spare.
    max_bits > _EXACT_FAST_VALUE_BITS && return nothing
    scaled = Vector{Int128}(undef, n)
    origin_small = typemax(Int128)
    @inbounds for i in 1:n
        v = signs[i] == Int8(0) ? Int128(0) :
            Int128(signs[i]) * (Int128(mantissas[i]) << (reduced[i] - reduced_min))
        scaled[i] = v
        v < origin_small && (origin_small = v)
    end
    offsets = Vector{UInt128}(undef, n)
    divisor_small = UInt128(0)
    @inbounds for i in 1:n
        d = UInt128(scaled[i] - origin_small)
        offsets[i] = d
        divisor_small == UInt128(1) || (divisor_small = gcd(divisor_small, d))
    end
    shift = reduced_min - raw_min           # >= 0 because reduced_i >= raw_i
    origin = BigInt(origin_small) << shift
    # The reference replaces a zero gcd (all offsets zero) by 1 *after* the
    # scaling, so that case must not pick up the 2^shift factor.
    divisor = divisor_small == UInt128(0) ? BigInt(1) : (BigInt(divisor_small) << shift)
    step = divisor_small == UInt128(0) ? UInt128(1) : divisor_small
    values = Matrix{UInt32}(undef, 3, n)
    max_offset = UInt128(0)
    @inbounds for i in 1:n
        o = offsets[i] ÷ step
        offsets[i] = o
        (o >> 96) == UInt128(0) || throw(_ExactFastDecline())
        values[1, i] = UInt32(o & UInt128(0xffffffff))
        values[2, i] = UInt32((o >> 32) & UInt128(0xffffffff))
        values[3, i] = UInt32((o >> 64) & UInt128(0xffffffff))
        o > max_offset && (max_offset = o)
    end
    return (ExactValueEncoding(raw_min, origin, divisor, values), offsets, max_offset)
end

# --------------------------------------------------------------- weights ---

@inline function _exact_store_magnitude!(magnitudes::AbstractMatrix{UInt32}, edge::Int,
                                         magnitude::UInt64)
    @inbounds magnitudes[1, edge] = UInt32(magnitude & UInt64(0xffffffff))
    @inbounds magnitudes[2, edge] = UInt32(magnitude >> 32)
    return nothing
end

"""
Reduce one row of weights in `T` (UInt64 or UInt128) arithmetic.

`buffer[slot]` holds `mantissa_slot << (reduced_slot - reduced_min)`, which the
caller has proven fits in `T`.  The row gcd of those magnitudes equals the
reference row gcd divided by `2^(reduced_min - raw_min)`, so the reduced
per-edge magnitudes are identical to the reference's.
"""
@inline function _exact_row_weights!(magnitudes::AbstractMatrix{UInt32},
                                     signs::Vector{Int8}, first_edge::Int, k::Int,
                                     row_signs::Vector{Int8}, buffer::Vector{T}) where {T <: Unsigned}
    divisor = T(0)
    @inbounds for slot in 1:k
        row_signs[slot] == Int8(0) && continue
        divisor == T(1) || (divisor = gcd(divisor, buffer[slot]))
    end
    divisor == T(0) && (divisor = T(1))
    @inbounds for slot in 1:k
        row_signs[slot] == Int8(0) && continue
        value = buffer[slot] ÷ divisor
        (value >> 64) == T(0) || throw(_ExactFastDecline())
        _exact_store_magnitude!(magnitudes, first_edge + slot - 1, UInt64(value))
        signs[first_edge + slot - 1] = row_signs[slot]
    end
    return divisor
end

"""
Fast weight encoder.  Returns an `ExactWeightEncoding`, or declines (`nothing`
when a row's magnitudes cannot be proven to fit in UInt128, `_ExactFastDecline`
when a reduced magnitude does not fit the payload's 64 bits).

Every `W` access here is `@inbounds`; `_exact_weights_consistent` has already
established that `nneighs[i] == length(neighs[i]) == length(weights[i])`.
"""
function _exact_encode_weights_fast(W, stat_code::Integer)
    n = Int(W.n)
    star = stat_code == 4
    offsets = Vector{Int64}(undef, n + 1)
    offsets[1] = Int64(1)
    max_k = 0
    @inbounds for i in 1:n
        k = Int(W.nneighs[i])
        k > max_k && (max_k = k)
        offsets[i + 1] = offsets[i] + Int64(k + (star ? 1 : 0))
    end
    total_edges = Int(offsets[n + 1]) - 1
    exponents = zeros(Int, n)
    gcds = fill(BigInt(1), n)
    magnitudes = zeros(UInt32, 2, total_edges)
    signs = zeros(Int8, total_edges)
    row_signs = Vector{Int8}(undef, max_k)
    row_reduced = Vector{Int}(undef, max_k)
    buffer64 = Vector{UInt64}(undef, max_k)
    buffer128 = Vector{UInt128}(undef, max_k)
    @inbounds for i in 1:n
        k = Int(W.nneighs[i])
        first_edge = Int(offsets[i])
        if star
            # Mirrors the reference: the focal slot and every neighbor slot
            # carry magnitude one, with a neutral row exponent and gcd.  The
            # weight values are never decoded, so they are never validated.
            exponents[i] = 0
            gcds[i] = BigInt(1)
            for slot in 0:k
                edge = first_edge + slot
                magnitudes[1, edge] = UInt32(1)
                signs[edge] = Int8(1)
            end
            continue
        end
        k == 0 && continue
        row = W.weights[i]
        raw_min = 0
        reduced_min = 0
        any_nonzero = false
        max_bits = 0
        for slot in 1:k
            s, m, er, ew = _exact_split(Float64(row[slot]))
            row_signs[slot] = s
            row_reduced[slot] = er
            buffer64[slot] = m
            # The reference takes the row exponent over *all* decoded weights,
            # and a zero weight decodes with exponent 0.
            slot == 1 ? (raw_min = ew) : (ew < raw_min && (raw_min = ew))
            s == Int8(0) && continue
            if !any_nonzero
                any_nonzero = true; reduced_min = er
            elseif er < reduced_min
                reduced_min = er
            end
        end
        if !any_nonzero
            exponents[i] = raw_min
            gcds[i] = BigInt(1)
            continue
        end
        for slot in 1:k
            row_signs[slot] == Int8(0) && continue
            bits = (row_reduced[slot] - reduced_min) + _exact_bitlen(buffer64[slot])
            bits > max_bits && (max_bits = bits)
        end
        max_bits > _EXACT_FAST_WEIGHT_BITS && return nothing
        shift = reduced_min - raw_min       # >= 0
        exponents[i] = raw_min
        if max_bits <= 64
            for slot in 1:k
                row_signs[slot] == Int8(0) && continue
                buffer64[slot] = buffer64[slot] << (row_reduced[slot] - reduced_min)
            end
            divisor = _exact_row_weights!(magnitudes, signs, first_edge, k,
                                          row_signs, buffer64)
            gcds[i] = shift == 0 ? BigInt(divisor) : (BigInt(divisor) << shift)
        else
            for slot in 1:k
                row_signs[slot] == Int8(0) && continue
                buffer128[slot] = UInt128(buffer64[slot]) << (row_reduced[slot] - reduced_min)
            end
            divisor = _exact_row_weights!(magnitudes, signs, first_edge, k,
                                          row_signs, buffer128)
            gcds[i] = shift == 0 ? BigInt(divisor) : (BigInt(divisor) << shift)
        end
    end
    return ExactWeightEncoding(exponents, gcds, offsets, magnitudes, signs)
end

# ------------------------------------------------------------ row bounds ---

"""
Fast row-bound proof.  Returns `(bounds, bound_limbs)`; `bounds` holds the
exact BigInts the reference produces and `bound_limbs` the same values as
256-bit words.  Both come from `_exact_row_bound`, which declines any row at or
above 2^255 before producing the word form.
"""
function _exact_row_bounds_fast(weights::ExactWeightEncoding, max_offset::UInt128,
                                geary::Bool)
    nrow = length(weights.offsets) - 1
    bounds = Vector{BigInt}(undef, nrow)
    bound_limbs = Vector{NTuple{4, UInt64}}(undef, nrow)
    range_big = geary ? BigInt(max_offset) * BigInt(max_offset) : BigInt(max_offset)
    # `max_offset < 2^96`, so the range term is below 2^192 and occupies at
    # most three of the four 64-bit words.
    lo, hi = _exact_words96(max_offset)
    range_words = geary ? _exact256_square96(lo, hi) : (lo, hi, UInt64(0), UInt64(0))
    magnitudes = weights.magnitudes
    last_total = UInt128(0)
    last_total_big = BigInt(0)
    @inbounds for i in 1:nrow
        first_edge = Int(weights.offsets[i])
        last_edge = Int(weights.offsets[i + 1]) - 1
        # Each magnitude is < 2^64 and a row holds at most n <= 2^31 edges
        # (n+1 for Gi*), so the total is < 2^96 and UInt128 cannot overflow.
        total = UInt128(0)
        for edge in first_edge:last_edge
            total += UInt128(magnitudes[1, edge]) | (UInt128(magnitudes[2, edge]) << 32)
        end
        # Rows usually repeat their magnitude total (one distinct row-standardised
        # weight per degree), so the BigInt conversion is memoised on the last one.
        if i == 1 || total != last_total
            last_total = total
            last_total_big = BigInt(total)
        end
        bounds[i], bound_limbs[i] = _exact_row_bound(total, last_total_big,
                                                     range_big, range_words)
    end
    return bounds, bound_limbs
end

# --------------------------------------------------------------- observed --

"""
Accumulate the observed statistics into 256-bit two's-complement words.

`_exact_row_bounds_fast` has already proven `sum(|w|) * max_offset^(1 or 2) <
2^255` for every row, and each term of the row sum is bounded by that same
product, so neither the individual terms nor any partial sum can leave the
signed 256-bit window; wrapping addition is therefore exact.

Every `W`-derived index used here (`neighs[...]`, and `offsets[j]` for a
neighbor `j`) is `@inbounds`; `_exact_weights_consistent` has already
established that each row lists exactly `nneighs[i]` neighbors and that every
one of them is in `1:n`.
"""
function _exact_observed_fast!(packed::Matrix{UInt32}, bound_limbs, defined,
                               weights::ExactWeightEncoding, offsets::Vector{UInt128},
                               W, stat_code::Integer)
    n = Int(W.n)
    magnitudes = weights.magnitudes
    signs = weights.signs
    star = stat_code == 4
    @inbounds for i in 1:n
        first_edge = Int(weights.offsets[i])
        last_edge = Int(weights.offsets[i + 1]) - 1
        acc = (UInt64(0), UInt64(0), UInt64(0), UInt64(0))
        if stat_code == 2
            xi = offsets[i]
            neighs = W.neighs[i]
            for edge in first_edge:last_edge
                sign = signs[edge]
                sign == Int8(0) && continue
                j = Int(neighs[edge - first_edge + 1])
                diff = Int128(xi) - Int128(offsets[j])
                magnitude = abs(diff) % UInt128
                lo, hi = _exact_words96(magnitude)
                term = _exact256_mul_small(_exact256_square96(lo, hi),
                                           UInt64(magnitudes[1, edge]) |
                                           (UInt64(magnitudes[2, edge]) << 32))
                sign < Int8(0) && (term = _exact256_negate(term))
                acc = _exact256_add(acc, term)
            end
        elseif stat_code == 1 || stat_code == 3 || star
            neighs = W.neighs[i]
            base = first_edge + (star ? 1 : 0)
            for edge in base:last_edge
                sign = signs[edge]
                sign == Int8(0) && continue
                j = Int(neighs[edge - base + 1])
                lo, hi = _exact_words96(offsets[j])
                term = _exact256_mul_96x64(lo, hi,
                                           UInt64(magnitudes[1, edge]) |
                                           (UInt64(magnitudes[2, edge]) << 32))
                sign < Int8(0) && (term = _exact256_negate(term))
                acc = _exact256_add(acc, term)
            end
        end
        _exact256_store!(packed, i, acc)
        if defined[i]
            magnitude = _exact256_isneg(acc) ? _exact256_negate(acc) : acc
            _exact256_gt(magnitude, bound_limbs[i]) && throw(_ExactFastDecline())
        end
    end
    return packed
end

# ---------------------------------------------------------------- defined --

"""
Fast defined mask.

With `origin` and `divisor` from the value encoding the reference's raw scaled
values are exactly `origin + divisor * offset_i`, so `raw_total` is
`n * origin + divisor * S` with `S = sum(offsets)`.  The Getis denominator
`raw_total - raw_scaled_i` is therefore zero exactly when
`divisor * (S - offset_i) == -(n - 1) * origin`, which is decided with one
BigInt division and `n` UInt128 comparisons.
"""
function _exact_defined_fast(W, stat_code::Integer, offsets::Vector{UInt128},
                             origin::BigInt, divisor::BigInt)
    n = Int(W.n)
    total = UInt128(0)                      # < n * 2^96 <= 2^127
    @inbounds for i in 1:n
        total += offsets[i]
    end
    mask = trues(n)
    @inbounds for i in 1:n
        W.nneighs[i] == 0 && (mask[i] = false)
    end
    if stat_code == 1
        nn = UInt128(n)
        @inbounds for i in 1:n
            nn * offsets[i] == total && (mask[i] = false)
        end
    elseif stat_code == 3
        target = -(BigInt(n) - 1) * origin
        if target >= 0
            quotient, remainder = divrem(target, divisor)
            if iszero(remainder) && quotient <= BigInt(typemax(UInt128))
                needle = UInt128(quotient)
                @inbounds for i in 1:n
                    (total - offsets[i]) == needle && (mask[i] = false)
                end
            end
        end
    elseif stat_code == 4
        iszero(BigInt(n) * origin + divisor * BigInt(total)) && (mask .= false)
    end
    if stat_code == 2 && n > 0 && all(==(offsets[1]), offsets)
        mask .= false
    end
    return mask
end

# ------------------------------------------------------------------ entry --

"""
Classify how (or whether) a `SpatialWeights` is structurally malformed.

Returns `:ok`, `:negative_degree`, `:inconsistent` (a length disagreement among
`n`, `nneighs[i]`, `neighs[i]` and `weights[i]`) or `:neighbor_domain` (a
neighbor index outside `1:n`).

This is the single place that decides what "malformed" means for the
accelerated path.  `_validate_weights_structure` turns it into the package's
user-facing exceptions before any payload is built, and
`_exact_weights_consistent` uses it as the precondition for the fast path's
`@inbounds` accesses.

Degrees *above* `n` are deliberately not a defect here.  `crand_local_gpu`
already rejects those with its own `max_k <= n - 1` message, and a direct
`_exact_tail_payload` call should decline rather than reject them.

The scan order mirrors the existing validators so the same graph produces the
same message it did before: every degree first (as `_checked_edges` does), then
row lengths and neighbor indices in row order (as `_gpu_csr_host` does).
"""
function _exact_weights_defect(W, n::Int)
    nneighs = W.nneighs
    neighs = W.neighs
    weights = W.weights
    (length(nneighs) == n && length(neighs) == n && length(weights) == n) ||
        return :inconsistent
    for i in 1:n
        nneighs[i] >= 0 || return :negative_degree
    end
    for i in 1:n
        k = Int(nneighs[i])
        row_neighs = neighs[i]
        length(row_neighs) == k || return :inconsistent
        for slot in 1:k
            j = row_neighs[slot]
            (1 <= j <= n) || return :neighbor_domain
        end
        length(weights[i]) == k || return :inconsistent
    end
    return :ok
end

"""
Precondition for every `@inbounds` access the fast path makes on `W`.

On top of `_exact_weights_defect === :ok` it also requires `nneighs[i] <= n`, so
a row holds at most `n + 1` edges and the UInt128 magnitude total in
`_exact_row_bounds_fast` provably cannot overflow.  Declining a wider row also
avoids the divergence where a weight vector longer than `nneighs[i]` is folded
into the reference's row gcd but not into the fast path's.
"""
function _exact_weights_consistent(W, n::Int)
    _exact_weights_defect(W, n) === :ok || return false
    for i in 1:n
        W.nneighs[i] <= n || return false
    end
    return true
end

"""
Machine-integer payload builder.

Returns `nothing`, or raises `_ExactFastDecline`, whenever it declines; the
caller then reruns `_exact_tail_payload_reference` from scratch, so guard
ordering, exception types and messages are always the reference's.
"""
function _exact_tail_payload_fast(data::AbstractVector{<:Real}, W;
                                  stat_code::Integer = 1,
                                  observed::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                                  defined::Union{Nothing, AbstractVector{Bool}} = nothing)
    (stat_code == 1 || stat_code == 2 || stat_code == 3 || stat_code == 4) || return nothing
    # Supplying observed values is a test-only path; it is not worth a second
    # implementation of the BigInt bookkeeping, so it goes to the reference.
    observed === nothing || return nothing
    n = Int(W.n)
    (0 <= n <= _EXACT_FAST_MAX_N) || return nothing
    # Offset-indexed or otherwise non-1-based data is the reference's to handle.
    firstindex(data) == 1 || return nothing
    length(data) == n || return nothing
    _exact_weights_consistent(W, n) || return nothing
    defined === nothing || length(defined) == n || return nothing
    encoded = _exact_encode_values_fast(data)
    encoded === nothing && return nothing
    values, offsets, max_offset = encoded
    weights = _exact_encode_weights_fast(W, stat_code)
    weights === nothing && return nothing
    row_bounds, bound_limbs = _exact_row_bounds_fast(weights, max_offset, stat_code == 2)
    defined_default = _exact_defined_fast(W, stat_code, offsets, values.origin, values.divisor)
    mask = defined === nothing ? defined_default : BitVector(defined)
    packed_obs = zeros(UInt32, 8, n)
    _exact_observed_fast!(packed_obs, bound_limbs, mask, weights, offsets, W, stat_code)
    return ExactTailPayload(values, weights, packed_obs, mask, row_bounds)
end

"""Build bounded exact-tail host payload; no permutation work occurs here."""
function _exact_tail_payload(data::AbstractVector{<:Real}, W;
                             stat_code::Integer = 1,
                             observed::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                             defined::Union{Nothing, AbstractVector{Bool}} = nothing)
    payload = try
        _exact_tail_payload_fast(data, W; stat_code=stat_code,
                                 observed=observed, defined=defined)
    catch err
        err isa _ExactFastDecline || rethrow()
        nothing
    end
    payload === nothing || return payload
    return _exact_tail_payload_reference(data, W; stat_code=stat_code,
                                         observed=observed, defined=defined)
end
