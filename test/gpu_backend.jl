# Select one optional vendor backend for the GPU tests.  This file is also
# safe to include from the standalone exact-tail helper.
const REQUESTED_BACKEND = lowercase(strip(get(ENV, "SPATIALDEPENDENCE_TEST_BACKEND", "")))
REQUESTED_BACKEND in ("", "cuda", "metal") ||
    throw(ArgumentError("SPATIALDEPENDENCE_TEST_BACKEND must be cuda or metal"))

const VENDOR_BACKEND = let
    candidate = nothing
    if REQUESTED_BACKEND in ("", "metal")
        try
            import Metal
            if Metal.functional()
                candidate = Metal.MetalBackend()
            elseif REQUESTED_BACKEND == "metal"
                throw(ArgumentError("SPATIALDEPENDENCE_TEST_BACKEND=metal requested, but Metal is not functional"))
            end
        catch error
            REQUESTED_BACKEND == "metal" && rethrow(error)
        end
    end
    if candidate === nothing && REQUESTED_BACKEND in ("", "cuda")
        try
            import CUDA
            if CUDA.functional()
                candidate = CUDA.CUDABackend()
            elseif REQUESTED_BACKEND == "cuda"
                throw(ArgumentError("SPATIALDEPENDENCE_TEST_BACKEND=cuda requested, but CUDA is not functional"))
            end
        catch error
            REQUESTED_BACKEND == "cuda" && rethrow(error)
        end
    end
    if candidate === nothing && REQUESTED_BACKEND == ""
        try
            import AMDGPU
            if AMDGPU.functional()
                candidate = AMDGPU.ROCBackend()
            end
        catch
        end
    end
    REQUESTED_BACKEND != "" && candidate === nothing &&
        throw(ArgumentError("requested test backend $REQUESTED_BACKEND is unavailable"))
    candidate
end
