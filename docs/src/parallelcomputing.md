```@meta
CurrentModule = SpatialDependence
```

# Parallel Computing

The package takes advantage of Julia multi-threading features to increase performance if Julia is started with multiple threads.

The accelerated local permutation functions also accept a KernelAbstractions backend. CUDA and
Metal runs use a deterministic stream for each observation and permutation, so repeating a run
with the same `seed` on the same backend, with the same inputs and library environment, gives the
same result. CPU and GPU runs use different Monte Carlo samples; compare their statistics and
inference for statistical agreement rather than expecting identical permutation draws. The
default Float32 p-value path uses bounded exact integer tail comparisons over its validated input
domain. Outside that domain, the default Float32 path rejects inputs it cannot represent under
that contract. Float64 accelerated p-values use floating-point comparisons and are not exact
integer tails. In both precisions, summaries and retained draws use backend arithmetic.

To count p-values with the native CPU statistic and its row tolerance on the accelerated samples,
pass `comparison=:cpu` with a backend:

```julia
result = localmoran(x, W; backend = CUDABackend(), seed = 42,
                    comparison = :cpu, return_perms = false)
```

This replays every sampled neighborhood on the CPU, replaces only p-values, and may add substantial
work proportional to the total sampled degree. GPU summaries and retained draws still use the
selected accelerator. With `precision=Float64`, this mode uses ordinary CPU centering for Moran
and Geary, so their scores and summaries can differ from the default shift-first Float64 path.

For a local Metal check on an Apple Silicon machine, run this from the package root:

```sh
testenv=$(mktemp -d)
SPDEP_REPO="$PWD" julia --startup-file=no --project="$testenv" -e 'using Pkg; Pkg.develop(path=ENV["SPDEP_REPO"]); Pkg.add(["GeoInterface", "KernelAbstractions", "RecipesBase", "SpatialDatasets", "StableRNGs", "Metal"])'
SPATIALDEPENDENCE_TEST_BACKEND=metal julia --startup-file=no --project="$testenv" test/runtests.jl
rm -rf "$testenv"
```

For an NVIDIA CUDA check on a machine with a functional CUDA device, use the same commands with
`CUDA` in the package list and `SPATIALDEPENDENCE_TEST_BACKEND=cuda`. The variable is optional;
without it, CPU tests always run and available vendor backends are probed opportunistically.
Misspelled or unavailable selected backends fail the test run instead of being skipped. Run
`test/runtests.jl` once so the GPU and exact-tail helper suites are included together.

See official Julia documentation on [starting Julia with multiple threats](https://docs.julialang.org/en/v1/manual/multi-threading/#Starting-Julia-with-multiple-threads): 

The following functions take advantage of multi-threading to increase their performance:

| Function             | Function Name | What is parallelized?                 |
|:---------------------|:--------------|:--------------------------------------|
| Polygon contiguity   | `polyneigh`   | Bouning box overlaps and polygon hits |
| Global Moran's I     | `moran`       | Permutation test                      |
| Global Geary's c     | `geary`       | Permutation test                      |
| Local Moran          | `localmoran`  | Conditional permutation test          |
| Local Geary          | `localgeary`  | Conditional permutation test          |
| Getis-Ord Statistics | `getisord`    | Conditional permutation test          |
