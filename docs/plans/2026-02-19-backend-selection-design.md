# Backend Selection Redesign

## Problem

1. **AVX-512 never activates** — `CMakeLists.txt` only passes `-mavx2` to `avx_cruncher.c`, so `__AVX512F__` is undefined and all AVX-512 code paths are compiled out.
2. **Selection logic is ad-hoc** — hardcoded skip rules in `main.c` don't implement a clean priority system. POCL CPU devices are treated as GPU backends and contend with AVX for CPU cores.

## Design: Priority-Based Backend Selection

Single selection pass in `main.c`. First backend that probes successfully wins; lower-priority backends are skipped.

| Priority | Condition | Hash crunchers | Enum threads |
|----------|-----------|---------------|-------------|
| 1 | Metal or discrete GPU | GPU only | all CPU cores |
| 2 | AVX-512 (runtime CPUID) | AVX-512, fill all cores | 2 (high prio) |
| 3 | iGPU (OpenCL, real GPU) | iGPU | 2-4 |
| 4 | AVX2 (runtime CPUID) | AVX2, fill all cores | 2 |
| 5 | Fallback | Scalar CPU, fill cores | 2 |

### Key Rules

- GPU backends don't compete for CPU cores -> maximize enumerators
- CPU-bound backends (AVX-512, AVX2, scalar) share cores with enumerators -> limit enumerators to 2
- POCL CPU-only devices are skipped entirely (AVX is always faster on same cores)
- OpenCL only used if it finds a real GPU device (not CL_DEVICE_TYPE_CPU)

## Runtime AVX-512 Detection

Compile `avx_cruncher.c` with `-mavx2 -mavx512f` so both code paths exist in the binary. Use `__builtin_cpu_supports("avx512f")` at runtime in probe functions:

- `avx512_cruncher_ops` — probe returns core count if CPUID confirms AVX-512, else 0
- `avx2_cruncher_ops` — probe returns core count if CPUID confirms AVX2, else 0

The `process_task` function's existing `#if` branches become runtime `if` checks (both paths compiled since both flags are enabled).

## OpenCL: Filter POCL CPU Devices

In `opencl_probe()`: after gathering devices, check `CL_DEVICE_TYPE`. Drop any `CL_DEVICE_TYPE_CPU` devices. If no GPU devices remain, return 0.

## Files Modified

| File | Change |
|------|--------|
| `CMakeLists.txt` | Add `-mavx512f` to avx_cruncher.c compile flags |
| `avx_cruncher.c/h` | Split into avx512/avx2 ops, runtime CPUID, runtime dispatch in process_task |
| `opencl_cruncher.c` | Filter out CPU-type devices |
| `main.c` | Priority-based backend selection, dynamic enum thread count |
