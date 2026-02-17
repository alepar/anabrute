# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

OpenCL-accelerated anagram brute-forcer for the [Trustpilot backend challenge](https://followthewhiterabbit.trustpilot.com/cs/step3.html). Given a seed phrase and target MD5 hashes, it finds anagram phrases whose MD5 matches the targets. Current throughput: ~1B hashes/sec on GPU (theoretical peak 10-50B).

## Build

```bash
mkdir -p build && cd build
cmake .. && make
```

Requires OpenCL. Two CMake targets:
- `anabrute` — main program
- `kernel_debug` — debug harness for the OpenCL kernel

**Note:** `CMakeLists.txt` has a hardcoded Windows NVIDIA OpenCL path for the `anabrute` target (line 11). For macOS/Linux, switch to the `find_package(OpenCL)` approach used by `kernel_debug` (uncomment line 10, comment line 11).

## Runtime

The binary reads files from the working directory:
- `input.dict` — word dictionary
- `input.hashes` — target MD5 hashes (one per line, 32 hex chars)
- `kernels/permut.cl` — OpenCL kernel source (loaded at runtime, not compiled in)

## Architecture

CPU-GPU producer-consumer pipeline with pthread-based concurrency:

1. **CPU cruncher threads** (`cpu_cruncher.c`) enumerate valid anagram word combinations from the dictionary using recursive backtracking. Each CPU thread processes a strided subset of the top-level dictionary. Results are packed into `permut_task` structs and batched into `tasks_buffer`s.

2. **Task buffer queue** (`task_buffers.c`) is a bounded, thread-safe producer-consumer queue (mutex + condvars). CPU threads produce buffers, GPU threads consume them. Size controlled by `TASKS_BUFFERS_SIZE` in `common.h`.

3. **GPU cruncher threads** (`gpu_cruncher.c`) consume task buffers, upload them to GPU memory, and dispatch OpenCL kernels. Each kernel invocation processes up to `PERMUT_TASKS_IN_KERNEL_TASK` tasks with up to `MAX_ITERS_IN_KERNEL_TASK` permutation iterations per task. Tasks that aren't fully permuted are carried over to the next kernel dispatch.

4. **OpenCL kernel** (`kernels/permut.cl`) — each work item takes one `permut_task`, generates word permutations using Heap's algorithm, constructs the candidate string, computes MD5, and checks against target hashes. Matches are written to `hashes_reversed` buffer.

5. **Main thread** (`main.c`) orchestrates everything: reads dict/hashes, initializes OpenCL devices, spawns threads, monitors progress, and prints matched hashes as they're found.

## Critical Sync Constraints

- `permut_task` struct layout in `task_buffers.h` **must exactly match** the layout in `kernels/permut.cl` (both define `MAX_STR_LENGTH=40`, `MAX_OFFSETS_LENGTH=16`). These constants are defined in both `common.h` and the kernel file and must stay in sync.
- The seed phrase in `seedphrase.c` (`"tyranousplutotwits"`) must match the `char_to_index` mapping (only characters present in the phrase have valid indices, `CHARCOUNT=12`).
- `MAX_WORD_LENGTH` in `common.h` controls the maximum number of words per anagram (not word length). Currently 5.

## Performance Characteristics

- **Current throughput:** ~1B hashes/sec (10-50x below theoretical GPU peak)
- **Bottleneck is GPU compute utilization, not memory bandwidth.** PCIe transfer is ~3% of dispatch time.
- **Main perf issues:** warp divergence from mixed `n` values in same dispatch, excessive kernel re-dispatch due to 512-iter cap, carryover logic overhead.
- The `permut_task` struct (96 bytes) is a bandwidth-efficient compressed representation — for `n=8` it encodes 40320 candidates (12,600x compression vs raw strings). Do NOT "optimize" by sending raw strings to GPU.
- See `IDEAS.md` for the batch-by-N optimization design that addresses these bottlenecks.

## Key Data Flow

```
Dictionary words (input.dict)
    |
    v
CPU threads: recursive backtracking over char_counts
    |  (enumerate word combinations that are anagrams of seed phrase)
    v
permut_task structs packed into tasks_buffer (256K tasks per buffer)
    |  (producer-consumer queue with mutex + condvars)
    v
GPU threads: upload buffer, dispatch OpenCL kernel
    |  (each work item: permute words via Heap's algo -> construct string -> MD5 -> compare)
    v
hashes_reversed buffer: matched results read back periodically
```

## Tuning

In `common.h`:
- `PERMUT_TASKS_IN_KERNEL_TASK` (default 256K) — number of tasks per GPU kernel dispatch. Peak performance at 256-512K; lower if kernel times out.
- `MAX_ITERS_IN_KERNEL_TASK` (default 512) — max permutation iterations per kernel task. Peak at ~512; lower if kernel times out. This is the main source of re-dispatch overhead for large `n` values.
- `TASKS_BUFFERS_SIZE` (default 64) — CPU-GPU queue depth, main RAM consumer.

## File Map

| File | Purpose |
|------|---------|
| `main.c` | Entry point, dict/hash loading, thread orchestration, progress display |
| `cpu_cruncher.c/h` | CPU-side anagram enumeration via recursive backtracking |
| `gpu_cruncher.c/h` | GPU-side task consumption, OpenCL kernel management, stats |
| `task_buffers.c/h` | Thread-safe producer-consumer queue, `permut_task` struct definition |
| `kernels/permut.cl` | OpenCL kernel: Heap's permutation + string construction + MD5 + hash comparison |
| `seedphrase.c/h` | Seed phrase and char-to-index mapping |
| `permut_types.c/h` | `char_counts`, `char_counts_strings` — character frequency tracking for anagram validation |
| `hashes.c/h` | MD5 hash parsing (ASCII hex <-> uint32 arrays) |
| `fact.c/h` | Factorial lookup table |
| `os.c/h` | Platform abstractions (`current_micros`, `num_cpu_cores`) |
| `common.h` | Shared constants and includes |
| `kernel_debug.c` | Standalone debug harness for testing the OpenCL kernel |
