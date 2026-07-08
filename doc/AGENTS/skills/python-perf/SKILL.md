---
name: python-perf
description: Performant Python for scientific computing — batching, NumPy temporaries, strides, dtype, chunking, escape to compiled kernels
trigger:
  glob:
    - "**/*.py"
    - "**/*bench*.py"
    - "**/*perf*.py"
---

## Core Principle

**Python is the harness, not the engine.** Call NumPy, BLAS, or compiled/OpenCL kernels on large batches to minimize harness overhead. PREFER vectorized primitives over Python-level hot loops — but vectorization that creates many full-size temporaries can be slower than a fused compiled loop. Measure before assuming.

## Optimization Workflow (MUST)

1. Preserve a correct reference implementation.
2. Measure end-to-end with `time.perf_counter()` (warm runs, representative sizes).
3. Isolate the hot region — separate orchestration, transfer, compilation, and kernel time.
4. Classify: Python-call overhead, allocation/temporaries, memory bandwidth, BLAS oversubscription, or need for compiled escape.
5. Form one explicit hypothesis; change one structural issue.
6. Validate numerics against the reference.
7. Benchmark again; keep the change only if the improvement is meaningful.

## Decision Hierarchy for Hot Paths

When a loop is hot, try in order:

1. **Suitable NumPy primitive or BLAS** — one call over the whole array.
2. **`out=` and `where=`** — avoid temporaries from advanced indexing.
3. **Broadcasting** — only when it does not create pathological intermediate sizes.
4. **Chunking** — when the full intermediate would exceed memory.
5. **Compiled escape** — OpenCL, C extension, or `numba` for fused or irregular inner loops.
6. **Simple Python loops** — orchestration, small collections (<100 items), code outside measured hot paths.

## Rules by Strength

### MUST

- Profile before optimizing; keep a reference result.
- Minimize Python function calls and per-element overhead in hot paths.
- Preallocate large buffers; reuse across iterations.
- Set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` to avoid BLAS oversubscription when also using OpenMP or multiprocessing.
- Distinguish views from copies (see below) before assuming zero-copy.

### PREFER

- Batch work: one kernel/ufunc call for N elements beats N calls for 1 element each.
- Avoid unnecessary temporaries (`out=`, `where=`, in-place only when safe).
- Use `float32` when accuracy permits — accidental `float64` doubles memory traffic.
- C-contiguous arrays for hot paths; call `np.ascontiguousarray` once if a foreign library requires it.
- Pass grid dimensions + origin/spacing to GPU kernels instead of materializing full coordinate arrays.

### CONSIDER / BENCHMARK

- Dense `meshgrid` vs sparse broadcasting vs kernel-side coordinate reconstruction.
- Chunked processing for memory-bound problems.
- In-place operations — can break aliasing, shared views, or expression fusion.
- Moving irregular inner loops to compiled code even when "vectorization" is possible.

### Repo case: smallDFT grid XC

`ThreadPoolExecutor` over grid tiles in `pyscf/smallDFT/parallel.py` capped at ~2× vs serial — GIL + per-tile Python overhead. Production path: **C/OpenMP in `libsmalldft`** only. Python role: `eval_ao`, libxc, ctypes dispatch, `GridWorkspace` AO cache. See skill:`cpu-perf` § Repo-specific smallDFT.

## Views vs Copies

| Operation | Usually |
|-----------|---------|
| Basic slicing `a[i:j]`, `a[:, 0]` | View |
| `reshape` | View when layout allows |
| `transpose`, `T` | Strided view (may be slow for downstream contiguous ops) |
| Boolean / integer advanced indexing `a[mask]`, `a[indices]` | **Copy** |
| `astype`, `ascontiguousarray`, `np.array(x)` | Copy |
| `np.multiply(..., out=)` | No intermediate if `out` preallocated |

## Common Patterns

### Grid iteration — avoid Python loops

```python
# WRONG: millions of Python iterations
for i in range(nx):
    for j in range(ny):
        result[i, j] = grid[i, j] * weight

# CORRECT: single ufunc
result = grid * weight
```

### Position generation — avoid dense meshgrid

Dense `meshgrid` materializes full coordinate arrays. For a 512³ grid, three `float64` arrays ≈ 3 GiB.

```python
# WRONG: three full Nx×Ny×Nz arrays
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

# PREFER: broadcasting (no full materialization)
xx = x[:, None, None]
yy = y[None, :, None]
zz = z[None, None, :]
result = f(xx, yy, zz)

# ALSO OK: sparse meshgrid
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij', sparse=True)

# GPU: pass x0, dx, (nx, ny, nz) and reconstruct inside kernel
```

### Conditional updates — avoid boolean indexing temporaries

```python
# WRONG: data[mask] creates a copy; assignment may create another
mask = data > threshold
result[mask] = data[mask] * scale

# CORRECT: fused ufunc with out= and where=
result = np.empty_like(data)  # or preallocated buffer
np.multiply(data, scale, out=result, where=mask)
# initialize unchanged slots if needed: result[~mask] = data[~mask]
```

### Accumulation — watch dtype and aliasing

```python
# PREFER: explicit dtype, preallocated output
acc = np.zeros(n, dtype=np.float64)  # widen accumulation if needed
np.add(acc, chunk, out=acc)  # in-place only when acc is not aliased to input

# WRONG: repeated tiny allocations in a loop
for i in range(n_chunks):
    acc = acc + process(chunk[i])  # new array every iteration
```

## Dtype, Strides, and Layout

- **Accidental `float64`**: literals and `np.zeros()` default to float64; specify `dtype=np.float32` when appropriate.
- **Fortran vs C order**: BLAS/LAPACK and some C extensions expect specific layout; check before hot loops.
- **Transposed arrays**: often a view with non-unit stride — subsequent contiguous ops may copy silently.
- **Object dtype**: avoid in numerical hot paths; every element is a Python object.
- **Broadcasting traps**: `a[:, None] * b[None, :]` is fine; `a[:, None, None] * b[None, :, None] * c[None, None, :]` on large grids can still allocate a huge result — chunk if needed.

## Chunking

When the working set exceeds available memory:

```python
for i0 in range(0, n, chunk_size):
    sl = slice(i0, min(i0 + chunk_size, n))
    process_chunk(data[sl], out=out[sl])
```

Chunk size: large enough to amortize Python overhead, small enough to fit in cache/RAM.

## PyOpenCL / GPU Orchestration

- Allocate buffers once; reuse across SCF iterations.
- Avoid `clEnqueueReadBuffer` / `clEnqueueWriteBuffer` inside hot loops.
- Cache compiled programs — kernel compilation is expensive.
- Batch kernel launches; avoid tiny kernels called from long Python loops.
- Use events for timing; separate first-run (compile) from steady-state.

## When Python Loops Are Fine

- File I/O, argument parsing, logging, plotting
- Small constants (<100 iterations) outside measured hot paths
- Building data structures once before the hot loop
- Correctness tests — prioritize clarity over micro-optimization (this skill does not auto-trigger on `tests/`)

## Anti-Patterns Checklist

- [ ] Python `for` over millions of elements doing arithmetic
- [ ] Calling NumPy ufuncs one element at a time in a loop
- [ ] Dense `meshgrid` when broadcasting or kernel-side coords suffice
- [ ] Boolean indexing creating copies in a tight loop
- [ ] Repeated `astype` / `ascontiguousarray` per iteration
- [ ] `float64` everywhere when `float32` meets accuracy needs
- [ ] BLAS using all cores while OpenMP also spawns threads
- [ ] Reading GPU results to Python every iteration for trivial ops

## Related Skills

- skill:`cpu-perf` — compiled C/C++ hot loops, cache, SIMD, OpenMP
- skill:`gpu-optimize` — OpenCL kernel optimization
- skill:`numerical-parity` — validate optimized path against reference
