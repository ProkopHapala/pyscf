---
name: gpu-optimize
description: Optimize OpenCL/CUDA kernels — coalesced access, data layout, gather ownership, local-memory tiling, register pressure, launch overhead, profiling
trigger:
  glob:
    - "**/*.cl"
    - "**/*.cu"
    - "**/kernels/**/*"
    - "**/OpenCL/**/*"
    - "**/*opencl*"
---

## Project Context

Typical workloads in this repo:

- OpenCL kernels (`pyscf/OpenCL/kernels.cl`, PyOpenCL host code)
- Scientific simulation: DFT grids, AO values, XC integration, molecular forces
- Many independent atoms, grid points, molecules, replicas, or small matrices
- Performance-critical paths use `float32`; correctness and debuggability beat clever micro-opts

Use these rules as strong defaults. Deviate only when profiling shows another design is faster.

## Optimization Workflow (MUST)

1. Preserve a correct reference implementation (CPU or high-precision GPU).
2. Measure end-to-end; use `clGetEventProfilingInfo` per kernel.
3. Count launches, host↔device bytes, approximate bytes read/written per output.
4. Classify bottleneck: launch/sync, bandwidth, latency, compute, divergence, register spill, insufficient parallelism.
5. Form one explicit hypothesis; change one structural issue.
6. Validate numerics against reference.
7. Benchmark representative sizes; keep change only if improvement is meaningful.
8. Document chosen workgroup size and data layout.

## Optimization Order

Do not tune arithmetic while the kernel is dominated by transfers or launch overhead.

1. Remove unnecessary host↔device transfers.
2. Remove unnecessary kernel launches.
3. Fix uncoalesced or random global-memory access.
4. Reduce total global-memory reads and writes per useful output.
5. Increase reuse in registers or workgroup-local memory.
6. Increase independent workgroups (batch replicas/systems).
7. Reduce register/private-memory pressure.
8. Reduce divergence and synchronization cost.
9. Optimize arithmetic and transcendental functions.

## Rules by Strength

### MUST

- Profile before optimizing; keep a reference result.
- **One owner per output element** — other work-items provide input, not contested writes.
- Neighboring work-items read neighboring addresses when possible.
- Persistent GPU buffers — allocate once, reuse across SCF/MD iterations.
- All work-items in a workgroup must reach every `barrier()` — never return or diverge around a required barrier.
- No cross-workgroup synchronization inside a normal kernel — use separate kernel launches.
- Guard global loads/stores at tile boundaries (`row < M`, `gid < n`).
- Query device and kernel limits before tuning workgroup size.

Query at runtime:

```text
CL_DEVICE_LOCAL_MEM_SIZE
CL_DEVICE_MAX_WORK_GROUP_SIZE
CL_KERNEL_WORK_GROUP_SIZE
CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
CL_KERNEL_LOCAL_MEM_SIZE
CL_KERNEL_PRIVATE_MEM_SIZE
```

### PREFER

- Gather over scatter; flat contiguous arrays over pointer chasing.
- Hot/cold field split — load only fields the kernel uses.
- Reuse loaded values in registers before loading more.
- Reconstruct cheap values (grid coords from index) instead of materializing large arrays.
- Tree reduction in workgroup-local memory over global atomics for large reductions.
- Batch independent systems until GPU has many runnable workgroups.

### CONSIDER / BENCHMARK

- `float4` / vector loads — only when all components are used and alignment is right.
- SoA vs AoS vs AoSoA — choose by access pattern, not dogma.
- Workgroup-local memory tiling — only when measured reuse amortizes barriers and resource cost.
- Atomics vs auxiliary-buffer + assembly vs per-owner recomputation.
- Kernel fusion — when intermediate stays in registers and pressure stays manageable.
- `native_*` / relaxed math — under documented accuracy budget.
- Kahan summation vs tree reduction for parallel accumulations.

## Memory Hierarchy (OpenCL terminology)

Use neutral vocabulary — do not mix CUDA "local memory" (spill space) with OpenCL `__local`:

| Level | OpenCL | Notes |
|-------|--------|-------|
| Registers | per-work-item private | fastest |
| Spill / private arrays | implementation-managed off-chip private memory | slow; not `__local` |
| Workgroup-shared | `__local` | shared within one workgroup; reduces residency if overused |
| Device/global | `__global` | GB-scale; ~100–400+ cycles latency |
| Cache | hardware-managed | coalescing + spatial locality matter |

- Register spills go to **private spill memory**, not into `__local`.
- Too much `__local` reduces resident workgroups (occupancy) — it does not spill transparently to global.
- Occupancy is not the goal; enough active work to hide latency without sacrificing registers/local memory unnecessarily.

## Bottleneck Decision Tree

### A: High memory traffic, bandwidth near peak → memory-bound

Reduce bytes per useful output:

- fuse producer + consumer; keep intermediates in registers
- tile with reuse; recompute cheap values instead of loading
- procedural grid coords / basis reconstruction
- hot/cold split; narrow storage (`float` vs `double`, 32-bit indices)
- compact active elements; eliminate redundant output arrays

Effective bandwidth = bytes transferred / runtime (not inferred from GFLOP/s alone).

### B: Low bandwidth, low compute → access or scheduling problem

Look for: uncoalesced access, random gathers, too few workgroups, dependency chains, divergence, barriers, atomic contention, register spills, serial inner loops, tiny kernels dominated by launch overhead.

### C: High compute utilization → arithmetic-bound

Fewer ops, CSE, algebraic simplification, lower precision under error budget, `native_*` functions, lookup tables vs transcendentals.

### D: Launch- or sync-bound

Batch systems/iterations; fuse adjacent kernels; device-side loops; reduce host round-trips; persistent buffers; avoid readback per iteration.

## Data-Oriented Design

Ask before choosing layout:

1. What is the unit of work?
2. Which fields does one work-item consume?
3. Which fields are hot in this kernel?
4. Who owns each output?
5. Can objects be reordered or compacted?
6. Is topology static enough to preprocess?

| Access pattern | Layout |
|----------------|--------|
| Neighbors read same field from neighbor objects | **SoA** |
| One work-item consumes nearly all fields of one object | **AoS** or packed `float4` |
| Fixed-width SIMD/workgroup blocks | **AoSoA** |
| Few hot fields in large records | **Hot/cold split** |

```c
// Hot every kernel
__global float4 *pos_type;
__global float4 *force_energy;

// Cold / topology
__global float4 *atom_params;
__global int4   *topology;
```

- `float3` in OpenCL structs is often 16-byte aligned (not packed 12 bytes).
- Do not put integer flags into float slots unless documented bit-level encoding.
- Vector loads (`float4`) help when all components are used; they hurt when only one scalar is needed.

## Coalesced Global-Memory Access

Ideal: `array[get_global_id(0)]` — neighbors access consecutive addresses.

```c
// Good
float4 p = positions[get_global_id(0)];

// Bad when stride is large
float4 p = positions[get_global_id(0) * stride];
```

Multidimensional grids: fastest-changing work-item dimension → fastest-changing array index (`ix = get_global_id(0)` for `ix + iy*nx + iz*nx*ny`).

Reorder data when cheaper than random reads every step: sort by cell/type, compact active lists, CSR neighbor lists, fixed neighbor slots for bounded valence.

## Converting Memory-Bound Kernels

```c
// Reuse in registers
float4 p = positions[i];
float3 d = p.xyz - center;
float r2 = dot(d, d);
// reuse d, r2 — do not reload positions[i]

// Several results per load
float e = native_exp(-a * r);
float e2 = e * e;
energy = D * (e2 - 2.0f * e);
force  = 2.0f * D * a * (e2 - e);

// Reconstruct coords — do not store full 3D grid unless reuse justifies it
int ix = get_global_id(0);
float x = x0 + ix * dx;
```

Fusion removes buffer + write + read + launch — but split again if fusion causes register spilling, divergence, or I-cache pressure.

## Gather Instead of Scatter

> Avoid **uncontrolled write contention**. Atomics are one implementation, not automatically wrong — benchmark against alternatives.

### Pattern 1: One owner per atom (gather neighbors)

```c
int ia = get_global_id(0);
float3 f = (float3)(0.0f);
for (int k = neighborOffset[ia]; k < neighborOffset[ia + 1]; k++) {
    int ja = neighbors[k];
    f += pairForce(ia, ja);
}
forces[ia] = (float4)(f, 0.0f);
```

May compute each pair twice (once per endpoint) — usually still better than contended atomics.

### Pattern 2: Interaction buffer + signed assembly

```c
// Kernel 1: one bond per work-item
int ibond = get_global_id(0);
int2 ij = bonds[ibond];
float3 f = computeBondForce(positions[ij.x], positions[ij.y]);
bondForce[ibond] = (float4)(f, 0.0f);

// Kernel 2: gather signed contributions — NOT f_i[b]+f_j[b] (that sums to zero!)
int ia = get_global_id(0);
float3 f = (float3)(0.0f);
for (int k = atomBondOffset[ia]; k < atomBondOffset[ia + 1]; k++) {
    int2 ref = atomBondRefs[k];   // ref.x = bond index, ref.y = sign (+1 or -1)
    f += convert_float(ref.y) * bondForce[ref.x].xyz;
}
forces[ia] = (float4)(f, 0.0f);
```

### Pattern 3: Fixed neighbor slots (molecular topology)

```c
int4 ng = neighbors[ia];
if (ng.x >= 0) f += pairForce(ia, ng.x);
if (ng.y >= 0) f += pairForce(ia, ng.y);
if (ng.z >= 0) f += pairForce(ia, ng.z);
if (ng.w >= 0) f += pairForce(ia, ng.w);
```

Benchmark atomics when: few contributions per output, low contention, auxiliary buffer traffic dominates.

## Ping-Pong Buffers

Separate src/dst for iterative updates — avoids read-write races.

```c
__kernel void iterate(__global const float4 *src, __global float4 *dst) {
    int i = get_global_id(0);
    dst[i] = update(src, i);
}
// Host alternates A→B, B→A each iteration
```

This is **Jacobi-style** (read all old, write all new). Gauss–Seidel consumes newly updated values and needs a different algorithm.

## Workgroup-Local Memory

Use `__local` only when measured reuse amortizes cooperative load + barriers, and resource cost does not kill residency.

```c
#define WG 128

__kernel void tiledInteraction(
    __global const float4 *source, __global float4 *output, const int n)
{
    __local float4 tile[WG];
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    float4 acc = (float4)(0.0f);

    for (int i0 = 0; i0 < n; i0 += WG) {
        int j = i0 + lid;
        tile[lid] = (j < n) ? source[j] : (float4)(0.0f);
        barrier(CLK_LOCAL_MEM_FENCE);

        int nt = min(WG, n - i0);
        for (int k = 0; k < nt; k++)
            acc += interaction(gid, tile[k]);

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (gid < n)
        output[gid] = acc;
}
```

Local-memory rules:

- Collaborative load; barrier after fill; barrier before overwrite.
- Never `return` before a barrier; never barrier inside divergent branch.
- Do not copy to `__local` just because it is "faster" — hardware cache may already capture reuse.

### Guarded tiled matmul template

```c
// row, col, kx, ky from get_global_id; TS = tile size; M, N, K dimensions
A_tile[ly][lx] = (row < M && kx < K) ? A[row*K + kx] : 0.0f;
B_tile[ly][lx] = (ky  < K && col < N) ? B[ky*N + col] : 0.0f;
// ... accumulate ...
if (row < M && col < N)
    C[row*N + col] = acc;
```

## Synchronization

- `barrier(CLK_LOCAL_MEM_FENCE)` synchronizes **one workgroup only** — also orders `__local` visibility.
- **No global barrier** between workgroups in a normal kernel.
- Workgroups execute in undefined order.
- Never communicate between workgroups via global memory without a valid atomic protocol.

```c
// BAD: early return skips barrier → deadlock
if (gid >= n) return;
barrier(CLK_LOCAL_MEM_FENCE);

// GOOD: all threads reach barrier; guard work separately
bool active = gid < n;
if (active) { /* load/compute */ }
barrier(CLK_LOCAL_MEM_FENCE);
if (active) { /* store */ }
```

## Register / Private-Memory Pressure

Reduce simultaneous live state:

- fewer accumulators; avoid large per-thread private arrays (`float buf[128]` spills)
- avoid excessive manual unrolling; avoid giant fused kernels
- split unrelated phases into separate kernels if profiling shows spills
- reuse variables when earlier values are dead
- lexical scopes *may* help compiler shorten lifetimes — not a reliable register-control mechanism

Inspect: `CL_KERNEL_PRIVATE_MEM_SIZE`, compiler resource reports, profiler spill metrics.

## Workgroup Size

Start with `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`. Benchmark multiples around 64, 128, 256 subject to kernel max and resource usage.

- NVIDIA: multiples of 32 are typical starting points — not an OpenCL universal rule (AMD wave32/wave64 varies).
- Small workgroups when: high register use, much `__local` memory, one workgroup per small system.
- Large workgroups when: lightweight work-items, local reduction benefits, too few workgroups to saturate GPU.
- For small-matrix kernels, 16–32 work-items handling one independent system may be correct — saturate via many systems.

## GPU Saturation

Provide many more workgroups than compute units for latency-bound kernels.

If one molecule/grid/matrix is too small, batch: multiple molecules, replicas, scan points, orientations, matrix instances.

Do not inflate work-item count when each does negligible work and launch overhead dominates.

## Host Runtime (PyOpenCL)

Persistent buffers — do not per-iteration:

- `clCreateBuffer`, `clEnqueueWriteBuffer`, `clEnqueueReadBuffer`
- kernel recompilation, Python dict/wrapper construction

```text
Bad:  launch → readback → Python modify → write → launch
Good: write input once → several kernels → read final result once
```

PyOpenCL launch overhead is significant — fuse tiny consecutive kernels only when same iteration space, single-use intermediate, fusion removes a global round-trip, and register pressure stays acceptable.

Device-side iteration: multiple steps in one kernel when sync is workgroup-local and host decisions are not needed per step.

## Branching

Avoid divergent branches in hot inner loops when measured cost is high.

Prefer: sort by type, separate kernels for different cases, fixed neighbor counts, arithmetic masks.

Simple boundary branches are fine — do not replace cheap predictable branches with expensive always-on arithmetic.

## Floating-Point Policy

OpenCL devices may support `half`, `float`, `double` — query and benchmark; GPU is not "always float32."

- Use lowest precision meeting error requirements.
- Separate storage precision from accumulation precision (`float` storage, `double` accum for sensitive reductions).
- Prefer tree reduction; sum similar magnitudes together.
- Kahan increases dependencies — benchmark vs tree reduction.
- Expect non-deterministic order in parallel reductions.
- `native_sqrt`, `native_exp`, etc. under documented accuracy budget.

## Workgroup Reduction

```c
#define WG 128
__kernel void reduceSum(__global const float *input, __global float *partial, const int n)
{
    __local float scratch[WG];
    int lid = get_local_id(0), gid = get_global_id(0);

    scratch[lid] = (gid < n) ? input[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = WG / 2; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        partial[get_group_id(0)] = scratch[0];
}
```

Second kernel (or CPU) reduces partials. For tiny final arrays, CPU reduction of partials may be fine.

## Profiling Checklist

Record per kernel: time, global/local size, launches, H↔D bytes, approx bytes R/W, approx FLOPs, end-to-end time.

| Symptom | Likely cause | Try |
|---------|-------------|-----|
| High time + high traffic | Memory-bound | Fewer loads/stores, fusion, tiling, compact layout |
| Low utilization + long per-thread loops | Insufficient parallelism | More workgroups, batch systems, decompose serial loops |
| Worse after fusion/unroll | Register spill / I-cache | Split kernel, reduce live state |
| Worse after adding `__local` | Insufficient reuse / occupancy | Remove local copy or shrink tile |
| Many tiny kernels | Launch overhead | Fuse, device-side loop, batch |

## Related Skills

- skill:`gpu-debug` — barrier deadlocks, CPU↔GPU tracing, gated debug macros
- skill:`port-to-opencl` — PyOpenCL workflow, kernel caching, buffer management
- skill:`python-perf` — Python harness overhead, when to keep work on GPU
- skill:`cpu-perf` — CPU reference paths, OpenMP, cache locality
- skill:`numerical-parity` — validate optimized GPU path against reference
