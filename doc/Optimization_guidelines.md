## Overall verdict

These are **good first-generation skills**, especially because they contain concrete transformation patterns rather than generic advice. The strongest ideas are:

* restructure scatter into gather/assembly;
* keep buffers resident and batch work;
* think about memory hierarchy and reuse;
* avoid repeated fork/join or kernel-launch overhead;
* recognize register pressure and insufficient parallelism;
* show actual code patterns.

The main weakness is that they mix three different things without distinguishing them:

1. **portable correctness rules**;
2. **generally useful heuristics**;
3. **hardware-specific guesses and folklore**.

An agent will interpret statements such as “must be multiple of 32,” “always use `float4`,” or “maximize local memory” as laws. Several such statements are false, overly absolute, or can actively make code slower.

My rough assessment:

* **CPU skill:** conceptually good, needs qualification and safer examples.
* **GPU skill:** contains the most valuable ideas, but also the most serious technical errors.
* **Python skill:** appropriately short, but much too categorical; one supposedly optimized example increases memory use dramatically.

The best redesign is not merely to add more tips. It is to make the skills teach a **diagnostic optimization process**.

---

# 1. GPU/OpenCL skill

The current GPU skill has a strong core: coalescing, gather-over-scatter, persistent buffers, tiling, register pressure, ping-pong buffers, reductions, and kernel-launch overhead are all relevant topics. 

However, I would substantially rewrite it.

## What is already good

### Gather instead of contested scatter

This is probably the best part of the file. The ownership principle should become even more explicit:

> Every output element should preferably have one owning work-item. Other work-items provide input data, not writes.

That principle naturally leads to:

* per-atom gather;
* CSR adjacency lists;
* auxiliary interaction buffers;
* segmented reductions;
* coloring;
* atomics only where contention is sufficiently low.

### Persistent data and batching

The advice to allocate buffers once, retain data on the GPU, batch operations, and avoid reading intermediate results to the CPU is excellent. It fits PyOpenCL particularly well.

### Recognition of register pressure

It is good that register pressure is included at all. Many introductory GPU guides discuss only coalescing and shared memory and then produce enormous kernels that spill.

### Explicit reusable patterns

Tiling, ping-pong, assembly, and reduction templates are precisely the sort of material a coding agent can apply.

---

## Critical technical corrections

### 1. Do not hard-code one GPU architecture

These statements should be removed:

* “NVIDIA: 48 KB local memory per compute unit”
* “workgroup size … must be multiple of 32”
* “1024 max”
* “registers: approximately 255 per thread”
* fixed 20-cycle and 400-cycle latency values
* “typical GPU has approximately 10k threads”

These may loosely resemble one NVIDIA generation, but this is an OpenCL skill covering NVIDIA, AMD, Intel, and potentially CPUs.

The skill should instruct the agent to query:

```text
CL_DEVICE_LOCAL_MEM_SIZE
CL_DEVICE_MAX_WORK_GROUP_SIZE
CL_DEVICE_MAX_WORK_ITEM_SIZES
CL_KERNEL_WORK_GROUP_SIZE
CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
CL_KERNEL_LOCAL_MEM_SIZE
CL_KERNEL_PRIVATE_MEM_SIZE
```

`CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` is explicitly a performance hint and may differ by kernel and device. AMD architectures can use 32- or 64-lane execution depending on architecture, so a universal multiple of 32 is not an OpenCL rule. ([Khronos Registry][1])

A better rule is:

> Start with the kernel-reported preferred multiple. Benchmark several multiples around 64, 128, and 256 work-items, subject to the kernel-specific maximum and resource usage.

For small-matrix kernels, even 16 or 32 work-items may be correct when one workgroup handles one independent system. Saturation can then come from many systems.

---

### 2. Correct the meaning of “local memory” and spilling

The present text says:

> too many live variables → spill to local memory
> too much local memory → spill to global

This mixes CUDA terminology with OpenCL terminology.

In **OpenCL**:

* `__private` is per-work-item storage;
* `__local` is workgroup-shared storage;
* `__global` is device-visible storage.

Register spills go into implementation-managed **private memory**, usually backed by off-chip device memory. They do not spill into OpenCL `__local` memory.

In CUDA documentation, confusingly, the corresponding off-chip per-thread spill space is called “local memory.” NVIDIA explicitly notes that this “local memory” is physically off-chip and is commonly used for spilled variables and dynamically indexed arrays. ([NVIDIA Docs][2])

Too much `__local`/shared memory normally:

* reduces the number of resident workgroups;
* lowers occupancy;
* or makes a launch configuration invalid.

It does **not normally spill transparently into global memory**.

Use neutral vocabulary throughout the skill:

```text
registers
per-thread private/spill memory
workgroup-shared memory (__local in OpenCL, __shared__ in CUDA)
device/global memory
cache
```

---

### 3. “Maximize local memory usage” is the wrong goal

Replace:

> Maximize shared/local memory usage for data reused across work-items.

with:

> Use workgroup-local memory only when measured reuse amortizes the cooperative load and barriers, and when its resource cost does not reduce useful residency excessively.

Local memory can hurt when:

* each value is used once;
* hardware caches already capture the reuse;
* the tile requires many barriers;
* local-memory bank conflicts occur;
* it reduces resident workgroups;
* subgroup shuffles would be cheaper;
* recomputation is cheaper than storage.

Occupancy is also not something to maximize blindly. NVIDIA’s current guidance explicitly says that higher occupancy does not always imply higher performance; the objective is enough active work to hide relevant latency without unnecessarily sacrificing registers or shared memory. ([NVIDIA Docs][2])

---

### 4. `float4` and SoA are not universal solutions

The current file says, in effect:

* prefer vector types;
* avoid structs;
* prefer SoA;
* pack heterogeneous properties into `float4`.

That is too dogmatic.

#### Coalescing happens across work-items

Thirty-two scalar `float` loads at consecutive addresses may coalesce perfectly well. A `float4` does not magically convert an uncoalesced access pattern into a coalesced one. It changes the amount and alignment of data requested by each work-item.

Vector loads help when:

* each work-item actually consumes all components;
* the address is appropriately aligned;
* the wider load does not fetch unused data;
* it does not increase register pressure unacceptably.

They hurt when only one or two components are used.

#### Use layout according to access pattern

The useful alternatives are:

* **AoS:** good when one thread consumes nearly all fields of one object;
* **SoA:** good when neighboring threads consume the same field from neighboring objects;
* **AoSoA:** often best when processing fixed-width blocks;
* **hot/cold split:** frequently more important than pure AoS versus SoA.

For example:

```cpp
// Hot every timestep
float4 pos_type[];
float4 force_energy[];

// Cold or rarely accessed
float4 parameters0[];
float4 parameters1[];
int4   topology[];
```

Do not put integer flags or type indices into floating-point slots unless this is an explicit bit-level representation with documented conversion. Separate `int4` and `float4` arrays are safer.

OpenCL also gives three-component vector types four-component size and alignment, so `float3` in structures is commonly 16 bytes, not a packed 12-byte object. ([Khronos Registry][3])

---

### 5. Braces do not reliably “release registers”

The example claiming that enclosing computations in `{}` releases registers should be removed.

Compilers perform liveness analysis independently of lexical scope. Scope can occasionally help a compiler or improve source clarity, but it is not a reliable register-control mechanism.

The shown example is especially misleading because `sum`, `prod`, and `norm` are all needed afterward, so those results remain live regardless of braces.

More reliable register-pressure advice is:

* reduce the number of simultaneous accumulators;
* avoid excessive manual unrolling;
* avoid large per-thread arrays;
* avoid dynamically indexed private arrays where possible;
* split unrelated computation phases if profiling shows spills;
* consider recomputing cheap values;
* inspect compiler resource reports;
* inspect generated code or profiler spill metrics;
* test whether reduced register use actually improves runtime.

Register usage, workgroup size, and shared-memory allocation jointly determine residency. More residency is useful only until latency is adequately hidden. ([NVIDIA Docs][2])

---

### 6. Do not say “avoid atomics”; say “avoid highly contended atomics”

Modern atomics are not intrinsically disastrous. Their cost depends primarily on:

* target memory level;
* data type;
* contention;
* spatial distribution;
* architecture;
* whether atomic aggregation can occur;
* whether replacing them requires another full buffer and kernel launch.

A two-pass auxiliary-array scheme may be much slower than atomics when:

* each output receives only a few contributions;
* collisions are rare;
* the auxiliary array is large;
* additional global traffic dominates;
* the second kernel has irregular gathers.

The skill should require benchmarking these alternatives:

1. direct atomic accumulation;
2. workgroup-local accumulation followed by fewer global atomics;
3. one-contribution-per-edge plus gather;
4. graph coloring;
5. per-owner direct evaluation, possibly recomputing symmetric interactions.

The real rule is:

> Avoid uncontrolled write contention. Atomics are one possible implementation, not automatically a mistake.

---

### 7. Kernel fusion needs a trade-off rule

“Fuse kernels whenever possible” is also too absolute.

Fusion can reduce:

* global round-trips;
* launch overhead;
* temporary buffers.

But it can increase:

* live ranges;
* register pressure;
* divergence;
* duplicated computation;
* barrier requirements;
* instruction-cache pressure;
* difficulty of scheduling independent phases.

A better rule is:

> Fuse adjacent bandwidth-bound kernels when the intermediate value can remain in registers and the fused kernel does not introduce unacceptable register pressure, divergence, or synchronization. Split kernels when doing so materially lowers live state or permits better parallel mapping.

This should be decided with measurements, not style preference.

---

### 8. Precision section is incorrect

Remove:

> GPU is always single-precision.

OpenCL devices may support `half`, `float`, and `double`; support and throughput are device-dependent. In OpenCL 3.x, double support is represented by the relevant feature or extension and must be queried. ([Khronos Registry][4])

The useful skill rules are:

* use the lowest precision that satisfies error requirements;
* separate storage precision from accumulation precision;
* consider `float` storage with `double` accumulation for sensitive reductions;
* consider pairwise or tree reductions before Kahan;
* test `native_*` or relaxed-math functions against an accuracy budget;
* document whether denormals, reassociation, and non-deterministic reductions are acceptable;
* do not expect bitwise reproducibility from parallel floating-point reductions.

Kahan summation is not automatically the best GPU solution; it increases dependencies and operation count. A balanced tree often improves both accuracy and parallelism.

The `%f` versus `%g` statement should simply be deleted. It is not a meaningful performance rule.

---

### 9. Replace “10k threads” with useful parallelism metrics

The skill currently confuses:

* total launched work-items;
* resident work-items;
* active workgroups;
* hardware execution lanes.

A kernel may launch millions of work-items while only a subset are resident at once.

Use rules such as:

* provide many more workgroups than compute units for latency-bound kernels;
* aim initially for several runnable workgroups per compute unit, unless one large workgroup intentionally consumes most resources;
* batch independent molecules, replicas, grid planes, matrix instances, or orientations when one problem is too small;
* do not inflate work-item count when each work-item then performs negligible work and launch overhead dominates;
* measure achieved occupancy and stall reasons rather than estimating “thread saturation” from global size.

---

## Two broken GPU examples

### Broken force assembly

The current assembly example does:

```c
total += f_i[b] + f_j[b];
```

But the arrays contain `f` and `-f`, so this sums to zero. 

The adjacency needs endpoint information:

```c
typedef struct {
    int bond;
    int sign;   // +1 if atom is endpoint i, -1 if endpoint j
} BondRef;

for(int k = offsets[iatom]; k < offsets[iatom + 1]; k++){
    BondRef r = refs[k];
    total += convert_float(r.sign) * bondForce[r.bond];
}
```

Or store one already signed contribution per adjacency entry.

This is exactly the kind of bug that makes a skill dangerous: an agent may copy the template because it is labeled as a reusable design pattern.

### Unsafe tiled matrix multiplication

The tiled example assumes:

* `N` is divisible by `TS`;
* global dimensions exactly match `N`;
* all tile accesses are valid;
* `TS × TS` is a legal workgroup;
* `TS` is a compile-time constant.

A reusable template must use guarded loads and stores:

```c
A_tile[ly][lx] = (row < M && kx < K) ? A[row*K + kx] : 0.0f;
B_tile[ly][lx] = (ky  < K && col < N) ? B[ky*N + col] : 0.0f;
...
if(row < M && col < N) C[row*N + col] = acc;
```

Also, a `32 × 32` workgroup is 1024 work-items and is often a poor default even where legal.

### Ping-pong is Jacobi-like, not Gauss–Seidel

Reading entirely from `src` and writing entirely to `dst` implements a Jacobi-style update. Conventional Gauss–Seidel consumes newly updated values and cannot be represented by this simple global ping-pong without changing the algorithm.

---

# 2. What is most importantly missing from the GPU skill

The largest omission is exactly the topic you mentioned: **how to transform a memory-bound algorithm**, rather than merely saying “coalesce memory.”

## Add a bottleneck decision tree

### Case A: DRAM bandwidth is near practical peak

The kernel is genuinely bandwidth-bound. Reduce bytes per useful output:

* fuse producer and consumer;
* keep intermediates in registers;
* tile values with reuse;
* use temporal blocking across iterations;
* recompute cheap quantities instead of loading them;
* procedurally generate coordinates or basis values;
* compress indices and flags;
* use narrower storage types;
* split hot and cold fields;
* compact active elements;
* eliminate redundant output arrays;
* use symmetry to reduce stored data;
* use matrix-free operators;
* batch multiple outputs per loaded input.

Effective bandwidth should be computed from known bytes read and written, not inferred from GFLOP/s. NVIDIA’s guide explicitly defines effective bandwidth from transferred bytes divided by runtime. ([NVIDIA Docs][2])

### Case B: DRAM bandwidth is low and compute use is low

This is usually not a simple bandwidth ceiling. Look for:

* uncoalesced accesses;
* cache misses from random gathers;
* insufficient workgroups;
* dependency chains;
* divergence;
* barriers;
* atomic contention;
* register spills;
* serialized loops inside each work-item;
* tiny kernels dominated by launch overhead.

### Case C: compute utilization is high

Then consider:

* fewer operations;
* common-subexpression reuse;
* algebraic simplification;
* lower precision;
* fast math under an error budget;
* lookup/interpolation versus transcendental functions;
* more efficient mapping to vector/matrix hardware.

### Case D: launch- or synchronization-bound

Consider:

* batching systems or iterations;
* fusion;
* command-buffer mechanisms when available;
* device-side looping for fixed iteration counts;
* replacing repeated global phases with local iterations;
* reducing host round-trips;
* replacing global synchronization with independent work ownership.

---

## Add data-oriented design as a first-class section

The current skill mentions SoA but not really data-oriented design.

The section should ask:

1. What is the unit of work?
2. Which data does one work-item consume?
3. Which data is shared by a subgroup or workgroup?
4. Which fields are hot in this kernel?
5. Who owns each output?
6. Can objects be reordered?
7. Can inactive objects be compacted?
8. Is topology static enough to preprocess?

Important patterns include:

* hot/cold separation;
* SoA, AoS and AoSoA chosen by access pattern;
* CSR adjacency for variable-degree graphs;
* ELLPACK or padded fixed-degree arrays for molecular topology;
* sorting particles by cell/type/material;
* compact active lists;
* 32-bit indices where ranges permit;
* storing relative offsets instead of 64-bit pointers;
* precomputed neighbor slots;
* separating geometry, topology, state, and parameters;
* one data layout per important kernel if conversion is amortized.

For molecular simulation, an AoSoA or fixed-neighbor-slot representation may be better than generic CSR because valence is small and bounded.

---

## Add synchronization and memory-model rules

This deserves its own skill or a major section.

Non-negotiable rules:

* a workgroup barrier synchronizes only one workgroup;
* there is no general global barrier between workgroups inside an ordinary kernel;
* all required work-items must encounter each dynamic barrier instance;
* never return or diverge around a required workgroup barrier;
* a barrier is also a memory-ordering operation only for the specified address spaces;
* a memory fence alone does not make other work-items wait;
* workgroups may execute in any order;
* never communicate through global memory between workgroups within a kernel unless using a formally valid atomic protocol that does not rely on all workgroups being simultaneously resident.

The OpenCL specification explicitly requires all work-items in a workgroup to reach the barrier before any continue. ([Khronos Registry][5])

Also include:

* double buffering within local memory;
* subgroup collectives;
* subgroup shuffle/broadcast;
* bank-conflict avoidance;
* deterministic versus non-deterministic accumulation;
* ownership tables for writes.

---

## Add profiling as a mandatory workflow

The existing profiling section is too small.

The agent should never begin with speculative code changes. Require:

1. preserve a correct reference implementation;
2. measure end-to-end runtime;
3. measure each kernel with event profiling;
4. count launches and transfers;
5. estimate bytes and FLOPs per output;
6. classify the bottleneck;
7. make one transformation;
8. validate numerics;
9. benchmark representative sizes;
10. retain the old path until the new path wins.

Useful quantities:

```text
time per kernel
launch count
host↔device bytes
effective DRAM bandwidth
arithmetic intensity = useful FLOPs / required bytes
registers or private memory per work-item
local memory per workgroup
active workgroups per compute unit
cache hit rates
branch/divergence efficiency
atomic contention
barrier/stall time
```

“Compare GFLOP/s with theoretical peak” is useful only for compute-bound kernels. AMD’s current guidance likewise begins with profiling and then separates memory, resource, and divergence concerns. ([ROCm Documentation][6])

---

# 3. CPU skill

The CPU skill has good coverage of locality, false sharing, SIMD-friendly code, preallocation, loop transformations, and OpenMP restructuring. 

It needs fewer absolute statements and more recognition that modern CPUs are sophisticated out-of-order machines.

## Good parts

Keep:

* flat contiguous storage over pointer chasing;
* spatial and temporal locality;
* blocking;
* false-sharing warning;
* aliasing awareness through `restrict`;
* preallocation;
* avoiding repeated OpenMP region creation;
* first-touch NUMA policy;
* compiler vectorization reports;
* reusing expensive subexpressions.

## Statements to change

### Hardware numbers should be illustrative only

The cache sizes, register counts, and latencies are not portable facts. In particular:

* “approximately 16 registers per thread” is not a useful CPU model;
* cache sizes vary considerably;
* cache latency depends on architecture and access conditions;
* 64-byte cache lines are common on x86 but should not be stated as universal;
* DRAM latency in nanoseconds is less actionable than bandwidth, memory-level parallelism, and cache miss rate.

Replace the table with:

> Query or document the target CPU. Treat 64-byte lines and typical x86 cache sizes as examples, not semantic assumptions.

### “Always access data in 64-byte aligned chunks” is wrong

Alignment of allocation and start addresses can help, but ordinary scalar and vector accesses do not need to be 64-byte chunks.

A better rule:

* align large arrays to at least the preferred SIMD width or cache-line boundary;
* keep the hot traversal contiguous;
* avoid frequent cache-line crossings for small fixed records;
* do not pad every object to 64 bytes unless it prevents real false sharing.

Excessive padding can significantly increase the working set.

### SoA is not always superior

Use the same AoS/SoA/AoSoA rule as on the GPU:

* SoA for field-wise loops;
* AoS when all fields are used together;
* AoSoA for SIMD blocks;
* hot/cold splitting for large records.

### Branchless is not automatically faster

Modern compilers can generate conditional moves, masked SIMD operations, or predication. Arithmetic branch elimination may perform unnecessary work.

The proposed “split into two loops” example is especially questionable: it makes two passes through memory and still contains branches.

The skill should say:

> First make branches predictable and vectorizable. Replace them with masks only when this reduces measured branch or divergence cost without excessive extra work.

### `alloca()` and dynamic stack arrays are not “effectively free”

They can:

* overflow the stack;
* prevent reuse;
* inhibit optimization;
* create large per-thread memory consumption;
* be non-portable.

Prefer fixed small local arrays, reusable scratch buffers, or thread-local arenas.

### `#pragma omp simd` is a correctness assertion

It is not merely a harmless hint. It tells the compiler that relevant loop-carried dependencies do not prevent SIMD execution. The skill should require the agent to verify aliasing and dependencies before adding it.

### Fix the blocked-matrix example

The example:

* reads `C` before showing initialization;
* does not handle dimensions that are not multiples of `BS`;
* says only the A tile should fit, although working-set sizing involves relevant A, B and C blocks;
* uses a loop order that is educational but not necessarily a good microkernel.

For a skill file, examples should be correctness-safe even if simplified.

### OpenMP energy example needs clearer semantics

The persistent parallel-region idea is good. But if `E` represents energy for the current iteration, it must be reset once per iteration before the reduction. Otherwise it accumulates over all iterations.

A robust pattern is:

```cpp
#pragma omp parallel
{
    for(int itr = 0; itr < niter; itr++){
        #pragma omp single
        E = 0.0;

        #pragma omp for reduction(+:E)
        for(int i = 0; i < n; i++){
            E += evalSingleAtom(i);
        }

        #pragma omp for
        for(int i = 0; i < n; i++){
            moveSingleAtom(i);
        }
    }
}
```

The persistent-region concept is valid OpenMP practice; just make the example’s state ownership explicit. ([OpenMP][7])

---

## Missing CPU topics

Add:

* performance counters and `perf stat`;
* cycles, instructions, IPC, branch misses, cache misses;
* effective memory bandwidth;
* roofline reasoning;
* vectorization reports from GCC/Clang;
* memory-bound versus dependency-bound loops;
* hardware prefetch-friendly strides;
* loop interchange;
* software prefetch only after profiling;
* TLB pressure and page locality;
* OpenMP scheduling choices;
* thread affinity;
* parallel reduction determinism;
* avoiding oversubscription with threaded BLAS;
* NUMA placement and cross-socket traffic;
* scalar cleanup and boundary loops;
* separate fast interior and safe boundary paths.

---

# 4. Python skill

The central principle—Python orchestrates and compiled code performs large numerical batches—is appropriate for your projects. 

But this skill currently encourages unnecessary memory allocation.

## Good parts

Keep:

* batch work;
* avoid millions of Python-level calls;
* preallocate large buffers;
* use in-place operations where safe;
* move genuine hot loops into NumPy, OpenCL, or compiled code.

## Problems

### “Vectorized operations only” is too absolute

Vectorization can produce several full-size temporary arrays and make a memory-bound computation slower than a fused compiled loop.

The appropriate hierarchy is:

1. use a suitable NumPy primitive or BLAS operation;
2. use `out=` and `where=` to avoid temporaries;
3. use broadcasting only when it does not create pathological results;
4. process in chunks if the full intermediate is too large;
5. use an OpenCL or compiled kernel for fused or irregular hot loops;
6. permit simple Python loops for orchestration, small collections, and code outside measured hot paths.

### The `meshgrid` example is not minimum allocation

This:

```python
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
```

creates full coordinate arrays by default. For a `512³` grid, three `float64` coordinate arrays alone require about 3 GiB.

Use broadcasting:

```python
xx = x[:, None, None]
yy = y[None, :, None]
zz = z[None, None, :]

result = f(xx, yy, zz)
```

or:

```python
xx, yy, zz = np.meshgrid(x, y, z, indexing="ij", sparse=True)
```

NumPy documents that dense meshgrid is the default, while `sparse=True` reduces coordinate-array dimensions for broadcasting. ([NumPy][8])

For GPU work, it is often still better to pass `x0`, `dx`, and grid dimensions and reconstruct coordinates inside the kernel.

### Boolean indexing creates copies

This:

```python
result[mask] = data[mask] * scale
```

usually creates a temporary `data[mask]`. Advanced indexing, including Boolean indexing, returns a copy. ([NumPy][9])

Prefer:

```python
np.multiply(data, scale, out=result, where=mask)
```

provided the unchanged part of `result` has been initialized appropriately. NumPy ufuncs support both `out=` and broadcast `where=` specifically for this kind of operation. ([NumPy][10])

### “Advanced slicing” conflates views and copies

Distinguish:

* basic slicing: generally a view;
* Boolean/integer advanced indexing: copy;
* reshape: view when possible, copy when required by layout;
* transpose: normally a strided view, which may then be slow for downstream contiguous operations.

### In-place is not automatically better

In-place operations can:

* overwrite an input still needed later;
* modify a shared view;
* trigger casting problems;
* interfere with expression fusion;
* force poor access ordering.

The rule should be “avoid unnecessary temporaries,” not “always operate in place.”

## Missing Python topics

Add:

* dtype selection, especially accidental `float64`;
* C versus Fortran contiguity;
* strides and transposed arrays;
* implicit copies from `astype`, `ascontiguousarray`, indexing, and foreign-library calls;
* broadcasting-generated large outputs;
* ufunc `out=` and `where=`;
* chunking;
* avoiding object dtype;
* BLAS thread oversubscription;
* measuring with `time.perf_counter` and repeated warm runs;
* separating host orchestration time, transfer time, compilation time, and kernel time;
* reusing PyOpenCL events and buffers;
* avoiding repeated kernel compilation;
* avoiding tiny NumPy operations in long Python loops.

I would also remove `**/tests/**/*` from the trigger. Correctness tests should prioritize clarity and diagnostic quality; only benchmark or performance tests should automatically activate this skill.

---

# 5. Recommended skill architecture

Rather than one giant “GPU optimization encyclopedia,” I suggest five coordinated skills.

## A. `perf-workflow`

Language-independent, always loaded during optimization work.

```text
1. Preserve a correct reference.
2. Measure end-to-end first.
3. Isolate the hot region.
4. Classify: launch, bandwidth, latency, compute, synchronization, allocation.
5. Estimate bytes, FLOPs, working set and parallelism.
6. Form one explicit hypothesis.
7. Change one structural issue.
8. Validate numerics and races.
9. Benchmark representative sizes.
10. Keep the simpler version unless the improvement is meaningful.
```

This prevents cargo-cult optimization.

## B. `data-oriented-design`

Shared by CPU and GPU:

```text
work ownership
hot/cold fields
AoS / SoA / AoSoA
contiguous traversal
reordering and sorting
compact active lists
CSR / ELL / fixed neighbor slots
index width
materialize versus recompute
batching
write ownership
```

## C. `opencl-perf`

Separate into:

1. execution mapping;
2. global-memory access;
3. workgroup-local memory;
4. subgroups;
5. register/private-memory pressure;
6. synchronization and atomics;
7. host runtime and transfers;
8. precision;
9. profiling;
10. safe reusable templates.

Then add short vendor notes:

```text
NVIDIA terminology and profiler
AMD wave32/wave64 and LDS
Intel subgroup/SIMD considerations
```

Do not mix vendor-specific constants into the portable core.

## D. `cpu-perf`

Concentrate on:

```text
cache and TLB locality
SIMD
aliasing
branch behavior
OpenMP
NUMA
memory bandwidth
profiling counters
```

## E. `numpy-perf`

Concentrate on:

```text
Python-call overhead
temporaries
strides
dtype
broadcasting
chunking
compiled escape paths
```

A separate `gpu-sync-debug` skill is also justified because synchronization errors are correctness problems, not merely performance problems.

---

# 6. How each rule should be phrased

Use three levels.

## MUST

Reserved for correctness or measurement requirements:

* preserve a reference result;
* profile before optimizing;
* all work-items must reach a required barrier;
* do not assume cross-workgroup synchronization;
* bounds-check padded global ranges;
* validate race freedom;
* query device and kernel limits.

## PREFER

Strong defaults:

* coalesced contiguous access;
* one owner per output;
* persistent buffers;
* batching;
* compact hot data;
* enough independent workgroups;
* register reuse for thread-private values.

## CONSIDER / BENCHMARK

Architecture- and workload-dependent choices:

* `float4`;
* SoA versus AoS;
* local-memory tiling;
* atomics versus auxiliary buffers;
* kernel fusion;
* unrolling;
* workgroup size;
* Kahan summation;
* fast math;
* recomputation versus storage.

The current files use “always,” “never,” “forbidden,” and “required” for many choices that are merely candidates. That will cause an agent to optimize according to slogans rather than evidence.

---

# 7. Highest-priority edits

I would make these first:

1. **Fix the force-assembly example**, because it currently sums every bond contribution to zero.
2. **Rewrite OpenCL memory terminology**, clearly distinguishing `__local` from private spill memory.
3. **Remove fixed workgroup, local-memory, register, and latency numbers.**
4. **Replace “maximize local memory,” “avoid atomics,” and “fuse whenever possible” with measured trade-off rules.**
5. **Add the bandwidth/latency/compute/launch bottleneck decision tree.**
6. **Add output ownership and synchronization semantics.**
7. **Replace the dense NumPy `meshgrid` example with sparse broadcasting.**
8. **Replace blanket SoA and `float4` rules with access-pattern-based AoS/SoA/AoSoA guidance.**
9. **Make profiling and correctness validation the first section of every performance skill.**
10. **Move data-oriented design into a shared skill rather than duplicating simplified SoA advice.**

The existing files have a valuable practical character. I would preserve that, but convert them from lists of asserted optimizations into **decision manuals: identify the bottleneck, select a transformation, understand its cost, and verify that it actually wins**.

[1]: https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetKernelWorkGroupInfo.html?utm_source=chatgpt.com "clGetKernelWorkGroupInfo(3) - Khronos Registry"
[2]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html "CUDA Best Practices Guide — CUDA C++ Best Practices Guide 13.3 documentation"
[3]: https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html?utm_source=chatgpt.com "The OpenCL™ C Specification - Khronos Registry"
[4]: https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html "The OpenCL™ C Specification"
[5]: https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html "The OpenCL™ Specification"
[6]: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html?utm_source=chatgpt.com "Performance guidelines — HIP 7.2.53211 Documentation"
[7]: https://www.openmp.org/spec-html/5.2/openmpse57.html?utm_source=chatgpt.com "parallel Construct"
[8]: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html "numpy.meshgrid — NumPy v2.5 Manual"
[9]: https://numpy.org/doc/stable/user/basics.indexing.html?utm_source=chatgpt.com "Indexing on ndarrays — NumPy v2.5 Manual"
[10]: https://numpy.org/doc/stable/reference/ufuncs.html "Universal functions (ufunc) — NumPy v2.5 Manual"

---

---

name: gpu-optimize
description: Optimize OpenCL/CUDA kernels for NVIDIA GPUs — coalesced memory access, data-oriented layout, local-memory tiling, register pressure, gather-based writes, kernel fusion, and GPU saturation
trigger:
glob:
- "**/*.cl"
- "**/*.cu"
- "**/apps_OCL/**/*"
- "**/apps_CUDA/**/*"
- "**/pyBall/OCL/**/*"
----------------------

# GPU Optimization

## Project Assumptions

These rules target our typical workloads:

* NVIDIA GPUs, especially GTX 3090-class hardware
* OpenCL or CUDA
* scientific simulation and graphics kernels
* many independent atoms, grid points, molecules, replicas, or small matrices
* performance-critical code uses `float32`
* correctness and debuggability remain more important than clever micro-optimization

Use these rules as strong defaults. Deviate only when profiling clearly shows that another design is faster.

# Core Rules

1. **Use single precision.**
2. **Avoid atomics in hot kernels.**
3. **Prefer gather over scatter.**
4. **Give each output element one owning work-item.**
5. **Make neighboring work-items read neighboring addresses.**
6. **Keep data on the GPU.**
7. **Reuse loaded data before loading more.**
8. **Use local memory only for data reused by multiple work-items.**
9. **Keep per-thread state small to avoid register spilling.**
10. **Minimize kernel launches and host-device synchronization.**
11. **Batch independent systems until the GPU is saturated.**
12. **Profile before and after every important optimization.**

# Optimization Order

Optimize in this order:

1. Remove unnecessary host-device transfers.
2. Remove unnecessary kernel launches.
3. Fix uncoalesced or random global-memory access.
4. Reduce the total number of global-memory reads and writes.
5. Increase data reuse using registers or local memory.
6. Increase the number of independent workgroups.
7. Reduce register pressure and private arrays.
8. Reduce divergence and synchronization.
9. Optimize arithmetic and transcendental functions.

Do not begin by replacing arithmetic instructions while the kernel is dominated by global-memory traffic or launch overhead.

# Data-Oriented Design

Design data around how kernels access it.

## Prefer Flat Arrays

Use flat arrays with explicit indexing.

```c
int i = get_global_id(0);
float4 p = positions[i];
```

Avoid:

* linked structures;
* pointer chasing;
* arrays of pointers;
* nested containers;
* large structures with rarely used fields.

## Prefer Structure of Arrays

When neighboring work-items use the same property, store that property contiguously.

```c
__global float4* positions;
__global float4* velocities;
__global float4* forces;
__global float*  charges;
__global int4*   neighbors;
```

Avoid large structures such as:

```c
struct Atom {
    float4 position;
    float4 velocity;
    float4 force;
    float charge;
    float mass;
    int type;
    int flags;
};
```

Large structures cause unnecessary loads and poor cache use.

## Split Hot and Cold Data

Put frequently used fields into compact hot arrays.

```c
__global float4* pos_type;
__global float4* force_energy;
```

Put infrequently used parameters into separate arrays.

```c
__global float4* atom_params;
__global int4* topology;
```

A kernel should load only the fields it actually needs.

## Use Packed Vector Types

Prefer:

* `float4`
* `float2`
* `int4`
* `int2`

for naturally grouped and aligned data.

Typical layout:

```c
float4 position;   // x, y, z, type or charge
float4 velocity;   // vx, vy, vz, inverse mass
float4 force;      // fx, fy, fz, energy
```

Do not repeatedly load a `float4` if only one scalar component is needed. Put frequently accessed scalar properties into separate arrays.

# Coalesced Global-Memory Access

The ideal pattern is:

```c
int i = get_global_id(0);
float x = array[i];
```

Neighboring work-items then access neighboring addresses.

## Good

```c
float4 p = positions[get_global_id(0)];
```

## Bad

```c
float4 p = positions[get_global_id(0) * stride];
```

when `stride` is large.

## Multidimensional Grids

Map the fastest-changing work-item dimension to the fastest-changing array index.

For:

```c
index = iz * nx * ny + iy * nx + ix;
```

use:

```c
ix = get_global_id(0);
iy = get_global_id(1);
iz = get_global_id(2);
```

The `x` dimension should be contiguous.

## Reorder Data When Possible

For irregular interactions:

* sort particles by spatial cell;
* group atoms by type;
* group active items together;
* store neighbors contiguously;
* use compact active lists;
* preprocess static topology.

It is often cheaper to reorder data once than to perform random reads in every timestep.

# Converting Memory-Bound Kernels

A memory-bound kernel is improved by reducing bytes transferred per useful result.

## Reuse Data in Registers

Load a value once and reuse it.

```c
float4 p = positions[i];
float3 d = p.xyz - center;
float r2 = dot(d, d);

// Reuse d and r2.
// Do not reload positions[i].
```

## Compute Several Results per Loaded Input

When one loaded value contributes to several outputs, calculate them together.

```c
float e = native_exp(-a * r);
float e2 = e * e;

energy = D * (e2 - 2.0f * e);
force  = 2.0f * D * a * (e2 - e);
```

## Recompute Cheap Values Instead of Loading Them

Arithmetic is often cheaper than global memory.

Prefer reconstructing:

* grid coordinates from index;
* periodic offsets;
* polynomial coefficients;
* simple geometric quantities;
* signs and masks;
* small rotation components;

rather than loading large precomputed arrays.

Example:

```c
int ix = get_global_id(0);
float x = x0 + ix * dx;
```

Do not store a full coordinate array unless it is reused enough to justify the memory traffic.

## Fuse Compatible Operations

Fuse adjacent operations when the intermediate value can remain in registers.

Instead of:

```text
kernel A: input -> temporary
kernel B: temporary -> output
```

prefer:

```text
kernel AB: input -> output
```

This removes:

* one temporary buffer;
* one global write;
* one global read;
* one kernel launch.

Do not create an enormous fused kernel. If fusion causes register spilling, excessive branching, or very low occupancy, split it again.

## Avoid Materializing Large Intermediate Grids

Do not store full multidimensional arrays when they can be reconstructed cheaply.

Examples:

* reconstruct 3D basis values from radial and angular components;
* generate grid positions from integer indices;
* evaluate splines directly from compact coefficients;
* keep only active cells;
* store per-type tables rather than per-atom copies.

## Use Narrow Storage Where Safe

Use:

* `float` instead of `double`;
* 32-bit indices instead of 64-bit indices;
* compact flags;
* packed neighbor indices;
* smaller lookup tables.

Do not sacrifice numerical correctness merely to reduce storage.

# Gather Instead of Scatter

## Hard Rule

Do not let many work-items write to the same output.

Avoid:

```c
atomic_add(&force[i], f);
atomic_add(&force[j], -f);
```

Atomics cause serialization, unpredictable performance, numerical-order dependence, and debugging difficulties.

## Pattern 1: One Owner per Atom

Each work-item owns one atom and gathers contributions from its neighbors.

```c
int ia = get_global_id(0);

float3 f = (float3)(0.0f);

for(int k = neighborOffset[ia]; k < neighborOffset[ia + 1]; k++){
    int ja = neighbors[k];
    f += pairForce(ia, ja);
}

forces[ia] = (float4)(f, 0.0f);
```

This may calculate an interaction twice, once from each endpoint. That is usually preferable to atomics.

## Pattern 2: Interaction Buffer Plus Assembly

Kernel 1 computes one interaction per work-item.

```c
int ibond = get_global_id(0);

int2 ij = bonds[ibond];
float3 f = computeBondForce(positions[ij.x], positions[ij.y]);

bondForce[ibond] = (float4)(f, 0.0f);
```

Kernel 2 gathers the signed contributions for each atom.

```c
int ia = get_global_id(0);

float3 f = (float3)(0.0f);

for(int k = atomBondOffset[ia]; k < atomBondOffset[ia + 1]; k++){
    int2 ref = atomBondRefs[k];
    int ibond = ref.x;
    float sign = ref.y > 0 ? 1.0f : -1.0f;

    f += sign * bondForce[ibond].xyz;
}

forces[ia] = (float4)(f, 0.0f);
```

## Pattern 3: Fixed Neighbor Slots

For molecules with small bounded coordination, fixed neighbor slots are often faster than general graph structures.

```c
int4 ng = neighbors[ia];

if(ng.x >= 0) f += pairForce(ia, ng.x);
if(ng.y >= 0) f += pairForce(ia, ng.y);
if(ng.z >= 0) f += pairForce(ia, ng.z);
if(ng.w >= 0) f += pairForce(ia, ng.w);
```

This is simple, predictable, and easy for the compiler to optimize.

# Ping-Pong Buffers

Use separate input and output buffers for iterative algorithms.

```c
__kernel void iterate(
    __global const float4* src,
    __global       float4* dst
){
    int i = get_global_id(0);
    dst[i] = update(src, i);
}
```

Alternate:

```text
iteration 0: A -> B
iteration 1: B -> A
iteration 2: A -> B
```

Use ping-pong buffers for:

* Jacobi iterations;
* diffusion;
* cellular updates;
* particle integration;
* relaxation methods;
* image filters;
* grid-based solvers.

Never read and write neighboring elements of the same buffer unless the algorithm explicitly guarantees race freedom.

# Local Memory

Use local memory when values are loaded once from global memory and reused by several work-items.

## Tiled Pattern

```c
#define WG 128

__kernel void tiledInteraction(
    __global const float4* source,
    __global       float4* output,
    const int n
){
    __local float4 tile[WG];

    int lid = get_local_id(0);
    int gid = get_global_id(0);

    float4 acc = (float4)(0.0f);

    for(int i0 = 0; i0 < n; i0 += WG){
        int j = i0 + lid;

        tile[lid] = (j < n) ? source[j] : (float4)(0.0f);
        barrier(CLK_LOCAL_MEM_FENCE);

        int nt = min(WG, n - i0);

        for(int k = 0; k < nt; k++){
            acc += interaction(gid, tile[k]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(gid < n){
        output[gid] = acc;
    }
}
```

## Local-Memory Rules

* Load tiles collaboratively.
* Each work-item should load one or a few contiguous values.
* Synchronize after filling the tile.
* Synchronize before overwriting the tile.
* Never return before a barrier.
* Never place a barrier inside a branch taken by only part of the workgroup.
* Keep local arrays compact.
* Use local memory only when the tile is reused.
* Do not copy data into local memory merely because local memory is faster.

# Synchronization

## Workgroup Barriers

A barrier synchronizes only work-items in one workgroup.

```c
barrier(CLK_LOCAL_MEM_FENCE);
```

Use barriers only when work-items exchange data through local memory.

## No Global Barrier Inside a Normal Kernel

Workgroups execute independently and in an undefined order.

Do not implement:

```text
workgroup 0 writes global data
workgroup 1 waits and reads it
```

Use separate kernels:

```text
kernel 1
global synchronization at kernel boundary
kernel 2
```

## Barrier Safety

All work-items in the workgroup must reach every barrier.

Bad:

```c
if(gid >= n) return;
barrier(CLK_LOCAL_MEM_FENCE);
```

Good:

```c
bool active = gid < n;

if(active){
    // guarded work
}

barrier(CLK_LOCAL_MEM_FENCE);

if(active){
    // guarded output
}
```

# Register Pressure

Registers are the fastest storage. Excessive per-thread state causes register spilling and reduces the number of active workgroups.

## Keep Live State Small

Avoid:

* large private arrays;
* too many accumulators;
* excessive manual unrolling;
* many simultaneously live vectors;
* giant fused kernels;
* repeatedly inlined large helper functions;
* dynamically indexed private arrays.

## Use Lexical Scopes

Enclose independent calculation phases in scopes so the compiler can shorten variable lifetimes.

```c
float resultA;
{
    float4 p = loadA(i);
    float4 q = transformA(p);
    resultA = evaluateA(q);
}

float resultB;
{
    float4 p = loadB(i);
    float4 q = transformB(p);
    resultB = evaluateB(q);
}
```

Do not keep temporary variables from one phase visible throughout the entire kernel.

## Reuse Variables

Prefer:

```c
float4 tmp = ...;
tmp = transform(tmp);
tmp = normalize(tmp);
```

over creating many separate intermediates when the earlier values are no longer needed.

## Avoid Large Per-Thread Arrays

Bad:

```c
float values[128];
```

Such arrays commonly spill into slow private memory.

Prefer:

* workgroup-local arrays when values are shared;
* global scratch buffers when necessary;
* streaming computation;
* small fixed arrays;
* several passes with less live state.

## Moderate Unrolling

Unrolling can improve arithmetic throughput but increases register use.

Unroll small fixed loops only when:

* the loop count is small;
* indexing becomes simpler;
* register use remains acceptable.

Do not fully unroll large loops.

## Split Kernels When Spilling Is Severe

If one kernel has several unrelated phases and uses too many registers, split it into smaller kernels despite the extra launch and temporary buffer.

Kernel fusion is good until it creates register spilling.

# Workgroup Size

For NVIDIA hardware, use workgroup sizes that are multiples of 32.

Good starting values:

* 32
* 64
* 128
* 256

Default to `64` or `128` unless the algorithm naturally requires another size.

Benchmark at least two reasonable sizes for important kernels.

Use smaller workgroups when:

* each work-item uses many registers;
* each workgroup uses much local memory;
* one workgroup processes one small independent system;
* synchronization cost dominates.

Use larger workgroups when:

* work-items are lightweight;
* local-memory reduction benefits from more participants;
* there are too few workgroups to saturate the GPU.

# GPU Saturation

A GPU needs many independent workgroups.

If one molecule, grid, or matrix is too small, batch:

* multiple molecules;
* multiple replicas;
* multiple orientations;
* multiple scan points;
* multiple frequency points;
* multiple optimization candidates;
* multiple small matrices.

Prefer:

```text
one workgroup per independent system
many systems in parallel
```

over:

```text
one system using only a few workgroups
```

Do not leave most of the GPU idle while one workgroup performs a long serial loop.

# Kernel Launches and Host Transfers

## Persistent Buffers

Allocate buffers once and reuse them.

Do not repeatedly call:

* `clCreateBuffer`;
* `clEnqueueWriteBuffer`;
* `clEnqueueReadBuffer`;
* kernel compilation;
* Python dictionary or wrapper construction;

inside the hot loop.

## Keep Intermediate Data on the GPU

Bad:

```text
launch kernel
read result to CPU
modify result in Python
write result to GPU
launch next kernel
```

Good:

```text
write input once
launch several kernels
read final result once
```

## Minimize Tiny Kernels

PyOpenCL launch overhead is significant.

Fuse small consecutive kernels when:

* they use the same iteration space;
* the intermediate is used only once;
* fusion removes a global-memory round trip;
* register pressure remains manageable.

## Device-Side Iteration

When iterations do not require host decisions, perform several iterations inside one kernel or use a persistent workgroup design.

Use this only when the required synchronization is local to each workgroup.

# Branching

Avoid divergent branches inside hot loops.

Prefer:

* homogeneous workgroups;
* sorting by atom or interaction type;
* separate kernels for fundamentally different cases;
* arithmetic masks;
* fixed neighbor counts;
* branch-free inner loops.

Simple boundary branches are acceptable. Do not replace a cheap predictable branch with a large amount of unnecessary arithmetic.

Move uncommon special cases into a separate kernel when possible.

# Floating-Point Policy

Performance-critical GPU kernels use `float`.

Use:

```c
float
float2
float4
native_sqrt
native_rsqrt
native_exp
native_sin
native_cos
```

when their accuracy is sufficient.

Avoid `double` unless explicitly required for:

* validation;
* reference calculations;
* numerically sensitive global accumulation;
* algorithms demonstrated to fail in single precision.

For reductions:

* prefer tree reduction;
* sum similarly sized values together;
* use compensated accumulation only when required;
* expect small differences caused by parallel reduction order.

# Reductions

Avoid global atomics.

## Workgroup Reduction

```c
#define WG 128

__kernel void reduceSum(
    __global const float* input,
    __global       float* partial,
    const int n
){
    __local float scratch[WG];

    int lid = get_local_id(0);
    int gid = get_global_id(0);

    scratch[lid] = (gid < n) ? input[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int stride = WG / 2; stride > 0; stride >>= 1){
        if(lid < stride){
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0){
        partial[get_group_id(0)] = scratch[0];
    }
}
```

Launch another reduction kernel for the partial results.

For very small final arrays, reading the partial sums to the CPU may be acceptable.

# Profiling

Always measure real kernel execution time using OpenCL event profiling.

Record:

* execution time;
* global work size;
* local work size;
* number of kernel launches;
* host-device transfer time;
* approximate bytes read and written;
* approximate floating-point operations;
* total end-to-end time.

## Diagnosis

### High time and large memory traffic

The kernel is probably memory-bound.

Reduce:

* loads;
* stores;
* intermediate arrays;
* unused fields;
* duplicate data;
* random access.

Increase:

* reuse;
* tiling;
* fusion;
* compactness.

### Low GPU utilization and long per-thread loops

The kernel lacks parallelism.

Increase:

* number of workgroups;
* number of independent systems;
* batching;
* work decomposition.

Reduce serial loops inside each work-item or workgroup.

### Performance becomes worse after fusion or unrolling

Suspect:

* register pressure;
* spilling;
* reduced occupancy;
* instruction explosion.

Reduce live state or split the kernel.

### Performance becomes worse after using local memory

Suspect:

* insufficient reuse;
* excessive barriers;
* too much local-memory allocation;
* reduced occupancy;
* poor cooperative loading.

Remove the local-memory copy or use a smaller tile.

# Required Optimization Workflow

1. Keep a correct reference implementation.
2. Measure the current implementation.
3. Identify the dominant kernel or transfer.
4. Decide whether it is limited by memory, computation, launch overhead, synchronization, or insufficient parallelism.
5. Make one structural optimization.
6. Compare numerical results with the reference.
7. Measure again.
8. Keep the change only if the improvement is meaningful.
9. Document the chosen workgroup size and data layout.
10. Retain useful timing and debug instrumentation.

# Related Skills

* skill:`cpu-perf`
* skill:`python-perf`
* skill:`gpu-debug`
* skill:`port-to-opencl`
* skill:`data-oriented-design`

---

---

name: data-oriented-design
description: Restructure scientific code around contiguous data, explicit ownership, batching, locality, and minimal memory traffic
-----------------------------------------------------------------------------------------------------------------------------------

# Data-Oriented Design

## Core Principle

Organize data according to how the hot loop processes it, not according to how the physical object is described conceptually.

The important questions are:

1. What is the unit of work?
2. Which fields are used together?
3. Which fields are accessed in every iteration?
4. Which values are reused?
5. Who owns each output?
6. Can objects be reordered?
7. Can inactive objects be removed from the hot loop?
8. Can values be reconstructed instead of stored?

# Hard Rules

* Use flat contiguous arrays.
* Avoid pointer chasing.
* Separate hot and cold data.
* Prefer one owner per output.
* Prefer gather over scatter.
* Batch many similar operations.
* Precompute static topology.
* Reuse buffers.
* Avoid unnecessary intermediate arrays.
* Reconstruct cheap values instead of loading them.
* Keep the hot working set compact.

# AoS, SoA, and Packed Blocks

## Prefer SoA for Field-Wise Processing

```c
float* x;
float* y;
float* z;
float* charge;
```

This is best when loops process one field across many objects.

## Prefer Packed Vectors for Closely Related Fields

```c
float4* position_type;
float4* velocity_mass;
```

This is best when each work-item normally consumes all packed components.

## Use AoSoA for SIMD or Workgroup Blocks

```c
struct AtomBlock {
    float x[BLOCK];
    float y[BLOCK];
    float z[BLOCK];
    float q[BLOCK];
};
```

This combines compact blocks with field-wise access.

# Hot and Cold Splitting

Bad:

```c
struct Atom {
    float3 position;
    float3 velocity;
    float3 force;
    float charge;
    float mass;
    int type;
    int flags;
    float parameters[32];
};
```

A force kernel may load or cache parts of this structure that it never uses.

Good:

```c
float4* position_type;
float4* force_energy;
float4* velocity_mass;

float4* rare_parameters;
int4* topology;
```

# Output Ownership

Every output should normally have one clear owner.

Examples:

* one GPU work-item owns one atom;
* one CPU thread owns one array range;
* one workgroup owns one small matrix;
* one grid work-item owns one output voxel;
* one assembly pass owns one force accumulator.

When several interactions contribute to one output, use:

* gather from neighbors;
* auxiliary interaction arrays;
* a reduction;
* graph coloring;
* fixed contribution slots;
* separate assembly passes.

Avoid arbitrary concurrent writes.

# Topology Layout

## Fixed Small Degree

For molecular bonds or small coordination numbers, use fixed-width slots:

```c
int4 neighbors;
```

or:

```c
int neighbors[MAX_NEIGHBORS];
```

Use `-1` for unused entries.

## Variable Degree

Use offsets plus a contiguous index array:

```c
offset[i] ... offset[i + 1]
neighbors[k]
```

This is CSR-style storage.

## Static Topology

Precompute:

* neighbor lists;
* bond-to-atom references;
* atom-to-bond references;
* signed contribution references;
* cell membership;
* type grouping;
* interpolation indices;
* constant coefficients.

Do not rediscover static relationships in every timestep.

# Active-Set Compaction

When only a fraction of items are active:

1. generate a compact list of active indices;
2. process only those indices;
3. rebuild the list only when activity changes significantly.

Do not branch over millions of inactive elements in every iteration.

# Reordering

Reorder objects to improve locality.

Useful sorting keys:

* spatial cell;
* atom type;
* material;
* interaction type;
* active/inactive state;
* molecule or replica;
* grid tile.

Keep mappings between reordered and original indices when required.

# Store Versus Recompute

Store a value when:

* it is expensive to compute;
* it is reused many times;
* its storage is compact;
* loading it is contiguous.

Recompute a value when:

* it is cheap;
* it would require a large array;
* it is used only once;
* loading it would be irregular;
* it can be derived from an index or small parameter set.

Typical values to recompute:

* grid coordinates;
* periodic shifts;
* signs;
* masks;
* polynomial terms;
* simple distances;
* basis indices;
* local transformation components.

# Loop and Kernel Fusion

Fuse operations when:

* they share the same iteration space;
* one produces a temporary used only by the next;
* the temporary can remain in registers or cache;
* fusion removes a large memory round trip.

Split operations when:

* the combined body becomes too large;
* register pressure becomes excessive;
* different phases need different parallel mappings;
* one phase is rare;
* synchronization becomes complicated.

# Performance Workflow

For every optimization:

1. identify the hot loop or kernel;
2. list every array it reads and writes;
3. estimate bytes transferred per output;
4. identify reused values;
5. identify output ownership;
6. remove unnecessary fields and intermediates;
7. improve contiguous access;
8. batch the work;
9. validate correctness;
10. measure the result.
