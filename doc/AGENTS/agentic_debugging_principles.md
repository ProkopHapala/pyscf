# Agentic Debugging Principles for Scientific Code

> Distilled from `Test_system_for_agentic_loops.chat.md`. Read the full discussion for rationale and war stories.

---

## 0. Core Philosophy

- **Debuggability over user experience.** Never hide bugs with graceful degradation.
- **Fail loud.** Crashes with full stack traces are preferable to silent corruption.
- **Root cause first.** Fix the fundamental issue; do not apply downstream workarounds.
- **Inventory before writing.** Review existing code for reusable functions before adding new ones.

---

## 1. Forcefield Debugging

### 1.1 NaN Padding — Invalid-Access Alarm Bell
- Fill all padding/ghost slots with `NaN` (or `inf`).
- Any accidental read poisons downstream math and makes the bug **visible immediately**.
- Never overload `NaN` with semantics like "pinned" — use a separate flag (e.g. `fixmask`).
- **Check**: after every GPU→CPU download, `assert np.isfinite(pos[real]).all()`.

### 1.2 Index Mapping — Bidirectional Consistency
For every forward mapping, verify the backward mapping exists:
- `neighs[i,k] = j` → `bkSlots[j,slot] = i` → `revSlot[i,k] = slot_in_j`
- Build a Python truth table from the original topology.
- Parse kernel logs and assert every expected pair appears with the correct action.

### 1.3 Conservation Laws — The Physics Floor
Run after every code change on a minimal system:
- **Linear momentum**: `|sum(m_i * v_i)|` conserved after one step.
- **Angular momentum**: `|sum(r_i × p_i) + sum(I_i * ω_i)|` conserved.
- **Energy drift** (MD only): track total energy; monotonic drift = unstable integrator.

**Technique**: start with *nodes-only* (no capping atoms) to test core rigid-body logic, then add caps.

### 1.4 Structured Per-Cluster Diagnostics
When corruption appears, know **which group failed first**:
- Print per-cluster: `n_real`, `COG`, `bbox_min`, `bbox_max`.
- The cluster with smallest `n_real` or most extreme bbox is usually the culprit.

### 1.5 Momentum Reset on Constraint Discontinuities
- After pin/unpin, drag start/end, teleport, or mode switch: **zero momentum buffers** before the next step.
- Stale momentum causes sudden jumps or divergence.

### 1.6 Kernel Debug Print with Targeting
- Gate OpenCL `printf` with compile-time flags and runtime verbosity.
- **Level 0**: silent (CI)
- **Level 1**: events (start/stop/toggle)
- **Level 2**: per-workgroup summaries (topology verification)
- **Level 3**: per-atom dumps (expensive; target with `DEBUG_GID_START/END`)
- Use component bitmasks to enable only collision, port, or rotation logs.

### 1.7 Isolation Before Combination
Test subsystems independently:
1. **Collision-only**: `ENABLE_COLL=1, ENABLE_PORT=0`
2. **Port-only**: `ENABLE_COLL=0, ENABLE_PORT=1`
3. **Combined**: both enabled
If combined fails but isolation passes, the bug is in interaction logic.

---

## 2. GUI Application Debugging

### 2.1 Strict Backend/Frontend Split
- Every UI action triggers **exactly one** backend mutation.
- Backend knows nothing about mouse coordinates, camera matrices, or widgets.
- GUI callbacks log *user actions*; backend methods log *internal decisions*. Do not mix.

### 2.2 Backend Must Be Headlessly Testable
- Every backend method must run without GUI, event loop, or OpenGL context.
- Write headless tests for topology mutations before touching the GUI.
- **KekuleBackend** is well-structured but currently lacks automated tests.

### 2.3 Structured Event Logging
Use tagged, machine-parseable prefixes:
```
[DRAG-START] idx=65 fixed=True invm=0.0833
[DRAG-CLAMP] idx=65 step=12.3 > 5.0; clamping
[FATAL] step_42: non-finite REAL atom positions at idx=[3,7,9]
```
Benefits: grep for regressions, replay sequences headlessly, assert "no `[FATAL]` during normal operation."

### 2.4 Fail-Loud Invariants After Every Mutation
After every topology operation, assert:
- All bond endpoints are alive.
- Neighbor list length matches bond count.
- No overlapping atoms (within tolerance).
- All positions finite.

This is critical because a single bad drag coordinate can corrupt the entire state.

### 2.5 Input Sanitization
Never trust GUI input (mouse positions, scroll deltas, spinbox values).
- Clamp drag steps to a user-configurable threshold.
- Print a warning when clamping fires — fail-loud, not silent.

### 2.6 Build Option Caching
- If backend uses OpenCL/CUDA/WebGPU with compile-time flags, cache the last flag tuple.
- Only recompile when flags actually change.
- Prevents multi-second GUI freezes and false "code is broken" panics.

---

## 3. Test Construction (Layer 0–5)

Build layers; do not write one giant integration test.

| Layer | Scope | Example |
|-------|-------|---------|
| 0 | **Topology assembly** (CPU only) | `add_ring(0,0)` → assert 6 atoms, 6 bonds |
| 1 | **Subsystem isolation** | Collision-only or port-only step; assert expected pairs |
| 2 | **Sequence replay** | Replay `[DRAG-START]…[DRAG-END]` from logs headlessly |
| 3 | **Conservation laws** | `|ΔP| < 1e-7`, `|ΔL| < 1e-7` after one step |
| 4 | **Combined dynamics + GUI** | Interactive mode with fail-loud invariant checks |
| 5 | **Platform parity** | Same tests on C++, OpenCL, WebGPU; accept 1e-4 relative diff |

---

## 4. Agentic Loop Protocol

When an agent modifies code:

1. **Inventory existing functions** before writing new ones.
2. **Write or update the headless test** first (if backend is touched).
3. **Run Layer 0** — topology/index checks. If these fail, everything downstream is garbage.
4. **Run Layer 1 or 2** — isolate the changed subsystem.
5. **Run Layer 3** — conservation laws must still hold.
6. **Run Layer 4** (GUI) — only after all headless tests pass.
7. **If corruption appears**: inspect structured per-cluster dump → identify smallest failing unit → trace back to last change touching that unit.
8. **Never suppress validation errors** — they are the fastest path to root cause.
9. **Never patch the renderer** to hide data corruption. Trace NaN/inf back to the computation.

---

## 5. Shared Utilities

- Consolidate duplicated helpers (`make_ports_from_neighs`, `write_xyz_frame`, `quat_rotate_vec`) into one module.
- A bug fix in one copy will not reach the others if they are duplicated.
- Composable test systems: build molecules in 3 lines, not 30.
