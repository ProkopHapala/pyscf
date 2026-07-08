"""gpu_timing.py — trustworthy OpenCL kernel timing for XC stage profiling.

Wall times without queue.finish() measure enqueue latency, not GPU work — misleading
for async OpenCL. This module pairs queue drain with optional clGetEventProfilingInfo
so benchmarks can report both total stage latency (wall) and device execution (CL).

Requires CommandQueue(PROFILING_ENABLE) — set in pyscf/OpenCL/__init__.py:init_device.

- **profile_kernel** — single NDRange launch; records wall + event time per kernel
- **profile_call** — multi-kernel Python callable (PBE chain, c2s matmul); wall only
- **event_elapsed_s** — clGetEventProfilingInfo wrapper with PyOpenCL fallback

Used by XCGridPlan.nr_rks_hermite_onthefly(profile=True) and precomp fused paths;
aggregated in xc_grid._finalize_gpu_timing → plan.last_timing.
"""
import time as _time

import pyopencl as cl

# === AUTO-DOC BEGIN ===
# enqueue_nd_range_profiled — launch + optional finish; returns completion event
# event_elapsed_s — GPU seconds from profiling events (0 if unavailable)
# profile_kernel — wall after finish + optional CL event time for one kernel
# profile_call — wall after finish for arbitrary GPU work (no per-kernel events)
# === AUTO-DOC END ===


def enqueue_nd_range_profiled(queue, kernel, global_size, local_size, wait_finish=True):
    '''Launch kernel; optionally block until complete. Returns completion event.'''
    evt = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size, wait_for=None)
    if wait_finish:
        queue.finish()
    return evt


def event_elapsed_s(evt):
    '''GPU execution time from profiling events (seconds), or 0 if unavailable.'''
    if evt is None:
        return 0.0
    try:
        start = evt.get_profiling_info(cl.profiling_info.START)
        end = evt.get_profiling_info(cl.profiling_info.END)
        return (end - start) * 1e-9
    except Exception:
        try:
            return (evt.profile.end - evt.profile.start) * 1e-9
        except Exception:
            return 0.0


def profile_kernel(queue, kernel, global_size, local_size, timing, wall_key, cl_key=None):
    '''Wall time with queue.finish() + optional OpenCL event time (true GPU kernel work).'''
    if timing is None:
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        queue.finish()
        return None
    t0 = _time.perf_counter()
    evt = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size, wait_for=None)
    queue.finish()
    timing[wall_key] = _time.perf_counter() - t0
    if cl_key is not None:
        timing[cl_key] = event_elapsed_s(evt)
    return evt


def profile_call(queue, fn, timing, wall_key, cl_key=None):
    '''Profile arbitrary GPU work (e.g. matmul chain) with finish before/after.'''
    if timing is None:
        fn()
        queue.finish()
        return
    queue.finish()
    t0 = _time.perf_counter()
    fn()
    queue.finish()
    timing[wall_key] = _time.perf_counter() - t0
    if cl_key is not None:
        timing[cl_key] = timing[wall_key]
