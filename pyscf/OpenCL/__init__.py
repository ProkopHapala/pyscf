import os
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from .tile_config import (
    get_active_tile_config, get_default_tile_config, set_active_tile_config,
    tile_config_from_env, sync_module_constants,
)

_KERNELS_FILE = os.path.join(os.path.dirname(__file__), 'kernels.cl')
_PBE_FILE = os.path.join(os.path.dirname(__file__), 'pbe.cl')

_ctx = None
_queue = None
_prg = None
_active_tile_config = None

def reset_opencl():
    '''Drop cached context/program (e.g. before recompiling with new tile flags).'''
    global _ctx, _queue, _prg, _active_tile_config
    _ctx = _queue = _prg = _active_tile_config = None

def init_device(device_name=None, quiet=False, tile_config=None, force_rebuild=False):
    '''Initialize OpenCL context and queue. Prefers NVIDIA by default.

    tile_config: TileConfig or dict of overrides; default from env then built-in defaults.
    force_rebuild: recompile kernels even if device already initialized.
    '''
    global _ctx, _queue, _prg, _active_tile_config
    if tile_config is not None:
        cfg = set_active_tile_config(tile_config)
    elif _active_tile_config is not None:
        cfg = _active_tile_config
    else:
        cfg = set_active_tile_config(tile_config_from_env())
    sync_module_constants()

    if _ctx is not None and not force_rebuild and _prg is not None and _active_tile_config == cfg:
        return _ctx, _queue

    rebuilding = _ctx is None or force_rebuild or _prg is None or _active_tile_config != cfg
    if rebuilding:
        from . import xc_grid as _xcg
        _xcg.clear_xc_plan_cache()

    platforms = cl.get_platforms()
    selected_device = None

    if device_name is not None:
        for plat in platforms:
            for dev in plat.get_devices():
                if device_name.lower() in dev.name.lower():
                    selected_device = dev
                    break
            if selected_device:
                break

    if selected_device is None:
        for plat in platforms:
            for dev in plat.get_devices():
                if 'nvidia' in dev.name.lower() or 'nvidia' in plat.name.lower():
                    selected_device = dev
                    break
            if selected_device:
                break

    if selected_device is None:
        for plat in platforms:
            devs = plat.get_devices()
            if devs:
                selected_device = devs[0]
                break

    if selected_device is None:
        raise RuntimeError('No OpenCL device found')

    _ctx = cl.Context([selected_device])
    _queue = cl.CommandQueue(_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    with open(_PBE_FILE, 'r') as f:
        pbe_src = f.read()
    with open(_KERNELS_FILE, 'r') as f:
        kernel_src = pbe_src + '\n' + f.read()
    build_opts = cfg.build_options()
    try:
        _prg = cl.Program(_ctx, kernel_src).build(options=build_opts)
    except cl.RuntimeError as e:
        build_log = getattr(e, 'build_log', None) or str(e)
        raise RuntimeError(f'OpenCL build failed ({build_opts}):\n{build_log}') from e
    _active_tile_config = cfg

    if not quiet:
        print(f'OpenCL device: {selected_device.name}')
        print(f'  tile build: {build_opts}')
    return _ctx, _queue

def get_ctx():
    if _ctx is None:
        init_device()
    return _ctx

def get_queue():
    if _queue is None:
        init_device()
    return _queue

def get_prg():
    if _prg is None:
        init_device()
    return _prg

def to_device(arr, dtype=np.float32):
    '''Transfer numpy array to device as float32.'''
    if arr.dtype != dtype:
        arr = arr.astype(dtype)
    return cl_array.to_device(get_queue(), np.ascontiguousarray(arr))

def to_host(darr):
    '''Transfer device array back to host.'''
    return darr.get()

def round_up(val, multiple):
    return ((val + multiple - 1) // multiple) * multiple

def get_device_mem_info():
    ctx = get_ctx()
    dev = ctx.devices[0]
    try:
        info = dev.get_info(cl.device_info.GLOBAL_MEM_SIZE)
        return int(info)
    except Exception:
        return 4 * 1024 * 1024 * 1024  # fallback 4GB
