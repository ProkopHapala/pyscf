import os
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

_KERNELS_FILE = os.path.join(os.path.dirname(__file__), 'kernels.cl')

_ctx = None
_queue = None
_prg = None

def init_device(device_name=None):
    '''Initialize OpenCL context and queue. Prefers NVIDIA by default.'''
    global _ctx, _queue, _prg
    if _ctx is not None:
        return _ctx, _queue

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
    _queue = cl.CommandQueue(_ctx)

    with open(_KERNELS_FILE, 'r') as f:
        kernel_src = f.read()
    _prg = cl.Program(_ctx, kernel_src).build()

    print(f'OpenCL device: {selected_device.name}')
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
