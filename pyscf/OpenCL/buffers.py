import numpy as np
import pyopencl as cl

class CLBuffer:
    '''Preallocated OpenCL buffer with explicit upload/download/finish.

    Avoids creating/destroying GPU buffers on the fly.
    '''
    def __init__(self, shape, dtype=np.float32, flags=cl.mem_flags.READ_WRITE):
        from . import get_ctx, get_queue
        self.ctx = get_ctx()
        self.queue = get_queue()
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.buf = cl.Buffer(self.ctx, flags, self.nbytes)
        self.host = np.zeros(self.shape, dtype=self.dtype)

    def upload(self, host_arr=None):
        '''Copy host data to device. If host_arr is None, upload internal host array.'''
        if host_arr is not None:
            self.host[:] = host_arr
        cl.enqueue_copy(self.queue, self.buf, np.ascontiguousarray(self.host, dtype=self.dtype))

    def download(self):
        '''Copy device data to internal host array and return it.'''
        cl.enqueue_copy(self.queue, self.host, self.buf)
        return self.host

    def finish(self):
        self.queue.finish()

    def __del__(self):
        if hasattr(self, 'buf'):
            self.buf.release()
