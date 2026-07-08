'''Grid-tile parallelism: embarrassingly parallel over iG, disjoint output slices.'''
from concurrent.futures import ThreadPoolExecutor

import numpy
from pyscf import lib

# Reused across nr_rks calls when n_workers unchanged.
_POOL = None
_POOL_N = 0

TILE_SIZE_DEFAULT = 4096


def default_n_workers():
    return max(1, lib.num_threads())


def default_tile_size(ngrids, n_workers):
    '''~equal tiles per worker, at least 512 points per tile.'''
    nw = max(1, int(n_workers))
    ts = (int(ngrids) + nw - 1) // nw
    return max(512, min(ts, TILE_SIZE_DEFAULT * 2))


def tile_ranges(ngrids, tile_size):
    tile_size = max(1, int(tile_size))
    for g0 in range(0, ngrids, tile_size):
        yield g0, min(g0 + tile_size, ngrids)


def _get_pool(n_workers):
    global _POOL, _POOL_N
    n_workers = max(1, int(n_workers))
    if n_workers <= 1:
        return None
    if _POOL is None or _POOL_N != n_workers:
        if _POOL is not None:
            _POOL.shutdown(wait=False)
        _POOL = ThreadPoolExecutor(max_workers=n_workers)
        _POOL_N = n_workers
    return _POOL


def get_pool(n_workers):
    return _get_pool(n_workers)


def shutdown_pool():
    global _POOL, _POOL_N
    if _POOL is not None:
        _POOL.shutdown(wait=True)
        _POOL = None
        _POOL_N = 0


def parallel_grid_fill(ngrids, tile_size, n_workers, fill_fn, executor=None):
    '''Run fill_fn(g0, g1) over grid tiles; each writes a disjoint output slice.

    fill_fn must be thread-safe (writes non-overlapping regions only).
    Serial path when n_workers <= 1 or only one tile.
    '''
    ranges = list(tile_ranges(ngrids, tile_size))
    if len(ranges) <= 1 or n_workers <= 1:
        for g0, g1 in ranges:
            fill_fn(g0, g1)
        return
    pool = executor or _get_pool(n_workers)
    if pool is None:
        for g0, g1 in ranges:
            fill_fn(g0, g1)
        return
    list(pool.map(lambda gr: fill_fn(gr[0], gr[1]), ranges))


def parallel_grid_reduce(ngrids, tile_size, n_workers, tile_fn, out, executor=None):
    '''tile_fn(g0,g1) -> partial array; summed into out (vmat path).'''
    ranges = list(tile_ranges(ngrids, tile_size))
    if len(ranges) <= 1 or n_workers <= 1:
        for g0, g1 in ranges:
            out += tile_fn(g0, g1)
        return out
    pool = executor or _get_pool(n_workers)
    if pool is None:
        for g0, g1 in ranges:
            out += tile_fn(g0, g1)
        return out
    for part in pool.map(lambda gr: tile_fn(gr[0], gr[1]), ranges):
        out += part
    return out
