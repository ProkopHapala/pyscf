"""Compile-time OpenCL XC tile parameters (passed as -D flags to kernels.cl).

Single source of truth for host launch geometry and CL build options.
Sweep with expamples_prokop/sweep_opencl_tiles.py.
"""
import os
from dataclasses import dataclass, asdict, fields


def _log2_pow2(n):
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f'tile size must be a positive power of 2, got {n}')
    return n.bit_length() - 1


@dataclass(frozen=True)
class TileConfig:
    NPTILE: int = 64
    NATILE: int = 2
    MAX_AO_ATOM: int = 16
    MAX_SHELL: int = 6
    WGS_VMAT: int = 256
    MAX_ITILE: int = 32  # legacy; OTF rho streams i-tiles (no private MAX_ITILE dumps)

    def __post_init__(self):
        for name in ('NPTILE', 'NATILE', 'MAX_AO_ATOM', 'WGS_VMAT'):
            _log2_pow2(getattr(self, name))
        if self.MAX_SHELL <= 0:
            raise ValueError(f'MAX_SHELL must be positive, got {self.MAX_SHELL}')
        if self.MAX_ITILE <= 0:
            raise ValueError(f'MAX_ITILE must be positive, got {self.MAX_ITILE}')
        pt = self.NPTILE * self.NATILE
        if self.WGS_VMAT < pt:
            raise ValueError(f'WGS_VMAT={self.WGS_VMAT} < NPTILE*NATILE={pt}')

    @property
    def WGS_TILED(self):
        return self.NPTILE * self.NATILE

    @property
    def LOG_NPTILE(self):
        return _log2_pow2(self.NPTILE)

    @property
    def LOG_NATILE(self):
        return _log2_pow2(self.NATILE)

    @property
    def LOG_MAX_AO_ATOM(self):
        return _log2_pow2(self.MAX_AO_ATOM)

    def min_natoms_overflow(self):
        '''Smallest natoms that exceeds MAX_ITILE i-tiles (rho prepass private arrays).'''
        return self.MAX_ITILE * self.NATILE + 1

    def build_options(self):
        defs = (
            ('NPTILE', self.NPTILE),
            ('LOG_NPTILE', self.LOG_NPTILE),
            ('NATILE', self.NATILE),
            ('LOG_NATILE', self.LOG_NATILE),
            ('MAX_AO_ATOM', self.MAX_AO_ATOM),
            ('LOG_MAX_AO_ATOM', self.LOG_MAX_AO_ATOM),
            ('MAX_SHELL', self.MAX_SHELL),
            ('WGS_VMAT', self.WGS_VMAT),
            ('MAX_ITILE', self.MAX_ITILE),
        )
        return ' '.join(f'-D{name}={val}' for name, val in defs)

    def cache_key(self):
        return tuple(getattr(self, f.name) for f in fields(self))


_DEFAULT = TileConfig()
_ACTIVE = _DEFAULT


def get_default_tile_config():
    return _DEFAULT


def get_active_tile_config():
    return _ACTIVE


def set_active_tile_config(cfg):
    global _ACTIVE
    if isinstance(cfg, dict):
        cfg = TileConfig(**cfg)
    elif not isinstance(cfg, TileConfig):
        raise TypeError(f'expected TileConfig or dict, got {type(cfg)}')
    _ACTIVE = cfg
    return cfg


def tile_config_from_env():
    '''Optional overrides: OPENCL_NPTILE, OPENCL_NATILE, OPENCL_WGS_VMAT, OPENCL_MAX_ITILE.'''
    kw = {}
    for env, key, cast in (
        ('OPENCL_NPTILE', 'NPTILE', int),
        ('OPENCL_NATILE', 'NATILE', int),
        ('OPENCL_WGS_VMAT', 'WGS_VMAT', int),
        ('OPENCL_MAX_ITILE', 'MAX_ITILE', int),
        ('OPENCL_MAX_AO_ATOM', 'MAX_AO_ATOM', int),
    ):
        if env in os.environ:
            kw[key] = cast(os.environ[env])
    if kw:
        return TileConfig(**{**asdict(_DEFAULT), **kw})
    return _DEFAULT


# Module-level shortcuts (match active config after init_device)
NPTILE = _ACTIVE.NPTILE
NATILE = _ACTIVE.NATILE
WGS_VMAT = _ACTIVE.WGS_VMAT
WGS_TILED = _ACTIVE.WGS_TILED


def sync_module_constants():
    global NPTILE, NATILE, WGS_VMAT, WGS_TILED
    cfg = _ACTIVE
    NPTILE = cfg.NPTILE
    NATILE = cfg.NATILE
    WGS_VMAT = cfg.WGS_VMAT
    WGS_TILED = cfg.WGS_TILED
