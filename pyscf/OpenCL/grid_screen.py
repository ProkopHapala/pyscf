"""Grid-point tile atom screening for sparse OpenCL XC (DFTB-style).

DFTB reference: pyBall/DFTB/Grid_dftb.py `build_tasks_gpu`, Grid_dftb.cl `count_atoms_per_block`.
Each atom carries a cutoff radius Rcut; a grid tile (bbox) keeps only atoms whose
sphere overlaps the tile (sphere-AABB test). Here a tile is one NPTILE chunk of
DFT grid points (not an 8^3 voxel block).

Cutoff per atom: max over its radial splines of the last radius where |R(r)| > eps*peak.
Optional margin accounts for angular factors (x^l y^m z^n).
"""
import numpy as np

from . import round_up


def shell_rcut_bohr(radial_values, r_bohr, eps=1e-7):
    '''Last radius (Bohr) where |R| >= eps * max|R| on the Hermite grid.'''
    rv = np.abs(np.asarray(radial_values, dtype=np.float64))
    r = np.asarray(r_bohr, dtype=np.float64)
    peak = float(rv.max()) if rv.size else 0.0
    if peak < 1e-30:
        return 0.0
    thresh = eps * peak
    above = np.nonzero(rv >= thresh)[0]
    if above.size == 0:
        return 0.0
    return float(r[above[-1]])


def compute_atom_rcut(plan, eps=1e-7, margin_bohr=0.5, l_angular_margin=True):
    '''Per-atom cutoff radius in Bohr (conservative for all shells on the atom).

    margin_bohr: added after radial tail (covers polynomial angular extent).
    l_angular_margin: scale margin by sqrt(l+1) per shell when taking max.
    '''
    natoms = plan.natoms
    rcut = np.zeros(natoms, dtype=np.float32)
    r_bohr = plan.r
    for ia in range(natoms):
        off = int(plan.atom_radial_offset[ia])
        ns = int(plan.atom_radial_offset[ia + 1] - off)
        rmax = 0.0
        for s in range(ns):
            ir = int(plan.atom_radial_list[off + s])
            l = int(plan.radial_l[ir])
            rs = shell_rcut_bohr(plan.radial_values[ir], r_bohr, eps=eps)
            if l_angular_margin and l > 0:
                rs += margin_bohr * np.sqrt(float(l + 1))
            else:
                rs += margin_bohr
            rmax = max(rmax, rs)
        rcut[ia] = rmax
    return rcut


def sphere_aabb_overlap(center, radius, bbox_min, bbox_max):
    '''True if sphere (center, radius) intersects axis-aligned box [min, max].'''
    closest = np.clip(center, bbox_min, bbox_max)
    diff = center - closest
    return float(np.dot(diff, diff)) < radius * radius


def build_gtile_atom_lists(coords, atom_coords, atom_rcut, nptile, natoms, pair_screen=True):
    '''Build per-gTile lists of active atoms (and optional active pairs).

    Returns dict with:
      gtile_atom_off  (n_gtile+1,) int32 CSR offsets into gtile_atom_list
      gtile_atom_list (n_active,) int32 atom indices
      gtile_pair_off  (n_gtile+1,) int32 offsets into gtile_pair_ij (if pair_screen)
      gtile_pair_ij   (n_pairs*2,) int32 flattened (ia, ja) with ia<=ja
      stats           dict of screening statistics
    '''
    coords = np.asarray(coords, dtype=np.float64)
    atom_coords = np.asarray(atom_coords, dtype=np.float64)
    atom_rcut = np.asarray(atom_rcut, dtype=np.float64)
    ngrids = coords.shape[0]
    n_gtile = round_up(ngrids, nptile) // nptile

    gtile_atom_off = np.zeros(n_gtile + 1, dtype=np.int32)
    gtile_atom_list = []
    gtile_pair_off = np.zeros(n_gtile + 1, dtype=np.int32)
    gtile_pair_ij = []
    natom_total = 0
    npair_total = 0
    na_max = 0

    for gtile in range(n_gtile):
        g0 = gtile * nptile
        g1 = min(g0 + nptile, ngrids)
        if g0 >= ngrids:
            gtile_atom_off[gtile + 1] = gtile_atom_off[gtile]
            gtile_pair_off[gtile + 1] = gtile_pair_off[gtile]
            continue
        pts = coords[g0:g1]
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        active = []
        for ia in range(natoms):
            if sphere_aabb_overlap(atom_coords[ia], atom_rcut[ia], bbox_min, bbox_max):
                active.append(ia)
        na = len(active)
        na_max = max(na_max, na)
        gtile_atom_list.extend(active)
        gtile_atom_off[gtile + 1] = gtile_atom_off[gtile] + na
        natom_total += na

        if pair_screen:
            pairs = []
            for ii, ia in enumerate(active):
                for jj in range(ii, len(active)):
                    ja = active[jj]
                    rij = atom_coords[ja] - atom_coords[ia]
                    if np.dot(rij, rij) <= (atom_rcut[ia] + atom_rcut[ja]) ** 2:
                        pairs.extend((ia, ja))
            gtile_pair_ij.extend(pairs)
            gtile_pair_off[gtile + 1] = gtile_pair_off[gtile] + len(pairs) // 2
            npair_total += len(pairs) // 2
        else:
            gtile_pair_off[gtile + 1] = gtile_pair_off[gtile]

    dense_atom_loops = n_gtile * natoms
    dense_pair_loops = n_gtile * natoms * natoms
    stats = {
        'n_gtile': n_gtile,
        'ngrids': ngrids,
        'natoms': natoms,
        'na_max': na_max,
        'na_mean': natom_total / max(n_gtile, 1),
        'npair_mean': npair_total / max(n_gtile, 1),
        'dense_atom_loops_per_gtile': natoms,
        'sparse_atom_factor': natoms / max(natom_total / max(n_gtile, 1), 1e-9),
        'dense_pair_loops_per_gtile': natoms * natoms,
        'sparse_pair_factor': (natoms * natoms) / max(npair_total / max(n_gtile, 1), 1e-9),
    }
    return {
        'gtile_atom_off': np.asarray(gtile_atom_off, dtype=np.int32),
        'gtile_atom_list': np.asarray(gtile_atom_list, dtype=np.int32),
        'gtile_pair_off': np.asarray(gtile_pair_off, dtype=np.int32),
        'gtile_pair_ij': np.asarray(gtile_pair_ij, dtype=np.int32),
        'atom_rcut': atom_rcut.astype(np.float32),
        'stats': stats,
    }
