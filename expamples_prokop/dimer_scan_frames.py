'''dimer_scan_frames.py — rigid dimer scan geometry from one relaxed XYZ + distance grid.

Fragment 2 [n0,natom) translates along the anchor vector so inter-fragment distance matches
each target r; fragment 1 stays fixed. Auto anchor = closest cross-fragment pair (prefers O···O).
Used by profile_dimer_scan.py when --geom is set (no pre-built trajectory).
'''
import re

import numpy as np


def read_xyz_atom(path):
    with open(path) as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + natom]:
        parts = line.split()
        if not re.match(r'^[A-Z][a-z]?$', parts[0]):
            continue
        atoms.append((parts[0], np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)))
    if len(atoms) != natom:
        raise ValueError(f'{path}: parsed {len(atoms)} atoms, header says {natom}')
    return atoms


def atoms_to_pyscf(atoms):
    return '; '.join(f'{el} {x:.10f} {y:.10f} {z:.10f}' for el, (x, y, z) in atoms)


def load_distances_file(path):
    rs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rs.append(float(line.split()[0]))
    if not rs:
        raise ValueError(f'no distances in {path}')
    return np.asarray(rs, dtype=float)


def _check_n0(n0, natom):
    if not (0 < n0 < natom):
        raise ValueError(f'n0={n0} invalid for natom={natom} (need 0 < n0 < natom)')


def closest_cross_fragment(atoms, n0, i_fixed=None, i_mobile=None, prefer_o=True):
    '''Return anchor pair (i_fixed in frag1, i_mobile in frag2). Auto: closest cross-fragment pair; prefer O···O if both sides have O.'''
    _check_n0(n0, len(atoms))
    if i_fixed is not None or i_mobile is not None:
        if i_fixed is None or i_mobile is None:
            raise ValueError('set both anchor-fixed and anchor-mobile, or neither')
        if not (0 <= i_fixed < n0):
            raise ValueError(f'anchor-fixed={i_fixed} must be in [0, {n0})')
        if not (n0 <= i_mobile < len(atoms)):
            raise ValueError(f'anchor-mobile={i_mobile} must be in [{n0}, {len(atoms)})')
        return i_fixed, i_mobile
    left = [(i, atoms[i][1]) for i in range(n0)]
    right = [(i, atoms[i][1]) for i in range(n0, len(atoms))]
    if prefer_o:
        left_o = [(i, p) for i, p in left if atoms[i][0] == 'O']
        right_o = [(i, p) for i, p in right if atoms[i][0] == 'O']
        if left_o and right_o:
            left, right = left_o, right_o
    best = min(((float(np.linalg.norm(pi - pj)), i, j) for i, pi in left for j, pj in right))
    return best[1], best[2]


def rigid_shift_frames(atoms, n0, i_fixed, i_mobile, distances):
    apos = np.array([a[1] for a in atoms], dtype=float)
    direction = apos[i_mobile] - apos[i_fixed]
    direction = direction / np.linalg.norm(direction)
    r0 = float(np.linalg.norm(apos[i_mobile] - apos[i_fixed]))
    mobile = list(range(n0, len(atoms)))
    out = []
    for r in distances:
        new = apos.copy()
        shift = direction * (float(r) - r0)
        for idx in mobile:
            new[idx] += shift
        framed = [(atoms[k][0], new[k]) for k in range(len(atoms))]
        out.append((float(r), atoms_to_pyscf(framed), float(r), i_fixed, i_mobile))
    return out, r0


def frames_from_relaxed_xyz(xyz_path, distances, n0, i_fixed=None, i_mobile=None, prefer_o=True):
    atoms = read_xyz_atom(xyz_path)
    i_fix, i_mob = closest_cross_fragment(atoms, n0, i_fixed, i_mobile, prefer_o=prefer_o)
    frames, r0 = rigid_shift_frames(atoms, n0, i_fix, i_mob, distances)
    meta = dict(n0=n0, natom=len(atoms), anchor_fixed=i_fix, anchor_mobile=i_mob, r0=r0, anchor_pair=f'{atoms[i_fix][0]}({i_fix})···{atoms[i_mob][0]}({i_mob})')
    return frames, meta
