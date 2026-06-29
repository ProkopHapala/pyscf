import numpy as np
from pyscf.data import nist


ANG = nist.BOHR
SP_CART_FACTOR = {0: 0.282094791773878143, 1: 0.488602511902919921}


def cart_powers(lmax):
    rows = []
    offs = [0]
    for l in range(lmax + 1):
        for ix in range(l, -1, -1):
            for iy in range(l - ix, -1, -1):
                rows.append((ix, iy, l - ix - iy))
        offs.append(len(rows))
    return np.asarray(rows, dtype=np.int32), np.asarray(offs, dtype=np.int32)


def _eval_radial(r, expn, coeff):
    return np.exp(-np.outer(r * r, expn)).dot(coeff)


def _eval_radial_dr(r, expn, coeff):
    e = np.exp(-np.outer(r * r, expn))
    return ((-2.0 * r[:, None] * expn[None, :]) * e).dot(coeff)


def _midpoint_fit(y, d, h, ymid):
    b = 8.0 / h[:, None] * (ymid - 0.5 * (y[:-1] + y[1:]))
    out = d.copy()
    n = y.shape[0]
    if n < 2:
        return out
    a = np.zeros((n - 1, n), dtype=np.double)
    i = np.arange(n - 1)
    a[i, i] = 1.0
    a[i, i + 1] = -1.0
    m = a.dot(a.T)
    rhs = a.dot(d) - b
    return d - a.T.dot(np.linalg.solve(m, rhs))


class MappedHermiteRadialBasis:
    def __init__(self, mol, r0_ang=0.01, du=0.02, rmax_ang=8.0, midpoint_fit=True):
        self.mol = mol
        self.r0 = float(r0_ang) / ANG
        self.du = float(du)
        self.rmax = float(rmax_ang) / ANG
        self.midpoint_fit = bool(midpoint_fit)
        self._build()

    def _build(self):
        mol = self.mol
        umax = np.log1p(self.rmax / self.r0)
        self.u = np.arange(0.0, umax + 2.0 * self.du, self.du, dtype=np.double)
        self.r = self.r0 * np.expm1(self.u)
        self.nrad = self.u.size
        nshell = mol.nbas
        nctr_max = max(mol.bas_nctr(ib) for ib in range(nshell))
        self.values = np.zeros((nshell, nctr_max, self.nrad), dtype=np.float32)
        self.du_values = np.zeros_like(self.values)
        self.dy_values = np.zeros_like(self.values)
        self.shell_nctr = np.empty(nshell, dtype=np.int32)
        self.shell_l = np.empty(nshell, dtype=np.int32)
        self.shell_atom = np.empty(nshell, dtype=np.int32)
        self.shell_cart0 = np.empty(nshell, dtype=np.int32)
        self.shell_ctr_ir = np.full(nshell * nctr_max, -1, dtype=np.int32)
        cart0 = 0
        for ib in range(nshell):
            l = mol.bas_angular(ib)
            nctr = mol.bas_nctr(ib)
            expn = mol.bas_exp(ib)
            coeff = mol._libcint_ctr_coeff(ib) * SP_CART_FACTOR.get(l, 1.0)
            y = _eval_radial(self.r, expn, coeff)
            dy_dr = _eval_radial_dr(self.r, expn, coeff)
            dy_du = dy_dr * (self.r + self.r0)[:, None]
            if self.midpoint_fit:
                um = self.u[:-1] + 0.5 * self.du
                rm = self.r0 * np.expm1(um)
                ym = _eval_radial(rm, expn, coeff)
                dy_du = _midpoint_fit(y, dy_du, np.full(self.nrad - 1, self.du), ym)
            self.values[ib, :nctr] = y.T.astype(np.float32)
            self.du_values[ib, :nctr] = dy_du.T.astype(np.float32)
            dy = np.diff(y, axis=0)  # y[i+1]-y[i] in float64
            self.dy_values[ib, :nctr, :-1] = dy.T.astype(np.float32)
            self.dy_values[ib, :nctr, -1] = 0.0  # padding
            self.shell_nctr[ib] = nctr
            self.shell_l[ib] = l
            self.shell_atom[ib] = mol.bas_atom(ib)
            self.shell_cart0[ib] = cart0
            cart0 += nctr * ((l + 1) * (l + 2) // 2)
        self.ncart = cart0
        self.lmax = int(self.shell_l.max()) if nshell else 0
        self.nradial = int(self.shell_nctr.sum())
        self.radial_values = np.empty((self.nradial, self.nrad), dtype=np.float32)
        self.radial_du_values = np.empty_like(self.radial_values)
        self.radial_dy_values = np.empty_like(self.radial_values)
        self.radial_l = np.empty(self.nradial, dtype=np.int32)
        self.radial_atom = np.empty(self.nradial, dtype=np.int32)
        self.radial_cart0 = np.empty(self.nradial, dtype=np.int32)
        ir = 0
        for ib in range(nshell):
            l = self.shell_l[ib]
            ncart_l = (l + 1) * (l + 2) // 2
            for ic in range(self.shell_nctr[ib]):
                self.shell_ctr_ir[ib * nctr_max + ic] = ir
                self.radial_values[ir] = self.values[ib, ic]
                self.radial_du_values[ir] = self.du_values[ib, ic]
                self.radial_dy_values[ir] = self.dy_values[ib, ic]
                self.radial_l[ir] = l
                self.radial_atom[ir] = self.shell_atom[ib]
                self.radial_cart0[ir] = self.shell_cart0[ib] + ic * ncart_l
                ir += 1
        self.radial_nodes = np.empty((self.nradial, self.nrad, 2), dtype=np.float32)
        self.radial_nodes[:, :, 0] = self.radial_values
        self.radial_nodes[:, :, 1] = self.radial_du_values
        atom_radial_offset = np.zeros(mol.natm + 1, dtype=np.int32)
        for ir in range(self.nradial):
            atom_radial_offset[self.radial_atom[ir] + 1] += 1
        for ia in range(mol.natm):
            atom_radial_offset[ia + 1] += atom_radial_offset[ia]
        atom_radial_list = np.empty(self.nradial, dtype=np.int32)
        count = np.zeros(mol.natm, dtype=np.int32)
        for ir in range(self.nradial):
            ia = self.radial_atom[ir]
            atom_radial_list[atom_radial_offset[ia] + count[ia]] = ir
            count[ia] += 1
        self.atom_radial_offset = atom_radial_offset
        self.atom_radial_list = atom_radial_list
        powers, power_offsets = cart_powers(self.lmax)
        self.cart_ixyz = np.empty((self.ncart, 3), dtype=np.int32)
        self.cart_shell = np.empty(self.ncart, dtype=np.int32)
        self.cart_ctr = np.empty(self.ncart, dtype=np.int32)
        iao = 0
        for ib in range(nshell):
            l = self.shell_l[ib]
            pows = powers[power_offsets[l]:power_offsets[l + 1]]
            for ic in range(self.shell_nctr[ib]):
                for p in pows:
                    self.cart_shell[iao] = ib
                    self.cart_ctr[iao] = ic
                    self.cart_ixyz[iao] = p
                    iao += 1
        self.atom_coords = np.asarray(mol.atom_coords(), dtype=np.float32)
        self.natoms = mol.natm
        atom_shell_offset = np.zeros(self.natoms + 1, dtype=np.int32)
        for ib in range(nshell):
            atom_shell_offset[mol.bas_atom(ib) + 1] += 1
        for ia in range(self.natoms):
            atom_shell_offset[ia + 1] += atom_shell_offset[ia]
        atom_shell_list = np.empty(nshell, dtype=np.int32)
        count = np.zeros(self.natoms, dtype=np.int32)
        for ib in range(nshell):
            ia = mol.bas_atom(ib)
            atom_shell_list[atom_shell_offset[ia] + count[ia]] = ib
            count[ia] += 1
        self.atom_shell_offset = atom_shell_offset
        self.atom_shell_list = atom_shell_list

    def eval_cart(self, coords):
        coords = np.asarray(coords, dtype=np.double)
        out = np.empty((coords.shape[0], self.ncart), dtype=np.float32)
        u = np.log1p(np.linalg.norm(coords[:, None, :] - self.mol.atom_coords()[None, :, :], axis=2) / self.r0)
        for iao in range(self.ncart):
            ib = self.cart_shell[iao]
            ia = self.shell_atom[ib]
            ic = self.cart_ctr[iao]
            ui = u[:, ia] / self.du
            i = np.floor(ui).astype(np.int64)
            i = np.clip(i, 0, self.nrad - 2)
            t = ui - i
            t = np.clip(t, 0.0, 1.0)
            t2 = t * t
            t3 = t2 * t
            y0 = self.values[ib, ic, i].astype(np.double)
            y1 = self.values[ib, ic, i + 1].astype(np.double)
            d0 = self.du_values[ib, ic, i].astype(np.double)
            d1 = self.du_values[ib, ic, i + 1].astype(np.double)
            radial = (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * self.du * d0 + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * self.du * d1
            dxyz = coords - self.mol.atom_coord(ia)
            ix, iy, iz = self.cart_ixyz[iao]
            out[:, iao] = radial * dxyz[:, 0]**ix * dxyz[:, 1]**iy * dxyz[:, 2]**iz
        return out

    def eval_sph(self, coords):
        return self.eval_cart(coords).dot(self.mol.cart2sph_coeff(normalized='sp').astype(np.float32))
