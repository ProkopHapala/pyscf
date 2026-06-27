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
        self.shell_nctr = np.empty(nshell, dtype=np.int32)
        self.shell_l = np.empty(nshell, dtype=np.int32)
        self.shell_atom = np.empty(nshell, dtype=np.int32)
        self.shell_cart0 = np.empty(nshell, dtype=np.int32)
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
            self.shell_nctr[ib] = nctr
            self.shell_l[ib] = l
            self.shell_atom[ib] = mol.bas_atom(ib)
            self.shell_cart0[ib] = cart0
            cart0 += nctr * ((l + 1) * (l + 2) // 2)
        self.ncart = cart0
        lmax = int(self.shell_l.max()) if nshell else 0
        powers, power_offsets = cart_powers(lmax)
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
