import numpy as np
from pyscf import gto
from pyscf.data import nist


ANG = nist.BOHR
SP_CART_FACTOR = {0: 0.282094791773878143, 1: 0.488602511902919921}


def cart_powers(l):
    out = []
    for ix in range(l, -1, -1):
        for iy in range(l - ix, -1, -1):
            out.append((ix, iy, l - ix - iy))
    return out


class HermiteRadialTable:
    def __init__(self, expn, coeff, dr, rmax, midpoint_fit=True):
        self.dr = float(dr)
        self.r = np.arange(0.0, rmax + 2 * dr, dr)
        self.y = self.eval_exact(self.r, expn, coeff)
        d = self.eval_deriv_exact(self.r, expn, coeff)
        if midpoint_fit:
            d = self.fit_midpoints(d, expn, coeff)
        self.d = d

    @staticmethod
    def eval_exact(r, expn, coeff):
        return np.exp(-np.outer(r * r, expn)).dot(coeff)

    @staticmethod
    def eval_deriv_exact(r, expn, coeff):
        return ((-2.0 * r[:, None] * expn[None, :]) * np.exp(-np.outer(r * r, expn))).dot(coeff)

    def fit_midpoints(self, d0, expn, coeff):
        n = self.r.size
        if n < 2:
            return d0
        rm = self.r[:-1] + 0.5 * self.dr
        ym = self.eval_exact(rm, expn, coeff)
        b = 8.0 / self.dr * (ym - 0.5 * (self.y[:-1] + self.y[1:]))
        a = np.zeros((n - 1, n))
        ii = np.arange(n - 1)
        a[ii, ii] = 1.0
        a[ii, ii + 1] = -1.0
        m = a.dot(a.T)
        rhs = a.dot(d0) - b
        return d0 - a.T.dot(np.linalg.solve(m, rhs))

    def __call__(self, r):
        r = np.asarray(r)
        u = r / self.dr
        i = np.floor(u).astype(np.int64)
        if np.any(i < 0) or np.any(i >= self.r.size - 1):
            raise ValueError('radial coordinate outside Hermite table')
        t = u - i
        t2 = t * t
        t3 = t2 * t
        y0 = self.y[i]
        y1 = self.y[i + 1]
        d0 = self.d[i]
        d1 = self.d[i + 1]
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        return h00[:, None] * y0 + h10[:, None] * self.dr * d0 + h01[:, None] * y1 + h11[:, None] * self.dr * d1


class RadialHermiteAO:
    def __init__(self, mol, dr_ang=0.1, rmax_ang=7.0, midpoint_fit=True):
        self.mol = mol
        self.dr = dr_ang / ANG
        self.rmax = rmax_ang / ANG
        self.tables = []
        for ib in range(mol.nbas):
            expn = mol.bas_exp(ib)
            coeff = mol._libcint_ctr_coeff(ib)
            self.tables.append(HermiteRadialTable(expn, coeff, self.dr, self.rmax, midpoint_fit=midpoint_fit))

    def eval_cart(self, coords):
        coords = np.asarray(coords, dtype=np.double)
        ao = np.empty((coords.shape[0], self.mol.nao_cart()), dtype=np.double)
        ia0 = 0
        for ib in range(self.mol.nbas):
            ia = self.mol.bas_atom(ib)
            l = self.mol.bas_angular(ib)
            nctr = self.mol.bas_nctr(ib)
            dxyz = coords - self.mol.atom_coord(ia)
            r = np.linalg.norm(dxyz, axis=1)
            radial = self.tables[ib](r) * SP_CART_FACTOR.get(l, 1.0)
            powers = cart_powers(l)
            for ic in range(nctr):
                for ix, iy, iz in powers:
                    ao[:, ia0] = radial[:, ic] * dxyz[:, 0]**ix * dxyz[:, 1]**iy * dxyz[:, 2]**iz
                    ia0 += 1
        return ao

    def eval_sph(self, coords):
        return self.eval_cart(coords).dot(self.mol.cart2sph_coeff(normalized='sp'))


def eval_cart_exact_python(mol, coords):
    coords = np.asarray(coords, dtype=np.double)
    ao = np.empty((coords.shape[0], mol.nao_cart()), dtype=np.double)
    ia0 = 0
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        nctr = mol.bas_nctr(ib)
        expn = mol.bas_exp(ib)
        coeff = mol._libcint_ctr_coeff(ib)
        dxyz = coords - mol.atom_coord(ia)
        r2 = np.einsum('gi,gi->g', dxyz, dxyz)
        radial = np.exp(-np.outer(r2, expn)).dot(coeff) * SP_CART_FACTOR.get(l, 1.0)
        for ic in range(nctr):
            for ix, iy, iz in cart_powers(l):
                ao[:, ia0] = radial[:, ic] * dxyz[:, 0]**ix * dxyz[:, 1]**iy * dxyz[:, 2]**iz
                ia0 += 1
    return ao


def report(name, ref, val):
    err = val - ref
    mask = np.abs(ref) > 1e-9
    rel = np.max(np.abs(err[mask]) / np.maximum(np.abs(ref[mask]), 1e-30)) if np.any(mask) else 0.0
    print(f'{name:28s} max_abs={np.max(np.abs(err)):.6e} rms={np.sqrt(np.mean(err * err)):.6e} max_rel_sig={rel:.6e}')


def main():
    rng = np.random.default_rng(123)
    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='cc-pvdz', unit='Angstrom', cart=False, verbose=0)
    centers = mol.atom_coords()
    n = 20000
    aidx = rng.integers(0, mol.natm, size=n)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    radius = (4.0 * rng.random(n) ** 0.7) / ANG
    coords = centers[aidx] + direction * radius[:, None]
    rmax_ang = np.max(np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=2)) * ANG + 0.2

    ref_cart = mol.eval_gto('GTOval_cart', coords)
    ref_sph = mol.eval_gto('GTOval_sph', coords)
    exact_cart = eval_cart_exact_python(mol, coords)
    exact_sph = exact_cart.dot(mol.cart2sph_coeff(normalized='sp'))
    print(f'npts={n} nao_sph={mol.nao_nr()} nao_cart={mol.nao_cart()} nbas={mol.nbas} rmax={rmax_ang:.3f} Ang')
    report('exact_python cart', ref_cart, exact_cart)
    report('exact_python sph', ref_sph, exact_sph)
    e_exact = np.max(np.abs(exact_sph - ref_sph))
    if e_exact > 1e-10:
        raise SystemExit(f'exact Python AO formula mismatch: {e_exact}')

    plain_01 = None
    mid_01 = None
    for dr in (0.1, 0.05, 0.025, 0.0125, 0.00625):
        tab_plain = RadialHermiteAO(mol, dr_ang=dr, rmax_ang=rmax_ang, midpoint_fit=False)
        tab_mid = RadialHermiteAO(mol, dr_ang=dr, rmax_ang=rmax_ang, midpoint_fit=True)
        plain_sph = tab_plain.eval_sph(coords)
        mid_sph = tab_mid.eval_sph(coords)
        e_plain = np.max(np.abs(plain_sph - ref_sph))
        e_mid = np.max(np.abs(mid_sph - ref_sph))
        rms_mid = np.sqrt(np.mean((mid_sph - ref_sph) ** 2))
        print(f'dr={dr:8.5f} Ang plain_max_abs={e_plain:.6e} midfit_max_abs={e_mid:.6e} midfit_rms={rms_mid:.6e}')
        if dr == 0.1:
            plain_01 = e_plain
            mid_01 = e_mid
    if not (np.isfinite(mid_01) and mid_01 < plain_01):
        raise SystemExit('midpoint-fitted Hermite table did not improve dr=0.1 Ang AO accuracy')


if __name__ == '__main__':
    main()
