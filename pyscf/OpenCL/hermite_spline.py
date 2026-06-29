'''Mapped log-grid Hermite splines for GTO radial factors.

Grid modes (see radial_grid):
  log   — uniform du in u = b·log1p(r/r0); node count grows with b if du fixed.
  power — fixed N nodes; r_i = r0·(rmax/r0)^((i/(N-1))^(1/b)); b>1 clusters near origin.

Fitted target: **contracted** radial R(r) = Σ_k c_k exp(-α_k r²) from libcint coefficients.

interp_space:
  'u' (production): Hermite in mapped u; knot tangents dy/du.
  'r': same r_i nodes; Hermite in physical r; tangents dy/dr.

tangents (knot derivative source):
  analytic   — dR/dr, d²R/dr² from exact GTO at each knot (no LSQ).
  quadrature — LSQ fit knot tangents to interior collocation samples.

origin_knot: prepend analytic knot at r=0 so the first physical segment [0,r0] has a
  left neighbor (fixes half-line boundary at the grid start).
'''
from __future__ import annotations

import numpy as np

from pyscf.data import nist

ANG = nist.BOHR
SP_CART_FACTOR = {0: 0.282094791773878143, 1: 0.488602511902919921}


def u_from_r(r, r0, map_b=1.0):
    map_b = float(map_b)
    return map_b * np.log1p(np.asarray(r, dtype=np.double) / float(r0))


def r_from_u(u, r0, map_b=1.0):
    map_b = float(map_b)
    return float(r0) * np.expm1(np.asarray(u, dtype=np.double) / map_b)


def mapped_u_grid(r0, du, rmax, map_b=1.0):
    r0 = float(r0)
    du = float(du)
    map_b = float(map_b)
    umax = map_b * np.log1p(float(rmax) / r0)
    u = np.arange(0.0, umax + 2.0 * du, du, dtype=np.double)
    r = r_from_u(u, r0, map_b)
    return u, r


def mapped_power_grid(r0, rmax, n_nodes, beta=1.0, map_b=1.0):
    '''Fixed node count; beta>1 packs nodes toward r0 via t^(1/beta) in log-radius.'''
    r0, rmax, beta, map_b = float(r0), float(rmax), float(beta), float(map_b)
    n = max(int(n_nodes), 2)
    t = np.linspace(0.0, 1.0, n, dtype=np.double)
    s = np.power(t, 1.0 / beta)
    r = r0 * np.power(rmax / r0, s)
    u = map_b * np.log1p(r / r0)
    return u, r


def mapped_uniform_grid(r0, rmax, n_nodes, map_b=1.0):
    '''Fixed N nodes uniformly spaced in physical r.'''
    r0, rmax, map_b = float(r0), float(rmax), float(map_b)
    n = max(int(n_nodes), 2)
    r = np.linspace(r0, rmax, n, dtype=np.double)
    u = map_b * np.log1p(r / r0)
    return u, r


def radial_grid(r0, rmax, *, map_b=1.0, du=None, n_nodes=None, grid='log'):
    '''Build (u, r, h_u, grid). uniform|power: fixed n_nodes; log: uniform du.'''
    grid = grid.lower()
    map_b = float(map_b)
    if grid == 'log':
        if du is None:
            raise ValueError('grid=log requires du')
        u, r = mapped_u_grid(r0, du, rmax, map_b)
    elif grid == 'power':
        if n_nodes is None:
            raise ValueError('grid=power requires n_nodes')
        u, r = mapped_power_grid(r0, rmax, n_nodes, beta=map_b, map_b=map_b)
    elif grid == 'uniform':
        if n_nodes is None:
            raise ValueError('grid=uniform requires n_nodes')
        u, r = mapped_uniform_grid(r0, rmax, n_nodes, map_b=map_b)
    else:
        raise ValueError(f'grid={grid!r}')
    return u, r, np.diff(u), grid


def reference_node_count(r0, du, rmax, map_b=1.0):
    return int(mapped_u_grid(r0, du, rmax, map_b)[0].size)


node_count = reference_node_count


def node_r_distribution(r0_ang, du_ref, rmax_ang, map_b, *, beta_ref=1.0, grid='power', n_nodes=None):
    '''Physical radii r_i (Å). power: β clusters origin; uniform: equal Δr; log: uniform du in u.'''
    r0 = float(r0_ang) / ANG
    rmax = float(rmax_ang) / ANG
    grid = grid.lower()
    n_ref = n_nodes if n_nodes is not None else reference_node_count(r0, du_ref, rmax, beta_ref)
    if grid == 'power':
        u, r = mapped_power_grid(r0, rmax, n_ref, beta=map_b, map_b=map_b)
        du_eff = float(np.median(np.diff(u))) if u.size > 1 else float(du_ref)
    elif grid == 'uniform':
        u, r = mapped_uniform_grid(r0, rmax, n_ref, map_b=map_b)
        du_eff = float(np.median(np.diff(u))) if u.size > 1 else float(du_ref)
    else:
        du_eff = float(du_ref)
        u, r = mapped_u_grid(r0, du_eff, rmax, map_b=map_b)
    return dict(i=np.arange(u.size), u=u, r_ang=r * ANG, du=du_eff, map_b=map_b, n=u.size, grid=grid)


def chain_rule_du(r, r0, dy_dr, d2y_dr2=None, map_b=1.0):
    '''dy/du and d2y/du2 for u = map_b * log1p(r/r0).'''
    map_b = float(map_b)
    dr_du = np.asarray(r + r0, dtype=np.double) / map_b
    if dy_dr.ndim > 1:
        dr_du = dr_du[:, None]
    dy_du = dy_dr * dr_du
    if d2y_dr2 is None:
        return dy_du
    d2y_du2 = (d2y_dr2 * dr_du + dy_dr) * dr_du
    return dy_du, d2y_du2


def contracted_radial_coeff(mol, ib):
    '''Contracted Cartesian radial coefficients (sum of primitives), not per-Gaussian.'''
    expn = mol.bas_exp(ib)
    coeff = mol._libcint_ctr_coeff(ib) * SP_CART_FACTOR.get(mol.bas_angular(ib), 1.0)
    return expn, coeff


def suggest_map_b(mol, ib, b_ref_alpha=10.0):
    '''Heuristic map_b from shell steepness: large α_max → larger b (finer u near origin).'''
    expn = mol.bas_exp(ib)
    alpha_max = float(np.max(expn))
    b = np.sqrt(alpha_max / float(b_ref_alpha))
    return float(np.clip(b, 0.25, 8.0))


def eval_radial(r, expn, coeff):
    '''Contracted radial: Σ_k c_k exp(-α_k r²).  coeff from libcint contraction.'''
    r = np.asarray(r, dtype=np.double)
    return np.exp(-np.outer(r * r, expn)).dot(coeff)


def eval_radial_dr(r, expn, coeff):
    r = np.asarray(r, dtype=np.double)
    e = np.exp(-np.outer(r * r, expn))
    return ((-2.0 * r[:, None] * expn[None, :]) * e).dot(coeff)


def eval_radial_d2r(r, expn, coeff):
    r = np.asarray(r, dtype=np.double)
    e = np.exp(-np.outer(r * r, expn))
    a = expn[None, :]
    return ((-2.0 * a + 4.0 * r[:, None] ** 2 * a * a) * e).dot(coeff)


def _y_exact_at_u(uu, r0, expn, coeff, ic=0, map_b=1.0):
    r = r_from_u(uu, r0, map_b)
    val = eval_radial(np.atleast_1d(r), expn, coeff)
    return float(val[0, ic] if val.ndim > 1 else val[ic])


def quadrature_nodes(n, rule='gauss_legendre'):
    '''Interior nodes t in (0, 1) and optional weights for interval [0, h].'''
    n = int(n)
    if n < 1:
        raise ValueError('n_quad must be >= 1')
    rule = rule.lower()
    if rule in ('gauss', 'gauss_legendre', 'legendre'):
        t, w = np.polynomial.legendre.leggauss(n)
        return (0.5 * (t + 1.0), 0.5 * w)
    if rule in ('chebyshev', 'cheb', 'gauss_chebyshev'):
        k = np.arange(1, n + 1, dtype=np.double)
        t = 0.5 * (1.0 - np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n)))
        w = np.full(n, np.pi / n)
        return t, w
    if rule in ('newton_cotes', 'cotes', 'uniform'):
        if n == 1:
            return np.array([0.5]), np.array([1.0])
        t = np.linspace(0.0, 1.0, n + 2)[1:-1]
        w = np.full(n, 1.0 / n)
        return t, w
    if rule in ('simpson', 'midpoint'):
        if n == 1:
            return np.array([0.5]), np.array([1.0])
        if n == 2:
            return np.array([0.25, 0.75]), np.array([0.5, 0.5])
        t = np.linspace(0.0, 1.0, n + 2)[1:-1]
        w = np.full(n, 1.0 / n)
        return t, w
    raise ValueError(f'unknown quadrature rule {rule!r}')


def cubic_basis(t):
    '''Cubic Hermite on t in [0,1]; matches kernels.cl hermite_eval_node.'''
    t = np.asarray(t, dtype=np.double)
    t1m = t - 1.0
    dy_unit = t * t * (3.0 - 2.0 * t)
    d0_unit = t * t1m * t1m
    d1_unit = t * t * t1m
    return dy_unit, d0_unit, d1_unit


def cubic_eval(t, h, y0, y1, d0, d1):
    dy = y1 - y0
    dy_u, d0_u, d1_u = cubic_basis(t)
    return y0 + dy_u * dy + d0_u * h * d0 + d1_u * h * d1


def cubic_eval_du(t, h, y0, y1, d0, d1):
    dy = y1 - y0
    t = np.asarray(t, dtype=np.double)
    t1m = t - 1.0
    dp_dt = 6.0 * t * (1.0 - t) * dy + (3.0 * t - 1.0) * t1m * h * d0 + t * (3.0 * t - 2.0) * h * d1
    return dp_dt / h


def quintic_basis(t):
    t = np.asarray(t, dtype=np.double)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    H00 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5
    H10 = t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5
    H20 = 0.5 * (t2 - 3.0 * t3 + 3.0 * t4 - t5)
    H01 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    H11 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5
    H21 = 0.5 * (t3 - 2.0 * t4 + t5)
    return H00, H10, H20, H01, H11, H21


def quintic_eval(t, h, y0, y1, d0, d1, c0, c1):
    H00, H10, H20, H01, H11, H21 = quintic_basis(t)
    h2 = h * h
    return H00 * y0 + h * H10 * d0 + h2 * H20 * c0 + H01 * y1 + h * H11 * d1 + h2 * H21 * c1


def quintic_eval_du(t, h, y0, y1, d0, d1, c0, c1):
    t = np.asarray(t, dtype=np.double)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    dH00 = -30.0 * t2 + 60.0 * t3 - 30.0 * t4
    dH10 = 1.0 - 18.0 * t2 + 32.0 * t3 - 15.0 * t4
    dH20 = 0.5 * (2.0 * t - 9.0 * t2 + 12.0 * t3 - 5.0 * t4)
    dH01 = 30.0 * t2 - 60.0 * t3 + 30.0 * t4
    dH11 = -12.0 * t2 + 28.0 * t3 - 15.0 * t4
    dH21 = 0.5 * (3.0 * t2 - 8.0 * t3 + 5.0 * t4)
    h2 = h * h
    dp_dt = dH00 * y0 + h * dH10 * d0 + h2 * dH20 * c0 + dH01 * y1 + h * dH11 * d1 + h2 * dH21 * c1
    return dp_dt / h


def midpoint_fit_cubic(y, d, h, ymid):
    '''Global correction of dy/du so cubic matches midpoint values (existing path).'''
    y = np.asarray(y, dtype=np.double)
    d = np.asarray(d, dtype=np.double)
    ymid = np.asarray(ymid, dtype=np.double)
    one_d = y.ndim == 1
    if one_d:
        y, d, ymid = y[:, None], d[:, None], ymid[:, None]
    h = np.asarray(h, dtype=np.double)
    if h.ndim == 0:
        h = np.full(y.shape[0] - 1, float(h))
    b = 8.0 / h[:, None] * (ymid - 0.5 * (y[:-1] + y[1:]))
    out = d.copy()
    n = y.shape[0]
    if n < 2:
        return out[:, 0] if one_d else out
    a = np.zeros((n - 1, n), dtype=np.double)
    i = np.arange(n - 1)
    a[i, i] = 1.0
    a[i, i + 1] = -1.0
    m = a.dot(a.T)
    rhs = a.dot(d) - b
    out = d - a.T.dot(np.linalg.solve(m, rhs))
    return out[:, 0] if one_d else out


def fit_cubic_quadrature(x, y, h, y_exact_fn, n_quad=3, rule='gauss_legendre', d_init=None, reg=0.0):
    '''Least-squares dy/dx at nodes from interior collocation (y at nodes fixed).'''
    n = y.size
    if d_init is None:
        d_init = np.zeros(n, dtype=np.double)
    h_arr = np.full(n - 1, float(h)) if np.size(h) == 1 else np.asarray(h, dtype=np.double)
    t_nodes, weights = quadrature_nodes(n_quad, rule)
    rows, rhs, wts = [], [], []
    for i in range(n - 1):
        y0, y1 = y[i], y[i + 1]
        hi = h_arr[i]
        for t_k, wt in zip(t_nodes, weights):
            dy_u, d0_u, d1_u = cubic_basis(t_k)
            y_tgt = float(y_exact_fn(x[i] + t_k * hi))
            rhs.append(y_tgt - y0 - dy_u * (y1 - y0))
            row = np.zeros(n, dtype=np.double)
            row[i] = d0_u * hi
            row[i + 1] = d1_u * hi
            rows.append(row)
            wts.append(np.sqrt(wt))
    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=np.double).ravel()
    sw = np.asarray(wts, dtype=np.double).ravel()
    A *= sw[:, None]
    b *= sw
    if reg > 0:
        A = np.vstack([A, reg * np.eye(n)])
        b = np.concatenate([b, reg * d_init])
    d, *_ = np.linalg.lstsq(A, b, rcond=None)
    return d


def fit_quintic_quadrature(x, y, h, y_exact_fn, n_quad=5, rule='gauss_legendre', d_init=None, c_init=None, reg=1e-6):
    '''Least-squares dy/dx and d2y/dx2 at nodes; y at nodes fixed.'''
    n = y.size
    if d_init is None:
        d_init = np.zeros(n, dtype=np.double)
    if c_init is None:
        c_init = np.zeros(n, dtype=np.double)
    h_arr = np.full(n - 1, float(h)) if np.size(h) == 1 else np.asarray(h, dtype=np.double)
    x0 = np.concatenate([d_init, c_init])
    t_nodes, weights = quadrature_nodes(n_quad, rule)
    rows, rhs, wts = [], [], []
    for i in range(n - 1):
        y0, y1 = y[i], y[i + 1]
        hi = h_arr[i]
        for t_k, wt in zip(t_nodes, weights):
            H00, H10, H20, H01, H11, H21 = quintic_basis(t_k)
            y_tgt = float(y_exact_fn(x[i] + t_k * hi))
            rhs.append(y_tgt - H00 * y0 - H01 * y1)
            row = np.zeros(2 * n, dtype=np.double)
            row[i] = hi * H10
            row[i + 1] = hi * H11
            row[n + i] = hi * hi * H20
            row[n + i + 1] = hi * hi * H21
            rows.append(row)
            wts.append(np.sqrt(wt))
    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=np.double).ravel()
    sw = np.asarray(wts, dtype=np.double).ravel()
    A *= sw[:, None]
    b *= sw
    if reg > 0:
        A = np.vstack([A, reg * np.eye(2 * n)])
        b = np.concatenate([b, reg * x0])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x[:n], x[n:]


def _segment_index(x_query, x_grid):
    ik = np.searchsorted(x_grid, x_query, side='right') - 1
    return np.clip(ik, 0, x_grid.size - 2)


def interp_cubic_knots(x_query, x_grid, y, d):
    x_query = np.asarray(x_query, dtype=np.double)
    x_grid = np.asarray(x_grid, dtype=np.double)
    ik = _segment_index(x_query, x_grid)
    h = x_grid[ik + 1] - x_grid[ik]
    t = np.clip((x_query - x_grid[ik]) / h, 0.0, 1.0)
    return cubic_eval(t, h, y[ik], y[ik + 1], d[ik], d[ik + 1])


def interp_cubic_du_knots(x_query, x_grid, y, d):
    x_query = np.asarray(x_query, dtype=np.double)
    x_grid = np.asarray(x_grid, dtype=np.double)
    ik = _segment_index(x_query, x_grid)
    h = x_grid[ik + 1] - x_grid[ik]
    t = np.clip((x_query - x_grid[ik]) / h, 0.0, 1.0)
    return cubic_eval_du(t, h, y[ik], y[ik + 1], d[ik], d[ik + 1])


def interp_cubic(u_query, u, y, d, h):
    '''Uniform-u fast path (production log grid).'''
    return interp_cubic_knots(u_query, u, y, d)


def interp_cubic_du(u_query, u, y, d, h):
    return interp_cubic_du_knots(u_query, u, y, d)


def interp_cubic_dr_u(r, u_grid, y, d, r0, map_b=1.0):
    '''dR/dr from cubic Hermite in mapped u.'''
    r = np.asarray(r, dtype=np.double)
    dR_du = interp_cubic_du_knots(u_from_r(r, r0, map_b), u_grid, y, d)
    return dR_du * float(map_b) / (r + r0)


def interp_cubic_dr_r(r_query, r_grid, y, d_dr):
    return interp_cubic_du_knots(r_query, r_grid, y, d_dr)


def interp_cubic_r(r_query, r_grid, y, d_dr):
    return interp_cubic_knots(r_query, r_grid, y, d_dr)


def interp_quintic_knots(x_query, x_grid, y, d, c):
    x_query = np.asarray(x_query, dtype=np.double)
    x_grid = np.asarray(x_grid, dtype=np.double)
    ik = _segment_index(x_query, x_grid)
    h = x_grid[ik + 1] - x_grid[ik]
    t = np.clip((x_query - x_grid[ik]) / h, 0.0, 1.0)
    return quintic_eval(t, h, y[ik], y[ik + 1], d[ik], d[ik + 1], c[ik], c[ik + 1])


def interp_quintic_du_knots(x_query, x_grid, y, d, c):
    x_query = np.asarray(x_query, dtype=np.double)
    x_grid = np.asarray(x_grid, dtype=np.double)
    ik = _segment_index(x_query, x_grid)
    h = x_grid[ik + 1] - x_grid[ik]
    t = np.clip((x_query - x_grid[ik]) / h, 0.0, 1.0)
    return quintic_eval_du(t, h, y[ik], y[ik + 1], d[ik], d[ik + 1], c[ik], c[ik + 1])


def interp_quintic(u_query, u, y, d, c, h):
    return interp_quintic_knots(u_query, u, y, d, c)


def memory_equivalent_du(du_cubic, bytes_cubic=8, bytes_quintic=16):
    '''Coarser du for quintic so node storage matches cubic float2 vs float4.'''
    return float(du_cubic) * (bytes_quintic / bytes_cubic)


def error_metrics(err, ref, rel_floor_frac=1e-6, rel_floor_abs=1e-15):
    '''Abs/rel error stats. rel uses max(|ref|, floor) to avoid blow-up near R(r)≈0.'''
    err = np.asarray(err, dtype=np.double).ravel()
    ref = np.asarray(ref, dtype=np.double).ravel()
    max_abs = float(np.max(np.abs(err)))
    idx_abs = int(np.argmax(np.abs(err)))
    peak = float(np.max(np.abs(ref))) if ref.size else 1.0
    floor = max(rel_floor_abs, rel_floor_frac * peak)
    denom = np.maximum(np.abs(ref), floor)
    rel = np.abs(err) / denom
    idx_rel = int(np.argmax(rel))
    return dict(max_abs=max_abs, idx_abs=idx_abs, max_rel=float(rel[idx_rel]), idx_rel=idx_rel, rel_floor=floor, ref_peak=peak, ref_at_max_abs=float(ref[idx_abs]), ref_at_max_rel=float(ref[idx_rel]))


def basis_radial_label(mol, ib, ic=0):
    ia = mol.bas_atom(ib)
    sym = mol.atom_symbol(ia)
    l = mol.bas_angular(ib)
    lname = 'spdfghi'[l] if l < 7 else f'l{l}'
    expn = mol.bas_exp(ib)
    exps = ','.join(f'{x:.3g}' for x in expn[:2])
    if expn.size > 2:
        exps += ',…'
    return f'{sym} {lname} shell={ib} ctr={ic}  exp=[{exps}]'


def grid_sample_points(r0_ang, du, rmax_ang, map_b=1.0, n_quad=5, quad_rule='gauss_legendre', grid='log', n_nodes=None):
    '''Node spacing + fit collocation points.'''
    r0 = float(r0_ang) / ANG
    rmax = float(rmax_ang) / ANG
    u, r, h_u, _ = radial_grid(r0, rmax, map_b=map_b, du=du, n_nodes=n_nodes, grid=grid)
    du_eff = float(h_u[0]) if h_u.size else float(du)
    t_q, _ = quadrature_nodes(n_quad, quad_rule)
    u_colloc = []
    for i in range(u.size - 1):
        for t in t_q:
            u_colloc.append(u[i] + t * h_u[i])
    return dict(u=u, r=r, r_ang=r * ANG, u_colloc=np.asarray(u_colloc), t_quad=t_q, du=du_eff, h_u=h_u, map_b=map_b, r0_ang=r0_ang, n_nodes=u.size, n_intervals=u.size - 1, grid=grid)


def normalize_tangents(tangents=None, fit=None):
    '''Resolve tangents mode; fit= is deprecated alias (exact → analytic).'''
    t = tangents if tangents is not None else fit
    if t is None:
        raise ValueError('tangents required')
    t = t.lower()
    if t in ('exact', 'midpoint'):
        t = 'analytic' if t == 'exact' else 'quadrature'
    if t not in ('analytic', 'quadrature'):
        raise ValueError(f'tangents={t!r}')
    return t


def _prepend_origin_knot(r, expn, coeff):
    '''Prepend r=0 with analytic R, R′, R″ so segment [0,r0] has a left neighbor.'''
    if r.size and r[0] < 1e-15:
        y = eval_radial(r, expn, coeff)
        return r, y, eval_radial_dr(r, expn, coeff), eval_radial_d2r(r, expn, coeff), False
    r = np.concatenate([np.array([0.0]), np.asarray(r, dtype=np.double)])
    y = eval_radial(r, expn, coeff)
    return r, y, eval_radial_dr(r, expn, coeff), eval_radial_d2r(r, expn, coeff), True


def _fit_u_cubic_midpoint(ic, y, dy_du, u, h_u, r0, map_b, expn, coeff):
    um = 0.5 * (u[:-1] + u[1:])
    ym = eval_radial(r_from_u(um, r0, map_b), expn, coeff)[:, ic]
    return midpoint_fit_cubic(y[:, ic], dy_du[:, ic], h_u, ym)


def _fit_u_cubic_analytic(ic, y, dy_du, **_):
    return dy_du[:, ic].copy()


def _y_exact_at_r(rr, expn, coeff, ic=0):
    val = eval_radial(np.atleast_1d(rr), expn, coeff)
    return float(val[0, ic] if val.ndim > 1 else val[ic])


def _fit_u_cubic_quadrature(ic, y, dy_du, u, h_u, r0, map_b, expn, coeff, n_quad, quad_rule, reg):
    y_exact = lambda uu, ic=ic: _y_exact_at_u(uu, r0, expn, coeff, ic, map_b)
    return fit_cubic_quadrature(u, y[:, ic], h_u, y_exact, n_quad=n_quad, rule=quad_rule, d_init=dy_du[:, ic], reg=reg)


def _fit_u_quintic(ic, y, dy_du, d2y_du2, u, h_u, r0, map_b, expn, coeff, tangents, n_quad, quad_rule, reg):
    d, c = dy_du[:, ic].copy(), d2y_du2[:, ic].copy()
    if tangents == 'quadrature':
        y_exact = lambda uu, ic=ic: _y_exact_at_u(uu, r0, expn, coeff, ic, map_b)
        d, c = fit_quintic_quadrature(u, y[:, ic], h_u, y_exact, n_quad=n_quad, rule=quad_rule, d_init=d, c_init=c, reg=reg)
    elif tangents != 'analytic':
        raise ValueError(f'quintic tangents={tangents!r}')
    return d, c


def _fit_r_quintic(ic, y, dy_dr, d2y_dr2, r, dr, expn, coeff, tangents, n_quad, quad_rule, reg):
    d, c = dy_dr[:, ic].copy(), d2y_dr2[:, ic].copy()
    if tangents == 'quadrature':
        y_exact = lambda rr, ic=ic: _y_exact_at_r(rr, expn, coeff, ic)
        d, c = fit_quintic_quadrature(r, y[:, ic], dr, y_exact, n_quad=n_quad, rule=quad_rule, d_init=d, c_init=c, reg=reg)
    elif tangents != 'analytic':
        raise ValueError(f'quintic tangents={tangents!r}')
    return d, c


def _fit_r_cubic_quadrature(ic, y, dy_dr, r, dr, expn, coeff, n_quad, quad_rule, reg):
    y_exact = lambda rr, ic=ic: _y_exact_at_r(rr, expn, coeff, ic)
    return fit_cubic_quadrature(r, y[:, ic], dr, y_exact, n_quad=n_quad, rule=quad_rule, d_init=dy_dr[:, ic], reg=reg)


def _fit_r_cubic_analytic(ic, y, dy_dr, **_):
    return dy_dr[:, ic].copy()


def _fill_d_tables(nctr, fit_fn, d_tables, args):
    for ic in range(nctr):
        d_tables[:, ic] = fit_fn(ic, *args)


def build_radial_tables_for_shell(mol, ib, r0_ang, du, rmax_ang, order='cubic', tangents='quadrature', fit=None, n_quad=5, quad_rule='gauss_legendre', reg=1e-6, map_b=1.0, interp_space='u', grid='log', n_nodes=None, origin_knot=True):
    '''Build spline tables. tangents: analytic|midpoint|quadrature; origin_knot: prepend r=0.'''
    r0 = float(r0_ang) / ANG
    map_b = float(map_b)
    interp_space = interp_space.lower()
    grid = grid.lower()
    tangents = normalize_tangents(tangents=tangents, fit=fit)
    rmax = float(rmax_ang) / ANG
    if grid == 'power' and n_nodes is None:
        n_nodes = reference_node_count(r0, du, rmax, map_b)
    if grid == 'uniform' and n_nodes is None:
        n_nodes = reference_node_count(r0, du, rmax, map_b)
    u, r, h_u, grid = radial_grid(r0, rmax, map_b=map_b, du=du, n_nodes=n_nodes, grid=grid)
    expn, coeff = contracted_radial_coeff(mol, ib)
    if origin_knot:
        r, y, dy_dr, d2y_dr2, added = _prepend_origin_knot(r, expn, coeff)
        u = map_b * np.log1p(r / r0)
        h_u = np.diff(u)
    else:
        y = eval_radial(r, expn, coeff)
        dy_dr = eval_radial_dr(r, expn, coeff)
        d2y_dr2 = eval_radial_d2r(r, expn, coeff)
        added = False
    dy_du, d2y_du2 = chain_rule_du(r, r0, dy_dr, d2y_dr2, map_b=map_b)
    nctr = y.shape[1]
    d_tables = np.zeros_like(y)
    c_tables = np.zeros_like(y)
    dr = np.diff(r)
    du_eff = float(h_u[0]) if h_u.size else float(du)
    if interp_space == 'u' and order == 'cubic' and tangents == 'analytic':
        _fill_d_tables(nctr, _fit_u_cubic_analytic, d_tables, (y, dy_du))
    elif interp_space == 'u' and order == 'cubic' and tangents == 'quadrature':
        _fill_d_tables(nctr, _fit_u_cubic_quadrature, d_tables, (y, dy_du, u, h_u, r0, map_b, expn, coeff, n_quad, quad_rule, reg))
    elif interp_space == 'u' and order == 'quintic':
        for ic in range(nctr):
            d_tables[:, ic], c_tables[:, ic] = _fit_u_quintic(ic, y, dy_du, d2y_du2, u, h_u, r0, map_b, expn, coeff, tangents, n_quad, quad_rule, reg)
    elif interp_space == 'r' and order == 'cubic' and tangents == 'analytic':
        _fill_d_tables(nctr, _fit_r_cubic_analytic, d_tables, (y, dy_dr))
    elif interp_space == 'r' and order == 'cubic' and tangents == 'quadrature':
        _fill_d_tables(nctr, _fit_r_cubic_quadrature, d_tables, (y, dy_dr, r, dr, expn, coeff, n_quad, quad_rule, reg))
    elif interp_space == 'r' and order == 'quintic':
        for ic in range(nctr):
            d_tables[:, ic], c_tables[:, ic] = _fit_r_quintic(ic, y, dy_dr, d2y_dr2, r, dr, expn, coeff, tangents, n_quad, quad_rule, reg)
    else:
        raise ValueError(f'unsupported combo interp_space={interp_space!r} order={order!r} tangents={tangents!r} grid={grid!r}')
    return dict(u=u, r=r, r0=r0, du=du_eff, h_u=h_u, map_b=map_b, y=y, d=d_tables, c=c_tables, order=order, tangents=tangents, fit=tangents, interp_space=interp_space, grid=grid, n_nodes=u.size, origin_knot=added, expn=expn, coeff=coeff)


def eval_radial_spline(r, tables, order='cubic'):
    '''Interpolate contracted radial at distances r.'''
    r = np.asarray(r, dtype=np.double)
    y, d, c = tables['y'], tables['d'], tables['c']
    ord_ = tables.get('order', order)
    out = np.zeros((r.size, y.shape[1]), dtype=np.double)
    r_grid = tables['r']
    if tables.get('interp_space', 'u') == 'r':
        for ic in range(y.shape[1]):
            if ord_ == 'quintic':
                out[:, ic] = interp_quintic_knots(r, r_grid, y[:, ic], d[:, ic], c[:, ic])
            else:
                out[:, ic] = interp_cubic_r(r, r_grid, y[:, ic], d[:, ic])
        return out
    map_b = float(tables.get('map_b', 1.0))
    u_q = u_from_r(r, tables['r0'], map_b)
    u_grid = tables['u']
    for ic in range(y.shape[1]):
        if ord_ == 'quintic':
            out[:, ic] = interp_quintic_knots(u_q, u_grid, y[:, ic], d[:, ic], c[:, ic])
        else:
            out[:, ic] = interp_cubic_knots(u_q, u_grid, y[:, ic], d[:, ic])
    return out


def eval_radial_spline_dr(r, tables, order='cubic'):
    '''dR/dr from spline tables.'''
    r = np.asarray(r, dtype=np.double)
    y, d, c = tables['y'], tables['d'], tables['c']
    ord_ = tables.get('order', order)
    out = np.zeros((r.size, y.shape[1]), dtype=np.double)
    if tables.get('interp_space', 'u') == 'r':
        r_grid = tables['r']
        for ic in range(y.shape[1]):
            if ord_ == 'quintic':
                out[:, ic] = interp_quintic_du_knots(r, r_grid, y[:, ic], d[:, ic], c[:, ic])
            else:
                out[:, ic] = interp_cubic_dr_r(r, r_grid, y[:, ic], d[:, ic])
        return out
    map_b = float(tables.get('map_b', 1.0))
    r0, u_grid = tables['r0'], tables['u']
    for ic in range(y.shape[1]):
        if ord_ == 'quintic':
            dR_du = interp_quintic_du_knots(u_from_r(r, r0, map_b), u_grid, y[:, ic], d[:, ic], c[:, ic])
            out[:, ic] = dR_du * map_b / (r + r0)
        else:
            out[:, ic] = interp_cubic_dr_u(r, u_grid, y[:, ic], d[:, ic], r0, map_b)
    return out


class MappedQuinticHermiteRadialBasis:
    '''Quintic Hermite radial tables on mapped log grid; float4 knots per node.'''

    def __init__(self, mol, r0_ang=0.01, du=0.04, rmax_ang=8.0, n_quad=5, quad_rule='gauss_legendre', reg=1e-6):
        self.mol = mol
        self.r0_ang = float(r0_ang)
        self.du_ang = float(du)
        self.rmax_ang = float(rmax_ang)
        self.n_quad = int(n_quad)
        self.quad_rule = quad_rule
        self.reg = float(reg)
        self.r0 = self.r0_ang / ANG
        self.du = self.du_ang
        self.rmax = self.rmax_ang / ANG
        self.u, self.r = mapped_u_grid(self.r0, self.du, self.rmax)
        self.nrad = self.u.size
        self._shell_tables = []
        for ib in range(mol.nbas):
            tab = build_radial_tables_for_shell(mol, ib, r0_ang, du, rmax_ang, order='quintic', fit='quadrature', n_quad=n_quad, quad_rule=quad_rule, reg=reg)
            self._shell_tables.append(tab)
