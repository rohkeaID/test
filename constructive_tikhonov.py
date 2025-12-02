"""
constructive_tikhonov.py

Constructive pipeline to obtain a diagonal Tikhonov regularizer Gamma
that (empirically) enforces per-coordinate box constraints |x_i| <= c_i for a
finite training set B = {b^(t)}.

Strategy (construct-based):
1. For each RHS b^(t) solve bounded least squares: minimize ||A x - b||^2 s.t. |x_i| <= c_i.
   This returns X_target that satisfies box constraints exactly (up to solver tolerance).
2. Derive per-coordinate candidate gamma values from the relation:
       (A^T A + diag(gamma)) x = A^T b  =>  gamma_i = (A^T b - A^T A x)_i / x_i  (if x_i != 0)
   For x_i == 0, we leave gamma candidate unspecified (NaN) and later fill conservatively.
3. Aggregate gamma candidates across RHS (median/mean/max), then clamp to sensible bounds.
4. Optionally project gamma to block-structure (average per-block) or smooth per-column.
5. Validate: solve for X_hat using the robust Tikhonov solver and report box violations, fidelity, q_i.

This file contains robust helper functions, diagnostics and a runnable demo on synthetic data.

"""

from __future__ import annotations
import numpy as np
import scipy.linalg as la
from scipy.optimize import lsq_linear
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warnings
from typing import Tuple, Dict, Any

warnings.filterwarnings("ignore")
np.set_printoptions(precision=6, suppress=True)

# -------------------------
# Robust solver (final version)
# -------------------------

def solve_tikhonov_robust(A: np.ndarray, Gamma_diag: np.ndarray, B: np.ndarray,
                          min_gamma: float = 1e-12,
                          jitter_list: tuple = (1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6),
                          eig_clip: float = 1e-12,
                          lsq_lapack_drivers: tuple = ('gelsy','gelsd','gelss'),
                          lsqr_maxiter: int = None,
                          normal_reg: float = 1e-12) -> np.ndarray:
    """
    Robust solver for (A^T A + diag(Gamma_diag)) X = A^T B with multi-stage fallbacks.

    Returns X (n x T).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Gamma_diag = np.asarray(Gamma_diag, dtype=np.float64).ravel()

    m, n = A.shape
    if Gamma_diag.size != n:
        raise ValueError(f"Gamma_diag length ({Gamma_diag.size}) must equal number of columns n ({n})")

    # enforce minimal gamma to avoid exact zeros
    Gamma_diag = np.maximum(Gamma_diag, min_gamma)

    At = A.T
    ATA = At.dot(A)    # n x n
    RHS = At.dot(B)    # n x T

    # base symmetric matrix
    M_base = ATA + np.diag(Gamma_diag)
    M_base = (M_base + M_base.T) / 2.0

    eig_exc = None
    lsqr_exc = None
    reg_exc = None

    # 1) Try Cholesky directly
    try:
        cho = la.cho_factor(M_base, overwrite_a=False, check_finite=False)
        X = la.cho_solve(cho, RHS, overwrite_b=False)
        return X
    except la.LinAlgError:
        pass

    # 2) Try diagonal jitter sequence
    for jitter in jitter_list:
        M_try = M_base.copy()
        M_try[np.diag_indices(n)] += jitter
        M_try = (M_try + M_try.T) / 2.0
        try:
            cho = la.cho_factor(M_try, overwrite_a=False, check_finite=False)
            X = la.cho_solve(cho, RHS, overwrite_b=False)
            return X
        except la.LinAlgError:
            continue

    # 3) Eigen-decomposition fallback (clip small/negative eigenvalues)
    try:
        eigvals, eigvecs = la.eigh(M_base)   # ascending eigenvalues
        eigvals_clipped = np.maximum(eigvals, eig_clip)
        inv_diag = 1.0 / eigvals_clipped
        Vt_RHS = eigvecs.T.dot(RHS)
        scaled = inv_diag[:, None] * Vt_RHS
        X = eigvecs.dot(scaled)
        return X
    except Exception as e_eig:
        eig_exc = e_eig

    # 4) Augmented least-squares fallback
    sqrtG = np.sqrt(Gamma_diag)         # length n
    A_aug = np.vstack([A, np.diag(sqrtG)])      # (m+n) x n
    B_aug = np.vstack([B, np.zeros((n, B.shape[1]), dtype=np.float64)])  # (m+n) x T

    # Row scaling to improve conditioning for SVD/QR solvers
    row_norms = np.linalg.norm(A_aug, axis=1)
    row_norms_safe = np.where(row_norms < 1e-16, 1.0, row_norms)
    S_inv = 1.0 / row_norms_safe
    A_aug_scaled = (A_aug.T * S_inv).T
    B_aug_scaled = (B_aug.T * S_inv).T

    # 4a) try scipy.linalg.lstsq with different lapack drivers
    try:
        for drv in lsq_lapack_drivers:
            try:
                sol, resids, rank, s = la.lstsq(A_aug_scaled, B_aug_scaled, lapack_driver=drv)
                return sol
            except Exception:
                continue
    except Exception:
        pass

    # 4b) Try iterative lsqr (sparse) for each RHS
    try:
        A_csr = sp.csr_matrix(A_aug_scaled)
        n_rhs = B_aug_scaled.shape[1]
        X_sol = np.zeros((n, n_rhs), dtype=np.float64)
        for j in range(n_rhs):
            b_j = B_aug_scaled[:, j]
            res_lsqr = spla.lsqr(A_csr, b_j, iter_lim=lsqr_maxiter)
            x_j = res_lsqr[0]
            X_sol[:, j] = x_j
        return X_sol
    except Exception as e_lsqr:
        lsqr_exc = e_lsqr

    # 4c) Regularized normal equations fallback
    try:
        N = A_aug_scaled.T.dot(A_aug_scaled)
        lam = max(normal_reg, 1e-15)
        N_reg = N + lam * np.eye(n)
        N_reg = (N_reg + N_reg.T) / 2.0
        RHS_reg = A_aug_scaled.T.dot(B_aug_scaled)
        try:
            cho = la.cho_factor(N_reg, overwrite_a=False, check_finite=False)
            X_reg = la.cho_solve(cho, RHS_reg, overwrite_b=False)
            return X_reg
        except la.LinAlgError:
            evals, evecs = la.eigh(N_reg)
            evals_clipped = np.maximum(evals, 1e-16)
            inv_diag = 1.0 / evals_clipped
            Vt_RHS = evecs.T.dot(RHS_reg)
            scaled = inv_diag[:, None] * Vt_RHS
            X_reg = evecs.dot(scaled)
            return X_reg
    except Exception as e_reg:
        reg_exc = e_reg

    # Nothing worked — raise informative RuntimeError
    raise RuntimeError(
        "solve_tikhonov_robust: all fallbacks failed.\n"
        f"Eigen fallback error: {repr(eig_exc)}\n"
        f"LSQR iterative error: {repr(lsqr_exc)}\n"
        f"Regularized normal eq error: {repr(reg_exc)}\n"
    )

# -------------------------
# Constructive pipeline functions
# -------------------------

def compute_target_xs(A: np.ndarray, B: np.ndarray, c_vec: np.ndarray, method: str = 'lsq_linear') -> np.ndarray:
    """
    For each column b in B solve: min ||A x - b||^2  s.t. -c_vec <= x <= c_vec.
    Returns X_target (n x T).

    Uses scipy.optimize.lsq_linear which is robust and handles bounds.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    c_vec = np.asarray(c_vec, dtype=np.float64).ravel()
    m, n = A.shape
    T = B.shape[1]
    X_target = np.zeros((n, T), dtype=np.float64)

    lb = -c_vec
    ub = c_vec
    for t in range(T):
        b = B[:, t]
        # lsq_linear uses a direct solver; set lsmr_tol='auto' for stable behavior
        res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=0)
        if not res.success:
            # fallback: try unbounded least squares and then clip
            x_ls, *_ = la.lstsq(A, b)  # returns (n,) approx
            x_clipped = np.clip(x_ls, lb, ub)
            X_target[:, t] = x_clipped
        else:
            X_target[:, t] = res.x
    return X_target


def gamma_from_targets(A: np.ndarray, B: np.ndarray, X_target: np.ndarray,
                       agg: str = 'median', gamma_bounds: Tuple[float,float]=(1e-12,1e12)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given target X_target (n x T), compute candidate gamma per coordinate and RHS:
        gamma_cands[:, t] = (A^T b^{(t)} - A^T A x^{(t)}) / x^{(t)}
    Then aggregate across t (median by default) to a single gamma vector.

    Returns (gamma, gamma_cands)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    n, T = X_target.shape

    At = A.T
    ATA = At.dot(A)
    ATB = At.dot(B)

    gamma_cands = np.full((n, T), np.nan, dtype=np.float64)
    for t in range(T):
        x_t = X_target[:, t]
        num = ATB[:, t] - ATA.dot(x_t)
        denom = x_t.copy()
        small = np.abs(denom) < 1e-12
        denom[small] = np.nan
        gamma_cands[:, t] = num / denom

    if agg == 'median':
        gamma = np.nanmedian(gamma_cands, axis=1)
    elif agg == 'mean':
        gamma = np.nanmean(gamma_cands, axis=1)
    elif agg == 'max':
        gamma = np.nanmax(gamma_cands, axis=1)
    else:
        raise ValueError('Unknown agg')

    # Replace NaN (arising where x_i==0) with conservative small gamma (or small positive)
    # Use a percentile of finite gamma_cands if available otherwise fallback to gamma_bounds[0]
    finite_vals = gamma[~np.isnan(gamma)]
    if finite_vals.size > 0:
        fill = np.percentile(np.abs(finite_vals), 25)
        fill = max(fill, gamma_bounds[0])
    else:
        fill = gamma_bounds[0]

    gamma = np.nan_to_num(gamma, nan=fill, posinf=gamma_bounds[1], neginf=gamma_bounds[0])
    # enforce positivity and bounds
    gamma = np.clip(gamma, gamma_bounds[0], gamma_bounds[1])
    return gamma, gamma_cands


def block_average_gamma(gamma: np.ndarray, Wn: np.ndarray) -> np.ndarray:
    """
    Optional smoothing: average gamma per column according to soft-mapping Wn.
    If Wn shape (n x K) and gamma given per-block (K) we can map; here gamma is per-col already.

    Provided for completeness; currently returns gamma unchanged.
    """
    return gamma.copy()


def validate_gamma(A: np.ndarray, B: np.ndarray, gamma: np.ndarray, c_vec: np.ndarray) -> Dict[str, Any]:
    """
    Validate candidate gamma:
      - solve Tikhonov and compute max box violation, fidelity and q vector
    Returns diagnostics dict.
    """
    X = solve_tikhonov_robust(A, gamma, B)
    Ax = A.dot(X)
    # compute fidelity relative to small-gamma baseline
    tiny_gamma = np.full(A.shape[1], 1e-16)
    X0 = solve_tikhonov_robust(A, tiny_gamma, B)
    fidelity_per_rhs = np.linalg.norm(A.dot(X) - A.dot(X0), axis=0)

    violations = np.maximum(np.abs(X) - c_vec[:, None], 0.0)
    max_violation = float(np.max(violations))
    n_violations = int(np.sum(violations > 1e-9))

    # q_i: row norms of J = (A^T A + G)^{-1} A^T
    ATA = A.T.dot(A)
    M = ATA + np.diag(gamma)
    M = (M + M.T)/2.0
    try:
        cho = la.cho_factor(M)
        J = la.cho_solve(cho, A.T)
    except la.LinAlgError:
        # fallback eigen inverse
        eigvals, eigvecs = la.eigh(M)
        eigvals_clipped = np.maximum(eigvals, 1e-12)
        inv_diag = 1.0 / eigvals_clipped
        J = eigvecs.dot(np.diag(inv_diag)).dot(eigvecs.T).dot(A.T)
    q = np.linalg.norm(J, axis=1)

    return {
        'X': X,
        'Ax': Ax,
        'fidelity_per_rhs': fidelity_per_rhs,
        'max_violation': max_violation,
        'n_violations': n_violations,
        'q': q,
    }

# -------------------------
# Utilities and demo
# -------------------------

def make_synthetic_A(n: int = 39, m: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    # construct spectrum with wide dynamic range
    s = np.geomspace(1e3, 1e-7, num=n)
    U = la.qr(rng.normal(size=(m, m)))[0][:, :n]
    V = la.qr(rng.normal(size=(n, n)))[0][:, :n]
    A = U.dot(np.diag(s)).dot(V.T)
    # add correlated blocks
    groups = [(0, 10), (10, 25), (25, n)]
    for (a,b) in groups:
        if a < n:
            b = min(b, n)
            base = rng.normal(size=(m,1))
            for j in range(a, b):
                A[:, j] += 0.85 * base.ravel() + 0.05 * rng.normal(size=m)
    return A


def demo_constructive(n: int = 39, m: int = 50, T: int = 6, seed: int = 1):
    rng = np.random.RandomState(seed)
    A = make_synthetic_A(n=n, m=m, seed=seed)
    X_true = rng.normal(size=(n, T)) * (np.linspace(1.0, 0.1, n)[:, None])
    B = A.dot(X_true)

    # box limits c based on true X_true magnitude (or user provided)
    c_vec = np.maximum(1.05 * np.max(np.abs(X_true), axis=1), 1e-6)

    print("Running constructive pipeline demo...")
    print(f"A shape: {A.shape}, B shape: {B.shape}")

    # 1) compute target xs
    X_target = compute_target_xs(A, B, c_vec)
    print("Computed X_target. stats: max abs", float(np.max(np.abs(X_target))))

    # 2) compute candidate gamma per coordinate
    gamma, gamma_cands = gamma_from_targets(A, B, X_target, agg='median', gamma_bounds=(1e-12,1e12))
    print("Gamma stats before clipping: min, median, max =", float(np.min(gamma)), float(np.median(gamma)), float(np.max(gamma)))

    # 3) clip gamma to reasonable bounds based on experience
    gamma = np.clip(gamma, 1e-8, 1e4)
    print("Gamma stats after clipping to [1e-8,1e4]: min, median, max =", float(np.min(gamma)), float(np.median(gamma)), float(np.max(gamma)))

    # 4) validate candidate
    diag = validate_gamma(A, B, gamma, c_vec)
    print("Validation: max_violation=", diag['max_violation'], "n_violations=", diag['n_violations'])
    print("Fidelity per RHS:", diag['fidelity_per_rhs'])

    return {
        'A': A, 'B': B, 'c_vec': c_vec,
        'X_target': X_target, 'gamma': gamma, 'gamma_cands': gamma_cands,
        'validation': diag
    }

# Only run demo if module is invoked directly
if __name__ == '__main__':
    out = demo_constructive()
    print('Demo finished. Keys:', list(out.keys()))
