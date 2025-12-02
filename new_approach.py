"""
spectral_tikhonov_convex.py

Convex spectral Tikhonov design:
  find t_i = 1/(s_i^2 + d_i) (0 < t_i <= 1/s_i^2) minimizing sum(1/t_i)
  (optionally + fidelity term) subject to box constraints |x_gamma[j,t]| <= c_j
  for a finite training set B.

Requires: numpy, scipy, cvxpy. Optional: sklearn.utils.extmath.randomized_svd for large problems.
"""
from __future__ import annotations
import numpy as np
import scipy.linalg as la
import cvxpy as cp
from typing import Dict, Any, Optional

# Optional randomized SVD for large m
try:
    from sklearn.utils.extmath import randomized_svd
    _HAS_RAND_SVD = True
except Exception:
    _HAS_RAND_SVD = False


# -------------------------
# Utilities: robust thin SVD with optional randomized mode
# -------------------------
def robust_thin_svd(A: np.ndarray, tol_rel: float = 1e-12, max_r: Optional[int] = None,
                    use_randomized: bool = False, random_state: int = 0):
    """
    Compute thin SVD of A and drop tiny singular values.
    If max_r is provided, cap number of retained modes to max_r.
    If use_randomized and sklearn is available, use randomized_svd for speed.
    Returns (U_k, s_k, Vt_k) where shapes are (m x r), (r,), (r x n).
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape

    if use_randomized and _HAS_RAND_SVD:
        # choose oversampling to capture spectrum; pick k = min(n,m,max_r or min(m,n))
        k_guess = min(n, m) if max_r is None else min(max_r, n, m)
        # do an initial randomized SVD with k_guess (it may be heavy), then threshold
        U, s, Vt = randomized_svd(A, n_components=k_guess, random_state=random_state)
    else:
        U, s, Vt = la.svd(A, full_matrices=False)

    smax = s[0]
    thresh = max(smax * tol_rel, smax * np.finfo(float).eps * 100.0)
    keep_mask = s > thresh
    if max_r is not None:
        # ensure we don't exceed max_r; keep largest max_r singulars that pass thresh
        idx = np.where(keep_mask)[0]
        if idx.size > max_r:
            idx = idx[:max_r]
            keep_mask[:] = False
            keep_mask[idx] = True
    if not np.any(keep_mask):
        keep_mask[0] = True
    Uk = U[:, keep_mask]
    sk = s[keep_mask]
    Vtk = Vt[keep_mask, :]
    return Uk, sk, Vtk


# -------------------------
# Build M matrix: maps t -> x entries (concatenated over t)
# M has shape (n*T) x r
# -------------------------
def build_M_matrix(Vtk: np.ndarray, s_k: np.ndarray, U_k_T_B: np.ndarray) -> np.ndarray:
    """
    Vtk: (r x n), s_k: (r,), U_k_T_B: (r x T)
    returns M of shape (n*T, r) where row block t corresponds to rows for RHS t.
    Entry M[row=(t*n + j), i] = v_{j,i} * s_i * alpha_i^{(t)}
    """
    V = Vtk.T  # n x r
    V_s = V * s_k[None, :]  # n x r
    r = V_s.shape[1]
    n = V_s.shape[0]
    T = U_k_T_B.shape[1]
    M = np.zeros((n * T, r), dtype=np.float64)
    for t in range(T):
        alpha_t = U_k_T_B[:, t]   # r
        # broadcast multiply: each row j -> V_s[j,:] * alpha_t
        M[t * n:(t + 1) * n, :] = V_s * alpha_t[None, :]
    return M


# -------------------------
# Fidelity C,d builder (optional)
# -------------------------
def build_fidelity_C_d(Uk: np.ndarray, s_k: np.ndarray, B: np.ndarray, Ax0: Optional[np.ndarray] = None):
    """
    Build C and d such that concatenated Ax_gamma = C t.
    C shape: (n*T, r), d shape: (n*T,)
    If Ax0 provided, d = Ax0.reshape(-1), else zeros.
    """
    r = len(s_k)
    T = B.shape[1]
    n = Uk.shape[0]
    alpha = Uk.T.dot(B)  # r x T
    C = np.zeros((n * T, r), dtype=np.float64)
    for i in range(r):
        u_i = Uk[:, i]  # n
        s2 = s_k[i] ** 2
        for t in range(T):
            a_it = alpha[i, t]
            C[t * n:(t + 1) * n, i] = s2 * a_it * u_i
    if Ax0 is None:
        d = np.zeros((n * T,), dtype=np.float64)
    else:
        d = Ax0.reshape(-1)
    return C, d


# -------------------------
# Main convex solver
# -------------------------
def spectral_convex_tikhonov(A: np.ndarray,
                             B: np.ndarray,
                             c_vec: np.ndarray,
                             rho_fid: float = 0.0,
                             weight_t_inv: Optional[np.ndarray] = None,
                             tol_svd_rel: float = 1e-12,
                             t_lower_eps: float = 1e-12,
                             solver: str = 'ECOS',
                             verbose: bool = False,
                             max_r: Optional[int] = None,
                             use_randomized_svd: bool = False) -> Dict[str, Any]:
    """
    Convex spectral Tikhonov design.

    Returns a dict containing t, d, Gamma, X_gamma, diagnostics.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    c_vec = np.asarray(c_vec, dtype=np.float64).ravel()
    m, n = A.shape
    T = B.shape[1]

    # 1) thin SVD (robust)
    Uk, s_k, Vtk = robust_thin_svd(A, tol_rel=tol_svd_rel, max_r=max_r, use_randomized=use_randomized_svd)
    r = len(s_k)
    if verbose:
        print(f"[svd] kept r={r}, s_max={s_k[0]:.3e}, s_min={s_k[-1]:.3e}")

    # 2) build M (n*T x r)
    U_k_T_B = Uk.T.dot(B)  # r x T
    M = build_M_matrix(Vtk, s_k, U_k_T_B)
    c_rep = np.tile(c_vec, T)

    # 3) optional fidelity term: build C,d using projection baseline Ax0
    C = None; d = None
    if rho_fid > 0:
        # baseline Ax0: project B onto column space of A (U U^T B)
        Ax0 = Uk.dot(Uk.T.dot(B))  # n x T
        C, d = build_fidelity_C_d(Uk, s_k, B, Ax0)

    # 4) CVX variable and constraints
    t = cp.Variable(r, pos=True)
    constraints = []
    # upper bounds t_i <= 1/s_i^2
    t_upper = 1.0 / (s_k ** 2 + 1e-30)
    constraints += [t <= t_upper]

    # box constraints: |M t| <= c_rep  --> two linear constraints
    # Use Parameter for M (to speed up) but set value immediately.
    M_param = cp.Parameter(M.shape, value=M, mutable=False)
    c_param = cp.Parameter(c_rep.shape, value=c_rep, mutable=False)
    constraints += [M_param @ t <= c_param, -M_param @ t <= c_param]

    # 5) objective: sum weight / t  + rho_fid * ||C t - d||^2
    if weight_t_inv is None:
        weight_t_inv = np.ones(r, dtype=np.float64)
    weight_param = cp.Parameter(weight_t_inv.shape, value=weight_t_inv, mutable=False)
    term1 = cp.sum(cp.multiply(weight_param, cp.inv_pos(t)))  # convex

    obj_exprs = [term1]
    if rho_fid > 0 and C is not None:
        C_param = cp.Parameter(C.shape, value=C, mutable=False)
        d_param = cp.Parameter(d.shape, value=d, mutable=False)
        obj_exprs.append(rho_fid * cp.sum_squares(C_param @ t - d_param))

    objective = cp.Minimize(cp.sum(obj_exprs))
    prob = cp.Problem(objective, constraints)

    # 6) Solve with robust fallback among solvers
    solved = False
    last_exc = None
    solvers_to_try = [solver, 'ECOS', 'OSQP', 'SCS']
    for s in solvers_to_try:
        try:
            if s == 'ECOS':
                prob.solve(solver=s, verbose=verbose, abstol=1e-8, reltol=1e-6, feastol=1e-6, warm_start=True, max_iters=10000)
            else:
                prob.solve(solver=s, verbose=verbose, warm_start=True)
            solved = True
            used_solver = s
            break
        except Exception as e:
            last_exc = e
            if verbose:
                print(f"[cvx] solver {s} failed: {e}. Trying next.")
            continue

    if not solved:
        raise RuntimeError(f"CVX solvers failed. Last exception: {repr(last_exc)}")

    if prob.status not in ("optimal", "optimal_inaccurate"):
        # Infeasible or other: return info
        raise RuntimeError(f"CVX problem status: {prob.status}")

    t_val = np.array(t.value, dtype=np.float64).ravel()
    t_val = np.maximum(t_val, t_lower_eps)

    # compute d and Gamma
    d_val = (1.0 / t_val) - s_k ** 2
    d_val = np.maximum(d_val, 0.0)

    V = Vtk.T  # n x r
    Gamma = V.dot(np.diag(d_val)).dot(V.T)
    Gamma = (Gamma + Gamma.T) / 2.0  # symmetrize

    # 7) Validate: form Mfull = A^T A + Gamma, solve for X_gamma
    ATA = A.T.dot(A)
    Mfull = ATA + Gamma
    Mfull = (Mfull + Mfull.T) / 2.0
    try:
        cho = la.cho_factor(Mfull)
        X_gamma = la.cho_solve(cho, A.T.dot(B))
    except la.LinAlgError:
        evals, evecs = la.eigh(Mfull)
        evals_clipped = np.maximum(evals, 1e-12)
        inv_mat = evecs.dot(np.diag(1.0 / evals_clipped)).dot(evecs.T)
        X_gamma = inv_mat.dot(A.T.dot(B))

    # violations and fidelity
    violations = np.maximum(np.abs(X_gamma) - c_vec[:, None], 0.0)
    max_violation = float(np.max(violations))
    n_viol = int(np.sum(violations > 1e-9))
    # baseline for fidelity: projection onto colspace (Ax0)
    X0 = la.pinv(A).dot(B)
    fidelity = np.linalg.norm(A.dot(X_gamma) - A.dot(X0), axis=0)

    diagnostics = {
        'solver_used': used_solver,
        'prob_status': prob.status,
        's_k': s_k,
        'r': r,
        't_upper': t_upper,
    }

    out = {
        't': t_val,
        'd': d_val,
        'Gamma': Gamma,
        'X_gamma': X_gamma,
        'max_violation': max_violation,
        'n_violations': n_viol,
        'fidelity_per_rhs': fidelity,
        'diagnostics': diagnostics
    }
    return out


# -------------------------
# Example usage / notes
# -------------------------
if __name__ == "__main__":
    # Demo with small synthetic A (for speed). For real A (m~10000) set use_randomized_svd=True and max_r small (e.g. 80).
    import numpy.random as npr
    npr.seed(1)
    n = 39
    m = 120
    T = 6
    A = np.random.randn(m, n)
    # make spectrum wide
    s = np.geomspace(1e3, 1e-7, num=n)
    U, _ = np.linalg.qr(np.random.randn(m, m))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    A = U[:, :n].dot(np.diag(s)).dot(V[:, :n].T)
    X_true = np.random.randn(n, T) * np.linspace(1.0, 0.1, n)[:, None]
    B = A.dot(X_true)
    c_vec = np.maximum(1.05 * np.max(np.abs(X_true), axis=1), 1e-6)

    out = spectral_convex_tikhonov(A, B, c_vec, rho_fid=0.0, solver='ECOS', verbose=True, max_r=60, use_randomized_svd=False)
    print("max_violation:", out['max_violation'], "n_viol:", out['n_violations'])
    print("Gamma diag (first 10 d):", out['d'][:10])
