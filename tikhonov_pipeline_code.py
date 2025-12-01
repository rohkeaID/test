# Below is the full code used in the demonstration run.
# It contains all functions: SVD diagnostics, block spectrum partition,
# mapping blocks to columns, leverage correction, Tikhonov solver,
# smooth hinge, loss+grad, optimization, tightening, and demo pipeline.

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

# ----------------------------
# Numerical and plotting setup
# ----------------------------
np.set_printoptions(precision=6, suppress=True)

import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# MODULE: svd_utils
# ----------------------------
# Functions: safe_svd, spectrum_diag

def safe_svd(A, tol_rel=1e-12):
    """
    Compute a thin SVD of matrix A with robust trimming of tiny singular values.

    Returns:
      U_keep : (m x r) matrix with kept left singular vectors
      s_keep : (r,) vector of kept singular values
      Vt_keep: (r x n) matrix with kept right singular vectors transposed
      diag   : dict with diagnostic values

    Comments / line-by-line rationale:
      - Use scipy.linalg.svd (thin) to get U, s, Vt. We never rely on normal equations.
      - Compute a threshold: either relative (s_max * tol_rel) or machine-eps scaled.
      - Keep singulars > threshold to avoid numerical rank-drop; but keep at least one.
    """
    U, s, Vt = la.svd(A, full_matrices=False)
    s_max = s[0] if s.size > 0 else 0.0
    eps = np.finfo(float).eps
    thresh = max(s_max * tol_rel, s_max * eps * 100.0)
    keep = s > thresh
    if not np.any(keep):
        keep[0] = True
    U_keep = U[:, keep]
    s_keep = s[keep]
    Vt_keep = Vt[keep, :]
    diag = {
        's_full': s,
        's_keep': s_keep,
        'keep_mask': keep,
        'cond': (s[0] / s_keep[-1]) if (s.size>0 and s_keep.size>0) else np.inf,
        'stable_rank': np.sum(s**2)/ (s[0]**2 + 1e-30),
        'log_gaps': -np.log(s[1:]/s[:-1]) if s.size>1 else np.array([])
    }
    return U_keep, s_keep, Vt_keep, diag


def spectrum_diag(s, energy_tail_cut=0.995, max_blocks=3, gap_multiplier=2.0):
    """
    Partition the singular-value spectrum indices into blocks.
    """
    if s.size <= 1:
        return [list(range(s.size))]
    log_gaps = -np.log(s[1:] / s[:-1])
    median_gap = np.median(log_gaps)
    gap_thresh = max(median_gap * gap_multiplier, np.percentile(log_gaps, 75))
    gap_idx = np.where(log_gaps >= gap_thresh)[0] + 1
    boundaries = [0]
    for gi in gap_idx:
        boundaries.append(int(gi))
        if len(boundaries)-1 >= max_blocks:
            break
    boundaries.append(len(s))
    blocks = []
    for i in range(len(boundaries)-1):
        i0 = int(boundaries[i])
        i1 = int(boundaries[i+1])
        blocks.append(list(range(i0, i1)))
    while len(blocks) > max_blocks:
        energies = [np.sum(s[b]**2) for b in blocks]
        pair_sums = [energies[i] + energies[i+1] for i in range(len(energies)-1)]
        merge_idx = int(np.argmin(pair_sums))
        blocks[merge_idx] = blocks[merge_idx] + blocks[merge_idx+1]
        del blocks[merge_idx+1]
    return blocks

# ----------------------------
# MODULE: column_mapping
# ----------------------------
# Functions: map_blocks_to_columns, compose_gamma_from_theta, column_leverage

def map_blocks_to_columns(Vt_keep, blocks, s=None, use_weighted=False):
    V = Vt_keep.T
    n, r = V.shape
    K = len(blocks)
    Wraw = np.zeros((n, K), dtype=float)
    for k, blk in enumerate(blocks):
        if len(blk) == 0:
            continue
        if use_weighted and (s is not None):
            Wraw[:, k] = np.sum((V[:, blk]**2) * (s[blk]**2)[None, :], axis=1)
        else:
            Wraw[:, k] = np.sum(V[:, blk]**2, axis=1)
    row_sums = Wraw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    Wn = Wraw / row_sums
    return Wn, Wraw


def compose_gamma_from_theta(theta, Wn, residual=None, gamma_bounds=(1e-12, 1e12)):
    gamma = Wn.dot(theta)
    if residual is not None:
        gamma = gamma + residual
    gamma = np.clip(gamma, gamma_bounds[0], gamma_bounds[1])
    return gamma


def column_leverage(Vt_keep, p=1.0):
    V = Vt_keep.T
    n, r = V.shape
    idx = np.arange(1, r+1)
    weights = idx**(-p)
    lev = np.sum((V**2) * weights[None, :], axis=1)
    lev_min, lev_max = lev.min(), lev.max()
    if lev_max - lev_min < 1e-30:
        return np.zeros_like(lev)
    lev_norm = (lev - lev_min) / (lev_max - lev_min)
    return lev_norm

# ----------------------------
# MODULE: tikhonov_solver
# ----------------------------
# Provides solve_tikhonov which avoids normal equations and uses robust solvers.

def solve_tikhonov(A, Gamma_diag, B):
    m, n = A.shape[0], A.shape[1]
    G = np.diag(Gamma_diag)
    At = A.T
    ATA = At.dot(A)
    M = ATA + G
    M = (M + M.T) / 2.0
    RHS = At.dot(B)
    try:
        cho = la.cho_factor(M, overwrite_a=False, check_finite=False)
        X = la.cho_solve(cho, RHS, overwrite_b=False)
    except la.LinAlgError:
        X = la.solve(M, RHS, assume_a='pos')
    return X

# ----------------------------
# MODULE: metrics_and_losses
# ----------------------------
# compute_metrics, smooth_hinge, loss_and_opt (for SciPy wrapper)

def compute_metrics(A, Gamma_diag, B, X, X0=None):
    Ax = A.dot(X)
    fidelity = None
    if X0 is not None:
        fidelity = np.linalg.norm(Ax - A.dot(X0), axis=0)
    n = A.shape[1]
    G = np.diag(Gamma_diag)
    ATA = A.T.dot(A)
    M = ATA + G
    M = (M + M.T) / 2.0
    try:
        cho = la.cho_factor(M)
        J = la.cho_solve(cho, A.T)
    except la.LinAlgError:
        J = la.solve(M, A.T)
    q = np.linalg.norm(J, axis=1)
    return {'Ax': Ax, 'fidelity': fidelity, 'q': q, 'J': J}


def smooth_hinge(u, kappa=50.0):
    return np.log1p(np.exp(kappa * u)) / (kappa + 1e-30)


def loss_and_grad(eta, A, B, c_vec, Wn, s_keep, Vt_keep,
                  rho_box=1e3, rho_fid=1.0, gamma_bounds=(1e-12,1e12), use_squares=False):
    theta = np.exp(eta)
    Gamma = compose_gamma_from_theta(theta, Wn, gamma_bounds=gamma_bounds)
    X = solve_tikhonov(A, Gamma, B)
    if use_squares:
        Jterm = np.sum(theta**2)
    else:
        Jterm = np.sum(theta)
    box_pen = 0.0
    T = B.shape[1]
    for t in range(T):
        abs_x = np.abs(X[:, t])
        box_pen += np.sum(smooth_hinge(abs_x - c_vec, kappa=200.0))
    X0 = solve_tikhonov(A, np.full(A.shape[1], 1e-16), B)
    fidelity_pen = np.sum(np.linalg.norm(A.dot(X) - A.dot(X0), axis=0)**2)
    L = Jterm + rho_box * box_pen + rho_fid * fidelity_pen
    return L

# ----------------------------
# MODULE: tightening
# ----------------------------
# Two strategies: Jacobian/LP (not implemented here fully) and bisection per-group

def tightening_bisection(A, theta, Wn, B, c_vec, groups=None, max_iter=30, tol=1e-3):
    K = len(theta)
    if groups is None:
        groups = [[k] for k in range(K)]
    theta_best = theta.copy()
    n = A.shape[1]
    for g in groups:
        lo, hi = 0.0, 1.0
        for it in range(max_iter):
            mid = (lo + hi) / 2.0
            theta_trial = theta_best.copy()
            for k in g:
                theta_trial[k] = theta_trial[k] * mid
            Gamma_trial = compose_gamma_from_theta(theta_trial, Wn)
            X_trial = solve_tikhonov(A, Gamma_trial, B)
            max_excess = np.max(np.abs(X_trial) - c_vec[:, None])
            if max_excess <= 1e-12:
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        for k in g:
            theta_best[k] = theta_best[k] * hi
    return theta_best

# ----------------------------
# MODULE: visualization
# ----------------------------
# Small plotting helpers used in demo

def plot_scree(s, title='Scree (singular values)'):
    plt.figure(figsize=(6,3))
    plt.semilogy(s, '-o')
    plt.title(title)
    plt.xlabel('index')
    plt.grid(True)
    plt.show()

def plot_cum_energy(s, title='Cumulative energy'):
    energy = np.cumsum(s**2) / (np.sum(s**2) + 1e-30)
    plt.figure(figsize=(6,3))
    plt.plot(energy, '-o')
    plt.title(title)
    plt.ylim(0,1.02)
    plt.grid(True)
    plt.show()

def plot_W_heatmap(Wn, title='Soft mapping weights W (columns x blocks)'):
    plt.figure(figsize=(6,4))
    plt.imshow(Wn, aspect='auto', cmap='viridis')
    plt.colorbar(label='block weight')
    plt.xlabel('block')
    plt.ylabel('column index')
    plt.title(title)
    plt.show()

# ----------------------------
# MODULE: synthetic demo / pipeline runner
# ----------------------------
# Functions: make_synthetic_A, demo_pipeline

def make_synthetic_A(n=39, m=50, structure='multiscale', seed=0):
    rng = np.random.RandomState(seed)
    if structure == 'multiscale':
        s = np.concatenate([
            np.geomspace(1e3, 1e0, num=8),
            np.geomspace(1e-2, 1e-7, num=max(0, n-8))
        ])
        s = s[:n]
    elif structure == 'lowrank':
        r = min(10, n)
        s = np.concatenate([np.linspace(1e3, 1e1, r), np.zeros(n-r)])
    else:
        s = np.geomspace(1e3, 1e-7, num=n)
    Urand = la.qr(rng.normal(size=(m, m)))[0][:, :len(s)]
    Vrand = la.qr(rng.normal(size=(n, n)))[0][:, :len(s)]
    A = Urand.dot(np.diag(s)).dot(Vrand.T)
    groups = [(0,10), (10,22), (22, n)]
    for (a,b) in groups:
        if a < n:
            b = min(b, n)
            base = rng.normal(size=(m,1))
            for j in range(a, b):
                A[:, j] += 0.85 * base.ravel() + 0.05 * rng.normal(size=m)
    return A

def demo_pipeline():
    m = 50; n = 39
    A = make_synthetic_A(n=n, m=m, seed=1)
    T = 6
    rng = np.random.RandomState(2)
    X_true = rng.normal(size=(n, T))
    B = A.dot(X_true)
    U_keep, s_keep, Vt_keep, diag = safe_svd(A, tol_rel=1e-12)
    s_full = diag['s_full']
    print('SVD diag: cond approx=%.3e, stable_rank=%.3f' % (diag['cond'], diag['stable_rank']))
    plot_scree(s_full[:min(len(s_full), 80)])
    plot_cum_energy(s_full[:min(len(s_full), 80)])
    blocks = spectrum_diag(s_full, energy_tail_cut=0.995, max_blocks=3)
    print('Spectrum blocks (indices):', blocks)
    d = min(6, Vt_keep.shape[0])
    Vcoords = (Vt_keep.T)[:, :d]
    pca = PCA(n_components=d)
    pca_feats = pca.fit_transform(Vcoords)
    kmeans = KMeans(n_clusters=min(3, max(1, d)), random_state=0).fit(pca_feats)
    clusters = [np.where(kmeans.labels_ == k)[0].tolist() for k in range(kmeans.n_clusters)]
    print('Column clusters sizes:', [len(c) for c in clusters])
    Wn, Wraw = map_blocks_to_columns(Vt_keep, blocks, s=s_full, use_weighted=False)
    plot_W_heatmap(Wn)
    K = len(blocks)
    s_arr = s_full
    avg_s_block = np.array([np.mean(s_arr[b]) if len(b) > 0 else s_arr[0] for b in blocks])
    theta0 = 1.0 * (s_arr[0]**2 / (avg_s_block**2 + 1e-30))
    theta0 = np.clip(theta0, 1e-12, 1e12)
    print('Initial theta (per block):', theta0)
    gamma_init = compose_gamma_from_theta(theta0, Wn)
    lev = column_leverage(Vt_keep, p=1.0)
    beta = 0.4
    gamma_levered = gamma_init * (1 - beta * lev)
    c_vec = np.maximum(0.9 * np.max(np.abs(X_true), axis=1), 1e-12)
    eta0 = np.log(theta0 + 1e-30)
    def obj(eta):
        return loss_and_grad(eta, A, B, c_vec, Wn, s_keep, Vt_keep, rho_box=1e3, rho_fid=1.0)
    bounds = [(np.log(1e-12), np.log(1e12))] * K
    res = opt.minimize(obj, eta0, method='L-BFGS-B', bounds=bounds, options={'maxiter':200, 'ftol':1e-8})
    eta_opt = res.x
    theta_opt = np.exp(eta_opt)
    gamma_opt = compose_gamma_from_theta(theta_opt, Wn)
    theta_tight = tightening_bisection(A, theta_opt, Wn, B, c_vec, groups=None)
    gamma_tight = compose_gamma_from_theta(theta_tight, Wn)
    X_before = solve_tikhonov(A, gamma_init, B)
    X_opt = solve_tikhonov(A, gamma_opt, B)
    X_tight = solve_tikhonov(A, gamma_tight, B)
    metrics_before = compute_metrics(A, gamma_init, B, X_before)
    metrics_opt = compute_metrics(A, gamma_opt, B, X_opt)
    metrics_tight = compute_metrics(A, gamma_tight, B, X_tight)
    print('
Metrics summary:')
    print('Max abs x before / opt / tight: %.6e / %.6e / %.6e' % (
        np.max(np.abs(X_before)), np.max(np.abs(X_opt)), np.max(np.abs(X_tight))))
    print('Sum gamma before / opt / tight: %.6e / %.6e / %.6e' % (
        np.sum(gamma_init), np.sum(gamma_opt), np.sum(gamma_tight)))
    plt.figure(figsize=(8,3))
    plt.plot(gamma_init, '-o', label='init')
    plt.plot(gamma_opt, '-s', label='opt')
    plt.plot(gamma_tight, '-^', label='tight')
    plt.yscale('log')
    plt.xlabel('column j')
    plt.ylabel('gamma_j')
    plt.legend()
    plt.title('Gamma per column (log scale)')
    plt.grid(True)
    plt.show()
    q_tight = metrics_tight['q']
    print('q_i (tight) sample (first 10):', q_tight[:10])
    violations_tight = np.maximum(np.abs(X_tight) - c_vec[:, None], 0.0)
    print('Train violations (tight) counts:', np.sum(violations_tight > 1e-9))
    diagnostics = {
        'A': A, 'B': B, 'X_true': X_true,
        's_full': s_full, 'blocks': blocks,
        'Wn': Wn, 'theta0': theta0, 'theta_opt': theta_opt, 'theta_tight': theta_tight,
        'gamma_init': gamma_init, 'gamma_opt': gamma_opt, 'gamma_tight': gamma_tight,
        'metrics_before': metrics_before, 'metrics_opt': metrics_opt, 'metrics_tight': metrics_tight,
    }
    outpath = '/mnt/data/tikhonov_diag_demo.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(diagnostics, f)
    print('Diagnostics saved to:', outpath)
    return diagnostics

if __name__ == '__main__':
    diag = demo_pipeline()
    print('
Demo finished. Keys in diagnostics:')
    print(sorted(diag.keys()))
