"""
svd_reg_lp_methods.py

Реализует три метода поиска диагональной в V-базисе регуляризации по SVD-модам,
чтобы регуляризованные решения для всех b^j попадали в заданные box-ограничения:

 - Method A: Active-set / constraint generation (рекомендуется)
 - Method B: Full LP (linprog / HiGHS) — прямая проверка всей системы ограничений
 - Method C: Cutting-plane / batch: добавляем наиболее нарушающие ограничения пакетами

Формулировка LP (переменные f_i, i=1..n):
    0 <= f_i <= 1/sigma_i
    Для всех j=1..J и компонент k=1..n:
        ell_k <= sum_i a_{k,i}^{(j)} * f_i <= ub_k,
    где a_{k,i}^{(j)} = v_{k,i} * z_i^{(j)}, z = U^T B

Цель: максимизировать sum_i sigma_i * f_i  (т.е. минимизировать регуляризацию)
(в linprog будем минимизировать -sigma^T f)

Зависимости:
    numpy, scipy (>=1.6 рекомендовано), tqdm (опционально)

Запуск:
    python svd_reg_lp_methods.py --A A.npy --B B.npy --ell ell.npy --ub ub.npy

Если файлов нет, скрипт сгенерирует демонстрационные данные (уменьшенный J).
"""

import argparse
import time
import numpy as np
from scipy.linalg import svd
from scipy import sparse
from scipy.optimize import linprog
from math import ceil
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

# -------------------------- Utilities ---------------------------------------

def load_arrays(A_path=None, B_path=None, ell_path=None, ub_path=None, synth_demo=False):
    """
    Загружает массивы A (m x n), B (m x J), и ell/ub (n,).
    Если A_path/B_path не заданы или файлов нет, генерируем демонстрационные массивы.
    """
    if (A_path is not None) and (B_path is not None):
        A = np.load(A_path)
        B = np.load(B_path)
        print(f"Loaded A {A.shape}, B {B.shape}")
    else:
        if not synth_demo:
            raise ValueError("Provide A.npy and B.npy or set synth_demo=True")
        rng = np.random.default_rng(123)
        m, n = 3000, 39
        J = 1000  # demo smaller J to keep runtime reasonable for local tests
        # build A with required spectrum:
        U_rnd = np.linalg.qr(rng.normal(size=(m, n)))[0]
        V_rnd = np.linalg.qr(rng.normal(size=(n, n)))[0]
        sigs = np.exp(np.linspace(np.log(1e3), np.log(1e-7), n))
        A = (U_rnd * sigs) @ V_rnd.T
        B = rng.normal(size=(m, J))
        print(f"Synthesized A {A.shape}, B {B.shape} (demo)")
    if ell_path is not None and ub_path is not None:
        ell = np.load(ell_path)
        ub = np.load(ub_path)
    else:
        # default symmetric box [-0.5, 0.5] (user should override)
        n = A.shape[1]
        ell = -0.5 * np.ones(n)
        ub = 0.5 * np.ones(n)
    return A, B, ell, ub

def compute_svd_and_precompute(A, B):
    """
    Compute economic SVD and precompute helpful matrices:
      U (m x n), s (n,), Vt (n x n)
      Z = U^T B (n x J)
      V = Vt.T (n x n)
    """
    print("Computing economic SVD...")
    U, s, Vt = svd(A, full_matrices=False)
    V = Vt.T
    print("Projecting B onto left singular vectors (Z = U^T B)...")
    Z = U.T @ B  # shape n x J
    return U, s, V, Z

# -------------------------- Constraint helpers ------------------------------

def compute_X_from_f(V, f, Z):
    """
    Given V (n x n), f (n,), Z (n x J) compute X = V @ ( (f[:,None] * Z) )
    Returns X shape (n x J), where column j is x^{(j)} in original coordinates.
    """
    E = (f[:, None] * Z)  # n x J
    X = V @ E
    return X

def find_violations(X, ell, ub, tol=1e-12):
    """
    Return indices of violations (j,k) where X[k,j] < ell_k - tol or > ub_k + tol
    Output as lists of tuples (j,k, value, type) type: 'below' or 'above'
    But for performance we return arrays of indices instead.
    """
    below = X < (ell[:, None] - tol)
    above = X > (ub[:, None] + tol)
    if not below.any() and not above.any():
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    # we return flattened lists of (j,k)
    idx_b = np.argwhere(below)
    idx_a = np.argwhere(above)
    # idx arrays have shape (p,2) with [k,j]
    # convert to (j,k)
    if idx_b.size > 0:
        jb = idx_b[:,1].astype(int)
        kb = idx_b[:,0].astype(int)
    else:
        jb = np.array([], dtype=int); kb = np.array([], dtype=int)
    if idx_a.size > 0:
        ja = idx_a[:,1].astype(int)
        ka = idx_a[:,0].astype(int)
    else:
        ja = np.array([], dtype=int); ka = np.array([], dtype=int)
    return (jb, kb, ja, ka)

# -------------------------- LP assembly helpers -------------------------------

def build_sparse_A_ub_for_indices(V, Z, indices_j, ell, ub):
    """
    Build sparse matrix A_ub and rhs b_ub for constraints of form:
       sum_i a_{k,i}^{(j)} f_i <= ub_k  (for each (j,k))
    and also for -sum_i a_{k,i}^{(j)} f_i <= -ell_k  (i.e., lower bounds)
    Here indices_j is list of j values to include (we include all k for each j)
    Returns:
       A_ub (2 * n * len(indices_j) x n) sparse CSR
       b_ub vector length 2 * n * len(indices_j)
       Also returns mapping rows -> (j,k, type)
    Memory: builds sparse with nnz = n*n*len(indices_j) dense in small n.
    """
    n = V.shape[0]
    Jsub = len(indices_j)
    rows = []
    cols = []
    data = []
    b = []
    row = 0
    mapping = []  # optional map row -> (j,k,type)
    for j in indices_j:
        z_j = Z[:, j]  # length n
        bvec = z_j  # we'll multiply by V to get a_{k,i} = v_{k,i} * z_i^{(j)}
        # A_j = V * bvec[None,:]  -> shape n x n
        A_j = V * bvec[np.newaxis, :]
        # Flatten rows
        # We'll append row-by-row
        for k in range(n):
            row_idx = row
            cols.extend(range(n))
            rows.extend([row_idx]*n)
            data.extend(A_j[k, :].tolist())
            b.append(ub[k])
            mapping.append((j,k,'upper'))
            row += 1
        for k in range(n):
            row_idx = row
            cols.extend(range(n))
            rows.extend([row_idx]*n)
            # -A_j row
            data.extend((-A_j[k, :]).tolist())
            b.append(-ell[k])
            mapping.append((j,k,'lower'))
            row += 1
    A_ub = sparse.csr_matrix((data, (rows, cols)), shape=(row, n))
    b_ub = np.array(b)
    return A_ub, b_ub, mapping

def build_sparse_A_ub_full(V, Z, ell, ub):
    """
    Build full A_ub and b_ub for all j=0..J-1, k=0..n-1.
    Careful: this can be very large: rows = 2 * n * J, cols = n.
    Return CSR sparse (rows x n) and b vector.
    """
    n, J = Z.shape
    rows = []
    cols = []
    data = []
    b = []
    row = 0
    print("Building full sparse constraint matrix: this may take memory/time...")
    for j in tqdm(range(J)):
        z_j = Z[:, j]
        A_j = V * z_j[np.newaxis, :]
        # upper bounds rows
        for k in range(n):
            cols.extend(range(n))
            rows.extend([row]*n)
            data.extend(A_j[k, :].tolist())
            b.append(ub[k])
            row += 1
        # lower bounds rows
        for k in range(n):
            cols.extend(range(n))
            rows.extend([row]*n)
            data.extend((-A_j[k, :]).tolist())
            b.append(-ell[k])
            row += 1
    A_ub = sparse.csr_matrix((data, (rows, cols)), shape=(row, n))
    b_ub = np.array(b)
    return A_ub, b_ub

# -------------------------- Solvers -----------------------------------------

def solve_lp_highs_full(V, Z, s, ell, ub, verbose=True):
    """
    Method B: build full LP and call linprog (HiGHS).
    Minimize c^T f  s.t. A_ub f <= b_ub, bounds.
    c = -sigma (to maximize sigma^T f)
    """
    n, J = Z.shape
    c = -s.copy()
    # bounds for f: 0 <= f_i <= 1/sigma_i
    bounds = [(0.0, 1.0/(si if si>0 else 1e-12)) for si in s]
    t0 = time.time()
    A_ub, b_ub = build_sparse_A_ub_full(V, Z, ell, ub)
    t1 = time.time()
    if verbose:
        print(f"Built full A_ub with shape {A_ub.shape}, time {t1-t0:.2f}s")
    # call linprog
    print("Calling linprog(method='highs') on full LP ...")
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'presolve': True})
    print("linprog status:", res.message, "success:", res.success, "niters:", res.nit)
    return res

def solve_lp_active_set(V, Z, s, ell, ub,
                        init_sample_J=200, max_iters=50, tol=1e-9,
                        add_top_k=500, verbose=True):
    """
    Method A: Active-set / constraint generation:
      - Start with initial subset of j (init_sample_J) chosen as the most violated by f0 (or random).
      - Solve LP on subset via linprog.
      - Verify on full dataset Z; if violated constraints found, add top `add_top_k` most violated j (all k for those j)
      - Iterate.
    Returns solution (res) and diagnostic.
    """
    n, J = Z.shape
    c = -s.copy()
    bounds = [(0.0, 1.0/(si if si>0 else 1e-12)) for si in s]
    # initial candidate f = f0 = 1/s
    f0 = 1.0 / (s + 0.0)
    # Evaluate violations for f0 across all j quickly (we count max abs violation per j)
    if verbose:
        print("Evaluating initial violations for f0 to pick seed subset...")
    X0 = compute_X_from_f(V, f0, Z)
    # compute max violation magnitude per column j
    below = (X0 < (ell[:, None] - tol))
    above = (X0 > (ub[:, None] + tol))
    viol_mag = np.maximum((ell[:, None] - X0).clip(min=0).max(axis=0),
                          (X0 - ub[:, None]).clip(min=0).max(axis=0))
    # pick top init_sample_J by viol_mag
    init_sample_J = min(init_sample_J, J)
    seed_cols = np.argsort(-viol_mag)[:init_sample_J]
    working_set = set(int(j) for j in seed_cols)
    if verbose:
        print(f"Initial working set size: {len(working_set)}")
    iter_count = 0
    last_num_constraints = 0
    while True:
        iter_count += 1
        cols = sorted(list(working_set))
        if verbose:
            print(f"\nActive-set iteration {iter_count}, working set size: {len(cols)}")
        # build sparse A_ub for these cols
        A_sub, b_sub, mapping = build_sparse_A_ub_for_indices(V, Z, cols, ell, ub)
        # solve LP on A_sub
        res = linprog(c=c, A_ub=A_sub, b_ub=b_sub, bounds=bounds, method='highs', options={'presolve': True})
        if verbose:
            print(f"LP on subset status: {res.message}; success: {res.success}; nit: {res.nit}")
        if not res.success:
            # if solver failed, return with info
            return None, {'status': 'solver_failed', 'message': res.message, 'res': res}
        f_sol = res.x
        # verify on full dataset
        Xfull = compute_X_from_f(V, f_sol, Z)
        # find all violations
        below_idx, below_kidx, above_idx, above_kidx = find_violations(Xfull, ell, ub, tol=tol)
        # find columns j with violations
        cols_viol_b = set(below_idx.tolist())
        cols_viol_a = set(above_idx.tolist())
        cols_viol = cols_viol_b.union(cols_viol_a)
        if verbose:
            print(f"Full-check violations columns count: {len(cols_viol)}")
        if len(cols_viol) == 0:
            return f_sol, {'status': 'optimal', 'iters': iter_count, 'constraints_in_final': A_sub.shape[0]}
        # else add up to add_top_k most violated columns (by magnitude)
        # compute violation magnitude per column
        viol_per_col = np.zeros(J)
        if len(cols_viol) > 0:
            # compute per-column max violation magnitude
            below_mag = (ell[:, None] - Xfull).clip(min=0)
            above_mag = (Xfull - ub[:, None]).clip(min=0)
            viol_per_col = np.maximum(below_mag.max(axis=0), above_mag.max(axis=0))
        # pick top add_top_k columns with largest violation not already in working_set
        pick_candidates = np.argsort(-viol_per_col)
        added = 0
        for jj in pick_candidates:
            if added >= add_top_k:
                break
            if viol_per_col[jj] <= tol:
                break
            if jj not in working_set:
                working_set.add(int(jj))
                added += 1
        if verbose:
            print(f"Added {added} new columns to working set.")
        # termination on iterations:
        if iter_count >= max_iters:
            return f_sol, {'status': 'max_iters_reached', 'iters': iter_count, 'constraints_in_final': A_sub.shape[0]}

def solve_lp_cutting_plane(V, Z, s, ell, ub,
                           initial_J=100, batch_add=100, max_iters=100,
                           tol=1e-9, verbose=True):
    """
    Method C: cutting-plane: iteratively solve LP on a small seed subset,
    find most violating columns j, add top batch_add, repeat.
    Similar to Active-set but selection strategy differs; returns solution or failure info.
    """
    n, J = Z.shape
    c = -s.copy()
    bounds = [(0.0, 1.0/(si if si>0 else 1e-12)) for si in s]
    # seed subset: pick initial_J columns randomly
    rng = np.random.default_rng(0)
    seed = list(rng.choice(J, size=min(initial_J, J), replace=False))
    working_set = set(seed)
    if verbose:
        print(f"Starting cutting-plane with initial_J={len(working_set)}")
    iter_count = 0
    while True:
        iter_count += 1
        cols = sorted(list(working_set))
        if verbose:
            print(f"\nCutting-plane iter {iter_count}, working set size {len(cols)}")
        A_sub, b_sub, mapping = build_sparse_A_ub_for_indices(V, Z, cols, ell, ub)
        res = linprog(c=c, A_ub=A_sub, b_ub=b_sub, bounds=bounds, method='highs', options={'presolve': True})
        if verbose:
            print(f"LP subset status: {res.message} success: {res.success}")
        if not res.success:
            return None, {'status': 'solver_failed', 'message': res.message, 'res': res}
        f_sol = res.x
        Xfull = compute_X_from_f(V, f_sol, Z)
        # compute per-column violation magnitude
        below_mag = (ell[:, None] - Xfull).clip(min=0)
        above_mag = (Xfull - ub[:, None]).clip(min=0)
        viol_mag = np.maximum(below_mag.max(axis=0), above_mag.max(axis=0))
        max_viol = viol_mag.max()
        if verbose:
            print(f"Max violation magnitude on full set: {max_viol:.3e}")
        if max_viol <= tol:
            return f_sol, {'status': 'optimal', 'iters': iter_count, 'working_set': len(working_set)}
        # get top batch_add columns by viol_mag
        order = np.argsort(-viol_mag)
        added = 0
        for jj in order:
            if viol_mag[jj] <= tol:
                break
            if jj not in working_set:
                working_set.add(int(jj))
                added += 1
                if added >= batch_add:
                    break
        if verbose:
            print(f"Added {added} columns this iter.")
        if iter_count >= max_iters:
            return f_sol, {'status': 'max_iters_reached', 'iters': iter_count, 'working_set': len(working_set)}

# -------------------------- Main runner -------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--A', default=None, help='Path to A.npy')
    parser.add_argument('--B', default=None, help='Path to B.npy')
    parser.add_argument('--ell', default=None, help='Path to ell.npy')
    parser.add_argument('--ub', default=None, help='Path to ub.npy')
    parser.add_argument('--demo', action='store_true', help='Generate demo data (smaller J)')
    parser.add_argument('--method', default='all', choices=['A','B','C','all'])
    args = parser.parse_args()

    A, B, ell, ub = load_arrays(args.A, args.B, args.ell, args.ub, synth_demo=args.demo)
    U, s, V, Z = compute_svd_and_precompute(A, B)

    n, J = Z.shape
    print(f"n={n}, J={J}")
    print("Singular values (first 10):", s[:10])

    # Quick necessary check (like earlier): check per (j,k) achievable max/min ranges
    print("Running quick necessary check (fast feasibility test)...")
    t_max = 1.0 / (s**2)
    # We'll compute per (j,k) Xmax and Xmin by using upper bounds t_max
    # Xmax_kj = sum_{i: a>0} a_{k,i}^{(j)} * t_max_i ; a_{k,i}^{(j)} = v_{k,i} * z_i^{(j)}
    violations = 0
    for j in range(J):
        z_j = Z[:, j]
        A_j = V * z_j[np.newaxis, :]
        Xmax = (np.maximum(A_j, 0) * t_max[np.newaxis, :]).sum(axis=1)
        Xmin = (np.minimum(A_j, 0) * t_max[np.newaxis, :]).sum(axis=1)
        if np.any(Xmax < ell) or np.any(Xmin > ub):
            violations += 1
            # optional: print an example and break
            print(f"Quick-check fail for j={j}; at least one (k) impossible")
            break
    if violations == 0:
        print("Quick necessary check passed: no obvious impossibility detected.")

    # Choose which methods to run
    methods = [args.method] if args.method != 'all' else ['A','B','C']

    results = {}
    if 'A' in methods:
        print("\n===== Running Method A (Active-set / constraint generation) =====")
        t0 = time.time()
        f_sol_A, infoA = solve_lp_active_set(V, Z, s, ell, ub,
                                             init_sample_J=200, max_iters=50,
                                             tol=1e-9, add_top_k=200, verbose=True)
        t1 = time.time()
        print("Method A done in {:.2f}s. Info: {}".format(t1-t0, infoA))
        results['A'] = (f_sol_A, infoA)
        if f_sol_A is not None:
            # compute w and L if desired
            wA = 1.0 / (f_sol_A + 1e-30) - s**2
            wA = np.maximum(wA, 0.0)
            print("Method A: first 10 w_i:", wA[:10])

    if 'B' in methods:
        print("\n===== Running Method B (Full LP via HiGHS) =====")
        t0 = time.time()
        resB = solve_lp_highs_full(V, Z, s, ell, ub, verbose=True)
        t1 = time.time()
        print("Method B done in {:.2f}s." .format(t1-t0))
        results['B'] = resB
        if resB.success:
            f_sol_B = resB.x
            wB = 1.0 / (f_sol_B + 1e-30) - s**2
            wB = np.maximum(wB, 0.0)
            print("Method B: first 10 w_i:", wB[:10])
        else:
            print("Method B failed or timed out: ", resB.message)

    if 'C' in methods:
        print("\n===== Running Method C (Cutting-plane / batch) =====")
        t0 = time.time()
        f_sol_C, infoC = solve_lp_cutting_plane(V, Z, s, ell, ub,
                                                initial_J=100, batch_add=200,
                                                max_iters=60, tol=1e-9, verbose=True)
        t1 = time.time()
        print("Method C done in {:.2f}s. Info: {}".format(t1-t0, infoC))
        results['C'] = (f_sol_C, infoC)
        if f_sol_C is not None:
            wC = 1.0 / (f_sol_C + 1e-30) - s**2
            wC = np.maximum(wC, 0.0)
            print("Method C: first 10 w_i:", wC[:10])

    # Save results to disk for later inspection
    import os
    os.makedirs("svd_reg_results", exist_ok=True)
    for key, val in results.items():
        if val is None:
            continue
        if key == 'B':
            # full solver result object
            np.save(f"svd_reg_results/res_{key}_status.npy", np.array([0]))
            # can't save all, but store message
            with open(f"svd_reg_results/res_{key}_message.txt", 'w') as f:
                f.write(str(val.message))
        else:
            f_sol, info = val
            if f_sol is not None:
                np.save(f"svd_reg_results/f_{key}.npy", f_sol)
                w = 1.0 / (f_sol + 1e-30) - s**2
                np.save(f"svd_reg_results/w_{key}.npy", np.maximum(w, 0.0))
                with open(f"svd_reg_results/info_{key}.txt", 'w') as f:
                    f.write(str(info))
    print("Results saved to ./svd_reg_results/")

if __name__ == "__main__":
    main()
