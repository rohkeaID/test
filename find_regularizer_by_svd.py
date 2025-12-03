# find_regularizer_by_svd.py
# См. подробные комментарии в теле скрипта — реализует quick necessary-check и QP для t.
import numpy as np
import os
import time
from scipy.linalg import svd
from scipy import sparse

try:
    import cvxpy as cp
except Exception as e:
    raise ImportError("Требуется установить cvxpy (pip install cvxpy). Ошибка: " + str(e))

np.set_printoptions(precision=6, suppress=True)


def load_or_synthesize():
    if os.path.exists("A.npy") and os.path.exists("B.npy"):
        A = np.load("A.npy")
        B = np.load("B.npy")
        print("Loaded A.npy and B.npy from disk.")
    else:
        print("A.npy / B.npy not found — генерирую синтетические данные меньше по размеру для демонстрации.")
        rng = np.random.default_rng(42)
        m, n = 3000, 39
        J = 15000
        U_rnd = np.linalg.qr(rng.normal(size=(m, n)))[0]
        V_rnd = np.linalg.qr(rng.normal(size=(n, n)))[0]
        sigs = np.exp(np.linspace(np.log(1e3), np.log(1e-7), n))
        A = (U_rnd * sigs) @ V_rnd.T
        B = rng.normal(size=(m, J))
        B = B[:, :500]  # демо-поднабор; замените реальным B
        print(f"Synthesized A shape {A.shape}, B shape {B.shape} (demo subset).")
    return A, B


def svd_economic(A):
    U, s, Vt = svd(A, full_matrices=False)
    return U, s, Vt


def quick_necessary_check(U, s, Vt, B, ell, ub):
    n = s.size
    J = B.shape[1]
    V = Vt.T
    C = U.T @ B  # n x J
    t_max = 1.0 / (s ** 2)
    m_viol = 0
    diagnostics = []
    start = time.time()
    for j in range(B.shape[1]):
        c_j = C[:, j]
        bvec = s * c_j
        A_j = V * bvec[np.newaxis, :]
        Xmax = (np.maximum(A_j, 0) * t_max[np.newaxis, :]).sum(axis=1)
        Xmin = (np.minimum(A_j, 0) * t_max[np.newaxis, :]).sum(axis=1)
        bad_max_idx = np.where(Xmax < ell)[0]
        bad_min_idx = np.where(Xmin > ub)[0]
        if bad_max_idx.size > 0 or bad_min_idx.size > 0:
            m_viol += bad_max_idx.size + bad_min_idx.size
            if len(diagnostics) < 10:
                diagnostics.append({
                    "j": j,
                    "bad_max_idx": bad_max_idx.tolist(),
                    "bad_min_idx": bad_min_idx.tolist(),
                    "Xmax_sample": Xmax[:5].tolist(),
                    "Xmin_sample": Xmin[:5].tolist(),
                })
    elapsed = time.time() - start
    feasible = (m_viol == 0)
    print(f"Quick necessary check completed in {elapsed:.2f}s. Violations count: {m_viol}.")
    return feasible, diagnostics


def solve_qp_for_t(V, s, C, ell, ub, t0=None, max_constraints=200000, use_slack=False):
    n, J = C.shape[0], C.shape[1]
    if t0 is None:
        t0 = 1.0 / (s ** 2)
    t_max = 1.0 / (s ** 2)
    total_constraints = 2 * n * J
    print(f"QP attempt: n={n}, J={J}, total linear constraints (upper+lower) = {total_constraints}.")
    if total_constraints > max_constraints:
        Jsub = int(max_constraints / (2 * n))
        Jsub = max(50, Jsub)
        rng = np.random.default_rng(0)
        cols = rng.choice(J, size=Jsub, replace=False)
        print(f"Total constraints {total_constraints} > {max_constraints}. Using random subset Jsub={Jsub} for initial solve.")
    else:
        cols = np.arange(J)
        Jsub = J

    # Build sparse block matrix A_sub (n*Jsub x n)
    rows = []
    data = []
    cols_idx = []
    row_ptr = 0
    for idx_j, j in enumerate(cols):
        c_j = C[:, j]
        bvec = s * c_j
        A_j = V * bvec[np.newaxis, :]
        A_j_coo = sparse.coo_matrix(A_j)
        rows.append(A_j_coo.row + row_ptr)
        cols_idx.append(A_j_coo.col)
        data.append(A_j_coo.data)
        row_ptr += n
    rows = np.concatenate(rows)
    cols_idx = np.concatenate(cols_idx)
    data = np.concatenate(data)
    A_sub = sparse.coo_matrix((data, (rows, cols_idx)), shape=(n * Jsub, n)).tocsr()
    ell_rep = np.tile(ell, Jsub)
    ub_rep = np.tile(ub, Jsub)

    t_var = cp.Variable(n)
    objective = cp.sum_squares(t_var - t0)
    constraints = [t_var >= 1e-16, t_var <= t_max]
    constraints += [A_sub @ t_var <= ub_rep, -A_sub @ t_var <= -ell_rep]
    if use_slack:
        s_pos = cp.Variable(A_sub.shape[0], nonneg=True)
        s_neg = cp.Variable(A_sub.shape[0], nonneg=True)
        constraints = [t_var >= 1e-16, t_var <= t_max,
                       A_sub @ t_var - s_pos <= ub_rep, -A_sub @ t_var - s_neg <= -ell_rep]
        beta = 1e6
        objective = cp.sum_squares(t_var - t0) + beta * cp.sum(s_pos + s_neg)

    prob = cp.Problem(cp.Minimize(objective), constraints)
    print("Starting cvxpy solve (this may take a while)...")
    start = time.time()
    try:
        prob.solve(solver=cp.OSQP, verbose=True, eps_abs=1e-6, eps_rel=1e-6, max_iter=200000)
    except Exception as e:
        print("Solver failed with exception:", e)
        return None, {"status": "solver_error", "error": str(e)}
    elapsed = time.time() - start
    print(f"QP solved in {elapsed:.2f}s. status = {prob.status}")

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None, {"status": prob.status}

    t_sol = t_var.value
    if Jsub < J:
        print("Verifying solution on full dataset...")
        violations = 0
        bad_examples = []
        BATCH = 1000
        for start_j in range(0, J, BATCH):
            end_j = min(J, start_j + BATCH)
            C_block = C[:, start_j:end_j]
            E = (s[:, None] * C_block) * t_sol[:, None]
            Xblock = V @ E
            below = Xblock < ell[:, None] - 1e-9
            above = Xblock > ub[:, None] + 1e-9
            if np.any(below) or np.any(above):
                violations += np.count_nonzero(below) + np.count_nonzero(above)
                if len(bad_examples) < 10:
                    idxs = np.argwhere(below | above)
                    for r, c in idxs[:10 - len(bad_examples)]:
                        bad_examples.append((start_j + c, int(r), Xblock[r, c]))
        print(f"Verification finished. Violations on full set: {violations}.")
        return t_sol, {"status": prob.status, "violations": violations, "bad_examples": bad_examples}
    else:
        return t_sol, {"status": prob.status, "violations": 0}


def main():
    A, B = load_or_synthesize()
    m, n = A.shape
    J = B.shape[1]
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    # Задайте реальные ell, ub вектор-границы (длиной n). Здесь пример:
    ell = -0.5 * np.ones(n)
    ub = 0.5 * np.ones(n)

    U, s, Vt = svd_economic(A)
    V = Vt.T
    print("Computed economic SVD. Singular values (first 10):", s[:10])

    feasible, diag = quick_necessary_check(U, s, Vt, B, ell, ub)
    if not feasible:
        print("Necessary condition failed for the given box and set of b's. Diagnostics (up to 10 cases):")
        for d in diag:
            print(d)
        print("\nConclusion: with diagonal-in-V regularization there is NO feasible t that forces all x^j into boxes.")
        return

    C = U.T @ B
    t0 = 1.0 / (s ** 2)
    t_sol, info = solve_qp_for_t(V, s, C, ell, ub, t0=t0, max_constraints=200000, use_slack=False)

    if t_sol is None:
        print("QP did not produce a solution. Info:", info)
        print("Trying QP with slack variables on subset (use_slack=True).")
        t_sol, info2 = solve_qp_for_t(V, s, C, ell, ub, t0=t0, max_constraints=200000, use_slack=True)
        print("Slack attempt info:", info2)
        if t_sol is not None:
            print("t (partial) found. Check info2['violations'].")
        return

    print("t solution found. Info:", info)
    w = 1.0 / t_sol - s ** 2
    w = np.maximum(w, 0.0)
    L = np.diag(np.sqrt(w)) @ V.T
    print("Computed w (first 10):", w[:10])
    np.save("t_solution.npy", t_sol)
    np.save("w_solution.npy", w)
    np.save("L_solution.npy", L)
    print("Saved t_solution.npy, w_solution.npy, L_solution.npy.")

