import numpy as np
from numpy.linalg import svd
from scipy.optimize import minimize, Bounds, LinearConstraint

# =====================================================
# 0. SVD и подготовка
# =====================================================

def prepare_svd(A, B):
    U, s, Vt = svd(A, full_matrices=False)
    V = Vt.T
    UB = U.T @ B
    alpha = np.sum(UB**2, axis=1)
    return U, s, V, UB, alpha


# =====================================================
# 1. Подбор регуляризации f (одна для всех b)
# =====================================================

def compute_X(f, U, V, UB):
    return V @ (f[:, None] * UB)

def max_violation(X, l, u):
    return max(
        np.max(np.maximum(l[:, None] - X, 0)),
        np.max(np.maximum(X - u[:, None], 0))
    )

def find_active_constraints(f, U, V, UB, l, u, tol=1e-8):
    X = compute_X(f, U, V, UB)
    active = []

    for k in range(X.shape[1]):
        xk = X[:, k]
        low = np.where(xk < l - tol)[0]
        high = np.where(xk > u + tol)[0]

        for j in low:
            active.append((k, j, "low"))
        for j in high:
            active.append((k, j, "high"))

    return active


def build_constraints(active, U, V, UB, l, u):
    Arows = []
    lb = []
    ub = []

    for (k, j, side) in active:
        z = UB[:, k]
        row = V[j, :] * z   # 1 × n

        Arows.append(row)
        if side == "low":
            lb.append(l[j])
            ub.append(np.inf)
        else:
            lb.append(-np.inf)
            ub.append(u[j])

    return np.array(Arows), np.array(lb), np.array(ub)




def fit_regularization(A, B, l, u, maxiter=300):
    U, s, V, UB, alpha = prepare_svd(A, B)
    n = A.shape[1]
    K = B.shape[1]

    # ---- целевая функция (RMS) ----
    def obj(f):
        return np.sum(alpha * (s**2 * f - 1.0)**2)

    def grad(f):
        return 2 * alpha * (s**2 * f - 1.0) * s**2

    # ---- линейные ограничения на x ----
    Acons = []
    lb = []
    ub = []

    for k in range(K):
        z = UB[:, k]
        Acons.append(V * z[None, :])
        lb.append(l)
        ub.append(u)

    Acons = np.vstack(Acons)
    lb = np.hstack(lb)
    ub = np.hstack(ub)

    lin_con = LinearConstraint(Acons, lb, ub)

    bounds = Bounds(0.0, 1.0 / s)
    f0 = 1.0 / s

    res = minimize(
        obj,
        f0,
        jac=grad,
        method="trust-constr",
        constraints=[lin_con],
        bounds=bounds,
        options=dict(maxiter=maxiter, verbose=2)
    )

    return res, f0, U, s, V, UB, alpha


def fit_regularization_constraint_generation(A, B, l, u, maxiter=500, max_outer=10):
    U, s, V, UB, alpha = prepare_svd(A, B)

    def obj(f):
        return np.sum(alpha * (s**2 * f - 1.0)**2)

    def grad(f):
        return 2 * alpha * (s**2 * f - 1.0) * s**2

    bounds = Bounds(0.0, 1.0 / s)
    f = 1.0 / s  # старт: минимальный RMS
    
    Active = []   # список ВСЕХ ограничений


    for it in range(max_outer):
        newly_active = find_active_constraints(f, U, V, UB, l, u)

        print(f"[Iter {it}] active constraints:", len(active))
        if not newly_active:
            print("All constraints satisfied. Converged to global optimum.")
            break
        
        Active.extend(newly_active)  # <<< КЛЮЧЕВО
        Active = list(set(Active))   # убрать дубли

        Acons, lb, ub = build_constraints(Active, U, V, UB, l, u)
        lin_con = LinearConstraint(Acons, lb, ub)

        res = minimize(
            obj, f, jac=grad,
            method="trust-constr",
            constraints=[lin_con],
            bounds=bounds,
            options=dict(maxiter=maxiter, verbose=2)
        )

        f = res.x

    return f, 1.0 / s, U, s, V, UB, alpha



# =====================================================
# 2. Метрики качества
# =====================================================

def metrics(A, B, f, U, s, V, l, u):
    X = V @ (f[:, None] * (U.T @ B))
    R = A @ X - B

    rms = np.sqrt(np.mean(R**2))
    viol = max(
        np.max(np.maximum(l[:, None] - X, 0)),
        np.max(np.maximum(X - u[:, None], 0))
    )
    return rms, viol, X


# =====================================================
# 3. Диагностика потери RMS по модам
# =====================================================

def rms_loss_per_mode(f, s, alpha):
    return alpha * (s**2 * f - 1.0)**2


# =====================================================
# 4. Диагностика активных ограничений
# =====================================================

def active_constraints(X, l, u, tol=1e-8):
    active_low = np.any(np.abs(X - l[:, None]) < tol, axis=1)
    active_up  = np.any(np.abs(X - u[:, None]) < tol, axis=1)
    return active_low, active_up


# =====================================================
# 5. СРАВНЕНИЕ: без регуляризации vs с регуляризацией
# =====================================================

def full_report(A, B, l, u):
    res, f_pinv, U, s, V, UB, alpha = fit_regularization(A, B, l, u)
    f_reg = res.x

    rms0, v0, _ = metrics(A, B, f_pinv, U, s, V, l, u)
    rms1, v1, Xreg = metrics(A, B, f_reg,  U, s, V, l, u)

    print("\n========== SUMMARY ==========")
    print(f"Pseudoinverse: RMS={rms0:.3e}, max violation={v0:.3e}")
    print(f"Regularized  : RMS={rms1:.3e}, max violation={v1:.3e}")

    # --- моды ---
    loss = rms_loss_per_mode(f_reg, s, alpha)
    idx = np.argsort(loss)[::-1]

    print("\nTop RMS-destroying modes:")
    for i in idx[:8]:
        print(f"mode {i:2d} | sigma={s[i]:.2e} | f={f_reg[i]:.2e} | loss={loss[i]:.2e}")

    # --- ограничения ---
    al, au = active_constraints(Xreg, l, u)
    print("\nActive lower bounds:", np.where(al)[0])
    print("Active upper bounds:", np.where(au)[0])

    return res






# Separated
import numpy as np
from numpy.linalg import svd
from scipy.optimize import minimize, LinearConstraint, Bounds

# =========================
# ВХОДНЫЕ ДАННЫЕ
# =========================
# A: (m, n) = (3000, 39)
# B: (m, K) = (3000, 15000)
# box constraints on x: l <= x <= u

def fit_svd_regularization(A, B, l, u):
    m, n = A.shape
    K = B.shape[1]

    # ---- SVD ----
    U, s, Vt = svd(A, full_matrices=False)
    V = Vt.T

    # ---- проекции b на U ----
    UB = U.T @ B                    # (n, K)
    alpha = np.sum(UB**2, axis=1)   # (n,)

    # ---- целевая функция ----
    def objective(f):
        return np.sum(alpha * (s**2 * f - 1.0)**2)

    def grad(f):
        return 2 * alpha * (s**2 * f - 1.0) * s**2

    # ---- ограничения на x ----
    # x = V diag(f) U^T b
    # => линейные по f
    A_cons = []
    lb_cons = []
    ub_cons = []

    for k in range(B.shape[1]):
        z = UB[:, k]
        M = V * z[None, :]           # (n, n)
        A_cons.append(M)
        lb_cons.append(l)
        ub_cons.append(u)

    A_cons = np.vstack(A_cons)
    lb_cons = np.hstack(lb_cons)
    ub_cons = np.hstack(ub_cons)

    lin_con = LinearConstraint(A_cons, lb_cons, ub_cons)

    # ---- bounds на f ----
    bounds = Bounds(0.0, 1.0 / s)

    # ---- старт: псевдообратная ----
    f0 = 1.0 / s

    res = minimize(
        objective,
        f0,
        jac=grad,
        method="trust-constr",
        constraints=[lin_con],
        bounds=bounds,
        options=dict(verbose=3, maxiter=500)
    )

    return res, f0, s, U, V


def compute_metrics(A, B, f, s, U, V, l, u):
    UB = U.T @ B
    X = V @ (f[:, None] * UB)

    # RMS
    R = A @ X - B
    rms = np.sqrt(np.mean(R**2))

    # violations
    viol_low = np.maximum(l[:, None] - X, 0)
    viol_high = np.maximum(X - u[:, None], 0)
    max_viol = max(np.max(viol_low), np.max(viol_high))

    return rms, max_viol, X

def compare_solutions(A, B, l, u, f_reg, f_pinv, s, U, V):
    rms_pinv, viol_pinv, _ = compute_metrics(A, B, f_pinv, s, U, V, l, u)
    rms_reg, viol_reg, _   = compute_metrics(A, B, f_reg,  s, U, V, l, u)

    print("=== COMPARISON ===")
    print(f"Pseudoinverse: RMS={rms_pinv:.3e}, max violation={viol_pinv:.3e}")
    print(f"Regularized  : RMS={rms_reg:.3e}, max violation={viol_reg:.3e}")


def rms_loss_per_mode(f, s, alpha):
    return alpha * (s**2 * f - 1.0)**2


loss = rms_loss_per_mode(f_reg, s, alpha)
idx = np.argsort(loss)[::-1]

print("Top modes killing RMS:")
for i in idx[:10]:
    print(f"mode {i:2d}: sigma={s[i]:.2e}, loss={loss[i]:.2e}")


def active_constraints(X, l, u, tol=1e-8):
    active_low  = np.any(np.abs(X - l[:, None]) < tol, axis=1)
    active_high = np.any(np.abs(X - u[:, None]) < tol, axis=1)
    return active_low, active_high


_, _, Xreg = compute_metrics(A, B, f_reg, s, U, V, l, u)
al, ah = active_constraints(Xreg, l, u)

print("Active lower bounds:", np.where(al)[0])
print("Active upper bounds:", np.where(ah)[0])



