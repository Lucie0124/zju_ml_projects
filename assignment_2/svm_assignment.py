"""
SVM assignment - hard/soft margin, primal and dual.
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import LinearConstraint, minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


# ---------- data loading ---------------------------

def encode_labels(y):
    # M (malignant) -> +1, B (benign) -> -1
    y = np.asarray(y).ravel()

    # .mat sometimes wraps each label in its own little array, so unwrap it
    if y.dtype == object:
        y = np.array([np.asarray(v).ravel()[0] for v in y])

    if y.dtype.kind in ("U", "S"):
        y = np.array([str(v).strip().upper() for v in y])
        return np.where(y == "M", 1.0, -1.0)

    # otherwise assume it's already numeric
    y = y.astype(float)
    return np.where(y > 0, 1.0, -1.0)


def df_to_xy(df):
    df = df.copy()
    drop = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if "id" in df.columns:
        drop.append("id")
    y = encode_labels(df["diagnosis"].to_numpy())
    Xdf = df.drop(columns=drop + ["diagnosis"])
    return Xdf.to_numpy(dtype=float), y


def load_data(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return df_to_xy(pd.read_csv(path))
    elif ext in (".pkl", ".pickle"):
        return df_to_xy(pd.read_pickle(path))
    elif ext == ".mat":
        m = loadmat(path)
        X = np.asarray(m["X"], dtype=float)
        y = encode_labels(m["y"])
        return X, y
    else:
        raise ValueError("don't know how to load " + ext)


# ---------- helpers -----------------

def predict(X, w, b):
    return np.where(X @ w - b >= 0, 1.0, -1.0)


def hard_primal_obj(w):
    return 0.5 * w.dot(w)


def soft_primal_obj(X, y, w, b, C):
    n = len(y)
    slack = np.maximum(0.0, 1.0 - y * (X @ w - b))
    return 0.5 * w.dot(w) + (C / n) * slack.sum()


def get_b(X, y, w, alpha, upper):
    # use the support vectors that sit exactly on the margin
    # for soft margin those are the ones with 0 < alpha < C/N
    if upper is None:
        on_margin = alpha > 1e-6
    else:
        on_margin = (alpha > 1e-6) & (alpha < upper - 1e-6)
    if not on_margin.any():
        on_margin = alpha > 1e-6
    if not on_margin.any():
        return 0.0
    # from y_i (w.x_i - b) = 1  ->  b = w.x_i - y_i, then average
    return np.mean(X[on_margin] @ w - y[on_margin])


# ---------- primal ----------------------------------------------------

def hard_primal(X, y):
    # min 0.5||w||^2  s.t.  y_i (w.x_i - b) >= 1
    # unknowns are stacked as z = [w, b]
    n, d = X.shape
    A = np.zeros((n, d + 1))
    A[:, :d] = y[:, None] * X
    A[:, d] = -y
    con = LinearConstraint(A, np.ones(n), np.inf)

    f = lambda z: 0.5 * z[:d].dot(z[:d])
    def grad(z):
        g = np.zeros_like(z)
        g[:d] = z[:d]
        return g

    res = minimize(f, np.zeros(d + 1), method="SLSQP", jac=grad,
                   constraints=[con], options={"ftol": 1e-9, "maxiter": 1000})
    w, b = res.x[:d], res.x[d]
    margin = y * (X @ w - b)
    sv = np.where(np.abs(margin - 1) < 1e-4)[0]
    return {"name": "hard-primal", "w": w, "b": b, "sv": sv,
            "primal": hard_primal_obj(w), "ok": res.success, "msg": res.message}


def soft_primal(X, y, C):
    # min 0.5||w||^2 + (C/n) sum(zeta)
    # z = [w, b, zeta]
    n, d = X.shape
    g = C / n

    # y_i(w.x_i - b) + zeta_i >= 1
    A1 = np.zeros((n, d + 1 + n))
    A1[:, :d] = y[:, None] * X
    A1[:, d] = -y
    A1[:, d + 1:] = np.eye(n)
    # zeta_i >= 0
    A2 = np.zeros((n, d + 1 + n))
    A2[:, d + 1:] = np.eye(n)

    A = np.vstack([A1, A2])
    lb = np.concatenate([np.ones(n), np.zeros(n)])
    con = LinearConstraint(A, lb, np.inf)

    def f(z):
        return 0.5 * z[:d].dot(z[:d]) + g * z[d + 1:].sum()
    def grad(z):
        out = np.zeros_like(z)
        out[:d] = z[:d]
        out[d + 1:] = g
        return out

    z0 = np.concatenate([np.zeros(d), [0.0], np.ones(n)])  # start feasible
    res = minimize(f, z0, method="SLSQP", jac=grad, constraints=[con],
                   options={"ftol": 1e-9, "maxiter": 1000})
    w, b = res.x[:d], res.x[d]
    zeta = np.maximum(res.x[d + 1:], 0.0)
    margin = y * (X @ w - b)
    sv = np.where(margin <= 1 + 1e-4)[0]
    return {"name": "soft-primal", "w": w, "b": b, "sv": sv, "slack": zeta,
            "primal": soft_primal_obj(X, y, w, b, C), "ok": res.success,
            "msg": res.message}


# ---------- dual ----------

def dual(X, y, C):
    # max  sum(a) - 0.5 a^T Q a   with Q_ij = y_i y_j x_i.x_j
    # s.t. sum(a_i y_i) = 0,  0 <= a_i (<= C/n for soft margin)
    n = X.shape[0]
    Z = y[:, None] * X
    Q = Z @ Z.T
    Q = 0.5 * (Q + Q.T)  # keep it symmetric

    upper = None if C is None else C / n
    bounds = [(0.0, upper)] * n

    # scipy minimises, so negate the dual objective
    f = lambda a: 0.5 * a @ Q @ a - a.sum()
    jac = lambda a: Q @ a - np.ones(n)
    eq = {"type": "eq", "fun": lambda a: y @ a, "jac": lambda a: y}

    res = minimize(f, np.zeros(n), method="SLSQP", jac=jac, bounds=bounds,
                   constraints=[eq], options={"ftol": 1e-10, "maxiter": 1000})
    alpha = np.clip(res.x, 0.0, upper)

    w = (alpha * y) @ X
    b = get_b(X, y, w, alpha, upper)

    dual_val = alpha.sum() - 0.5 * alpha @ Q @ alpha
    if C is None:
        primal_val = hard_primal_obj(w)
        name = "hard-dual"
    else:
        primal_val = soft_primal_obj(X, y, w, b, C)
        name = "soft-dual"

    sv = np.where(alpha > 1e-6)[0]
    return {"name": name, "w": w, "b": b, "alpha": alpha, "sv": sv,
            "primal": primal_val, "dual": dual_val,
            "gap": primal_val - dual_val, "ok": res.success, "msg": res.message}


# ---------- libsvm comparison ----------

def fit_libsvm(X, y, C):
    # sklearn's SVC is the libsvm wrapper. It uses f = w.x + intercept,
    # so our b = -intercept. Our objective has C/n, libsvm has plain C,
    # so the matching penalty is C/n.
    from sklearn.svm import SVC
    n = X.shape[0]
    C_sk = (1e6 / n) if C is None else (C / n)
    clf = SVC(kernel="linear", C=C_sk, shrinking=False, tol=1e-9)
    clf.fit(X, y)
    w = clf.coef_.ravel()
    b = -clf.intercept_[0]
    alpha = np.zeros(n)
    alpha[clf.support_] = np.abs(clf.dual_coef_).ravel()
    return clf, w, b, alpha


# ---------- printing ------------------

def show(res, Xtr, ytr, Xte, yte):
    print("\n[%s]  %s" % (res["name"], "OK" if res["ok"] else "WARN"))
    print("  train acc :", round(accuracy_score(ytr, predict(Xtr, res["w"], res["b"])), 4))
    print("  test  acc :", round(accuracy_score(yte, predict(Xte, res["w"], res["b"])), 4))
    print("  #SV       :", len(res["sv"]))
    print("  ||w||     :", round(np.linalg.norm(res["w"]), 6))
    print("  b         :", round(res["b"], 6))
    print("  primal    :", round(res["primal"], 6))
    if "dual" in res:
        print("  dual      :", round(res["dual"], 6))
        print("  gap       : %.3e" % res["gap"])
    cm = confusion_matrix(yte, predict(Xte, res["w"], res["b"]), labels=[-1, 1])
    print("  confusion (rows true [-1,+1], cols pred [-1,+1]):")
    print("   ", str(cm).replace("\n", "\n    "))


def show_vs_libsvm(res, clf, w, b, alpha, Xte, yte):
    print("    vs libsvm:")
    print("      ||dw|| = %.3e" % np.linalg.norm(res["w"] - w))
    print("      |db|   = %.3e" % abs(res["b"] - b))
    if "alpha" in res:
        print("      ||da|| = %.3e" % np.linalg.norm(res["alpha"] - alpha))
    f_mine = Xte @ res["w"] - res["b"]
    f_lib = Xte @ w - b
    print("      max|df| = %.3e" % np.max(np.abs(f_mine - f_lib)))
    print("      corr    = %.6f" % np.corrcoef(f_mine, f_lib)[0, 1])
    print("      acc mine/lib = %.4f / %.4f" % (
        accuracy_score(yte, predict(Xte, res["w"], res["b"])),
        accuracy_score(yte, clf.predict(Xte))))


# ---------- main -----------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.csv")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", default="all",
                   choices=["hard-primal", "hard-dual", "soft-primal",
                            "soft-dual", "all"])
    args = p.parse_args()

    X, y = load_data(args.data)
    print("loaded", args.data, "->", X.shape[0], "samples,", X.shape[1], "features")
    print("class balance: +1(M) =", int((y == 1).sum()),
          " -1(B) =", int((y == -1).sum()))

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    # standardize with training stats only
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr)
    Xte = sc.transform(Xte)

    n = len(ytr)
    print("C =", args.C, " C/N =", round(args.C / n, 6),
          " mode =", args.mode)

    models = []
    if args.mode in ("hard-primal", "all"):
        models.append(hard_primal(Xtr, ytr))
    if args.mode in ("hard-dual", "all"):
        models.append(dual(Xtr, ytr, None))
    if args.mode in ("soft-primal", "all"):
        models.append(soft_primal(Xtr, ytr, args.C))
    if args.mode in ("soft-dual", "all"):
        models.append(dual(Xtr, ytr, args.C))

    for res in models:
        show(res, Xtr, ytr, Xte, yte)
        C_for_ref = None if res["name"].startswith("hard") else args.C
        clf, w, b, alpha = fit_libsvm(Xtr, ytr, C_for_ref)
        show_vs_libsvm(res, clf, w, b, alpha, Xte, yte)

    # quick check that primal and dual landed in the same place
    if args.mode == "all":
        hp, hd, sp, sd = models
        print("\nprimal vs dual (should be tiny):")
        print("  hard: ||dw|| = %.3e  |db| = %.3e" % (
            np.linalg.norm(hp["w"] - hd["w"]), abs(hp["b"] - hd["b"])))
        print("  soft: ||dw|| = %.3e  |db| = %.3e" % (
            np.linalg.norm(sp["w"] - sd["w"]), abs(sp["b"] - sd["b"])))


if __name__ == "__main__":
    main()
