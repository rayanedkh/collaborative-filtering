
import numpy as np
import os
from tqdm import tqdm, trange
import argparse


import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

# ---------- Kernels ----------
def rbf_kernel(u, v, gamma=1.0):
    diff = u - v
    return np.exp(-gamma * np.dot(diff, diff))

def mlp_kernel(u, v, w=1.0, b=0.0):
    num = w * np.dot(u, v) + b
    den = np.sqrt((w * np.dot(u, u) + b + 1.0) * (w * np.dot(v, v) + b + 1.0))
    val = np.arcsin(num / den)
    return val

# ---------- Helpers ----------
def center_kernel(K):
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n

def compute_pairwise_kernel(X, mask, kernel='rbf', kernel_params=None):
    """
    Compute Gram matrix K where for pair (i,j) kernel is computed only on
    features known in both i and j (mask rows indicate known features).
    X: (n, p) with NaN for missing (but we pass filled X_est)
    mask: boolean (n, p) True where value is observed in original data (not current fill)
    kernel: 'rbf' or 'mlp' or a callable
    kernel_params: dict of kernel parameters
    """
    if kernel_params is None:
        kernel_params = {}
    n, p = X.shape
    K = np.zeros((n, n))
    # choose kernel function
    if kernel == 'rbf':
        kfunc = lambda a, b: rbf_kernel(a, b, **kernel_params)
    elif kernel == 'mlp':
        kfunc = lambda a, b: mlp_kernel(a, b, **kernel_params)
    elif callable(kernel):
        kfunc = kernel
    else:
        raise ValueError("Unknown kernel")

    for i in range(n):
        xi = X[i]
        mi = mask[i]
        for j in range(i, n):
            xj = X[j]
            mj = mask[j]
            common = mi & mj
            if np.sum(common) == 0:
                val = 0.0
            else:
                val = kfunc(xi[common], xj[common])
            K[i, j] = val
            K[j, i] = val
    return K

# ---------- Objective L(K, C) ----------
def cross_entropy_obj(K, C):
    """
    L = 1/2 * ( log|C| + trace(K C^{-1}) )
    C must be positive definite (we add small reg if necessary).
    """
    # regularize C a little for numerical stability
    eps = 1e-9
    n = C.shape[0]
    C_reg = C + eps * np.eye(n)
    sign, logdet = np.linalg.slogdet(C_reg)
    if sign <= 0:
        # numerical fallback
        logdet = np.log(np.linalg.det(C_reg) + 1e-12)
    invC = np.linalg.inv(C_reg)
    tr = np.trace(K @ invC)
    return 0.5 * (logdet + tr)

# ---------- Main algorithm ----------
def missing_kpca_sanguinetti(X_orig,
                             q=None,
                             kernel='rbf',
                             kernel_params=None,
                             max_outer_iter=20,
                             tol=1e-6,
                             max_inner_optit=200,
                             verbose=True):
    """
    Implements Algorithm 1 from Sanguinetti & Lawrence (2006).
    X_orig: (n, p) with np.nan for missing entries.
    q: latent dimension (if None, choose so that ~95% of variance retained for initial K)
    kernel/kernel_params: kernel choice and its params (e.g., {'gamma':0.5} for RBF)
    Returns: X_completed, info dict
    """
    X = np.array(X_orig, dtype=float)
    n, p = X.shape
    mask_obs = ~np.isnan(X_orig)   # True where original observed
    
    # 1) initialize missing with column means
    col_means = np.nanmean(X, axis=0)
    # where column mean may be nan (entire column missing), set 0
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X_est = X.copy()
    inds = np.where(np.isnan(X_est))
    X_est[inds] = np.take(col_means, inds[1])

    if kernel_params is None:
        kernel_params = {}

    prev_obj = np.inf
    for outer in trange(max_outer_iter):
        # 2) compute kernel on current X_est (pairwise using observed-common dims)
        K = compute_pairwise_kernel(X_est, mask_obs, kernel=kernel, kernel_params=kernel_params)

        # center K
        Kc = center_kernel(K)

        # 3) eigendecompose Kc
        eigvals, eigvecs = eigh(Kc)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # choose q if None: pick smallest q s.t. 95% variance
        if q is None:
            total = np.sum(np.maximum(eigvals, 0.0))
            if total <= 0:
                q = min(5, n-1)
            else:
                csum = np.cumsum(np.maximum(eigvals, 0.0))
                q = int(np.searchsorted(csum / total, 0.95)) + 1
                q = min(max(q, 1), n-1)

        # estimate sigma^2 as mean of leftover eigenvalues (like PPCA)
        if n > q:
            sigma2 = np.maximum(np.mean(eigvals[q:]), 1e-8)
        else:
            sigma2 = 1e-8

        # build C = X X^T + sigma2 * I using top-q components per paper
        # In paper: X_latent = U_q * sqrt(V_q - sigma2 I)
        Vq = eigvals[:q]
        Uq = eigvecs[:, :q]
        # ensure nonnegative diag
        Lambda_sq = np.maximum(Vq - sigma2, 0.0)
        # C = Uq diag(Lambda_sq) Uq^T + sigma2 I
        C = (Uq * Lambda_sq) @ Uq.T + sigma2 * np.eye(n)

        # compute current objective
        obj_now = cross_entropy_obj(K, C)
        if verbose:
            print(f"Outer it {outer:2d}: q={q}, sigma2={sigma2:.4e}, obj={obj_now:.6e}")

        # check relative change
        if np.abs(prev_obj - obj_now) < tol * max(1.0, np.abs(prev_obj)):
            if verbose:
                print("Converged by objective change.")
            break
        prev_obj = obj_now

        # 4) minimize objective w.r.t missing entries
        # pack current missing entries into vector z
        missing_idx = np.where(~mask_obs)
        num_missing = len(missing_idx[0])
        if num_missing == 0:
            if verbose:
                print("No missing entries.")
            break

        z0 = X_est[missing_idx].copy()  # initial guess

        # objective function for optimizer: given z vector, compute L(K(z), C(z))
        def obj_z(z):
            # fill X_est candidate
            Xc = X_est.copy()
            Xc[missing_idx] = z
            # compute K on Xc
            Kcand = compute_pairwise_kernel(Xc, mask_obs, kernel=kernel, kernel_params=kernel_params)
            Kc_c = center_kernel(Kcand)
            eigvals_c, eigvecs_c = eigh(Kc_c)
            idxc = np.argsort(eigvals_c)[::-1]
            eigvals_c = eigvals_c[idxc]
            eigvecs_c = eigvecs_c[:, idxc]
            # choose same q and sigma2 recomputed
            if n > q:
                sigma2_c = np.maximum(np.mean(eigvals_c[q:]), 1e-8)
            else:
                sigma2_c = 1e-8
            Vq_c = eigvals_c[:q]
            Uq_c = eigvecs_c[:, :q]
            Lambda_sq_c = np.maximum(Vq_c - sigma2_c, 0.0)
            Cc = (Uq_c * Lambda_sq_c) @ Uq_c.T + sigma2_c * np.eye(n)
            val = cross_entropy_obj(Kc_c, Cc)
            return val

        # bounds can be optionally provided (e.g., ratings between 1 and 5)
        # here we put no bounds; if you have rating scale, you should set bounds accordingly
        res = minimize(obj_z, z0, method='L-BFGS-B', options={'maxiter': max_inner_optit, 'disp': False})
        if not res.success and verbose:
            print("Warning: inner optimization did not fully converge:", res.message)

        # update X_est with optimized missing entries
        X_est[missing_idx] = res.x

        # optionally check objective after update (already in outer loop next iter)

    info = {'q': q, 'sigma2': sigma2, 'obj': prev_obj}
    return X_est, info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    

    # Any method you want
    X = table.copy()
    table_est, _ = missing_kpca_sanguinetti(X)
    table_completed = table.copy()
    table_completed[np.isnan(table)] = table_est[np.isnan(table)]


    # Save the completed table 
    np.save("output.npy", table_completed) ## DO NOT CHANGE THIS LINE