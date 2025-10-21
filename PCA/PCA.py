
import numpy as np
import os
from tqdm import tqdm, trange
import argparse


def indapca(X, use_correlation=False, n_components=None):
    """
    Implémentation de l'algorithme InDaPCA selon Podani et al., 2021.
    """
    X = np.array(X, dtype=float)
    n, p = X.shape
    W = (~np.isnan(X)).astype(float)


    means = np.nansum(W * X, axis=0) / np.sum(W, axis=0)
    X_centered = X - means

    # Calcul pairwise de la matrice C
    C = np.zeros((p, p))
    for i in trange(p):
        xi = X_centered[:, i]
        wi = W[:, i]
        for h in range(p):
            xh = X_centered[:, h]
            wh = W[:, h]
            mask = wi * wh
            n_valid = np.sum(mask)
            if n_valid < 2:
                C[i, h] = 0.0
                continue

            xi_masked = xi[mask == 1]
            xh_masked = xh[mask == 1]
            
            mean_xi = np.mean(xi_masked)
            mean_xh = np.mean(xh_masked)
            cov = np.sum((xi_masked - mean_xi) * (xh_masked - mean_xh)) / (n_valid - 1)
            C[i, h] = cov

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    if n_components is None:
        n_components = p
    eigvecs_k = eigvecs[:, :n_components]
    eigvals_k = eigvals[:n_components]

    # Corrélations variables/composantes
    variable_correlations = eigvecs_k * np.sqrt(np.abs(eigvals_k))

    # Scores des observations
    scores = np.zeros((n, n_components))
    for j in range(n):
        valid = W[j, :] == 1
        scores[j, :] = np.dot(X_centered[j, valid], eigvecs_k[valid, :])

    return eigvals_k, eigvecs_k, scores, variable_correlations

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
    names_genre = np.load("namesngenre.npy", allow_pickle=True)
    all_genres = set()
    for _, genres in names_genre:
        for g in genres.split('|'):
            all_genres.add(g)

    k = len(all_genres)
    eigvals, eigvecs_k, scores, variable_correlations = indapca(table, use_correlation=True, n_components=k)

    # Reconstuction
    means = np.nanmean(table, axis=0)
    table_est_std = np.dot(scores, eigvecs_k.T)
    table_est = table_est_std + means
    table_completed = table.copy()
    table_completed[np.isnan(table)] = table_est[np.isnan(table)]
    table_completed = np.clip(np.round(table_est * 2) / 2, 0, 5)


    # Save the completed table 
    np.save("output.npy", table_completed) ## DO NOT CHANGE THIS LINE