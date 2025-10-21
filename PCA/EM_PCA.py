
import numpy as np
import os
from tqdm import tqdm, trange
import argparse


def indapca_iterative(X, n_components=None, max_iter=10, tol=1e-4, scale=True):
    X = np.array(X, dtype=float)
    n, p = X.shape
    W = (~np.isnan(X)).astype(float)

    # Initialisation
    means = np.nanmean(X, axis=0)
    X_filled = np.where(np.isnan(X), np.expand_dims(means, 0), X)


    if scale:
        stds = np.nanstd(X, axis=0)
        stds[stds == 0] = 1
        X_filled = (X_filled - means) / stds
    else:
        stds = np.ones_like(means)

    for iteration in trange(max_iter):
        # Centrage
        X_centered = X_filled - np.nanmean(X_filled, axis=0)

        # Covariance
        C = np.cov(X_centered, rowvar=False)

        # PCA
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        if n_components is None:
            n_components = p
        eigvecs_k = eigvecs[:, :n_components]
        scores = np.dot(X_centered, eigvecs_k)

        # Reconstruction
        X_recon = np.dot(scores, eigvecs_k.T)
        X_recon += np.nanmean(X_filled, axis=0)
        X_recon = np.clip(np.round(X_recon * 2) / 2, 0, 5)

        '''alpha = iteration / max_iter
        X_recon = (1 - alpha) * X_recon + alpha * np.clip(np.round(X_recon * 2) / 2, 0, 5)
        alpha = 1 / (1 + np.exp(-k * (iteration / (max_iter - 1) - 0.5)))
        X_recon = (1 - alpha) * X_recon + alpha * np.clip(np.round(X_recon * 2) / 2, 0, 5)'''

        # Mise à jour des valeurs manquantes
        prev = X_filled.copy()
        X_filled[W == 0] = X_recon[W == 0]


    # Revenir à l’échelle originale
    X_completed = X_filled * stds + means if scale else X_filled + means

    return X_completed, eigvals[:n_components], eigvecs_k, scores

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
    '''names_genre = np.load("namesngenre.npy", allow_pickle=True)
    all_genres = set()
    for _, genres in names_genre:
        for g in genres.split('|'):
            all_genres.add(g)

    k = len(all_genres)'''
    k = 8 # obtained with opti_PCA.py

    user_means = np.nanmean(table, axis=1, keepdims=True)
    table_centered = table - user_means
    table_est, eigvals, eigvecs_k, scores = indapca_iterative(table_centered, n_components=k,scale=True)
    table_est = table_est + user_means
    table_completed = table.copy()
    table_completed[np.isnan(table)] = table_est[np.isnan(table)]
    table_completed = np.clip(np.round(table_est * 2) / 2, 0, 5)


    # Save the completed table 
    np.save("output.npy", table_completed) ## DO NOT CHANGE THIS LINE