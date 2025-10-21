import numpy as np
import os
from tqdm import trange
import argparse
import matplotlib.pyplot as plt

def indapca_iterative(X, n_components=None, max_iter=50, tol=1e-4, scale=True):
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

    for iteration in trange(max_iter, leave=False):
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

        # Mise à jour des valeurs manquantes
        prev = X_filled.copy()
        X_filled[W == 0] = X_recon[W == 0]

        diff = np.nanmean((X_filled - prev)**2)
        if diff < tol:
            break

    # Revenir à l’échelle originale
    X_completed = X_filled * stds + means if scale else X_filled + means

    return X_completed, eigvals[:n_components], eigvecs_k, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table and plot reconstruction error for different k.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                      help="Name of the npy of the ratings table to complete")
    args = parser.parse_args()

    # Open Ratings table
    print('Ratings loading...')
    table = np.load(args.name)
    print('Ratings Loaded.')
    test_set = np.load("ratings_test.npy")  

    mask_test = ~np.isnan(test_set)

    

    # Préparation des données
    user_means = np.nanmean(table, axis=1, keepdims=True)
    table_centered = table - user_means

    # Liste pour stocker les erreurs
    ks = list(range(5, 101))
    errors = []

    print("Calcul des erreurs pour k = 5 à 100...")
    for k in trange(5,101):
        table_est, eigvals, eigvecs_k, scores = indapca_iterative(table_centered, n_components=k, scale=True)
        table_est = table_est + user_means

        diff = table_est - test_set
        rmse = np.sqrt(np.sum((diff[mask_test])**2) / np.sum(mask_test))
        errors.append(rmse)
    
    best_k = ks[np.argmin(errors)]
    best_rmse = np.min(errors)

    print(f"\n Meilleur k = {best_k} avec RMSE = {best_rmse:.4f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(ks, errors, marker='o')
    plt.title("Erreur de reconstruction en fonction de k")
    plt.xlabel("Nombre de composantes principales (k)")
    plt.ylabel("Erreur quadratique moyenne (MSE)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("error_vs_k.png")
    plt.show()

    print("Graphique enregistré sous 'error_vs_k.png'")