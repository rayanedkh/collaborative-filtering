
import numpy as np
import os
from tqdm import tqdm, trange
import argparse


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
    # Matrix Factorization with ALS

    # Hyperparameters à optimiser 
    n_iters = 300      # nombre d'itérations ALS
    lambda_i = 0.1    # régularisation pour I (films)
    lambda_u = 0.1    # régularisation pour U (utilisateurs)

    names_genre = np.load("namesngenre.npy", allow_pickle=True)
    all_genres = set()
    for _, genres in names_genre:
        for g in genres.split('|'):
            all_genres.add(g)

    k = 80 #len(all_genres) # nombre de facteurs latents (nombre de genres différents)

    R = table.copy()
    n_items, n_users = R.shape

    mask = ~np.isnan(R)
    R[np.isnan(R)] = 0

    # Initialisation aléatoire (petite valeur) adapter le coeff
    rng = np.random.default_rng(0)
    I = 0.1*rng.standard_normal((n_items, k))
    U = 0.1*rng.standard_normal((n_users, k))

    eye_k = np.eye(k)

    for it in trange(n_iters, desc="ALS iterations"): #essayer d'aller jusqu'à convergence
        # Update I
        for i in range(n_items):
            users_i = np.where(mask[i, :])[0]
            if len(users_i) == 0:
                continue
            U_sub = U[users_i, :]
            r_i = R[i, users_i]
            A = U_sub.T @ U_sub + lambda_i * eye_k
            b = U_sub.T @ r_i
            I[i, :] = np.linalg.solve(A, b)

        # Update U
        for u in range(n_users):
            items_u = np.where(mask[:, u])[0]
            if len(items_u) == 0:
                continue
            I_sub = I[items_u, :]
            r_u = R[items_u, u]
            A = I_sub.T @ I_sub + lambda_u * eye_k
            b = I_sub.T @ r_u
            U[u, :] = np.linalg.solve(A, b)

    print("ALS training completed.")

    # Reconstruction
    R_pred = I @ U.T
    table = np.where(mask, table, R_pred)

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
