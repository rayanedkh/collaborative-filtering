
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

    # === Train mask ===
    mask_train = ~np.isnan(table)
    if (table == 0).any(): 
        mask_train &= (table != 0)
    u_train, i_train = np.where(mask_train)
    y_train = table[u_train, i_train].astype(np.float32)


    m, n = table.shape
    k = 32
    rng = np.random.default_rng(0)

    # === Hyperparamètres ===
    lr = 2e-4 
    lambda_U = 0.8
    lambda_I = 0.3
    lambda_b = 0.02
    epochs = 400

    # === Initialisation ===
    U = 0.1 * rng.standard_normal((m, k), dtype=np.float32)
    I = 0.1 * rng.standard_normal((n, k), dtype=np.float32)
    b_i = np.zeros(m)
    b_u = np.zeros(n)
    mu = np.nanmean(table)

    rmse_train_hist = []

    # === Entraînement SGD ===
    for epoch in trange(epochs, desc="SGD Training"):
        perm = rng.permutation(len(y_train))
        for uu, ii, yy in zip(u_train[perm], i_train[perm], y_train[perm]): 
            # prédiction
            pred = mu + b_i[uu] + b_u[ii] + U[uu].dot(I[ii])
            e = yy - pred

            # mises à jour
            b_i[uu] += lr * (e - lambda_b * b_i[uu])
            b_u[ii] += lr * (e - lambda_b * b_u[ii])
            U[uu] += lr * (e * I[ii] - lambda_U * U[uu])
            I[ii] += lr * (e * U[uu] - lambda_I * I[ii])

        # reconstruction complète
        R_hat = mu + b_i[:, None] + b_u[None, :] + U @ I.T

        # RMSE train
        rmse_train = np.sqrt(np.mean((y_train - R_hat[u_train, i_train]) ** 2))
        rmse_train_hist.append(rmse_train)

    table_completed = np.where(mask_train,table,np.clip(np.round(R_hat * 2) / 2, 0, 5))

    # Save the completed table 
    np.save("output.npy", table_completed) ## DO NOT CHANGE THIS LINE