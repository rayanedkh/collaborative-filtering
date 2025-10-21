import numba as nb
import numpy as np
import pandas as pd

from .kernels import (
    kernel_linear,
    kernel_sigmoid,
    kernel_rbf,
    kernel_linear_sgd_update,
    kernel_sigmoid_sgd_update,
    kernel_rbf_sgd_update,
)

from .recommender_base import RecommenderBase

from typing import Tuple, Union


class KernelMF(RecommenderBase):
    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 100,
        kernel: str = "linear",
        gamma: Union[str, float] = "auto",
        reg: float = 1,
        lr: float = 0.01,
        init_mean: float = 0,
        init_sd: float = 0.1,
        min_rating: int = 0,
        max_rating: int = 5,
        verbose: int = 1,
    ):
        if kernel not in ("linear", "sigmoid", "rbf"):
            raise ValueError("Kernel must be one of linear, sigmoid, or rbf")

        super().__init__(min_rating=min_rating, max_rating=max_rating, verbose=verbose)

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.kernel = kernel
        self.gamma = 1 / n_factors if gamma == "auto" else gamma
        self.reg = reg
        self.lr = lr
        self.init_mean = init_mean
        self.init_sd = init_sd
        return

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = self._preprocess_data(X=X, y=y, type="fit")
        self.global_mean = X["rating"].mean()

        # Initialize vector bias parameters
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)

        # Initialize latent factor parameters of matrices P and Q
        self.user_features = np.random.normal(
            self.init_mean, self.init_sd, (self.n_users, self.n_factors)
        )
        self.item_features = np.random.normal(
            self.init_mean, self.init_sd, (self.n_items, self.n_factors)
        )

        # Perform stochastic gradient descent
        (
            self.user_features,
            self.item_features,
            self.user_biases,
            self.item_biases,
            self.train_rmse,
        ) = _sgd(
            X=X.to_numpy(dtype=np.float64),
            global_mean=self.global_mean,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            user_features=self.user_features,
            item_features=self.item_features,
            n_epochs=self.n_epochs,
            kernel=self.kernel,
            gamma=self.gamma,
            lr=self.lr,
            reg=self.reg,
            min_rating=self.min_rating,
            max_rating=self.max_rating,
            verbose=self.verbose,
        )
        self.is_fitted_ = True
        self.train_rmse_ = self.train_rmse[-1] if len(self.train_rmse) > 0 else None
        return self


    def predict(self, X: pd.DataFrame, bound_ratings: bool = True) -> list:
        # If empty return empty list
        if X.shape[0] == 0:
            return []

        X = self._preprocess_data(X=X, type="predict")

        # Get predictions
        predictions, predictions_possible = _predict(
            X=X.to_numpy(dtype=np.float64),
            global_mean=self.global_mean,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            user_features=self.user_features,
            item_features=self.item_features,
            min_rating=self.min_rating,
            max_rating=self.max_rating,
            kernel=self.kernel,
            gamma=self.gamma,
            bound_ratings=bound_ratings,
        )

        self.predictions_possible = predictions_possible
        return predictions
    
    

@nb.njit()
def _calculate_rmse(
    X: np.ndarray,
    global_mean: float,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_features: np.ndarray,
    item_features: np.ndarray,
    min_rating: float,
    max_rating: float,
    kernel: str,
    gamma: float,
):
    n_ratings = X.shape[0]
    errors = np.zeros(n_ratings)

    # Iterate through all user-item ratings and calculate error
    for i in range(n_ratings):
        user_id, item_id, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        user_bias = user_biases[user_id]
        item_bias = item_biases[item_id]
        user_feature_vec = user_features[user_id, :]
        item_feature_vec = item_features[item_id, :]

        # Calculate predicted rating for given kernel
        if kernel == "linear":
            rating_pred = kernel_linear(
                global_mean=global_mean,
                user_bias=user_bias,
                item_bias=item_bias,
                user_feature_vec=user_feature_vec,
                item_feature_vec=item_feature_vec,
            )

        elif kernel == "sigmoid":
            rating_pred = kernel_sigmoid(
                global_mean=global_mean,
                user_bias=user_bias,
                item_bias=item_bias,
                user_feature_vec=user_feature_vec,
                item_feature_vec=item_feature_vec,
                a=min_rating,
                c=max_rating - min_rating,
            )

        elif kernel == "rbf":
            rating_pred = kernel_rbf(
                user_feature_vec=user_feature_vec,
                item_feature_vec=item_feature_vec,
                gamma=gamma,
                a=min_rating,
                c=max_rating - min_rating,
            )

        # Calculate error
        errors[i] = rating - rating_pred

    rmse = np.sqrt(np.square(errors).mean())
    return rmse


@nb.njit()
def _sgd(
    X: np.ndarray,
    global_mean: float,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_features: np.ndarray,
    item_features: np.ndarray,
    n_epochs: int,
    kernel: str,
    gamma: float,
    lr: float,
    reg: float,
    min_rating: float,
    max_rating: float,
    verbose: int,
    update_user_params: bool = True,
    update_item_params: bool = True,
):
    train_rmse = []

    for epoch in range(n_epochs):
        # Shuffle dataset before each epoch
        np.random.shuffle(X)

        # Iterate through all user-item ratings
        for i in range(X.shape[0]):
            user_id, item_id, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]

            if kernel == "linear":
                kernel_linear_sgd_update(
                    user_id=user_id,
                    item_id=item_id,
                    rating=rating,
                    global_mean=global_mean,
                    user_biases=user_biases,
                    item_biases=item_biases,
                    user_features=user_features,
                    item_features=item_features,
                    lr=lr,
                    reg=reg,
                    update_user_params=update_user_params,
                    update_item_params=update_item_params,
                )

            elif kernel == "sigmoid":
                kernel_sigmoid_sgd_update(
                    user_id=user_id,
                    item_id=item_id,
                    rating=rating,
                    global_mean=global_mean,
                    user_biases=user_biases,
                    item_biases=item_biases,
                    user_features=user_features,
                    item_features=item_features,
                    lr=lr,
                    reg=reg,
                    a=min_rating,
                    c=max_rating - min_rating,
                    update_user_params=update_user_params,
                    update_item_params=update_item_params,
                )

            elif kernel == "rbf":
                kernel_rbf_sgd_update(
                    user_id=user_id,
                    item_id=item_id,
                    rating=rating,
                    user_features=user_features,
                    item_features=item_features,
                    lr=lr,
                    reg=reg,
                    gamma=gamma,
                    a=min_rating,
                    c=max_rating - min_rating,
                    update_user_params=update_user_params,
                    update_item_params=update_item_params,
                )

        # Calculate error and print
        rmse = _calculate_rmse(
            X=X,
            global_mean=global_mean,
            user_biases=user_biases,
            item_biases=item_biases,
            user_features=user_features,
            item_features=item_features,
            min_rating=min_rating,
            max_rating=max_rating,
            kernel=kernel,
            gamma=gamma,
        )
        train_rmse.append(rmse)

        if verbose == 1:
            print("Epoch ", epoch + 1, "/", n_epochs, " -  train_rmse:", rmse)

    return user_features, item_features, user_biases, item_biases, train_rmse


@nb.njit()
def _predict(
    X: np.ndarray,
    global_mean: float,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_features: np.ndarray,
    item_features: np.ndarray,
    min_rating: int,
    max_rating: int,
    kernel: str,
    gamma: float,
    bound_ratings: bool,
):
    n_factors = user_features.shape[1]
    predictions = []
    predictions_possible = []

    for i in range(X.shape[0]):
        user_id, item_id = int(X[i, 0]), int(X[i, 1])
        user_known = user_id != -1
        item_known = item_id != -1

        # Default values if user or item are not known
        user_bias = user_biases[user_id] if user_known else 0
        item_bias = item_biases[item_id] if item_known else 0
        user_feature_vec = (
            user_features[user_id, :] if user_known else np.zeros(n_factors)
        )
        item_feature_vec = (
            item_features[item_id, :] if item_known else np.zeros(n_factors)
        )

        # Calculate predicted rating given kernel
        if kernel == "linear":
            rating_pred = kernel_linear(
                global_mean=global_mean,
                user_bias=user_bias,
                item_bias=item_bias,
                user_feature_vec=user_feature_vec,
                item_feature_vec=item_feature_vec,
            )

        elif kernel == "sigmoid":
            rating_pred = kernel_sigmoid(
                global_mean=global_mean,
                user_bias=user_bias,
                item_bias=item_bias,
                user_feature_vec=user_feature_vec,
                item_feature_vec=item_feature_vec,
                a=min_rating,
                c=max_rating - min_rating,
            )

        elif kernel == "rbf":
            rating_pred = kernel_rbf(
                user_feature_vec=user_feature_vec,
                item_feature_vec=item_feature_vec,
                gamma=gamma,
                a=min_rating,
                c=max_rating - min_rating,
            )

        # Bound ratings to min and max rating range
        if bound_ratings:
            if rating_pred > max_rating:
                rating_pred = max_rating
            elif rating_pred < min_rating:
                rating_pred = min_rating

        predictions.append(rating_pred)
        predictions_possible.append(user_known and item_known)

    return predictions, predictions_possible
