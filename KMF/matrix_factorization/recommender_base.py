import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Union


class RecommenderBase(BaseEstimator, RegressorMixin, metaclass=ABCMeta):
    """
    Abstract base class for all recommender models.
    All subclasses should implement the fit() and predict() methods
    """
    @abstractmethod
    def __init__(self, min_rating: float = 0, max_rating: float = 5, verbose: int = 0):
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.verbose = verbose
        return

    @property
    def known_users(self):
        return set(self.user_id_map.keys())

    @property
    def known_items(self):
        return set(self.item_id_map.keys())

    def contains_user(self, user_id: Any) -> bool:
        return user_id in self.known_users

    def contains_item(self, item_id: Any) -> bool:
        return item_id in self.known_items

    def _preprocess_data(
        self, X: pd.DataFrame, y: pd.Series = None, type: str = "fit"):
        X = X.loc[:, ["user_id", "item_id"]]

        if type != "predict":
            X["rating"] = y

        if type == "fit":
                        # Check for duplicate user-item ratings
            if X.duplicated(subset=["user_id", "item_id"]).sum() != 0:
                raise ValueError("Duplicate user-item ratings in matrix")

            # Shuffle rows
            X = X.sample(frac=1, replace=False)
            # Create mapping of user_id and item_id to assigned integer ids
            user_ids = X["user_id"].unique()
            item_ids = X["item_id"].unique()
            self.user_id_map = {user_id: i for (i, user_id) in enumerate(user_ids)}
            self.item_id_map = {item_id: i for (i, item_id) in enumerate(item_ids)}
            self.n_users = len(user_ids)
            self.n_items = len(item_ids)

        # Remap user id and item id to assigned integer ids
        X.loc[:, "user_id"] = X["user_id"].map(self.user_id_map)
        X.loc[:, "item_id"] = X["item_id"].map(self.item_id_map)

        if type == "predict":
            # Replace missing mappings with -1
            X.fillna(-1, inplace=True)
        return X

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    @abstractmethod
    def predict(self, X: pd.DataFrame, bound_ratings: bool = True) -> list:
        return []


