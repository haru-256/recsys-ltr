import pathlib
from typing import Literal, Any

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OrdinalEncoder


class PrmPublicDataset(Dataset):
    USER_CATEGORICAL_DIMS = 3
    ITEM_CATEGORICAL_DIMS = 5
    ITEM_DENSE_DIMS = 19
    INIT_RANK_LIST_LENGTH = 30

    def __init__(
        self,
        data_dir: pathlib.Path,
        type: Literal["train", "valid", "test"] = "train",
        nums: int = 100000,
    ) -> None:
        assert data_dir.exists()

        self.data_dir = data_dir
        self.nums = nums
        if type == "train":
            data_path = data_dir / "data.train.csv"
        elif type == "val":
            data_path = data_dir / "data.val.csv"
        elif type == "test":
            data_path = data_dir / "data.test.csv"
        else:
            raise ValueError(f"invalid type: {type}")
        assert data_path.exists()
        df = pd.read_csv(data_path, nrows=self.nums)
        columns = ["user_profile", "item_categorical", "item_dense", "label"]
        df[columns] = df[columns].applymap(lambda x: eval(x))

        self.labels = df["label"]
        self.uid = df["uid"]
        self.user_categorical = df["user_profile"]
        self.item_categorical = df["item_categorical"]
        self.item_dense = df["item_dense"]

        # calc cardinalities for each categorical features
        # TODO: This is not efficient. We should calculate this when build dataset.
        self.user_category_cardinalities = self.get_cardinalities("user_categorical")
        self.user_categorical_feature_names = sorted(self.user_category_cardinalities.keys())
        self.item_category_cardinalities = self.get_cardinalities("item_categorical")
        self.item_categorical_feature_names = sorted(self.item_category_cardinalities.keys())

    def __getitem__(self, index):
        uid = np.asarray(self.uid[index], dtype=np.int64)
        user_categorical = np.asarray(self.user_categorical[index], dtype=np.int64)
        item_categorical = np.asarray(self.item_categorical[index], dtype=np.int64)
        item_dense = np.asarray(self.item_dense[index], dtype=np.float32)
        label = np.asarray(self.labels[index], dtype=np.float32)

        # TODO: It is not efficient to check data for each call.
        assert uid.size == 1
        assert len(user_categorical) == self.USER_CATEGORICAL_DIMS
        assert np.all(
            item_categorical.shape == (self.INIT_RANK_LIST_LENGTH, self.ITEM_CATEGORICAL_DIMS)
        )
        assert np.all(item_dense.shape == (self.INIT_RANK_LIST_LENGTH, self.ITEM_DENSE_DIMS))
        assert len(label) == self.INIT_RANK_LIST_LENGTH

        return (uid, user_categorical, item_categorical, item_dense), label

    def __len__(self):
        return len(self.labels)

    def get_cardinalities(self, feature_name: str):
        cardinalities = {}
        if feature_name == "user_categorical":
            dims = self.USER_CATEGORICAL_DIMS
            features = self.user_categorical
        elif feature_name == "item_categorical":
            dims = self.ITEM_CATEGORICAL_DIMS
            features = self.item_categorical
        else:
            raise ValueError(f"invalid feature_name: {feature_name}")
        for dim in range(dims):
            cardinality = set()
            for feature in features:
                feature = np.asarray(feature).reshape(-1, dims)
                cardinality.update(set(feature[:, dim]))
            cardinalities[f"{feature_name}_{dim}"] = len(cardinality)

        return cardinalities


class CategoryEncoder:
    def __init__(self) -> None:
        self.unknown_value = 0
        self.missing_value = 1

    def fit(self, category_df: pd.DataFrame, fillna_value: str = "#nan") -> None:
        category2idx_dict_: dict[str, dict[Any, int]] = {}
        for category_column in category_df.columns:
            category2idx = {"#unknown": self.unknown_value, "#missing": self.missing_value}
            category2idx.update(
                {
                    h: i
                    # NOTE: start 2 because 0 is for unknown, 1 is for missing.
                    for i, h in enumerate(
                        sorted(list(set(category_df[category_column].fillna(fillna_value)))), 2
                    )
                }
            )
            category2idx_dict_[category_column] = category2idx
        self.category2idx_dict_ = category2idx_dict_
        self.category_cardinalities_ = {k: len(v) for k, v in self.category2idx_dict_.items()}

    def transform(self, category_df: pd.DataFrame) -> pd.DataFrame:
        category_features = {}
        for category_column in category_df.columns:
            category_feature = category_df[category_column].map(
                lambda x: self.category2idx_dict_[category_column].get(x, self.unknown_value)
                if not pd.isna(x)
                else self.missing_value
            )
            category_features[category_column] = category_feature

        encoded_category_df = pd.DataFrame(category_features)
        return encoded_category_df
