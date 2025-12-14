"""
Data preprocessing utilities module. Defines classes and functions for handling
feature information and preprocessing datasets.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import map_array


@dataclass
class FeatureInfo:
    """
    ### FeatureInfo

    A class to store information about a feature in the dataset.

    **Attributes**

    - `name` : str
        - The name of the feature/column.
    - `index` : int
        - The position of the feature in the input array.
    - `type` : str, optional
        - The type of the feature: `numerical`, `categorical`, or `category_as
        numeric`.
    - `bins` : np.ndarray, optional
        - The bin edges for numerical features or categorical features treated as
        numeric.
    - `categories` : np.ndarray, optional
        - The unique categories for categorical features.
    - `target_statistics` : dict, optional
        - A mapping from category to its target statistic for categorical features
        encoded as numeric.
    """

    name: str
    index: int
    type: str = None
    bins: np.ndarray = None
    categories: np.ndarray = None
    target_statistics: dict = None


class DataPreprocessor:
    def __init__(
        self,
        max_categories: int = 100,
        max_bins: int = 255,
        n_permutations: int = 10,
        prior_strength: float = 1.0,
        seed: int = None,
    ):
        """
        ### DataPreprocessor

        A data preprocessor for handling numerical and categorical features. It's used
        during both training and prediction.

        **Parameters**
        - `max_categories` : int, default=100
            - Maximum number of unique categories for a feature to be treated as
            categorical. Otherwise, it is encoded as numerical using label encoding.
        - `max_bins` : int, default=255
            - Number of bins to use for numerical features and categorical features
            treated as numeric.
        - `n_permutations` : int, default=10
            - Number of random permutations to use for ordered target statistics
            encoding of categorical features.
        - `seed` : int, optional
            - Random seed for reproducibility.
        """
        self.max_categories = max_categories
        self.max_bins = max_bins
        self.n_permutations = n_permutations
        self.prior_strength = prior_strength
        self.random = np.random.default_rng(seed)
        # A list of FeatureInfo objects, one for each feature in the dataset. To be populated
        # during the `fit` method.
        self.features_info_ = None

    def fit(
        self, X: pd.DataFrame, y: np.ndarray = None, sample_weight: np.ndarray = None
    ) -> "DataPreprocessor":
        """
        Create histograms of features in the dataset. Categorical and numerical
        features are handled differently.
        """
        # Create feature info for each column
        features_info = []
        # Identify categorical features
        categorical_features = X.select_dtypes(include=["category", "object", "string"]).columns
        # Process each feature individually
        for index, col in enumerate(X.columns):
            info = FeatureInfo(name=col, index=index)
            values = None
            category_as_numeric = False
            # Handle categorical features
            if col in categorical_features:
                # Use pandas' categorical type to get the unique categories.
                # NaN values are considered as a separate category (-1 code)
                category_col = X[col].astype("category")
                info.categories = category_col.cat.categories
                # If too many categories, use CatBoost-style target encoding
                if len(info.categories) > self.max_categories:
                    category_as_numeric = True
                    assert y is not None and sample_weight is not None
                    target_mean = np.average(y, weights=sample_weight)
                    # Convert categories to codes for grouping
                    codes = category_col.cat.codes.to_numpy()
                    # Aggregate $y * w$ and $w$ by category. Ignore -1 (NaN) codes.
                    valid = codes >= 0
                    w_sum = np.bincount(codes[valid], weights=(y * sample_weight)[valid])
                    w_total = np.bincount(codes[valid], weights=sample_weight[valid])
                    r"""
                    Compute TS for category $c$ as:
                    $$TS(c) = \frac{\sum_{i \in c} w_i y_i + \alpha \cdot \mu}{\sum_{i \in c} w_i + \alpha}$$
                    where $\mu$ is the global weighted mean of the target, and $\alpha$ is
                    the prior strength.
                    """
                    info.target_statistics = dict(
                        enumerate(
                            (w_sum + self.prior_strength * target_mean)
                            / (w_total + self.prior_strength)
                        )
                    )
                    # Map the codes to target statistics for binning in the next step
                    values = map_array(codes, info.target_statistics)
                # It's a categorical feature. No binning needed.
                else:
                    info.type = "categorical"
            # Bin numerical and categorical-as-numeric features
            if col not in categorical_features or category_as_numeric:
                if values is None:
                    values = X[col].to_numpy()
                # Compute quantile-based bins with no interpolation
                quantiles = np.nanquantile(
                    values,
                    np.linspace(0.0, 1.0, endpoint=True, num=self.max_bins + 1),
                    method="nearest",
                )
                info.bins = np.unique(quantiles).astype("float64")
                # Set feature type
                if col in categorical_features:
                    info.type = "category_as_numeric"
                else:
                    info.type = "numerical"
            features_info.append(info)

        self.features_info_ = features_info
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: np.ndarray = None,
        sample_weight: np.ndarray = None,
        mode="test",
    ) -> np.ndarray:
        """
        Transform the dataset into a numerical numpy array based on the fitted
        feature information.
        """
        array = X[[info.name for info in self.features_info_]].values.copy()
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.to_numpy()
            y = y.astype("float64")

        for i, info in enumerate(self.features_info_):
            if info.type == "categorical":
                # Transform categorical features to integer codes. The reason we use
                # integer code rather than keeping original string values is that the transformed
                # dataset is supposed to be a homogeneous float64 numpy array.
                array[:, i] = pd.Categorical(X[info.name], categories=info.categories).codes
            # CatBoost-style target statistics encoding. Train and test modes differ.
            elif info.type == "category_as_numeric":
                category_col = pd.Categorical(X[info.name], categories=info.categories)
                # If test mode, use the precomputed target statistics mapping
                if mode == "test":
                    array[:, i] = map_array(category_col.codes, info.target_statistics)
                # For training mode, compute ordered target statistics averaged over
                # multiple random permutations
                elif mode == "train":
                    assert y is not None and sample_weight is not None
                    target_mean = np.average(y, weights=sample_weight)
                    codes = category_col.codes.astype("int64")
                    encodings = np.empty((len(X), self.n_permutations))
                    n_cats = max(codes) + 1
                    # We impose a random order on the data and for every sample $i$,
                    # compute the target statistic using only samples that come before
                    # it in this random order. This is to avoid target leakage.
                    for p in range(self.n_permutations):
                        group_sum = np.zeros(n_cats)
                        group_weight = np.zeros(n_cats)
                        for idx in self.random.permutation(len(X)):
                            c = codes[idx]
                            # If the category is -1 (NaN), assign NaN and continue
                            if c < 0:
                                encodings[idx, p] = np.nan
                                continue
                            # Same formula as in fit(), but using only previous samples
                            encodings[idx, p] = (
                                group_sum[c] + self.prior_strength * target_mean
                            ) / (group_weight[c] + self.prior_strength)
                            group_sum[c] += y[idx] * sample_weight[idx]
                            group_weight[c] += sample_weight[idx]
                    # Average over all permutations
                    array[:, i] = np.mean(encodings, axis=1)

        array = array.astype("float64")
        if y is not None:
            return array, y
        return array
