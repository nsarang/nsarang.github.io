# **Required imports**
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# DataPreprocessor handles feature encoding and binning and train/test transformations
from .data import DataPreprocessor
from .distributions import Distribution

# TreeGrower handles the logic of growing individual trees
from .grower import TreeGrower
from .utils import trunc


class GradientBoostedModel:
    def __init__(
        self,
        distribution: Distribution,
        learning_rate: float = 0.1,
        n_trees: int = 100,
        n_leaves: int = 20,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_weight_leaf: float = 20,
        min_gain_to_split: float = 0.0,
        l2_regularization: float = 1.0,
        feature_fraction: float = 1.0,
        samples_fraction: float = 1.0,
        goss_top_rate: float = None,
        goss_bottom_rate: float = None,
        goss_rescale_bottom: bool = False,
        gamma: float = 0.0,
        path_smoothing_strength: float = 0.0,
        max_bins: int = 255,
        max_categories: int = 100,
        n_permutations: int = 10,
        prior_strength: float = 1.0,
        seed: int = None,
        verbose: bool = True,
    ):
        """
        ### Gradient Boosted Model (GBM)

        This is the user-facing class for training and making predictions.

        **Parameters**
        - `distribution` : Distribution
            - The loss distribution to optimize.
        - `learning_rate` : float, default=0.1
            - The learning rate (shrinkage factor) for each tree's contribution.
        - `n_trees` : int, default=100
            - The number of trees to grow in the ensemble.
        - `n_leaves` : int, default=20
            - The maximum number of leaves per tree.
        - `max_depth` : int, default=6
            - The maximum depth of each tree.
        - `min_samples_split` : int, default=2
            - The minimum number of samples required to split a node.
        - `min_weight_leaf` : float, default=20
            - The minimum sum of instance weights required in a leaf node, for a split to be considered.
        - `min_gain_to_split` : float, default=0.0
            - The minimum gain required to perform a split.
        - `l2_regularization` : float, default=1.0
            - L2 regularization term on leaf values.
        - `feature_fraction` : float, default=1.0
            - The fraction of features to subsample for each tree. Values between 0 and 1. This helps with regularization.
        - `samples_fraction` : float, default=1.0
            - The fraction of training samples to subsample for each tree. Values between 0 and 1.
        - `goss_top_rate` : float, optional
            - Whether to use GOSS sampling. If set, specifies the top fraction of samples (by absolute gradient) to keep.
        - `goss_bottom_rate` : float, optional
            - If using GOSS, specifies the bottom fraction of samples (by absolute gradient) to keep.
        - `goss_rescale_bottom` : bool, optional
            - Whether to rescale the weights of the bottom samples in GOSS to maintain the overall weight sum.
        - `gamma` : float, default=0.0
            - Regularization term for tree complexity (number of leaves).
        - `path_smoothing_strength` : float, default=0.0
            - Strength of path smoothing to apply to leaf values after tree is grown.
        - `max_bins` : int, default=255
            - The maximum number of bins to use for numerical features.
        - `max_categories` : int, default=100
            - The maximum number of unique categories for a feature to be treated as categorical. Otherwise, it is encoded as numerical using label encoding.
        - `n_permutations` : int, default=10
            - Number of random permutations to use for ordered target statistics encoding of categorical features (CatBoost-style).
        - `prior_strength` : float, default=1.0
            - The strength of the prior for target statistics encoding.
        - `seed` : int, optional
            - Random seed for reproducibility.
        - `verbose` : bool, default=True
            - Whether to print progress messages during training.
        """
        self.distribution = distribution
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.n_leaves = n_leaves
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_weight_leaf = min_weight_leaf
        self.min_gain_to_split = min_gain_to_split
        self.l2_regularization = l2_regularization
        self.feature_fraction = feature_fraction
        self.samples_fraction = samples_fraction
        self.goss_top_rate = goss_top_rate
        self.goss_bottom_rate = goss_bottom_rate
        self.goss_rescale_bottom = goss_rescale_bottom
        self.gamma = gamma
        self.path_smoothing_strength = path_smoothing_strength
        self.max_bins = max_bins
        self.max_categories = max_categories
        self.n_permutations = n_permutations
        self.prior_strength = prior_strength
        self.seed = seed
        self.verbose = verbose

        # Attributes to be set during fitting
        self.data_processor_ = None
        self.initial_params_ = None
        self.trees_ = []
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        Fit the Gradient Boosted Model to the data. This method should give you a high-level
        overview of the training process.

        - `X` : pd.DataFrame
            - The input features.
        - `y` : np.ndarray
            - The target values.
        - `sample_weight` : np.ndarray, optional
            - Sample weights for each instance.
        """
        if sample_weight is None:
            sample_weight = np.ones(len(X))

        # Preprocess data and compute feature info
        self.data_processor_ = DataPreprocessor(
            max_categories=self.max_categories,
            max_bins=self.max_bins,
            n_permutations=self.n_permutations,
            prior_strength=self.prior_strength,
            seed=self.seed,
        )
        self.data_processor_.fit(X, y, sample_weight)
        # Transform input data into numerical array
        inputs, targets = self.data_processor_.transform(X, y, sample_weight, mode="train")

        # Initialize tree grower. This handles the logic of growing individual trees.
        grower = TreeGrower(
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            min_weight_leaf=self.min_weight_leaf,
            max_leaves=self.n_leaves,
            max_depth=self.max_depth,
            goss_top_rate=self.goss_top_rate,
            goss_bottom_rate=self.goss_bottom_rate,
            goss_rescale_bottom=self.goss_rescale_bottom,
            feature_fraction=self.feature_fraction,
            samples_fraction=self.samples_fraction,
            min_samples_split=self.min_samples_split,
            min_gain_to_split=self.min_gain_to_split,
            gamma=self.gamma,
            distribution=self.distribution,
            seed=self.seed,
        )

        # Initialize the ensemble. Query the Distribution object for initial parameters.
        self.initial_params_ = self.distribution.init_params(y)
        predictions = np.full((len(y), len(self.initial_params_)), self.initial_params_)
        # Print initial loss
        if self.verbose:
            loss = -self.distribution.log_prob(y, predictions).sum()
            print(f"Initial loss: {trunc(loss)}")

        # Grow trees one by one
        for i in tqdm(range(self.n_trees)):
            tree = grower.grow(
                inputs=inputs,
                features_info=self.data_processor_.features_info_,
                targets=targets,
                sample_weight=sample_weight,
                current_predictions=predictions,
            )
            # If no valid tree could be grown, stop early.
            # Could happen due to lack of enough data or no tangible gain.
            if tree is None:
                break

            # Apply path smoothing if specified
            if self.path_smoothing_strength > 0.0:
                tree.apply_smoothing(self.path_smoothing_strength)

            # Append the new tree and update predictions
            self.trees_.append(tree)
            predictions += self.learning_rate * tree.predict(inputs)

            if self.verbose:
                loss = -self.distribution.log_prob(y, predictions).sum()
                print(f"Loss after tree {i + 1}: {trunc(loss)}")

        self.is_fitted_ = True
        return self

    def predict(self, X, return_type="predict"):
        """
        Make inferences using the fitted model.

        - `X` : pd.DataFrame
            - The input features.
        - return_type : str or None, default="predict"
            - Transformation to apply to raw predictions. Options depend on the distribution.
            It basically passes the raw predictions to the distribution's corresponding method.
            If None, returns raw predictions.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.data_processor_.transform(X)
        # Start with the initial parameters and add contributions from each tree
        raw_prediction = np.full((len(X), len(self.initial_params_)), self.initial_params_)
        for tree in self.trees_:
            raw_prediction += self.learning_rate * tree.predict(X)

        # Return the desired output type. Calls the corresponding implementation in the Distribution class.
        if return_type is not None:
            return getattr(self.distribution, return_type)(raw_prediction)
        return raw_prediction
