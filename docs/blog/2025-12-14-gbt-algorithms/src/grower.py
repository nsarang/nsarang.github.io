from dataclasses import dataclass
from queue import PriorityQueue
from typing import List, Tuple, Union

import numpy as np

from .data import FeatureInfo
from .tree import Node, Tree
from .utils import groupby_sum_2d


@dataclass
class SplitCandidate:
    """
    ### SplitCandidate

    A class to store information about a potential node split during tree growth.

    **Attributes**
    - `gain` : float
        - The gain achieved by this split.
    - `feature` : str
        - The feature on which to split.
    - `feature_index` : int
        - The index of the feature in the input array.
    - `split_point` : Union[float, Tuple]
        - For numerical features, the threshold value. For categorical features, a tuple of
          two arrays representing the left and right category groups.
    - `nulls_to_left` : bool
        - Whether null values go to the left child.
    - `gradient_sum` : np.ndarray
        - The sum of gradients for samples in the node.
    - `hessian_sum` : np.ndarray
        - The sum of hessians for samples in the node.
    - `left_grad_sum` : np.ndarray
        - The sum of gradients for samples going to the left child.
    - `left_hess_sum` : np.ndarray
        - The sum of hessians for samples going to the left child.
    """

    gain: float
    feature: str = None
    feature_index: int = None
    split_point: Union[float, Tuple] = None
    nulls_to_left: bool = None
    gradient_sum: np.ndarray = None
    hessian_sum: np.ndarray = None
    left_grad_sum: np.ndarray = None
    left_hess_sum: np.ndarray = None

    @property
    def right_grad_sum(self) -> np.ndarray:
        return self.gradient_sum - self.left_grad_sum

    @property
    def right_hess_sum(self) -> np.ndarray:
        return self.hessian_sum - self.left_hess_sum


@dataclass
class TreeGrower:
    """
    ### TreeGrower

    A class to grow decision trees. The core logic for finding the best splits and constructing
    the tree structure is implemented here.

    **Parameters**
    - `distribution` : Objective
        - The objective function defining the loss and gradient/hessian computations.
    - `learning_rate` : float
        - The learning rate for boosting.
    - `max_leaves` : int
        - The maximum number of leaves in the tree.
    - `max_depth` : int
        - The maximum depth of the tree.
    - `min_samples_split` : int
        - The minimum number of samples required to split a node.
    - `min_gain_to_split` : float
        - The minimum gain required to perform a split.
    - `l1_regularization` : float
        - The L1 regularization term for leaf value calculation.
    - `l2_regularization` : float
        - The L2 regularization term for leaf value calculation.
    - `min_weight_leaf` : float
        - The minimum sum of instance weights required in a leaf node.
    - `feature_fraction` : float
        - The fraction of features to consider when looking for the best split.
    - `samples_fraction` : float
        - The fraction of samples to consider when growing the tree.
    - `gamma` : float
        - The minimum loss reduction required to make a split.
    - `goss_top_rate` : float
        - The top rate for Gradient-based One-Side Sampling (GOSS).
    - `goss_bottom_rate` : float
        - The bottom rate for GOSS.
    - `goss_rescale_bottom` : bool
        - Whether to rescale the bottom samples in GOSS.
    - `objective_weight` : List[float]
        - Weights for multi-output objectives.
    - `n_workers` : int
        - The number of parallel workers to use for finding splits.
    - `seed` : int
        - Random seed for reproducibility.
    """

    distribution: any
    learning_rate: float
    max_leaves: int
    max_depth: int = 100
    min_samples_split: int = 2
    min_gain_to_split: float = 0.0
    l1_regularization: float = 0.0
    l2_regularization: float = 0.0
    min_weight_leaf: float = float("-inf")
    feature_fraction: float = 1.0
    samples_fraction: float = 1.0
    gamma: float = 0.0
    goss_top_rate: float = None
    goss_bottom_rate: float = None
    goss_rescale_bottom: bool = False
    objective_weight: List[float] = None
    n_workers: int = None
    seed: int = None

    def __post_init__(self):
        """Initialize the random state for rng operations"""
        self.random = np.random.RandomState(self.seed)

    def grow(
        self,
        inputs: np.ndarray,
        features_info: List["FeatureInfo"],
        targets: np.ndarray,
        sample_weight: np.ndarray,
        current_predictions: np.ndarray,
        **kwargs,
    ) -> Tree:
        """
        Train a decision tree using the provided dataset.

        **Parameters**
        - `inputs` : np.ndarray
            - The input feature array.
        - `features_info` : List[FeatureInfo]
            - List of FeatureInfo objects describing each feature.
        - `targets` : np.ndarray
            - The target values.
        - `sample_weight` : np.ndarray
            - Sample weights for each instance.
        - `current_predictions` : np.ndarray
            - The current predictions from the ensemble. The goal of adding a new tree is to
              correct these predictions.
        """
        # Subsample features and samples if required
        inputs, features_info, targets, sample_weight, current_predictions = self.subsample_dataset(
            inputs,
            features_info,
            targets,
            sample_weight,
            current_predictions,
        )
        # Default sample weights to 1 if not provided
        if sample_weight is None:
            sample_weight = np.ones(len(inputs), dtype=np.float64)

        # Compute gradients and hessians for the samples in the node
        gradient, hessian = self.distribution.gradient_hessian(
            targets,
            current_predictions,
            sample_weight,
        )
        hessian = np.maximum(hessian, 1e-6)

        # GOSS Sampling
        if self.goss_top_rate is not None and self.goss_bottom_rate is not None:
            (
                gradient,
                hessian,
                inputs,
                targets,
                sample_weight,
            ) = self.apply_goss_sampling(gradient, hessian, inputs, targets, sample_weight)

        # Initialize the tree with a root node
        root = Node(
            depth=0,
            l2_regularization=self.l2_regularization,
            sample_indices=np.arange(len(inputs)),
        )
        tree = Tree(root=root, **kwargs)

        # Priority queue to store split candidates. Higher gain splits have higher priority.
        candidates = PriorityQueue()

        # Start growing the tree
        leaves_to_process = [root]
        n_processed = 0
        for _ in range(self.max_leaves):
            for node in leaves_to_process:
                # Check if node can be split
                if (
                    node.depth >= self.max_depth
                    or len(node.sample_indices) < self.min_samples_split
                ):
                    continue

                # Find the best split for the given node
                split_candidate = self.find_best_split(
                    inputs[node.sample_indices],
                    features_info,
                    gradient[node.sample_indices],
                    hessian[node.sample_indices],
                    sample_weight[node.sample_indices],
                )
                # Add the split proposal to the queue if it satisfies min gain
                if (split_candidate is not None) and (
                    split_candidate.gain > self.min_gain_to_split
                ):
                    # The queue stores tuples of the form (-gain, node_index, node, split_candidate) since
                    # PriorityQueue in Python sorts in ascending order.
                    candidates.put(
                        (
                            -split_candidate.gain,
                            n_processed,  # tie-breaker
                            node,
                            split_candidate,
                        )
                    )
                n_processed += 1
            leaves_to_process = []
            if candidates.empty():
                break

            # Get the global best split candidate
            _, _, node, split_info = candidates.get()
            self.split_node(node, inputs, features_info, split_info)

            # Add children to the list for processing in the next iteration
            leaves_to_process.extend(node.children.values())

        if len(tree) == 0:
            print("No splits were made; returning None")
            return None
        return tree

    def find_best_split(
        self, inputs, features_info, gradient, hessian, sample_weight
    ) -> SplitCandidate:
        """
        Find the best split for the given node. This is a helper method that can evaluate each
        feature in parallel and aggregate the results.
        """
        tasks = [
            (
                info,
                inputs[:, index],
                gradient,
                hessian,
                sample_weight,
            )
            for index, info in enumerate(features_info)
        ]

        # Parallelize feature processing if n_workers > 1. Otherwise, process sequentially.
        if self.n_workers is not None and self.n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                feature_results = list(executor.map(lambda t: self._process_feature(*t), tasks))
        else:
            feature_results = [self._process_feature(*task) for task in tasks]

        # Return the best result
        return max(
            (r for r in feature_results if r is not None), key=lambda r: r.gain, default=None
        )

    def _process_feature(
        self,
        info: "FeatureInfo",
        values: np.ndarray,
        gradient: np.ndarray,
        hessian: np.ndarray,
        sample_weight: np.ndarray,
    ):
        """
        Find the best split for a given feature. This part is the backbone of the optimization.
        """
        # Handle categorical and numerical features differently
        if info.type == "categorical":
            # Calculate group-wise sums of gradients and hessians for each category
            cats, cat_indices = np.unique(values, return_inverse=True)
            groups = cats
            grad_group, hess_group, weight_group = groupby_sum_2d(
                n_groups=len(groups),
                group_indices=cat_indices,
                gradients=gradient,
                hessians=hessian,
                sample_weight=sample_weight,
            )
            # Sort categories by gain to convert categorical split into a
            # ordered split problem: categories[:k] vs categories[k:].
            fisher_order = self.calculate_gain(grad_group, hess_group)
            sort_indices = np.argsort(fisher_order)
            # Reorder arrays based on the calculated order
            groups, grad_group, hess_group, weight_group = (
                groups[sort_indices],
                grad_group[sort_indices],
                hess_group[sort_indices],
                weight_group[sort_indices],
            )
        else:
            # For numerical features, bin the values first and then compute group-wise sums
            bins = info.bins
            bin_indices = np.digitize(values, bins, right=True)
            bin_indices[np.isnan(values)] = len(bins)
            groups = np.concatenate([bins, [np.nan]])

            grad_group, hess_group, weight_group = groupby_sum_2d(
                n_groups=len(groups),
                group_indices=bin_indices,
                gradients=gradient,
                hessians=hessian,
                sample_weight=sample_weight,
            )

        # Separate out the null group
        null_group = np.isnan(groups)
        grad_null_sum = grad_group[null_group].sum(axis=0)
        hess_null_sum = hess_group[null_group].sum(axis=0)
        weight_null_sum = weight_group[null_group].sum(axis=0)
        # Calculate cumulative sums excluding the null group
        groups = groups[~null_group]
        grad_group_csum = np.cumsum(grad_group[~null_group], axis=0)
        hess_group_csum = np.cumsum(hess_group[~null_group], axis=0)
        weight_group_csum = np.cumsum(weight_group[~null_group], axis=0)

        if len(groups) <= 1:
            return None

        # Node total sums
        node_grad_sum = grad_group_csum[-1] + grad_null_sum
        node_hess_sum = hess_group_csum[-1] + hess_null_sum
        node_weight_sum = weight_group_csum[-1] + weight_null_sum

        # Calculate split gains for every possible split point and null direction
        grad_group_csum = grad_group_csum[:-1]
        hess_group_csum = hess_group_csum[:-1]
        weight_group_csum = weight_group_csum[:-1]

        parent_gain = self.calculate_gain(node_grad_sum, node_hess_sum)
        gains_with_nulls_left = (
            self.calculate_gain(
                grad_group_csum + grad_null_sum,
                hess_group_csum + hess_null_sum,
            )
            + self.calculate_gain(
                node_grad_sum - (grad_group_csum + grad_null_sum),
                node_hess_sum - (hess_group_csum + hess_null_sum),
            )
            - parent_gain
        )
        gains_with_nulls_right = (
            self.calculate_gain(
                grad_group_csum,
                hess_group_csum,
            )
            + self.calculate_gain(
                node_grad_sum - grad_group_csum,
                node_hess_sum - hess_group_csum,
            )
            - parent_gain
        )

        # Determine the best split for this feature
        split_gain_combined = np.concatenate([gains_with_nulls_left, gains_with_nulls_right])

        # Enforce minimum weight per leaf constraint
        valid_null_left = (weight_group_csum + weight_null_sum >= self.min_weight_leaf) & (
            node_weight_sum - (weight_group_csum + weight_null_sum) >= self.min_weight_leaf
        )
        valid_null_right = (weight_group_csum >= self.min_weight_leaf) & (
            node_weight_sum - weight_group_csum >= self.min_weight_leaf
        )
        split_gain_combined[~np.concatenate([valid_null_left, valid_null_right])] = -np.inf

        # Find the overall best split and store the details in a SplitCandidate object
        best_gain_idx = np.argmax(split_gain_combined)
        feature_gain = split_gain_combined[best_gain_idx]

        nulls_to_left = True if best_gain_idx < (len(groups) - 1) else False
        split_idx = best_gain_idx % (len(groups) - 1)

        left_grad_sum = grad_group_csum[split_idx]
        left_hess_sum = hess_group_csum[split_idx]
        if nulls_to_left:
            left_grad_sum += grad_null_sum
            left_hess_sum += hess_null_sum

        if info.type == "categorical":
            split_point = (
                groups[: split_idx + 1],
                groups[split_idx + 1 :],
            )
        else:
            split_point = groups[split_idx]

        return SplitCandidate(
            gain=feature_gain,
            feature=info.name,
            feature_index=info.index,
            split_point=split_point,
            nulls_to_left=nulls_to_left,
            gradient_sum=node_grad_sum,
            hessian_sum=node_hess_sum,
            left_grad_sum=left_grad_sum,
            left_hess_sum=left_hess_sum,
        )

    def split_node(
        self,
        node: Node,
        inputs: np.ndarray,
        features_info: List["FeatureInfo"],
        split_info: SplitCandidate,
    ):
        """
        Split the given node using the provided split information. Creates left and right child nodes and
        updates the parent node accordingly.
        """
        # Update the split criteria for the parent node
        node.update(
            feature=split_info.feature,
            feature_index=split_info.feature_index,
            split_point=split_info.split_point,
            nulls_to_left=split_info.nulls_to_left,
            gradient_sum=split_info.gradient_sum,
            hessian_sum=split_info.hessian_sum,
        )
        # Determine the feature index in the subsampled dataset
        feature_index = next(i for i, f in enumerate(features_info) if f.name == split_info.feature)
        # Find out which samples go to the left and right child nodes. This information is used later
        # for further splits.
        mask = node.criterion(inputs[node.sample_indices, feature_index], input_type="feature")
        left_child = Node(
            parent=node,
            depth=node.depth + 1,
            l2_regularization=node.l2_regularization,
            sample_indices=node.sample_indices[mask],
            gradient_sum=split_info.left_grad_sum,
            hessian_sum=split_info.left_hess_sum,
        )
        right_child = Node(
            parent=node,
            depth=node.depth + 1,
            l2_regularization=node.l2_regularization,
            sample_indices=node.sample_indices[~mask],
            gradient_sum=split_info.right_grad_sum,
            hessian_sum=split_info.right_hess_sum,
        )
        node.children["left"] = left_child
        node.children["right"] = right_child

    def calculate_gain(self, gradients, hessians):
        r"""
        Calculate the per-node gain based on gradients and hessians. The gain is calculated as:

        $$\text{gain} = \frac{(T(G))^2}{H + \lambda}$$

        where:
        - $T(G) = \text{sign}(G) \cdot \max(0, |G| - \alpha)$ is the soft-thresholding operator
        - $G$ is the sum of gradients
        - $H$ is the sum of hessians
        - $\alpha$ is the L1 regularization parameter (induces sparsity)
        - $\lambda$ is the L2 regularization parameter (prevents overfitting)

        **Parameters**
        - `gradients` : np.ndarray
            - The sum of gradients for the node(s) with shape `(n_nodes, n_outputs)`.
        - `hessians` : np.ndarray
            - The sum of hessians for the node(s) with shape `(n_nodes, n_outputs)`.
        **Returns**
        - `gain_outputs` : np.ndarray
            - The calculated gain for each node, shape `(n_nodes,)`.
        """
        if self.l1_regularization > 0:
            gradients = np.sign(gradients) * np.maximum(
                0, np.abs(gradients) - self.l1_regularization
            )
        gain_outputs = (gradients**2) / (hessians + self.l2_regularization + 1e-16)
        if self.objective_weight is not None:
            gain_outputs *= self.objective_weight / np.sum(self.objective_weight)
        return gain_outputs.sum(axis=-1)

    def subsample_dataset(
        self,
        inputs: np.ndarray,
        features_info: List["FeatureInfo"],
        targets: np.ndarray,
        sample_weight: np.ndarray,
        current_predictions: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """
        Subsample features and samples based on the specified fractions. This is useful for
        creating a variety of trees in the ensemble and preventing overfitting.
        """
        # Take a random subset of features if specified
        if self.feature_fraction < 1:
            n_features = inputs.shape[1]
            n_subsample_features = int(n_features * self.feature_fraction)
            feature_indices = self.random.choice(
                n_features, size=n_subsample_features, replace=False
            )
            inputs = inputs[:, feature_indices]
            features_info = [features_info[i] for i in feature_indices]

        # Subsample data points if specified
        if self.samples_fraction < 1:
            n_samples = inputs.shape[0]
            subsample_size = int(n_samples * self.samples_fraction)
            subsample_indices = self.random.choice(n_samples, size=subsample_size, replace=False)
            inputs = inputs[subsample_indices]
            targets = targets[subsample_indices]
            if sample_weight is not None:
                sample_weight = sample_weight[subsample_indices]
            current_predictions = current_predictions[subsample_indices]

        return (
            inputs,
            features_info,
            targets,
            sample_weight,
            current_predictions,
        )

    def apply_goss_sampling(self, gradient, hessian, inputs, targets, sample_weight):
        r"""
        Apply Gradient-based One-Side Sampling. The idea is to retain instances with large gradients
        while randomly sampling from instances with small gradients. This helps focus the model
        on hard-to-predict instances while still maintaining a representative sample of the overall data.
        
        **Input:** $g$ (gradients), $h$ (hessians), $X$ (inputs), $y$ (targets), 
        $w$ (sample weights), $a$ (top rate), $b$ (bottom rate), $\text{rescale}$ (rescale-bottom flag)

        $$
        \begin{aligned}
        & n = |g|,\ n_{\text{top}} = \lfloor a n \rfloor,\ n_{\text{bot}} = \lfloor b n \rfloor \\
        & r_i = \lVert g_i \rVert_1,\ \pi = \operatorname{argsort}(r;\ \text{descending}) \\
        & T = \{\pi_1,\dots,\pi_{n_{\text{top}}}\},\ C = \{\pi_{n_{\text{top}}+1},\dots,\pi_n\} \\
        & B \sim \text{UniformSubset}(C,\ n_{\text{bot}}),\ S = T \cup B \\
        & \text{if rescale:} \\
        &\quad s = \frac{\sum_{i \notin T} w_i}{\sum_{i \in B} w_i},\
        \forall i \in B:\ (g_i,h_i,w_i) \leftarrow s (g_i,h_i,w_i)
        \end{aligned}
        $$

        **Output:** $ \{(x_i,y_i,g_i,h_i,w_i)\}_{i \in S} $
        """
        n_samples = len(gradient)
        n_top = int(n_samples * self.goss_top_rate)
        n_bottom = int(n_samples * self.goss_bottom_rate)

        # Sort the samples by absolute gradient values in descending order
        abs_grad = np.abs(gradient).sum(axis=1)
        sorted_indices = np.argsort(-abs_grad)
        # Select top and bottom samples
        top_indices = sorted_indices[:n_top]
        bottom_indices = self.random.choice(sorted_indices[n_top:], n_bottom, replace=False)
        selected_indices = np.concatenate([top_indices, bottom_indices])

        # I added an argument to make it optional whether to rescale the bottom instances or not. I found
        # that in some cases, not rescaling gives better results.
        scale = 1.0
        if self.goss_rescale_bottom:
            scale = (np.sum(sample_weight) - np.sum(sample_weight[top_indices])) / np.sum(
                sample_weight[bottom_indices]
            )

        gradient = gradient[selected_indices]
        hessian = hessian[selected_indices]
        sample_weight = sample_weight[selected_indices]

        gradient[n_top:] *= scale
        hessian[n_top:] *= scale
        sample_weight[n_top:] *= scale

        return (
            gradient,
            hessian,
            inputs[selected_indices],
            targets[selected_indices],
            sample_weight,
        )
