from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np


@dataclass
class Node:
    """
    A class representing a node in a decision tree.

    **Attributes**
    - `depth` : int
        - The depth of the node in the tree.
    - `sample_indices` : List[int]
        - The indices of training samples that reach this node. Used during training.
    - `parent` : Node, optional
        - The parent node. None for the root node.
    - `children` : Dict[str, Node]
        - The child nodes, typically with keys "left" and "right".
    - `gradient_sum` : np.ndarray
        - The sum of gradients for samples in this node.
    - `hessian_sum` : np.ndarray
        - The sum of hessians for samples in this node.
    - `l1_regularization` : float
        - The L1 regularization term.
    - `l2_regularization` : float
        - The L2 regularization term.
    - `feature` : str, optional
        - Feature name used for splitting at this node.
    - `feature_index` : int, optional
        - Index of the feature in the input array.
    - `split_point` : Union[float, Tuple], optional
        - The split point for numerical features or a tuple of categories for categorical features.
    - `nulls_to_left` : bool, optional
        - Whether null values go to the left child.
    - `smoothed_value` : np.ndarray, optional
        - The smoothed value for the node, used for prediction. Only set for leaf nodes.
    - `info` : Dict[str, Any]
        - Additional information about the node.
    """

    depth: int
    sample_indices: List[int] = field(default_factory=list)
    parent: "Node" = None
    children: Dict[str, "Node"] = field(default_factory=dict)
    gradient_sum: np.ndarray = None
    hessian_sum: np.ndarray = None
    l1_regularization: float = 0.0
    l2_regularization: float = 0.0
    feature: str = None
    feature_index: int = None
    split_point: Union[float, Tuple] = None
    nulls_to_left: bool = None
    smoothed_value: np.ndarray = None
    info: Dict[str, Any] = field(default_factory=dict)

    def value(self, smooth: bool = True) -> float:
        r"""
        Calculate the leaf value using the formula:
        $$ \text{Value} = - \frac{\text{sign}(G) \cdot \max(|G| - \lambda_{\text{L1}}, 0)}{H + \lambda_{\text{L2}}} $$
        """
        if smooth and self.smoothed_value is not None:
            return self.smoothed_value

        gradient = self.gradient_sum
        if self.l1_regularization:
            gradient = np.sign(gradient) * np.maximum(
                np.abs(gradient) - self.l1_regularization, 0.0
            )
        return -gradient / (self.hessian_sum + self.l2_regularization + 1e-16)

    @property
    def type(self) -> str:
        """
        Returns the type of the node: `InternalNode`, `Leaf`, or `VirtualNode`.
        VirtualNode is a special node used only during training to represent nodes that
        have not been instantiated yet.
        """
        if self.children:
            return "InternalNode"
        elif self.gradient_sum is not None and self.hessian_sum is not None:
            return "Leaf"
        else:
            return "VirtualNode"

    def criterion(self, array: np.ndarray, input_type: str = "all") -> np.ndarray:
        """
        Returns a boolean mask indicating which samples go to the left child.

        **Parameters**
        - `array` : np.ndarray
            - The input feature array.
        - `input_type` : str, default="all"
            - Specifies whether the input array contains all features ("all") or just the feature
            used for splitting ("feature").
        """
        if input_type == "all":
            array = array[:, self.feature_index]
        null_mask = np.isnan(array) & self.nulls_to_left
        if isinstance(self.split_point, tuple):
            return null_mask | np.isin(array, self.split_point[0])
        else:
            return null_mask | (array <= self.split_point)

    def __repr__(self):
        """String representation of the node. Used for visualization and debugging."""
        if self.children:
            return f"{self.type}(feature='{self.feature}', split_point={self.split_point}, samples={len(self.sample_indices)})"
        elif self.gradient_sum is not None and self.hessian_sum is not None:
            return f"{self.type}(value={self.value().round(decimals=4)}, samples={len(self.sample_indices)})"
        else:
            return f"{self.type}(samples={len(self.sample_indices)})"

    def update(self, **kwargs):
        """Helper method to update a dataclass's fields."""
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Node has no attribute '{key}'")
            setattr(self, key, value)


class Tree:
    def __init__(self, root: Node = None):
        """A structure representing a decision tree."""
        self.root = root

    def predict(self, x: np.ndarray, node: Node = None):
        """Make predictions for the input samples `x` starting from the given node (or root if None).
        We traverse the tree from the root to the leaves based on the split criteria at each node, and aggregate
        the leaf values to produce the final predictions.

        **Parameters**
        - `x` : np.ndarray
            - The input samples for which predictions are to be made, with shape (n_samples, n_features).
        - `node` : Node, optional
            - The current node in the tree from which to start predictions. If None, starts from the root node.

        **Returns**
        - `output` : np.ndarray
            - The leaf values for each input sample, with shape (n_samples, n_outputs).
        """
        node = node or self.root
        n_samples = len(x)
        # If the node is an internal node, split the samples and recurse
        if node.children:
            mask = node.criterion(x)
            n_output = len(node.children["left"].value())
            output = np.zeros((n_samples, n_output))
            output[mask] = self.predict(x[mask], node.children["left"])
            output[~mask] = self.predict(x[~mask], node.children["right"])
        # If in the leaf node, return the leaf value for all samples
        else:
            output = np.tile(node.value(), (n_samples, 1))
        return output

    def __len__(self):
        """Returns the number of leaves in the tree."""
        return len(self.leaves)

    def __iter__(self):
        """Iterator to traverse all nodes in the tree using depth-first search."""

        def dfs(node: Node):
            """
            Utility for traversal of the tree. Since it's a recursive generator, we need
            to use `yield` instead of `return` to yield nodes one by one. On a similar note,
            `yield from` is used for calling the generator recursively.
            It's a rarely used syntax but I find it a neat feature of Python :)
            """
            yield node
            for child in node.children.values():
                yield from dfs(child)

        if self.root:
            yield from dfs(self.root)

    @property
    def leaves(self) -> List[Node]:
        """Returns a list of all leaf nodes in the tree. We use the __iter__ method to traverse the tree."""
        return [node for node in self if node.type == "Leaf"]

    def apply_smoothing(self, beta: float):
        r"""
        Apply exponential path smoothing to all leaf values (in-place).

        Blends each leaf with ancestors using exponential decay: leaf gets weight 1,
        parent gets $\beta$, grandparent gets $\beta^2$, etc.

        $$v_{\text{smoothed}} = \frac{\sum_{i=0}^{d} \beta^{d-i} \cdot v_i}{\sum_{i=0}^{d} \beta^{d-i}}$$

        **Parameters**
        - `beta` : float in (0, 1], default=0.5
            - Decay factor. Lower values = stronger regularization toward shallow predictions.
        """

        def dfs(node: Node, weighted_sum: np.ndarray, weight_total: float):
            # Add current node's contribution
            weighted_sum = weighted_sum + node.value(smooth=False)
            weight_total = weight_total + 1.0

            if node.type == "Leaf":
                node.smoothed_value = weighted_sum / weight_total
            # Discount contributions from ancestors and recurse
            else:
                weighted_sum = weighted_sum * beta
                weight_total = weight_total * beta
                for child in node.children.values():
                    dfs(child, weighted_sum, weight_total)

        initial_sum = np.zeros_like(self.root.value(smooth=False))
        dfs(self.root, initial_sum, 0.0)

    def print(self, node=None, level=0, prefix="Root: "):
        """Print the tree structure in a readable format."""
        node = node or self.root
        print(" " * (level * 4) + prefix + str(node))
        if node.children:
            self.print(node.children.get("left"), level + 1, prefix="L--- ")
            self.print(node.children.get("right"), level + 1, prefix="R--- ")
