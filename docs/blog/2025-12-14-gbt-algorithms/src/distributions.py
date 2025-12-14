from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .utils import sigmoid, to_2d


class Distribution(ABC):
    """Base class for loss distributions."""

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        """Number of parameters/output dimensions required by the distribution."""

    @abstractmethod
    def log_prob(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute the log probability of the true values given the predictions."""

    @abstractmethod
    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        """Return interpretable predictions (e.g., [mu, sigma] for Gaussian, [prob] for Bernoulli)."""

    @abstractmethod
    def gradient_hessian(
        self, y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the gradient and hessian of the loss function w.r.t. predictions."""

    @abstractmethod
    def init_params(self, y: np.ndarray) -> np.ndarray:
        """Initialize the parameters based on the target values."""

    @abstractmethod
    def sample(
        self, y_pred: np.ndarray, n_samples: int, random_state: np.random.RandomState
    ) -> np.ndarray:
        """Generate samples from the distribution given predictions."""


class Gaussian(Distribution):
    def __init__(self, learn_variance: bool = False):
        """
        Gaussian distribution objective for regression tasks. Uses natural
        parameterization for faster convergence.

        **Parameters**
        - `learn_variance` : bool, default=False
            - Whether to learn the variance parameter or assume it is fixed.
            If False: equivalent to Squared loss (homoscedastic regression).
            If True: models both mean and variance (heteroscedastic regression).
        """
        self.learn_variance = learn_variance
        if not learn_variance:
            self.eta2_fixed = -0.5

    @property
    def n_outputs(self) -> int:
        return 2 if self.learn_variance else 1

    def log_prob(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        r"""
        Log probability of Gaussian distribution. Meaning, for each data point, we calculate
        $\log P(y_{\text{true}} \mid \theta)$ where $\theta$ are the parameters predicted by the model. During learning,
        we find the optimal $\theta$ that maximizes this log probability over the training set.

        This implementation uses the natural parameterization of the Gaussian distribution, where
        the learnable parameters are:
        
        $$\begin{aligned}
        \eta_1 &= \frac{\mu}{\sigma^2} \\
        \eta_2 &= -\frac{1}{2\sigma^2}
        \end{aligned}$$

        In practice, we don't learn $\eta_2$ directly, since it must be negative, but the model output
        is unbounded. Instead, we model $z$ where $\eta_2 = -\exp(z)$. This ensures $\eta_2$ is always negative, since
        $\exp(z) > 0$ for all real $z$.
        """
        # Convert the raw predictions ($\eta_1, z$) to ($\mu, \sigma$)
        mu, sigma = self._format_prediction(y_pred).T
        # Compute log probability using the standard Gaussian formula
        log_prob = (
            -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((y_true - mu) ** 2) / (sigma**2)
        )
        return log_prob

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        r"""
        Return predictions $(\mu, \sigma)$.
        Output Shape: `(n_samples, 1 or 2)` depending on whether variance is learnable.
        """
        return self._format_prediction(y_pred)[:, : self.n_outputs]

    def gradient_hessian(
        self, y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the gradient and hessian of the negative log likelihood loss
        w.r.t. the natural parameters $(\eta_1, \eta_2)$.
        """
        eta1, eta2 = self._format_prediction(y_pred, natural=True).T

        grad = np.zeros_like(y_pred)
        hess = np.zeros_like(grad)

        # Derivatives w.r.t $\eta_1$
        grad[:, 0] = -y_true - eta1 / (2 * eta2)
        hess[:, 0] = -1 / (2 * eta2)

        # If learning variance, then $z$ is also a learnable parameter
        if self.learn_variance:
            # Derivatives w.r.t $\eta_2$
            grad_eta2 = -(y_true**2) + (eta1**2 / (4 * eta2**2)) - 1 / (2 * eta2)
            hess_eta2 = -(eta1**2) / (2 * eta2**3) + 1 / (2 * eta2**2)

            # Derivatives w.r.t $z$, which is the raw output of the model
            # Chain rule: $\frac{dL}{dz} = \frac{dL}{d\eta_2} \cdot \frac{d\eta_2}{dz}$
            # where $\frac{d\eta_2}{dz} = -\exp(z) = \eta_2$
            grad[:, 1] = grad_eta2 * eta2
            hess[:, 1] = hess_eta2 * (eta2**2) + grad_eta2 * eta2

        if weight is not None:
            weight = to_2d(weight)
            grad *= weight
            hess *= weight

        return grad, hess

    def init_params(self, y):
        r"""Returns initial $(\eta_1, \eta_2)$ or $(\eta_1)$ if variance is fixed."""
        mu = np.mean(y)
        if self.learn_variance:
            sigma = np.std(y)
            eta1, eta2 = self.to_natural(mu, sigma)
            z = np.log(-eta2)  # Inverse of eta2 = -exp(z)
            return np.array([eta1, z])
        return np.array([mu])

    def mean(self, y_pred: np.ndarray) -> np.ndarray:
        r"""Mean is $\mu$ parameter of the Gaussian."""
        mu, _ = self._format_prediction(y_pred).T
        return mu

    def stdev(self, y_pred: np.ndarray) -> np.ndarray:
        r"""Standard deviation is $\sigma$ parameter of the Gaussian."""
        _, sigma = self._format_prediction(y_pred).T
        return sigma

    def sample(self, y_pred, n_samples: int, random_state: np.random.RandomState) -> np.ndarray:
        """Generate samples from the Gaussian distribution given predictions."""
        mu, sigma = self._format_prediction(y_pred).T
        return random_state.normal(mu, sigma, size=(mu.shape[0], n_samples))

    def entropy(self, y_pred: np.ndarray) -> np.ndarray:
        """Compute the entropy of the Gaussian distribution given predictions."""
        _, sigma = self._format_prediction(y_pred).T
        return 0.5 * np.log(2 * np.pi * np.e * sigma**2)

    def _format_prediction(self, y_pred: np.ndarray, natural: bool = False) -> np.ndarray:
        r"""Convert raw predictions to $(\mu, \sigma)$ or $(\eta_1, \eta_2)$."""
        eta1 = y_pred[:, 0]
        if self.learn_variance:
            z = y_pred[:, 1]
            # model $z$ instead of $\eta_2$ directly to ensure negativity
            eta2 = -np.exp(z)
        else:
            eta2 = np.full_like(eta1, self.eta2_fixed)

        if natural:
            return np.column_stack([eta1, eta2])
        mu, sigma = self.from_natural(eta1, eta2)
        return np.column_stack([mu, sigma])

    @staticmethod
    def to_natural(mu, sigma):
        r"""Helper function to convert $(\mu, \sigma)$ to $(\eta_1, \eta_2)$."""
        eta1 = mu / sigma**2
        eta2 = -1 / (2 * sigma**2)
        return eta1, eta2

    @staticmethod
    def from_natural(eta1, eta2):
        r"""Helper function to convert $(\eta_1, \eta_2)$ to $(\mu, \sigma)$."""
        mu = -eta1 / (2 * eta2)
        sigma = np.sqrt(-1 / (2 * eta2))
        return mu, sigma


class Bernoulli(Distribution):
    """
    Logistic objective for binary classification tasks.
    """

    @property
    def n_outputs(self) -> int:
        return 1

    def log_prob(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        r"""
        Log probability of Bernoulli distribution.

        The Bernoulli distribution is:
        $$P(y \mid p) = p^y (1-p)^{1-y}$$

        Where $p$ is the probability of success, and $y \in \{0, 1\}$. Taking the log, we get:
        $$\log P(y \mid p) = y \log p + (1-y) \log(1-p)$$

        Which is the standard binary cross-entropy loss.

        **Logit Parameterization:**

        We model $z \in \mathbb{R}$ (logit) instead of $p$ directly, using the sigmoid transform:
        $$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

        The rationale behind this is that since the model outputs are unbounded, using sigmoid
        squashes them into the valid probability range (0, 1).

        By substituting $p = \sigma(z)$ into the log probability, we get:
        $$\log P(y \mid z) = y \cdot z - \log(1 + e^{z})$$
        """
        logit = y_pred[:, 0]
        return y_true * logit - np.logaddexp(0, logit)

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        r"""
        Returns the probability $p$ of the positive class.
        Output Shape: `(n_samples, 1)`
        """
        logit = y_pred[:, 0]
        prob = sigmoid(logit)
        return prob.reshape(-1, 1)

    def gradient_hessian(
        self, y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        The derivatives of the loss function (negative log likelihood) w.r.t. the model output
        (logit parameter $z$) are:

        - Gradient: $\sigma(\text{z}) - y$
        - Hessian: $\sigma(\text{z}) \cdot (1 - \sigma(\text{z}))$

        where $\sigma$ is the sigmoid function.
        """
        logit = y_pred[:, 0]
        prob = sigmoid(logit)

        grad = np.zeros_like(y_pred)
        hess = np.zeros_like(grad)

        grad[:, 0] = prob - y_true
        hess[:, 0] = prob * (1 - prob)

        if weight is not None:
            weight = to_2d(weight)
            grad *= weight
            hess *= weight

        return grad, hess

    def init_params(self, y):
        r"""
        Returns initial logit based on mean probability (a priori).
        Note: $$p = \sigma(\text{z}) = \frac{1}{1 + e^{-\text{z}}} \implies \text{z} = \log \frac{p}{1-p}$$
        """
        mean_prob = np.clip(np.mean(y), 1e-7, 1 - 1e-7)
        logit = np.log(mean_prob / (1 - mean_prob))
        return np.array([logit])

    mean = predict

    def sample(self, y_pred, n_samples: int, random_state: np.random.RandomState) -> np.ndarray:
        """Generate samples from the Bernoulli distribution given predictions."""
        logit = y_pred[:, 0]
        prob = sigmoid(logit)
        return random_state.binomial(1, prob, size=(len(prob), n_samples))
