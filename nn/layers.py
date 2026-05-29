"""Layers with hand-written forward and backward passes.

Each layer caches what it needs on the forward pass and returns input-gradients
on the backward pass, plus parameter-gradients where it has parameters. There is
no automatic differentiation here on purpose: the point is to show the chain rule
applied by hand, then prove it correct with a numerical gradient check (see tests).
"""
from __future__ import annotations

import numpy as np


class Linear:
    """Fully-connected layer: y = x @ W + b."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        # He initialisation keeps activations from vanishing/exploding through ReLU.
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # dL/dW = x^T · dy ; dL/db = sum over the batch ; dL/dx = dy · W^T
        self.dW = self._x.T @ dy
        self.db = dy.sum(axis=0)
        return dy @ self.W.T

    def params_and_grads(self):
        yield self.W, self.dW
        yield self.b, self.db


class ReLU:
    """Rectified linear unit: passes positives through, zeros the rest."""

    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # Gradient flows only where the input was positive.
        return dy * self._mask

    def params_and_grads(self):
        return iter(())


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray):
    """Combined softmax + cross-entropy.

    Returns (loss, dlogits). Done together for numerical stability: subtracting the
    row-max before exp avoids overflow, and the gradient simplifies to (p - y) / N.
    `labels` is an integer class index per row.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)

    n = logits.shape[0]
    log_likelihood = -np.log(probs[np.arange(n), labels] + 1e-12)
    loss = log_likelihood.mean()

    dlogits = probs.copy()
    dlogits[np.arange(n), labels] -= 1.0
    dlogits /= n
    return loss, dlogits
