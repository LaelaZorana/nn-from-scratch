"""A multilayer perceptron built from the hand-written layers, plus an Adam optimizer."""
from __future__ import annotations

import numpy as np

from .layers import Linear, ReLU, softmax_cross_entropy


class MLP:
    """Linear -> ReLU -> Linear -> ReLU -> Linear classifier."""

    def __init__(self, sizes=(784, 256, 128, 10), seed: int = 0):
        rng = np.random.default_rng(seed)
        self.layers = [
            Linear(sizes[0], sizes[1], rng), ReLU(),
            Linear(sizes[1], sizes[2], rng), ReLU(),
            Linear(sizes[2], sizes[3], rng),
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dlogits: np.ndarray) -> None:
        for layer in reversed(self.layers):
            dlogits = layer.backward(dlogits)

    def loss_and_grad(self, x: np.ndarray, y: np.ndarray):
        logits = self.forward(x)
        loss, dlogits = softmax_cross_entropy(logits, y)
        self.backward(dlogits)
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).argmax(axis=1)

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def params_and_grads(self):
        for layer in self.layers:
            yield from layer.params_and_grads()

    # --- save / load -------------------------------------------------------
    def state(self) -> dict:
        out = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                out[f"W{i}"] = layer.W
                out[f"b{i}"] = layer.b
        return out

    def load_state(self, state: dict) -> None:
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                layer.W = state[f"W{i}"]
                layer.b = state[f"b{i}"]


class Adam:
    """Adam optimizer over a model's (param, grad) pairs."""

    def __init__(self, model: MLP, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.model = model
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.t = 0
        self._m = [np.zeros_like(p) for p, _ in model.params_and_grads()]
        self._v = [np.zeros_like(p) for p, _ in model.params_and_grads()]

    def step(self) -> None:
        self.t += 1
        for i, (p, g) in enumerate(self.model.params_and_grads()):
            self._m[i] = self.b1 * self._m[i] + (1 - self.b1) * g
            self._v[i] = self.b2 * self._v[i] + (1 - self.b2) * (g * g)
            m_hat = self._m[i] / (1 - self.b1 ** self.t)
            v_hat = self._v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
