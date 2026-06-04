"""Tests for the from-scratch network.

The important one is `test_gradient_check`: it verifies the hand-written backward
passes against numerical finite-difference gradients. If the chain rule were wired
up wrong, this test would catch it. This is the "build it AND prove it" guarantee.
"""
from __future__ import annotations

import numpy as np

from nn.layers import softmax_cross_entropy
from nn.model import MLP, Adam


def _loss_only(model: MLP, x, y) -> float:
    """Forward pass + loss, without touching gradients."""
    logits = model.forward(x)
    loss, _ = softmax_cross_entropy(logits, y)
    return loss


def test_softmax_cross_entropy_uniform():
    # Uniform logits over C classes -> loss should equal ln(C).
    logits = np.zeros((4, 10))
    y = np.array([0, 1, 2, 3])
    loss, dlogits = softmax_cross_entropy(logits, y)
    assert abs(loss - np.log(10)) < 1e-6
    assert dlogits.shape == logits.shape


def test_forward_shape():
    model = MLP(sizes=(784, 64, 32, 10), seed=1)
    x = np.random.default_rng(0).standard_normal((8, 784)).astype(np.float32)
    assert model.forward(x).shape == (8, 10)
    assert model.predict(x).shape == (8,)
    probs = model.probabilities(x)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_gradient_check():
    """Analytic gradients must match finite differences to high precision."""
    rng = np.random.default_rng(42)
    model = MLP(sizes=(6, 5, 4, 3), seed=2)
    x = rng.standard_normal((4, 6))
    y = rng.integers(0, 3, size=4)

    # Analytic gradients (snapshot them; FD will perturb params in place afterwards).
    model.loss_and_grad(x, y)
    analytic = [g.copy() for _, g in model.params_and_grads()]

    eps = 1e-5
    for idx, (p, _) in enumerate(model.params_and_grads()):
        flat = p.ravel()
        ga = analytic[idx].ravel()
        # Check a handful of random coordinates per parameter tensor.
        coords = rng.choice(flat.size, size=min(5, flat.size), replace=False)
        for c in coords:
            orig = flat[c]
            flat[c] = orig + eps
            lp = _loss_only(model, x, y)
            flat[c] = orig - eps
            lm = _loss_only(model, x, y)
            flat[c] = orig
            numeric = (lp - lm) / (2 * eps)
            denom = max(1e-8, abs(numeric) + abs(ga[c]))
            rel_err = abs(numeric - ga[c]) / denom
            assert rel_err < 1e-4, f"grad mismatch at param {idx} coord {c}: {rel_err:.2e}"


def test_overfit_tiny_batch():
    """A tiny batch should be driven to near-zero loss: proves the loop learns."""
    rng = np.random.default_rng(0)
    model = MLP(sizes=(20, 32, 16, 4), seed=0)
    opt = Adam(model, lr=1e-2)
    x = rng.standard_normal((8, 20))
    y = rng.integers(0, 4, size=8)
    first = model.loss_and_grad(x, y)
    for _ in range(300):
        model.loss_and_grad(x, y)
        opt.step()
    last = _loss_only(model, x, y)
    assert last < 0.05, f"expected near-zero loss, got {last:.4f} (started {first:.4f})"
    assert (model.predict(x) == y).all()
