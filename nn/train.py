"""Train the from-scratch MLP on MNIST and save the weights.

Usage:  python -m nn.train [--epochs 6] [--batch 128] [--lr 1e-3]
Saves to weights/mnist_mlp.npz.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .data import load_mnist
from .model import MLP, Adam


def accuracy(model: MLP, x: np.ndarray, y: np.ndarray, batch: int = 1000) -> float:
    correct = 0
    for i in range(0, len(x), batch):
        preds = model.predict(x[i:i + batch])
        correct += int((preds == y[i:i + batch]).sum())
    return correct / len(x)


def train(epochs: int = 6, batch: int = 128, lr: float = 1e-3, seed: int = 0,
          out: str = "weights/mnist_mlp.npz") -> float:
    x_train, y_train, x_test, y_test = load_mnist()
    model = MLP(seed=seed)
    opt = Adam(model, lr=lr)
    rng = np.random.default_rng(seed)
    n = len(x_train)

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        x_train, y_train = x_train[order], y_train[order]
        t0 = time.time()
        running = 0.0
        steps = 0
        for i in range(0, n, batch):
            xb, yb = x_train[i:i + batch], y_train[i:i + batch]
            loss = model.loss_and_grad(xb, yb)
            opt.step()
            running += loss
            steps += 1
        acc = accuracy(model, x_test, y_test)
        print(f"epoch {epoch}  loss={running / steps:.4f}  test_acc={acc:.4f}  ({time.time() - t0:.1f}s)")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **model.state())
    print(f"saved weights -> {out}  (final test acc {acc:.4f})")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.parse_args()
    args = ap.parse_args()
    train(epochs=args.epochs, batch=args.batch, lr=args.lr)
