# nn-from-scratch

**🔗 Live demo:** [try it on Hugging Face Spaces](https://huggingface.co/spaces/LaelaZ/nn-from-scratch) — draw a digit and a hand-written NumPy network classifies it.

A small neural network I built **from scratch in NumPy** — no PyTorch, no TensorFlow, no autograd
framework. I wrote every forward pass, every backward pass, the softmax cross-entropy, and the Adam
optimizer by hand, because I wanted to understand the chain rule by implementing it rather than
calling `.backward()`.

It trains a multilayer perceptron on MNIST to **~97.7% test accuracy** in about 9 seconds on a laptop CPU.

## Why this repo is more than "it trains"

The interesting part isn't the accuracy — it's that the hand-written gradients are **verified
correct**. `tests/test_nn.py` includes a finite-difference **gradient check**: it perturbs each
parameter by a tiny amount, measures how the loss actually changes, and confirms that matches the
analytic gradient the backward pass computed. If the chain rule were wired up wrong anywhere, that
test would fail. (This is the same build-it-and-prove-it discipline I apply to evaluation work.)

```
tests/test_nn.py
  test_softmax_cross_entropy_uniform   # ln(C) sanity check
  test_forward_shape                   # shapes + probabilities sum to 1
  test_gradient_check                  # analytic grads == finite differences
  test_overfit_tiny_batch              # the loop can drive a tiny batch to ~0 loss
```

## Architecture

784 → 256 → 128 → 10, ReLU activations, He initialisation, softmax cross-entropy loss, Adam optimizer.

## Run it

```bash
pip install -r requirements.txt
python -m nn.train --epochs 6        # trains, saves weights/mnist_mlp.npz
pytest -q                            # runs the gradient check + other tests
python app.py                        # launches the draw-a-digit demo locally
```

## Layout

```
nn/
  layers.py   # Linear, ReLU, softmax_cross_entropy — hand-written forward + backward
  model.py    # MLP + Adam optimizer
  data.py     # MNIST download + idx parsing
  train.py    # training loop
tests/        # gradient check + sanity tests
app.py        # Gradio demo (draw a digit / load a test image)
weights/      # trained weights (shipped)
examples/     # a few real MNIST test digits
```

Part of my ML portfolio. License: MIT.
