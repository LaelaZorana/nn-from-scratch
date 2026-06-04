---
title: Neural Network From Scratch (NumPy)
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Neural Network From Scratch (NumPy)

A multilayer perceptron written entirely by hand in NumPy: every forward and backward
pass, the softmax cross-entropy, and the Adam optimizer. No PyTorch, no TensorFlow.

It reaches **~97.7% accuracy on MNIST**, and its hand-written backprop is verified against
**finite-difference gradients** in the test suite (so the chain rule is provably wired up
correctly, not just "it seems to train"). Draw a digit, or load a real MNIST test image.

Architecture: 784 → 256 → 128 → 10, ReLU activations, He initialisation, Adam.

**Source & full docs:** https://github.com/LaelaZorana/nn-from-scratch
