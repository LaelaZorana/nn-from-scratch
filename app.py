"""
Gradio demo for the from-scratch neural network.

Draw a digit (or load a real MNIST test image) and a multilayer perceptron —
written by hand in NumPy, no deep-learning framework — classifies it. The same
network's backprop is verified against finite differences in the test suite.

Run locally:   pip install -r requirements.txt && python app.py
On Hugging Face Spaces this file is the entry point (app_file: app.py).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import gradio as gr

from nn.model import MLP

ACCENT = "#0891b2"  # cyan

MODEL = MLP()
_weights = Path("weights/mnist_mlp.npz")
if _weights.exists():
    MODEL.load_state(dict(np.load(_weights)))

CSS = """
:root { --accent: %s; }
.gradio-container { max-width: 1080px !important; }
#hero { background: linear-gradient(135deg, var(--accent), #0f172a);
        color:#fff; border-radius:18px; padding:26px 30px; margin-bottom:6px; }
#hero h1 { margin:0 0 8px 0; font-size:1.7rem; font-weight:800; letter-spacing:-.01em; }
#hero p { margin:0; opacity:.93; font-size:1.0rem; line-height:1.5; max-width:760px; }
#hero .pill { display:inline-block; background:rgba(255,255,255,.16); border-radius:999px;
        padding:3px 11px; font-size:.74rem; font-weight:700; margin-bottom:12px; letter-spacing:.04em; }
.footer { margin-top:20px; padding-top:14px; border-top:1px solid rgba(128,128,128,.25);
        font-size:.88rem; text-align:center; opacity:.92; }
.footer a { text-decoration:none; font-weight:700; color:var(--accent); }
""" % ACCENT

FOOTER = """
<div class="footer">
🧠 Neural network built from scratch (NumPy, no framework) by <b>Laela Zorana</b> &nbsp;·&nbsp;
<a href="https://github.com/LaelaZorana/nn-from-scratch">Source on GitHub</a> &nbsp;·&nbsp;
see also the eval toolkit:
🔍 <a href="https://huggingface.co/spaces/LaelaZ/ai-agent-scenario-qc">Scenario QC</a> ·
⚖️ <a href="https://huggingface.co/spaces/LaelaZ/rlhf-pairwise-rater">RLHF Rater</a> ·
📦 <a href="https://huggingface.co/spaces/LaelaZ/scorm-qa-validator">SCORM QA</a>
</div>
"""


def _predict_28(arr28: np.ndarray) -> dict:
    """arr28: (28,28) float in [0,1], white digit on black. Returns {label: prob}."""
    x = arr28.reshape(1, 784).astype(np.float32)
    probs = MODEL.probabilities(x)[0]
    return {str(i): float(probs[i]) for i in range(10)}


def _mnist_normalize(ink: np.ndarray) -> np.ndarray:
    """MNIST-style preprocessing: crop to the digit, scale to 20px, center by mass in 28x28."""
    from PIL import Image  # lazy: only needed at draw-time (installed on the Space)

    ink = ink.astype(np.float32)
    if ink.max() > 0:
        ink = ink / ink.max() * 255.0
    ys, xs = np.where(ink > 30)
    if len(xs) == 0:
        return np.zeros((28, 28), np.float32)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = ink[y0:y1 + 1, x0:x1 + 1]

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    small = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize(new, Image.BILINEAR), np.float32)

    canvas = np.zeros((28, 28), np.float32)
    sh, sw = small.shape
    top, left = (28 - sh) // 2, (28 - sw) // 2
    canvas[top:top + sh, left:left + sw] = small
    return canvas / 255.0


def classify_drawing(value):
    """Handle gr.Sketchpad output (dict with 'composite', or a raw array)."""
    if value is None:
        return {}
    img = value["composite"] if isinstance(value, dict) else value
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 4:      # RGBA -> alpha marks the strokes
        ink = img[..., 3]
    elif img.ndim == 3:                          # RGB -> dark strokes on light bg
        ink = 255 - img[..., :3].mean(axis=2)
    else:                                        # already grayscale
        ink = img if img.mean() < 128 else 255 - img
    return _predict_28(_mnist_normalize(ink))


def load_example(digit: str):
    arr = np.load(f"examples/digit_{digit}.npy").astype(np.float32)
    return (arr * 255).astype(np.uint8), _predict_28(arr)


EXAMPLE_DIGITS = [str(i) for i in range(10) if Path(f"examples/digit_{i}.npy").exists()]

theme = gr.themes.Soft(primary_hue="cyan", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

with gr.Blocks(title="Neural Net From Scratch (NumPy)", theme=theme, css=CSS) as demo:
    gr.HTML(
        '<div id="hero"><span class="pill">DEEP LEARNING · NO FRAMEWORK</span>'
        "<h1>🧠 Neural Network From Scratch</h1>"
        "<p>A multilayer perceptron written by hand in NumPy — every forward and backward pass, "
        "the softmax, the Adam optimizer. No PyTorch, no TensorFlow. It hits ~97.7% on MNIST, and "
        "its backprop is verified against finite-difference gradients in the test suite. "
        "Draw a digit below, or load a real test image.</p></div>"
    )

    with gr.Tab("✏️ Draw a digit"):
        with gr.Row():
            sketch = gr.Sketchpad(label="Draw 0–9", type="numpy", image_mode="RGBA",
                                  canvas_size=(280, 280),
                                  brush=gr.Brush(default_size=16, colors=["#000000"], color_mode="fixed"))
            draw_out = gr.Label(num_top_classes=3, label="Prediction")
        gr.Markdown("*Draw thick and centered for best results — the model was trained on centered MNIST digits.*")
        sketch.change(classify_drawing, inputs=sketch, outputs=draw_out)

    with gr.Tab("🔢 Try a test image"):
        with gr.Row():
            with gr.Column():
                digit_dd = gr.Dropdown(EXAMPLE_DIGITS, value=(EXAMPLE_DIGITS[0] if EXAMPLE_DIGITS else None),
                                       label="Real MNIST test digit")
                ex_img = gr.Image(label="Input (28×28)", height=200, image_mode="L")
            ex_out = gr.Label(num_top_classes=3, label="Prediction")
        digit_dd.change(load_example, inputs=digit_dd, outputs=[ex_img, ex_out])
        demo.load(load_example, inputs=digit_dd, outputs=[ex_img, ex_out])

    gr.HTML(FOOTER)
    gr.Markdown("*Runs the actual package (`nn/`) — the same code the test suite gradient-checks. "
                "Architecture: 784 → 256 → 128 → 10, ReLU, softmax cross-entropy, Adam.*")


if __name__ == "__main__":
    demo.launch()
