"""
Gradio demo for the from-scratch neural network.

Draw a digit (or load a real MNIST test image) and a multilayer perceptron,
written by hand in NumPy with no deep-learning framework, classifies it. The UI
is bespoke, not the stock label bars: the winning digit fills a hero card with a
confidence ring, and every class from 0 to 9 renders as an animated bar chart.
All of it is drawn by the Python function as HTML into a single gr.HTML panel.
Inference runs the real package (nn/), the same code the test suite
gradient-checks.

Run locally:   pip install -r requirements.txt && python app.py
On Hugging Face Spaces this file is the entry point (app_file: app.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import gradio as gr

from nn.model import MLP

ACCENT = "#0891b2"  # cyan

MODEL = MLP()
_weights = Path("weights/mnist_mlp.npz")
if _weights.exists():
    MODEL.load_state(dict(np.load(_weights)))


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


# --------------------------------------------------------------------------- #
# Custom HTML rendering (replaces stock gr.Label bars)
# --------------------------------------------------------------------------- #

EMPTY_STATE = """
<div class="nn-empty">
  <div class="nn-empty-emoji">✏️</div>
  <div class="nn-empty-text">Draw a digit or pick a test image, and the network names it.</div>
  <div class="nn-empty-sub">Ten classes, the digits 0 through 9.</div>
</div>
"""


def _render(dist: dict) -> str:
    """Render the prediction as a hero digit card plus an animated 0 to 9 bar chart (HTML)."""
    if not dist:
        return EMPTY_STATE

    ordered = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_p = ordered[0]
    conf = round(top_p * 100)

    hero = f"""
    <div class="nn-hero" style="--c:{ACCENT}">
      <div class="nn-hero-digit">{top_label}</div>
      <div class="nn-hero-body">
        <div class="nn-hero-label">It reads a {top_label}</div>
        <div class="nn-hero-sub">{conf}% confident</div>
      </div>
      <div class="nn-hero-ring" style="--p:{top_p:.4f}">
        <span>{conf}<small>%</small></span>
      </div>
    </div>
    """

    # Bar chart in natural digit order 0 to 9, so it reads like a keypad.
    rows = []
    for i in range(10):
        label = str(i)
        p = dist.get(label, 0.0)
        pct = round(p * 100)
        win = " nn-row-win" if label == top_label else ""
        bar_color = ACCENT if label == top_label else "#cbd5e1"
        rows.append(f"""
        <div class="nn-row{win}">
          <div class="nn-row-name">{label}</div>
          <div class="nn-track">
            <div class="nn-fill" style="width:{p*100:.2f}%;background:{bar_color}"></div>
          </div>
          <div class="nn-pct">{pct}%</div>
        </div>
        """)
    chart = f'<div class="nn-chart">{"".join(rows)}</div>'
    return f'<div class="nn-result">{hero}{chart}</div>'


def classify_drawing(value) -> str:
    """Handle gr.Sketchpad output (dict with 'composite', or a raw array)."""
    if value is None:
        return EMPTY_STATE
    img = value["composite"] if isinstance(value, dict) else value
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 4:      # RGBA -> alpha marks the strokes
        ink = img[..., 3]
    elif img.ndim == 3:                          # RGB -> dark strokes on light bg
        ink = 255 - img[..., :3].mean(axis=2)
    else:                                        # already grayscale
        ink = img if img.mean() < 128 else 255 - img
    arr = _mnist_normalize(ink)
    if arr.max() <= 0:
        return EMPTY_STATE
    return _render(_predict_28(arr))


def load_example(digit: str):
    arr = np.load(f"examples/digit_{digit}.npy").astype(np.float32)
    return (arr * 255).astype(np.uint8), _render(_predict_28(arr))


EXAMPLE_DIGITS = [str(i) for i in range(10) if Path(f"examples/digit_{i}.npy").exists()]


CSS = """
:root {
  --nn-bg1:#ecfeff; --nn-bg2:#eff6ff; --nn-ink:#0f172a; --nn-muted:#64748b;
  --nn-card:#ffffff; --nn-line:rgba(15,23,42,.08); --nn-accent:%s;
  --nn-font:'Plus Jakarta Sans','Inter',system-ui,sans-serif;
}
/* Light lock: HF Spaces default to dark mode, but this UI is designed light.
   Override Gradio's dark theme variables so it renders light everywhere. */
:root, .dark, gradio-app.dark {
  color-scheme: light !important;
  --body-background-fill:#ffffff !important;
  --background-fill-primary:#ffffff !important;
  --background-fill-secondary:#f6f6fb !important;
  --block-background-fill:#ffffff !important;
  --block-label-background-fill:#ffffff !important;
  --input-background-fill:#ffffff !important;
  --neutral-950:#16131f !important;
  --border-color-primary:rgba(20,16,40,.12) !important;
  --body-text-color:var(--nn-ink) !important;
  --body-text-color-subdued:var(--nn-muted) !important;
  --block-title-text-color:var(--nn-ink) !important;
  --block-info-text-color:var(--nn-muted) !important;
}
html, body, gradio-app, .dark, .gradio-container { background:#ffffff !important; }
/* Sketchpad: whiten the editor stage behind the transparent drawing canvas */
.image-container, .image-container *, .image-container canvas, .empty.wrap { background:#ffffff !important; }
.gradio-container, .gradio-container * { color: var(--nn-ink); }
.gradio-container { max-width: 880px !important; background:
  radial-gradient(1200px 500px at 15%% -10%%, var(--nn-bg1), transparent 60%%),
  radial-gradient(1000px 500px at 110%% 10%%, var(--nn-bg2), transparent 55%%) !important; }
.gradio-container, .gradio-container * { font-family: var(--nn-font); }

/* Header */
#nn-head { text-align:center; padding: 18px 8px 4px; }
#nn-head .nn-pill { display:inline-block; background:#0f172a; color:#fff; border-radius:999px;
  padding:5px 13px; font-size:.7rem; font-weight:700; letter-spacing:.08em; margin-bottom:14px; }
#nn-head h1 { margin:0; font-size:2.05rem; font-weight:800; letter-spacing:-.02em; color:var(--nn-ink);
  background:linear-gradient(90deg,#0891b2,#0ea5e9,#0f172a); -webkit-background-clip:text;
  background-clip:text; -webkit-text-fill-color:transparent; }
#nn-head p { margin:10px auto 0; max-width:600px; color:var(--nn-muted); font-size:1.02rem; line-height:1.55; }

/* Inputs */
#nn-go { border-radius:14px !important; font-weight:800 !important; font-size:1rem !important;
  background:linear-gradient(135deg,#0891b2,#0ea5e9) !important; border:none !important; color:#fff !important;
  box-shadow:0 10px 26px rgba(8,145,178,.32) !important; transition:transform .12s ease, box-shadow .12s ease !important; }
#nn-go:hover { transform:translateY(-1px); box-shadow:0 14px 32px rgba(8,145,178,.42) !important; }

/* Results panel */
.nn-result { animation: nn-fade .35s ease both; }
@keyframes nn-fade { from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:none} }

.nn-hero { display:flex; align-items:center; gap:20px; padding:22px 26px; border-radius:20px;
  background:var(--nn-card); border:1px solid var(--nn-line); position:relative; overflow:hidden;
  box-shadow:0 18px 44px color-mix(in srgb, var(--c) 22%%, transparent); }
.nn-hero::before { content:""; position:absolute; inset:0; opacity:.10;
  background:radial-gradient(420px 160px at 8%% 0%%, var(--c), transparent 70%%); }
.nn-hero-digit { font-size:70px; line-height:1; font-weight:800; color:var(--c);
  font-variant-numeric:tabular-nums; min-width:64px; text-align:center;
  filter:drop-shadow(0 6px 12px color-mix(in srgb,var(--c) 40%%,transparent)); }
.nn-hero-body { flex:1; }
.nn-hero-label { font-size:1.65rem; font-weight:800; color:var(--nn-ink); letter-spacing:-.01em; }
.nn-hero-sub { color:var(--nn-muted); font-size:.98rem; margin-top:2px; }
.nn-hero-ring { width:64px; height:64px; border-radius:50%%; display:grid; place-items:center; flex-shrink:0;
  background:conic-gradient(var(--c) calc(var(--p)*360deg), color-mix(in srgb,var(--c) 16%%, #fff) 0); }
.nn-hero-ring span { width:50px; height:50px; border-radius:50%%; background:var(--nn-card); display:grid; place-items:center;
  font-weight:800; color:var(--nn-ink); font-size:1.02rem; }
.nn-hero-ring small { font-size:.62rem; font-weight:700; color:var(--nn-muted); }

.nn-chart { margin-top:14px; padding:16px 22px; border-radius:18px; background:var(--nn-card);
  border:1px solid var(--nn-line); box-shadow:0 10px 30px rgba(15,23,42,.05); }
.nn-row { display:flex; align-items:center; gap:14px; padding:5px 0; }
.nn-row-name { width:22px; text-align:center; font-weight:800; color:var(--nn-muted); font-size:1rem;
  font-variant-numeric:tabular-nums; }
.nn-row-win .nn-row-name { color:var(--nn-accent); }
.nn-track { flex:1; height:13px; border-radius:999px; background:#eef2f7; overflow:hidden; }
.nn-fill { height:100%%; border-radius:999px; transform-origin:left; animation: nn-grow .65s cubic-bezier(.2,.8,.2,1) both; }
@keyframes nn-grow { from{transform:scaleX(0)} to{transform:scaleX(1)} }
.nn-pct { width:42px; text-align:right; font-variant-numeric:tabular-nums; font-weight:700; color:var(--nn-muted); font-size:.9rem; }
.nn-row-win .nn-pct { color:var(--nn-ink); }

/* Empty state */
.nn-empty { text-align:center; padding:42px 20px; border-radius:20px; background:var(--nn-card);
  border:1px dashed var(--nn-line); }
.nn-empty-emoji { font-size:2.6rem; }
.nn-empty-text { margin-top:10px; font-weight:700; color:var(--nn-ink); font-size:1.05rem; }
.nn-empty-sub { margin-top:4px; color:var(--nn-muted); font-size:.92rem; }

/* Footer */
.nn-footer { margin-top:22px; padding-top:16px; border-top:1px solid var(--nn-line);
  text-align:center; font-size:.88rem; color:var(--nn-muted); line-height:1.9; }
.nn-footer a { text-decoration:none; font-weight:700; color:var(--nn-accent); }
.nn-meta { text-align:center; color:var(--nn-muted); font-size:.82rem; margin-top:10px; }
""" % ACCENT


FOOTER = """
<div class="nn-footer">
🧠 Neural network built from scratch in NumPy, no framework, by <b>Laela Zorana</b><br>
<a href="https://github.com/LaelaZorana/nn-from-scratch">Source on GitHub</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/distilbert-emotion">Emotion Classifier</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/cnn-gradcam">CNN + Grad-CAM</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/ai-agent-scenario-qc">Scenario QC</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/rlhf-pairwise-rater">RLHF Rater</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/scorm-qa-validator">SCORM QA</a>
</div>
"""


theme = gr.themes.Soft(
    primary_hue="cyan", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), gr.themes.GoogleFont("Inter"),
          "system-ui", "sans-serif"],
)

# Hugging Face Spaces default to dark mode. This bespoke UI is designed light, so
# force the light theme on load (one redirect if the param is missing, then never again).
FORCE_LIGHT = """
function() {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light');
    window.location.replace(u.href);
  }
}
"""

with gr.Blocks(title="Neural Net From Scratch (NumPy)", theme=theme, css=CSS, js=FORCE_LIGHT) as demo:
    gr.HTML(
        '<div id="nn-head"><span class="nn-pill">DEEP LEARNING · NO FRAMEWORK</span>'
        "<h1>A neural network I wrote by hand</h1>"
        "<p>This is a multilayer perceptron written from scratch in NumPy, every forward "
        "and backward pass, the softmax, the Adam optimizer. No PyTorch, no TensorFlow. "
        "It hits about 97.7% on MNIST, and its backprop is checked against finite-difference "
        "gradients in the test suite. Draw a digit, or load a real test image.</p></div>"
    )

    with gr.Tab("✏️ Draw a digit"):
        with gr.Row():
            sketch = gr.Sketchpad(label="Draw 0 to 9", type="numpy", image_mode="RGBA",
                                  canvas_size=(280, 280),
                                  brush=gr.Brush(default_size=16, colors=["#000000"], color_mode="fixed"))
            draw_out = gr.HTML(EMPTY_STATE)
        gr.Markdown("*Draw thick and centered for best results. The model was trained on centered MNIST digits.*")
        sketch.change(classify_drawing, inputs=sketch, outputs=draw_out)

    with gr.Tab("🔢 Try a test image"):
        with gr.Row():
            with gr.Column():
                digit_dd = gr.Dropdown(EXAMPLE_DIGITS, value=(EXAMPLE_DIGITS[0] if EXAMPLE_DIGITS else None),
                                       label="Real MNIST test digit")
                ex_img = gr.Image(label="Input (28x28)", height=200, image_mode="L")
            ex_out = gr.HTML(EMPTY_STATE)
        digit_dd.change(load_example, inputs=digit_dd, outputs=[ex_img, ex_out])
        demo.load(load_example, inputs=digit_dd, outputs=[ex_img, ex_out])

    gr.HTML(FOOTER)
    gr.HTML('<div class="nn-meta">Runs the actual package (nn/), the same code the test suite '
            'gradient-checks. Architecture: 784 to 256 to 128 to 10, ReLU, softmax cross-entropy, Adam.</div>')


if __name__ == "__main__":
    demo.launch()
