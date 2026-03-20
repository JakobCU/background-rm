# Background Remover

A Python desktop app for removing backgrounds from images. Supports both manual color-based removal and AI-powered automatic subject detection.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Features

### Manual Mode (Color-Based)
- **Eyedropper tool** — click directly on the image to sample the background color
- **Color picker** dialog for precise color selection
- **Tolerance** slider (0–255) — controls how far a pixel's color can deviate from the target and still be removed
- **Edge softness** slider (0–50) — gradual alpha fade at boundaries instead of hard cutoffs

### Auto Detect Mode (AI)
- **AI-powered subject detection** using [rembg](https://github.com/danielgatis/rembg) (U2-Net) — similar to iPhone's "lift subject from background"
- Multiple model options:
  - `u2net` — best quality, general purpose
  - `u2netp` — lightweight/faster
  - `u2net_human_seg` — optimized for people
  - `isnet-general-use` — general purpose alternative
  - `silueta` — lightweight alternative
- **Alpha matting** for refined edges (hair, fur, translucent objects) with adjustable foreground/background thresholds

### General
- Live preview on a checkerboard background so transparency is clearly visible
- Supports PNG, JPG, JPEG, BMP, GIF, and WebP input
- Exports as PNG with full alpha transparency

## Installation

```bash
git clone https://github.com/<your-username>/background-rm.git
cd background-rm
pip install -r requirements.txt
```

For GPU acceleration (requires CUDA), replace `rembg[cpu]` with `rembg[gpu]` in `requirements.txt` before installing.

## Usage

```bash
python app.py
```

1. Click **Open Image** to load an image
2. Choose a mode:
   - **Manual** — use the eyedropper or color picker to select the background color, adjust tolerance/softness, then click **Remove Background**
   - **Auto Detect** — select a model, optionally enable alpha matting, then click **Auto Detect Subject** (first run downloads the model, ~170 MB)
3. Click **Save Result** to export the image as PNG

## Requirements

- Python 3.8+
- Pillow
- NumPy
- rembg (for auto detection)
