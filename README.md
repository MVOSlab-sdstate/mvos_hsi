<p align="center">
  <img src="logo1.png" width="300">
</p>

<p align="center">
  <strong>End-to-end Python library for preprocessing agricultural hyperspectral leaf images</strong><br>
  Calibration · Leaf segmentation · Data augmentation · Spectral visualization
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green">
  <img src="https://img.shields.io/badge/python-3.9+-blue">
  <img src="https://img.shields.io/pypi/v/mvos_hsi" alt="PyPI">
  <img src="https://img.shields.io/pypi/dm/mvos_hsi" alt="Downloads">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey">
</p>

---

## Overview

Hyperspectral imaging (HSI) captures hundreds of contiguous spectral bands per pixel,
enabling non-destructive quantification of plant traits — leaf water content, pigment
concentration, and early stress indicators — that are invisible to standard cameras. Yet
turning raw sensor output into analysis-ready data is still largely done through ad-hoc,
lab-specific scripts that are hard to share and difficult to reproduce.

**MVOS_HSI** consolidates the entire preprocessing pipeline into a single installable
Python package. It handles raw ENVI calibration, vegetation-index-based leaf
segmentation and cropping, geometric data augmentation, and spectral visualization —
so you can generate reproducible, ML-ready hyperspectral datasets without writing
custom glue code or manually cropping leaves. Outputs integrate with both Python
(NumPy, scikit-learn, TensorFlow) and MATLAB workflows.

The library can be used as an importable Python API or run entirely from the command line.

> **Paper:** Aggarwal R., Yadav P.K., Qin J., Burks T.F., Kim M.S. — *MVOS_HSI: A Python library for preprocessing agricultural crop hyperspectral image data* · [GitHub](https://github.com/MVOSlab-sdstate/mvos_hsi) · Contact: pappu.yadav@sdstate.edu

---

## Pipeline

```
Raw ENVI cubes (.hdr / .img)
         │
         ▼
┌─────────────────────────────────────────────┐
│  1. Calibrate                               │
│     • Dark-reference subtraction            │
│     • Spectral binning  (k bands → 1)       │
│     • Spatial binning   (k×k → 1 pixel)     │
│  Output: <stem>_R.mat,  <stem>_F.mat        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  2. Clip                                    │
│     • Compute vegetation index image        │
│       (NDVI / CI-RedEdge / GCI)             │
│     • Threshold (Otsu auto or manual)       │
│     • Extract per-leaf hypercubes           │
│  Output: clipped_hypercubes/<stem>_leafN    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  3. Augment                                 │
│     • Flip, rotate, shear, scale            │
│     • Transform applied across ALL bands    │
│  Output: augmented_hypercubes/              │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
         ML-ready dataset

         (optional side step)
         4. Plot spectra
            Pixel / ROI / leaf-center profiles
```

---

## Features

| Module | What it does | Key parameters |
|--------|--------------|----------------|
| **Calibration** | Dark-reference subtraction; spectral & spatial binning; writes `.mat` outputs | `spectral_bin`, `spatial_bin` |
| **Clipping** | Vegetation-index segmentation; Otsu or manual threshold; square or tight crop | `index`, `threshold_mode`, `crop_mode`, `crop_size` |
| **Augmentation** | Geometry-preserving transforms across all wavelength channels | `num_aug`, `flip`, `rotate`, `shear`, `scale` |
| **Plotting** | Center-pixel, pixel, ROI, and multi-sample spectral profiles | `stem`, `leaves`, `ylim`, `save` |

---

## Installation

```bash
pip install mvos_hsi
```

We recommend installing into a dedicated virtual environment and pinning versions for reproducibility:

```bash
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
python -m venv .venv && .venv\Scripts\activate       # Windows

pip install mvos_hsi
pip freeze > requirements.txt
```

**Dependencies:** Python 3.9+, NumPy, SciPy, Matplotlib, Spectral Python (SPy), scikit-image, imgaug

---

## Expected Input Format

MVOS_HSI expects raw hyperspectral cubes as **ENVI pairs** (`.hdr` + `.img`).
Each sample should have paired reflectance (`_R`) and fluorescence (`_F`) cubes,
plus a matching dark-reference acquisition captured under the same sensor settings
(exposure, gain, binning):

```
dataset/
├── SampleA_R.hdr   ←─ reflectance cube header
├── SampleA_R.img   ←─ reflectance cube data
├── SampleA_F.hdr   ←─ fluorescence cube header
├── SampleA_F.img   ←─ fluorescence cube data
├── SampleB_R.hdr
├── SampleB_R.img
├── ...
├── Dark_R.hdr      ←─ dark reference (reflectance)
├── Dark_R.img
├── Dark_F.hdr      ←─ dark reference (fluorescence)
└── Dark_F.img
```

Pass `dark_base = "dataset/Dark"` — the package appends `_R`/`_F` automatically.

**Wavelength metadata** (required for clipping and plotting) can be provided as:
- A MATLAB `.mat` file containing a `wavelength` variable (e.g. `data_uf.mat`)
- A single-column CSV file (e.g. `wavelength.csv`)

If neither is provided, the package falls back to band-index-based selection.

---

## Python Usage

### Step 0 — Setup

```python
from pathlib import Path
import mvos_hsi

ROOT            = Path(r"C:\path\to\dataset")
DARK_BASE       = ROOT / "Dark"                    # no _R/_F suffix
WAVELENGTHS_MAT = Path(r"C:\path\to\data_uf.mat")  # contains 'wavelength'
WAVELENGTHS_CSV = None                              # or Path("wavelength.csv")
CLIPS_OUTDIR    = ROOT / "clipped_hypercubes"
```

---

### Step 1 — Calibration

Applies dark-reference subtraction and optional binning to produce calibrated `.mat` files.

When only a dark reference is available (no white reference), MVOS_HSI performs dark subtraction:

```
calibrated(λ) = raw(λ) − dark(λ)
```

Full reflectance normalization `R(λ) = (raw − dark) / (white − dark)` is also supported
when a white reference is recorded.

```python
mvos_hsi.calibrate_folder(
    folder       = str(ROOT),
    dark_base    = str(DARK_BASE),
    spectral_bin = 3,   # 1 = none | 2 = half bands | 3 = one-third bands
    spatial_bin  = 3,   # averages over k×k spatial neighborhoods
)
# Writes per sample:
#   <stem>_R.mat  →  variable: R_plant
#   <stem>_F.mat  →  variable: F_plant
```

> **Tip:** Match `spatial_bin` to the binning used during acquisition.
> `spectral_bin=3` reduces a 300-band cube to 100 bands — a reasonable default for
> leaf-level work. Setting `spectral_bin=1` disables spectral binning.

---

### Step 2 — Clipping

Detects and extracts individual leaves as separate hyperspectral hypercubes using a
vegetation index to separate leaf tissue from background.

```python
clip_result = mvos_hsi.clip_folder(
    folder          = str(ROOT),
    index           = "ndvi",    # see vegetation index table below
    wavelengths_mat = str(WAVELENGTHS_MAT),
    wavelengths_csv = str(WAVELENGTHS_CSV) if WAVELENGTHS_CSV else None,
    threshold_mode  = "auto",    # "auto" = Otsu  |  "manual"
    threshold_value = 0.45,      # only used when threshold_mode="manual"
    min_area        = 100,       # discard connected regions smaller than N pixels
    crop_mode       = "square",  # "square" | "tight"
    crop_size       = 30,        # side length (px) for square crops
    pad             = 0,         # extra padding (px) for tight crops
    outdir          = None,      # default: ROOT/clipped_hypercubes
)

# Returns: { "<stem>": ["<outdir>/<stem>_leaf1", ...], ... }
# Each leaf saved as: <stem>_leafN.hdr + <stem>_leafN.img
```

**Vegetation index options:**

| Index | Formula | Best for |
|-------|---------|----------|
| `ndvi` | (NIR − Red) / (NIR + Red) | General vegetation vs. background — robust default |
| `ciredge` | (NIR / Red-edge) − 1 | Stressed or senescent leaves; red-edge chlorophyll |
| `gci` | (NIR / Green) − 1 | Canopy chlorophyll content estimation |

**Thresholding:**
- `"auto"` — Otsu's method selects a threshold by maximising between-class variance. Recommended for most datasets.
- `"manual"` — use `threshold_value` directly when Otsu over- or under-segments (e.g., very dark backgrounds or low-contrast scenes).

**Crop modes:**
- `"square"` — pads each detected leaf to a fixed `crop_size × crop_size` window. Best for ML models that require uniform spatial input dimensions.
- `"tight"` — crops tightly to the leaf bounding box ± `pad` pixels. Best when preserving spatial detail matters more than uniform size.

---

### Step 3 — Augmentation

Expands training data by applying geometric transforms to each clipped hypercube.
Every transform is applied **consistently across all wavelength channels**, preserving
spectral signatures while introducing realistic spatial pose variations.

```python
mvos_hsi.augment_folder(
    folder  = str(CLIPS_OUTDIR),
    num_aug = 3,             # augmented variants generated per cube
    outdir  = None,          # default: CLIPS_OUTDIR/augmented_hypercubes
    flip    = True,          # random horizontal flip
    rotate  = (-10, 10),     # rotation range in degrees  |  None to disable
    shear   = (-16, 16),     # shear range in degrees     |  None to disable
    scale   = None,          # e.g. (0.95, 1.05) for mild zoom  |  None to disable
)
# Writes: augmented_hypercubes/<stem>_leafN_augK.hdr + .img
```

> **Tip:** Start conservative — `num_aug=3`, `rotate=(-10,10)`, `shear=(-16,16)`.
> Enable `scale` only if your model needs zoom invariance. Overly aggressive
> augmentation can distort fine spectral-spatial patterns.

---

### Step 4 — Spectral Plotting

#### Single sample — leaf-center spectra

Plots the center-pixel spectrum of one or more leaves for quick quality control.

```python
mvos_hsi.plot_leaf_center(
    clipped_dir     = str(CLIPS_OUTDIR),
    stem            = "H_P1_V4_B",       # sample stem used during clipping
    leaves          = [1, 2],            # which leaf indices to plot
    wavelengths_mat = str(WAVELENGTHS_MAT),
    wavelengths_csv = None,
    title           = "Center spectra — H_P1_V4_B leaves 1 & 2",
    ylim            = None,              # e.g. (0.0, 1.0) to fix y-axis
    save            = None,              # path to save PNG, or None
    show            = True,
)
```

#### Multi-sample comparison (CLI)

```bash
mvos-hsi plotting leaf-multi \
  --clipped-dir     "C:\path\to\clipped_hypercubes" \
  --item            H_P1_V4_B:1 \
  --item            H_P1_V6_B:3 \
  --wavelengths-mat "C:\path\to\data_uf.mat"
```

---

## Command-Line Interface (CLI)

After installation, `mvos-hsi` is available as a global command.

```bash
mvos-hsi --help
```

> **Windows:** use `^` for line continuation. **Linux/macOS:** use `\`.

### Calibration

```bash
mvos-hsi calibration folder \
  --folder "C:\path\to\dataset" \
  --dark   "C:\path\to\dataset\Dark" \
  --k 3 \
  --spatial 3
```

### Clipping

```bash
mvos-hsi clipping folder \
  --folder          "C:\path\to\dataset" \
  --index           ndvi \
  --wavelengths-mat "C:\path\to\data_uf.mat" \
  --threshold-mode  auto \
  --crop-mode       square \
  --crop-size       30
```

Output: `<folder>/clipped_hypercubes/<stem>_leafN.hdr` / `.img`

### Augmentation

```bash
mvos-hsi augmentation folder \
  --folder "C:\path\to\dataset\clipped_hypercubes" \
  --num 3 \
  --flip \
  --rotate -10 10 \
  --shear  -16 16
```

Output: `<folder>/augmented_hypercubes/`

### Spectral Plotting

```bash
# Single sample
mvos-hsi plotting leaf \
  --clipped-dir     "C:\path\to\clipped_hypercubes" \
  --stem            H_P1_V4_B \
  --leaf            1 3 \
  --wavelengths-mat "C:\path\to\data_uf.mat"

# Multi-sample comparison
mvos-hsi plotting leaf-multi \
  --clipped-dir     "C:\path\to\clipped_hypercubes" \
  --item            H_P1_V4_B:1 \
  --item            H_P1_V6_B:3 \
  --wavelengths-mat "C:\path\to\data_uf.mat"
```

---

## Output Summary

| Step | Output location | Format | Contents |
|------|----------------|--------|----------|
| Calibration | Next to input ENVI files | `.mat` | `R_plant`, `F_plant` arrays |
| Clipping | `<root>/clipped_hypercubes/` | ENVI `.hdr` + `.img` | Per-leaf hypercubes |
| Augmentation | `.../clipped_hypercubes/augmented_hypercubes/` | ENVI `.hdr` + `.img` | Augmented leaf variants |
| Plotting | User-specified path or screen | `.png` / on-screen | Spectral profile figures |

---

## Limitations

- Preprocessing quality depends strongly on acquisition conditions. Dark-reference images must be captured under the same sensor settings (exposure, gain, binning) as the sample imagery.
- Performance may decrease for data collected under challenging conditions: variable or unstable illumination, severe shadowing, highly heterogeneous backgrounds, or sensor-specific artifacts not yet modelled.
- Full reflectance normalization requires both a dark and a white reference. When only a dark reference is available, MVOS_HSI performs dark subtraction only, producing reflectance-*like* values suitable for segmentation and ML training.

---

## Sample Dataset

A ready-to-use sample dataset — including multiple hyperspectral images, dark reference images, and a `.mat` wavelength file — is available to download. Use it to explore all pipeline stages before working with your own data.

**[Download sample dataset (Google Drive)](https://drive.google.com/drive/folders/1S7Q1xkLRZeDtlIYSJ0i3ZpOVSlOGjay-?usp=sharing)**

---

## Citation

If you use MVOS_HSI in your research, please cite:

```bibtex
@article{aggarwal2024mvos_hsi,
  title   = {{MVOS\_HSI}: A Python library for preprocessing agricultural crop hyperspectral image data},
  author  = {Aggarwal, Rishik and Yadav, Pappu Kumar and Qin, Jianwei and Burks, Thomas F. and Kim, Moon S.},
  year    = {2024},
  note    = {v1.0.0. \url{https://github.com/MVOSlab-sdstate/mvos_hsi}}
}
```

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE) for details.

| | |
|---|---|
| Repository | https://github.com/MVOSlab-sdstate/mvos_hsi |
| PyPI | https://pypi.org/project/mvos_hsi |
| Support | pappu.yadav@sdstate.edu |
| Institution | Machine Vision and Optical Sensor (MVOS) Lab · South Dakota State University |
