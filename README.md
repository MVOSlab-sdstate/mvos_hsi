# mvos_hsi

`mvos_hsi` is a Python package for working with hyperspectral leaf images in the MVOS lab workflow.

It provides a full pipeline:

1. **Calibration** of raw ENVI cubes (sample + dark) into calibrated reflectance/fluorescence.
2. **Clipping** of individual leaves into smaller hyperspectral hypercubes using NDVI / CI-RedEdge / GCI.
3. **Augmentation** of clipped hypercubes (flip/rotate/shear/scale) to expand training data.
4. **Spectral plotting** tools to generate spectral profiles from pixels, ROIs, or leaf centers.

The goal is to mirror the existing MATLAB calibration + clipping pipeline in a reproducible, Pythonic way.

---

## 1. Installation

From PyPI:

```bash
pip install mvos_hsi
```
## 2. Expected input data

The package assumes:

- Raw hyperspectral cubes are stored as **ENVI** pairs (`.hdr` / `.img`).
- Each sample has:
  - A reflectance-like cube: `<stem>_R.hdr` + `<stem>_R.img`
  - A fluorescence-like cube: `<stem>_F.hdr` + `<stem>_F.img`
- There is at least one **dark reference** cube with base name like:

  ```text
  <root>/Dark_R.hdr
  <root>/Dark_R.img
  <root>/Dark_F.hdr
  <root>/Dark_F.img
  ```

  Where the **dark base** path is:

  ```text
  <root>/Dark
  ```

Optionally, you have wavelength information from:

- A MATLAB file with `wavelength` (e.g. `data_uf.mat`), and/or  
- A CSV file with one column of wavelength values (e.g. `wavelength.csv`).

---

## 3. Basic Python usage

Below is a minimal end-to-end example.

```python
import os
from pathlib import Path
import mvos_hsi

# Root folder containing raw ENVI cubes (*_R.hdr/img, *_F.hdr/img)
ROOT = r"C:\path\to\your\dataset"
DARK_BASE = os.path.join(ROOT, "Dark")  # base path, WITHOUT _R/_F

# Optional wavelength info
WAVELENGTHS_MAT = r"C:\path\to\data_uf.mat"  # contains 'wavelength'
WAVELENGTHS_CSV = None                          # or r"C:\path\to\wavelength.csv"

# ---------------------------------------------------------------------
# 1) Calibration: raw ENVI -> calibrated MAT files
# ---------------------------------------------------------------------
mvos_hsi.calibrate_folder(
    folder=ROOT,
    dark_base=DARK_BASE,
    spectral_bin=3,   # 1 = no binning; 2 = half; 3 = one-third
    spatial_bin=3,    # to mirror MATLAB IFOV cropping/binning
)

# This creates, for each sample:
#   <stem>_R.mat (with R_plant)
#   <stem>_F.mat (with F_plant)
# next to the original ENVI files.

# ---------------------------------------------------------------------
# 2) Clipping: detect leaves and extract leaf hypercubes
# ---------------------------------------------------------------------
clip_result = mvos_hsi.clip_folder(
    folder=ROOT,
    index="ndvi",                # one of: "ndvi", "ciredge", "gci"
    wavelengths_mat=WAVELENGTHS_MAT,
    wavelengths_csv=WAVELENGTHS_CSV,
    threshold_mode="auto",       # "auto" (Otsu) or "manual"
    threshold_value=0.45,        # used only if threshold_mode == "manual"
    min_area=100,                # minimum region size in pixels
    crop_mode="square",          # "square" or "tight"
    crop_size=30,                # square size if crop_mode="square"
    pad=0,                       # padding (pixels) if crop_mode="tight"
    outdir=None,                 # default: ROOT/clipped_hypercubes
)

# clip_result is a dict:
#   { "<stem>": [ "<outdir>/<stem>_leaf1", "<outdir>/<stem>_leaf2", ... ], ... }
# Each leaf is stored as an ENVI cube: <stem>_leafN.hdr/.img

clips_outdir = Path(ROOT) / "clipped_hypercubes"

# ---------------------------------------------------------------------
# 3) Augmentation: increase training data
# ---------------------------------------------------------------------
mvos_hsi.augment_folder(
    folder=str(clips_outdir),
    num_aug=3,                   # number of augmented variants per cube
    outdir=None,                 # default: <clips_outdir>/augmented_hypercubes
    flip=True,                   # horizontal flip
    rotate=(-10, 10),            # rotation range in degrees, or None
    shear=(-16, 16),             # shear range in degrees, or None
    scale=None,                  # e.g. (0.95, 1.05) for mild zoom
)

# Augmented cubes are saved in:
#   ROOT/clipped_hypercubes/augmented_hypercubes

# ---------------------------------------------------------------------
# 4) Spectral plotting: leaf center spectra
# ---------------------------------------------------------------------
# Example: plot center-pixel spectra for leaf 1 and 2 of one sample
mvos_hsi.plot_leaf_center(
    clipped_dir=str(clips_outdir),
    stem="H_P1_V4_B",            # sample stem used during clipping
    leaves=[1, 2],               # which leaf indices to plot
    wavelengths_mat=WAVELENGTHS_MAT,
    wavelengths_csv=WAVELENGTHS_CSV,
    title="Center pixel spectra for H_P1_V4_B leaves 1 & 2",
    ylim=None,                   # e.g. (0, 1.0) if you want to fix y-axis
    save=None,                   # path to save PNG, or None to skip saving
    show=True,                   # True to display the plot
)
```

---

## 4. Command-line interface (CLI)

`mvos_hsi` also installs a console command:

```bash
mvos-hsi --help
```

### 4.1. Calibration from CLI

Calibrate a whole folder:

```bash
mvos-hsi calibration folder ^
  --folder "C:\path\to\dataset" ^
  --dark   "C:\path\to\dataset\Dark" ^
  --k 3 ^
  --spatial 3
```

This creates `<stem>_R.mat` and `<stem>_F.mat` next to the raw ENVI cubes.

### 4.2. Clipping from CLI

Clip all calibrated samples:

```bash
mvos-hsi clipping folder ^
  --folder "C:\path\to\dataset" ^
  --index ndvi ^
  --wavelengths-mat "C:\path\to\data_uf.mat" ^
  --threshold-mode auto ^
  --crop-mode square ^
  --crop-size 30
```

Clipped leaf cubes are saved under:

```text
C:\path	o\dataset\clipped_hypercubes
```

Each leaf: `<stem>_leafN.hdr` / `.img`.

### 4.3. Augmentation from CLI

Augment all clipped hypercubes:

```bash
mvos-hsi augmentation folder ^
  --folder "C:\path\to\dataset\clipped_hypercubes" ^
  --num 3 ^
  --flip ^
  --rotate -10 10 ^
  --shear -16 16
```

Augmented cubes go under:

```text
C:\path	o\dataset\clipped_hypercubesugmented_hypercubes
```

### 4.4. Plotting from CLI

Plot center-pixel spectra for specific leaves:

```bash
mvos-hsi plotting leaf ^
  --clipped-dir "C:\path\to\dataset\clipped_hypercubes" ^
  --stem H_P1_V4_B ^
  --leaf 1 3 ^
  --wavelengths-mat "C:\path\to\data_uf.mat"
```

Plot center-pixel spectra across samples:

```bash
mvos-hsi plotting leaf-multi ^
  --clipped-dir "C:\path\to\dataset\clipped_hypercubes" ^
  --item H_P1_V4_B:1 ^
  --item H_P1_V6_B:3 ^
  --wavelengths-mat "C:\path\to\data_uf.mat"
```

> On Linux/macOS you can use `\` instead of `^` for line continuation.

---

## 5. Example dataset layout

A small example dataset can be stored in:

```text
example_data/
  corn_demo/
    raw/
      H_P1_V4_B_R.hdr
      H_P1_V4_B_R.img
      H_P1_V4_B_F.hdr
      H_P1_V4_B_F.img
      Dark_R.hdr
      Dark_R.img
      Dark_F.hdr
      Dark_F.img
    # (optional) wavelength files:
    # data_uf.mat
    # wavelength.csv
```

You can then test the pipeline using:

```python
import os
from pathlib import Path
import mvos_hsi

ROOT = r"path	o\mvos_hsi\example_data\corn_demo
aw"
DARK_BASE = os.path.join(ROOT, "Dark")
WAVELENGTHS_MAT = r"path	o\mvos_hsi\example_data\corn_demo\data_uf.mat"

mvos_hsi.calibrate_folder(
    folder=ROOT,
    dark_base=DARK_BASE,
    spectral_bin=3,
    spatial_bin=3,
)

mvos_hsi.clip_folder(
    folder=ROOT,
    index="ndvi",
    wavelengths_mat=WAVELENGTHS_MAT,
)
```

---

## 6. How to add the example dataset to this repo (for developers)

> This section is for developers maintaining the repository.  
> End users installing via `pip install mvos_hsi` do **not** need to do this.

1. Create the folder structure locally:

   ```bash
   mkdir -p example_data/corn_demo/raw
   ```

2. Copy a small sample of your data into `example_data/corn_demo/raw/`:

   - One sample cube:
     - `H_P1_V4_B_R.hdr`, `H_P1_V4_B_R.img`
     - `H_P1_V4_B_F.hdr`, `H_P1_V4_B_F.img`
   - One dark cube:
     - `Dark_R.hdr`, `Dark_R.img`
     - `Dark_F.hdr`, `Dark_F.img`
   - Optionally: `data_uf.mat` and/or `wavelength.csv` (small versions).

3. Add and commit the example data:

   ```bash
   git add example_data
   git commit -m "Add small example hyperspectral dataset"
   git push
   ```

Now the repository contains:

- The code  
- A clear README  
- A small example dataset for testing the full pipeline.

---

## 7. License

This project is released under the MIT License (see `LICENSE` file).
