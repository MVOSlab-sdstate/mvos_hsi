# mvos_hsi

**MVOS Hyperspectral Imaging Utilities**  
Python package for calibration, clipping, and augmentation of hyperspectral leaf images, ported from MATLAB scripts.

---

## Features
- **Calibration**
  - Single image or batch folder calibration
  - Dark and white reference correction
  - Spectral binning & spatial binning options
  - Outputs calibrated reflectance and fluorescence cubes
  - Saves calibrated cubes as `.mat` files (`*_R.mat`, `*_F.mat`)

- **Clipping**
  - Detect leaves using vegetation indices:
    - NDVI (Normalized Difference Vegetation Index)
    - CI-RedEdge (Chlorophyll Index Red-Edge)
    - GCI (Green Chlorophyll Index)
  - Thresholding via Otsu (auto) or manual (e.g., NDVI > 0.45)
  - Flexible cropping:
    - **Square** windows (fixed size, e.g., 30×30 pixels)
    - **Tight** bounding boxes fit to each leaf
  - Saves clipped cubes as ENVI files (`.hdr` + `.img`) in `clipped_hypercubes/`

- **Augmentation**
  - Apply data augmentation to calibrated or clipped cubes
  - Options:
    - Per image (single `.mat` or `.hdr`)
    - Per folder
    - Per class (each subfolder = class label)
  - User-defined number of augmentations
  - Saves augmented cubes in `augmented_hypercubes/`

---

## Installation
Clone this repository and install in **editable mode**:

```bash
conda activate hyperspectral
pip install -e .
