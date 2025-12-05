# mvos_hsi

**MVOS Hyperspectral Imaging Utilities**

`mvos_hsi` is a Python package developed at the  
**Machine Vision and Optical Sensors (MVOS) Lab, South Dakota State University (SDSU)**  
for preprocessing and managing hyperspectral imaging (HSI) data.

The package provides a reproducible pipeline for:

- Calibrating raw hyperspectral cubes  
- Clipping sample regions (e.g., leaves)  
- Applying data augmentation  
- Generating graphs for spectral comparisons
- Organizing outputs into clean, ML-ready datasets  

It was originally designed for leaf-level agricultural experiments (e.g., nitrogen status in corn leaves) but is broadly applicable to other hyperspectral imaging tasks.


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
    - Square windows (fixed size, e.g., 30×30 pixels)
    - Tight bounding boxes fit to each leaf
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

### From PyPI 

```bash
pip install mvos_hsi
