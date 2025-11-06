from __future__ import annotations
import os, glob
from typing import Optional, List, Dict, Tuple

import numpy as np
import cv2
from spectral import envi
from scipy.io import loadmat
import csv

# ---------- helpers ----------
def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def _load_wavelengths_from_mat(path: str) -> np.ndarray:
    mat = loadmat(path)
    # expects a key 'wavelength' (as used in your scripts)
    # ndvi_image.py / ci_rededge_image.py / gci_image.py load it this way
    # :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
    wl = mat.get("wavelength")
    if wl is None:
        raise ValueError("No 'wavelength' key found in MAT file.")
    return wl.flatten()

def _load_wavelengths_from_csv(path: str) -> np.ndarray:
    vals = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                vals.append(float(row[0]))
            except Exception:
                pass
    if not vals:
        raise ValueError("No numeric wavelengths parsed from CSV.")
    return np.asarray(vals, dtype=float)

def _nearest_band(wavelengths: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(wavelengths - target_nm)))

def _compute_index(
    cube: np.ndarray,
    index: str,
    wavelengths: Optional[np.ndarray] = None,
    *,
    # fallback band indices if wavelengths are not provided:
    red_idx: Optional[int] = None,
    green_idx: Optional[int] = None,
    rededge_idx: Optional[int] = None,
    nir_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Compute NDVI, CI-RedEdge, or GCI using either wavelengths (preferred) or explicit band indices.
    """
    index = index.lower().strip()

    if wavelengths is not None:
        if index == "ndvi":
            red = cube[:, :, _nearest_band(wavelengths, 660.0)]
            nir = cube[:, :, _nearest_band(wavelengths, 850.0)]
            # (nir - red) / (nir + red) from your ndvi script :contentReference[oaicite:7]{index=7}
            idx = (nir - red) / (nir + red + 1e-12)
        elif index in ("ci_rededge", "ciredge", "ci-rededge"):
            re  = cube[:, :, _nearest_band(wavelengths, 710.0)]
            nir = cube[:, :, _nearest_band(wavelengths, 850.0)]
            # (nir / rededge) - 1  from your ci_rededge script :contentReference[oaicite:8]{index=8}
            idx = (nir / (re + 1e-12)) - 1.0
        elif index == "gci":
            g   = cube[:, :, _nearest_band(wavelengths, 550.0)]
            nir = cube[:, :, _nearest_band(wavelengths, 850.0)]
            # (nir / green) - 1  from your gci script :contentReference[oaicite:9]{index=9}
            idx = (nir / (g + 1e-12)) - 1.0
        else:
            raise ValueError("index must be one of: ndvi, ciredge, gci")
        return idx

    # no wavelengths provided -> rely on explicit band indices
    if index == "ndvi":
        if red_idx is None or nir_idx is None:
            raise ValueError("ndvi requires red_idx and nir_idx when no wavelengths are provided.")
        red, nir = cube[:, :, red_idx], cube[:, :, nir_idx]
        return (nir - red) / (nir + red + 1e-12)
    elif index in ("ci_rededge", "ciredge", "ci-rededge"):
        if rededge_idx is None or nir_idx is None:
            raise ValueError("ciredge requires rededge_idx and nir_idx when no wavelengths are provided.")
        re, nir = cube[:, :, rededge_idx], cube[:, :, nir_idx]
        return (nir / (re + 1e-12)) - 1.0
    elif index == "gci":
        if green_idx is None or nir_idx is None:
            raise ValueError("gci requires green_idx and nir_idx when no wavelengths are provided.")
        g, nir = cube[:, :, green_idx], cube[:, :, nir_idx]
        return (nir / (g + 1e-12)) - 1.0
    else:
        raise ValueError("index must be one of: ndvi, ciredge, gci")

def _make_mask(
    idx_img: np.ndarray,
    *,
    mode: str,               # "auto" (Otsu) or "manual"
    threshold: Optional[float] = None,
    min_area: int = 100
) -> np.ndarray:
    """
    Return a binary mask where leaf pixels are True.
    - 'auto': compute Otsu threshold on the index image (fast, robust).
    - 'manual': use provided numeric 'threshold' (e.g., NDVI > 0.45 like your auto script). :contentReference[oaicite:10]{index=10}
    """
    img = idx_img.astype(np.float32)
    if mode == "auto":
        # OpenCV Otsu on a normalized 8-bit copy
        norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, bin_img = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        if threshold is None:
            raise ValueError("manual mode requires 'threshold'.")
        bin_img = (img > float(threshold)).astype(np.uint8) * 255

    # clean-up small specks; fill small holes
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)

    # contours
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bin_img, dtype=np.uint8)
    for c in contours:
        if cv2.contourArea(c) >= float(min_area):
            cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    return mask.astype(bool)

def _save_envi(cube: np.ndarray, out_stem: str, wavelengths: Optional[np.ndarray] = None):
    meta = {
        "description": "Clipped leaf hypercube",
        "samples": cube.shape[1],
        "lines": cube.shape[0],
        "bands": cube.shape[2],
        "data type": 4,   # float32
        "interleave": "bil",
    }
    if wavelengths is not None:
        meta["wavelength"] = wavelengths.tolist()
    envi.save_image(out_stem + ".hdr", cube.astype(np.float32), metadata=meta)

def _clip_regions(
    cube: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str,           # "square" or "tight"
    size: int = 30,      # used for square
    pad: int = 0         # optional padding for tight
) -> List[Tuple[int, int, int, int]]:
    """
    Return list of bounding boxes (x, y, w, h) for each leaf region.
    - 'square': make a size×size box centered on each contour’s center.
    - 'tight':   use the contour’s bounding rect with optional padding.
    """
    # contours on mask
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape
    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if mode == "square":
            cx, cy = x + w // 2, y + h // 2
            half = size // 2
            xs, ys = max(cx - half, 0), max(cy - half, 0)
            xe, ye = min(cx + half, W), min(cy + half, H)
            # ensure exact size with clipping
            w2, h2 = xe - xs, ye - ys
            # pad if needed
            if w2 < size or h2 < size:
                pad_x = max(0, size - w2)
                pad_y = max(0, size - h2)
                xs = max(xs - pad_x // 2, 0); xe = min(xs + size, W)
                ys = max(ys - pad_y // 2, 0); ye = min(ys + size, H)
            boxes.append((xs, ys, min(size, W - xs), min(size, H - ys)))
        else:  # tight
            xs = max(x - pad, 0); ys = max(y - pad, 0)
            xe = min(x + w + pad, W); ye = min(y + h + pad, H)
            boxes.append((xs, ys, xe - xs, ye - ys))
    return boxes

# ---------- public API ----------
def clip_sample(
    sample_mat_stem: str,
    *,
    index: str,                        # "ndvi", "ciredge", "gci"
    wavelengths_mat: Optional[str] = None,
    wavelengths_csv: Optional[str] = None,
    threshold_mode: str = "auto",      # "auto" or "manual"
    threshold_value: Optional[float] = None,
    min_area: int = 100,
    crop_mode: str = "square",         # "square" or "tight"
    crop_size: int = 30,               # for square
    pad: int = 0,                      # for tight
    outdir: Optional[str] = None
) -> List[str]:
    """
    Clip one calibrated sample saved as <stem>_R.mat (and optionally F in <stem>_F.mat).
    Saves each clip as ENVI .hdr/.img under 'clipped_hypercubes' (or --outdir).
    Returns list of saved file stems.
    """
    # load calibrated reflectance cube from <stem>_R.mat (key 'R_plant' created earlier)
    from scipy.io import loadmat
    R_path = sample_mat_stem + "_R.mat"
    if not os.path.exists(R_path):
        raise FileNotFoundError(f"Missing calibrated MAT: {R_path}")

    m = loadmat(R_path)
    if "R_plant" not in m:
        raise KeyError("MAT file must contain key 'R_plant'.")
    cube = m["R_plant"]  # (H, W, B) float32

    # wavelengths (optional but preferred)
    wavelengths = None
    if wavelengths_mat:
        wavelengths = _load_wavelengths_from_mat(wavelengths_mat)
    elif wavelengths_csv:
        wavelengths = _load_wavelengths_from_csv(wavelengths_csv)

    # index image
    idx_img = _compute_index(cube, index=index, wavelengths=wavelengths)

    # mask (auto Otsu or manual threshold like NDVI>0.45 from your auto script) :contentReference[oaicite:11]{index=11}
    mask = _make_mask(
        idx_img,
        mode=("auto" if threshold_mode == "auto" else "manual"),
        threshold=(None if threshold_mode == "auto" else float(threshold_value)),
        min_area=int(min_area),
    )

    # bounding boxes
    boxes = _clip_regions(cube, mask, mode=crop_mode, size=int(crop_size), pad=int(pad))

    # output dir
    if outdir is None:
        outdir = os.path.join(os.path.dirname(R_path), "clipped_hypercubes")
    _ensure_outdir(outdir)

    # save clips
    saved = []
    base_name = os.path.basename(sample_mat_stem)
    for i, (x, y, w, h) in enumerate(boxes, 1):
        clip = cube[y:y+h, x:x+w, :]
        stem = os.path.join(outdir, f"{base_name}_leaf{i}")
        _save_envi(clip, stem, wavelengths=wavelengths)
        saved.append(stem)
    return saved

def clip_folder(
    folder: str,
    *,
    index: str,
    wavelengths_mat: Optional[str] = None,
    wavelengths_csv: Optional[str] = None,
    threshold_mode: str = "auto",
    threshold_value: Optional[float] = None,
    min_area: int = 100,
    crop_mode: str = "square",
    crop_size: int = 30,
    pad: int = 0,
    outdir: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Find all <stem>_R.mat in a folder and produce clipped hypercubes for each.
    Returns dict: {<stem>: [saved_stems...], ...}
    """
    stems = []
    for p in sorted(glob.glob(os.path.join(folder, "*_R.mat"))):
        stems.append(p[:-6])  # drop "_R.mat"

    out: Dict[str, List[str]] = {}
    for stem in stems:
        out[stem] = clip_sample(
            stem,
            index=index,
            wavelengths_mat=wavelengths_mat,
            wavelengths_csv=wavelengths_csv,
            threshold_mode=threshold_mode,
            threshold_value=threshold_value,
            min_area=min_area,
            crop_mode=crop_mode,
            crop_size=crop_size,
            pad=pad,
            outdir=outdir,
        )
    return out

# ---------- CLI ----------
def main(argv: Optional[List[str]] = None):
    import argparse, json
    p = argparse.ArgumentParser(prog="mvos-hsi-clip", description="Clip calibrated HSI cubes into leaf hypercubes")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("clip", help="Clip one sample (expects <stem>_R.mat)")
    s1.add_argument("--stem", required=True, help="Base path without _R.mat/_F.mat (e.g., C:/data/Leaf)")
    s1.add_argument("--index", required=True, choices=["ndvi", "ciredge", "gci"], help="Index for mask")
    s1.add_argument("--wavelengths-mat", help="MAT file with 'wavelength' vector")
    s1.add_argument("--wavelengths-csv", help="CSV file with wavelength per row")
    s1.add_argument("--threshold-mode", default="auto", choices=["auto", "manual"])
    s1.add_argument("--threshold-value", type=float, help="If manual, e.g., 0.45 for NDVI")
    s1.add_argument("--min-area", type=int, default=100)
    s1.add_argument("--crop-mode", default="square", choices=["square", "tight"])
    s1.add_argument("--crop-size", type=int, default=30)
    s1.add_argument("--pad", type=int, default=0)
    s1.add_argument("--outdir", help="Output directory (default: <folder>/clipped_hypercubes)")

    s2 = sub.add_parser("clip-folder", help="Clip all samples in a folder (expects many <stem>_R.mat)")
    s2.add_argument("--folder", required=True)
    s2.add_argument("--index", required=True, choices=["ndvi", "ciredge", "gci"])
    s2.add_argument("--wavelengths-mat")
    s2.add_argument("--wavelengths-csv")
    s2.add_argument("--threshold-mode", default="auto", choices=["auto", "manual"])
    s2.add_argument("--threshold-value", type=float)
    s2.add_argument("--min-area", type=int, default=100)
    s2.add_argument("--crop-mode", default="square", choices=["square", "tight"])
    s2.add_argument("--crop-size", type=int, default=30)
    s2.add_argument("--pad", type=int, default=0)
    s2.add_argument("--outdir")

    args = p.parse_args(argv)

    if args.cmd == "clip":
        saved = clip_sample(
            args.stem,
            index=args.index,
            wavelengths_mat=args.wavelengths_mat,
            wavelengths_csv=args.wavelengths_csv,
            threshold_mode=args.threshold_mode,
            threshold_value=args.threshold_value,
            min_area=args.min_area,
            crop_mode=args.crop_mode,
            crop_size=args.crop_size,
            pad=args.pad,
            outdir=args.outdir,
        )
        print(json.dumps({"saved": saved}, indent=2))
    else:
        res = clip_folder(
            args.folder,
            index=args.index,
            wavelengths_mat=args.wavelengths_mat,
            wavelengths_csv=args.wavelengths_csv,
            threshold_mode=args.threshold_mode,
            threshold_value=args.threshold_value,
            min_area=args.min_area,
            crop_mode=args.crop_mode,
            crop_size=args.crop_size,
            pad=args.pad,
            outdir=args.outdir,
        )
        print(json.dumps({k: v for k, v in res.items()}, indent=2))
