from __future__ import annotations
import os, csv
from typing import Optional, Tuple, List, Dict, Sequence
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from spectral import open_image

# -------------------- wavelength helpers --------------------

def load_wavelengths(
    *,
    wavelengths_mat: Optional[str] = None,
    wavelengths_csv: Optional[str] = None,
    envi_header_meta: Optional[Dict] = None
) -> Optional[np.ndarray]:
    """
    Load wavelengths from:
      1) a MAT file containing key 'wavelength' (preferred)
      2) a CSV with one wavelength per row (or a 'Wavelength (nm)' column)
      3) ENVI header 'wavelength' field if present in metadata

    Returns float array of shape (bands,) or None if unavailable.
    """
    # 1) MAT
    if wavelengths_mat:
        m = sio.loadmat(wavelengths_mat)
        wl = m.get("wavelength", None)
        if wl is None:
            raise ValueError(f"No 'wavelength' key in {wavelengths_mat}")
        return np.asarray(wl).ravel().astype(float)

    # 2) CSV
    if wavelengths_csv:
        vals = []
        with open(wavelengths_csv, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and any("wavelength" in h.lower() for h in header):
                idx = [i for i, h in enumerate(header) if "wavelength" in h.lower()][0]
                for row in reader:
                    if row and row[idx]:
                        vals.append(float(row[idx]))
            else:
                if header:
                    try:
                        vals.append(float(header[0]))
                    except Exception:
                        pass
                for row in reader:
                    if row and row[0]:
                        vals.append(float(row[0]))
        if not vals:
            raise ValueError(f"No numeric wavelengths found in {wavelengths_csv}")
        return np.asarray(vals, dtype=float)

    # 3) ENVI header
    if envi_header_meta and "wavelength" in envi_header_meta:
        wl = np.asarray(envi_header_meta["wavelength"], dtype=float).ravel()
        return wl

    return None

# -------------------- cube loaders --------------------

def _load_cube_any(path: str, key: Optional[str]) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Load a hyperspectral cube from:
      - Calibrated .mat -> keys: R_plant (default), F_plant, R_Leaf, etc.
      - ENVI .hdr -> return data + header metadata
    Returns: (cube: float32 [H,W,B], meta: dict or None)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        m = sio.loadmat(path)
        pick_order = [key] if key else ["R_plant", "F_plant", "R_Leaf", "F_Leaf", "cube", "data"]
        chosen = None
        for k in pick_order:
            if k and k in m:
                chosen = m[k]; break
        if chosen is None:
            raise KeyError(f"No suitable cube key found in {path}. "
                           f"Tried {pick_order}. Pass --key to specify.")
        arr = np.asarray(chosen, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Cube in {path} must be 3D (H,W,B). Got shape {arr.shape}")
        return arr, None

    elif ext == ".hdr":
        ds = open_image(path)
        cube = np.asarray(ds.load(), dtype=np.float32)
        meta = getattr(ds, "metadata", {}) or {}
        return cube, meta

    else:
        raise ValueError("Unsupported file type (use .mat or .hdr)")

# -------------------- spectrum extraction --------------------

def _spectrum_pixel(cube: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w, b = cube.shape
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"Pixel (x={x}, y={y}) out of bounds for cube {w}x{h}")
    return cube[y, x, :].astype(np.float32)

def _spectrum_roi(cube: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    h, w, b = cube.shape
    xs, xe = sorted((max(0, x0), min(w, x1)))
    ys, ye = sorted((max(0, y0), min(h, y1)))
    if xs == xe or ys == ye:
        raise ValueError("Empty ROI after clipping.")
    roi = cube[ys:ye, xs:xe, :]
    return roi.reshape(-1, roi.shape[-1]).mean(axis=0).astype(np.float32)

def _center_xy(cube: np.ndarray) -> Tuple[int, int]:
    h, w, _ = cube.shape
    return w // 2, h // 2

# -------------------- plotting (API) --------------------

def plot_pixel(
    image: str,
    *,
    x: int,
    y: int,
    key: Optional[str] = None,
    wavelengths_mat: Optional[str] = None,
    wavelengths_csv: Optional[str] = None,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    save: Optional[str] = None,
    show: bool = True
):
    """Plot the spectrum at a single pixel (x,y) from an image (.mat or .hdr)."""
    cube, meta = _load_cube_any(image, key)
    wl = load_wavelengths(wavelengths_mat=wavelengths_mat, wavelengths_csv=wavelengths_csv, envi_header_meta=meta)
    spec = _spectrum_pixel(cube, x, y)

    plt.figure(figsize=(9, 5))
    if wl is None:
        plt.plot(np.arange(spec.size), spec, label=f"{os.path.basename(image)} @ ({x},{y})")
        plt.xlabel("Band index")
    else:
        plt.plot(wl, spec, label=f"{os.path.basename(image)} @ ({x},{y})")
        plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance / Intensity")
    if ylim: plt.ylim(*ylim)
    plt.title(title or "Spectral profile (pixel)")
    plt.legend(); plt.tight_layout()
    if save: plt.savefig(save, dpi=300)
    if show: plt.show()
    plt.close()

def plot_roi(
    image: str,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    key: Optional[str] = None,
    wavelengths_mat: Optional[str] = None,
    wavelengths_csv: Optional[str] = None,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    save: Optional[str] = None,
    show: bool = True
):
    """Plot the mean spectrum over a rectangular ROI [x0:x1, y0:y1]."""
    cube, meta = _load_cube_any(image, key)
    wl = load_wavelengths(wavelengths_mat=wavelengths_mat, wavelengths_csv=wavelengths_csv, envi_header_meta=meta)
    spec = _spectrum_roi(cube, x0, y0, x1, y1)

    plt.figure(figsize=(9, 5))
    if wl is None:
        plt.plot(np.arange(spec.size), spec, label=f"{os.path.basename(image)} ROI")
        plt.xlabel("Band index")
    else:
        plt.plot(wl, spec, label=f"{os.path.basename(image)} ROI")
        plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance / Intensity")
    if ylim: plt.ylim(*ylim)
    plt.title(title or "Spectral profile (ROI mean)")
    plt.legend(); plt.tight_layout()
    if save: plt.savefig(save, dpi=300)
    if show: plt.show()
    plt.close()

def plot_leaf_center(
    clipped_dir: str,
    *,
    stem: str,
    leaves: Sequence[int],
    wavelengths_mat: Optional[str] = None,
    wavelengths_csv: Optional[str] = None,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    save: Optional[str] = None,
    show: bool = True
):
    """
    Plot the spectrum from the CENTER PIXEL of one or more clipped leaves.
    Looks for files like: <clipped_dir>/<stem>_leaf<N>.hdr
    """
    curves = []
    labels = []
    wl_global = None

    for n in leaves:
        leaf_hdr = os.path.join(clipped_dir, f"{stem}_leaf{int(n)}.hdr")
        if not os.path.exists(leaf_hdr):
            raise FileNotFoundError(f"Missing clipped leaf: {leaf_hdr}")

        ds = open_image(leaf_hdr)
        cube = np.asarray(ds.load(), dtype=np.float32)
        meta = getattr(ds, "metadata", {}) or {}
        wl = load_wavelengths(wavelengths_mat=wavelengths_mat, wavelengths_csv=wavelengths_csv, envi_header_meta=meta)
        if wl_global is None:
            wl_global = wl

        cx, cy = _center_xy(cube)
        spec = _spectrum_pixel(cube, cx, cy)
        curves.append(spec)
        labels.append(f"{stem}_leaf{int(n)} @ center ({cx},{cy})")

    plt.figure(figsize=(10, 6))
    for spec, lab in zip(curves, labels):
        if wl_global is None:
            plt.plot(np.arange(spec.size), spec, label=lab)
            xlabel = "Band index"
        else:
            plt.plot(wl_global, spec, label=lab)
            xlabel = "Wavelength (nm)"
    plt.xlabel(xlabel)
    plt.ylabel("Reflectance / Intensity")
    if ylim: plt.ylim(*ylim)
    plt.title(title or "Spectral profile (center pixel of clipped leaf)")
    plt.legend()
    plt.tight_layout()
    if save: plt.savefig(save, dpi=300)
    if show: plt.show()
    plt.close()

def plot_leaf_centers_multi(
    clipped_dir: str,
    items: List[Tuple[str, int]],   # [("H_P1_V4_B", 1), ("H_P1_V6_B", 3), ...]
    *,
    wavelengths_mat: Optional[str] = None,
    wavelengths_csv: Optional[str] = None,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Plot spectra from the CENTER pixel of multiple (stem, leaf#) pairs on one figure.
    Looks for <clipped_dir>/<stem>_leaf<leaf>.hdr for each item.
    """
    curves, labels = [], []
    wl_global = None

    for stem, leaf in items:
        leaf_hdr = os.path.join(clipped_dir, f"{stem}_leaf{int(leaf)}.hdr")
        if not os.path.exists(leaf_hdr):
            raise FileNotFoundError(f"Missing clipped leaf: {leaf_hdr}")

        ds = open_image(leaf_hdr)
        cube = np.asarray(ds.load(), dtype=np.float32)
        meta = getattr(ds, "metadata", {}) or {}
        wl = load_wavelengths(wavelengths_mat=wavelengths_mat, wavelengths_csv=wavelengths_csv, envi_header_meta=meta)
        if wl_global is None:
            wl_global = wl

        w, h = cube.shape[1], cube.shape[0]
        cx, cy = w // 2, h // 2
        spec = cube[cy, cx, :].astype(np.float32)
        curves.append(spec)
        labels.append(f"{stem} - leaf{leaf} @ center ({cx},{cy})")

    plt.figure(figsize=(10, 6))
    for spec, lab in zip(curves, labels):
        if wl_global is None:
            plt.plot(np.arange(spec.size), spec, label=lab)
            xlabel = "Band index"
        else:
            plt.plot(wl_global, spec, label=lab)
            xlabel = "Wavelength (nm)"
    plt.xlabel(xlabel)
    plt.ylabel("Reflectance / Intensity")
    if ylim: plt.ylim(*ylim)
    plt.title(title or "Leaf center spectra (multi-stem)")
    plt.legend()
    plt.tight_layout()
    if save: plt.savefig(save, dpi=300)
    if show: plt.show()
    plt.close()

# -------------------- CLI --------------------

def main(argv: Optional[List[str]] = None):
    import argparse
    p = argparse.ArgumentParser(prog="mvos-hsi-plot", description="Plot spectral profiles from hyperspectral images")
    sub = p.add_subparsers(dest="cmd", required=True)

    # pixel
    s1 = sub.add_parser("pixel", help="Plot a spectrum at a pixel (x,y)")
    s1.add_argument("--image", required=True, help="Path to .mat (calibrated) or .hdr (ENVI)")
    s1.add_argument("--x", type=int, required=True)
    s1.add_argument("--y", type=int, required=True)
    s1.add_argument("--key", help="For .mat: data key (default tries R_plant/F_plant/R_Leaf/...)")
    s1.add_argument("--wavelengths-mat")
    s1.add_argument("--wavelengths-csv")
    s1.add_argument("--title")
    s1.add_argument("--ylim", nargs=2, type=float, help="ymin ymax")
    s1.add_argument("--save", help="Output PNG path")
    s1.add_argument("--no-show", action="store_true")

    # roi
    s2 = sub.add_parser("roi", help="Plot a spectrum averaged over a rectangle (x0,y0)-(x1,y1)")
    s2.add_argument("--image", required=True)
    s2.add_argument("--x0", type=int, required=True)
    s2.add_argument("--y0", type=int, required=True)
    s2.add_argument("--x1", type=int, required=True)
    s2.add_argument("--y1", type=int, required=True)
    s2.add_argument("--key")
    s2.add_argument("--wavelengths-mat")
    s2.add_argument("--wavelengths-csv")
    s2.add_argument("--title")
    s2.add_argument("--ylim", nargs=2, type=float)
    s2.add_argument("--save")
    s2.add_argument("--no-show", action="store_true")

    # leaf-center
    s3 = sub.add_parser("leaf", help="Plot spectrum from center pixel of clipped leaf(s)")
    s3.add_argument("--clipped-dir", required=True, help="Folder containing clipped hypercubes (ENVI)")
    s3.add_argument("--stem", required=True, help="Sample stem used during clipping (e.g., H_P1_V4_B)")
    s3.add_argument("--leaf", nargs="+", type=int, required=True, help="Leaf number(s), e.g., --leaf 1 3 5")
    s3.add_argument("--wavelengths-mat")
    s3.add_argument("--wavelengths-csv")
    s3.add_argument("--title")
    s3.add_argument("--ylim", nargs=2, type=float)
    s3.add_argument("--save")
    s3.add_argument("--no-show", action="store_true")

    # leaf-multi (different stems on one plot)
    s4 = sub.add_parser("leaf-multi", help="Plot center spectra from multiple (stem, leaf) pairs")
    s4.add_argument("--clipped-dir", required=True)
    s4.add_argument("--item", action="append", required=True,
                    help="Pair as STEM:LEAF (repeatable). Example: --item H_P1_V4_B:1 --item H_P1_V6_B:3")
    s4.add_argument("--wavelengths-mat")
    s4.add_argument("--wavelengths-csv")
    s4.add_argument("--title")
    s4.add_argument("--ylim", nargs=2, type=float)
    s4.add_argument("--save")
    s4.add_argument("--no-show", action="store_true")

    args = p.parse_args(argv)

    ylim = tuple(args.ylim) if getattr(args, "ylim", None) else None
    show = not getattr(args, "no_show", False)

    if args.cmd == "pixel":
        plot_pixel(
            args.image,
            x=args.x, y=args.y,
            key=args.key,
            wavelengths_mat=args.wavelengths_mat,
            wavelengths_csv=args.wavelengths_csv,
            title=args.title,
            ylim=ylim,
            save=args.save,
            show=show,
        )
    elif args.cmd == "roi":
        plot_roi(
            args.image,
            x0=args.x0, y0=args.y0, x1=args.x1, y1=args.y1,
            key=args.key,
            wavelengths_mat=args.wavelengths_mat,
            wavelengths_csv=args.wavelengths_csv,
            title=args.title,
            ylim=ylim,
            save=args.save,
            show=show,
        )
    elif args.cmd == "leaf":
        plot_leaf_center(
            args.clipped_dir,
            stem=args.stem,
            leaves=args.leaf,
            wavelengths_mat=args.wavelengths_mat,
            wavelengths_csv=args.wavelengths_csv,
            title=args.title,
            ylim=ylim,
            save=args.save,
            show=show,
        )
    elif args.cmd == "leaf-multi":
        items = []
        for s in args.item:
            stem, leaf = s.rsplit(":", 1)
            items.append((stem, int(leaf)))
        plot_leaf_centers_multi(
            args.clipped_dir,
            items,
            wavelengths_mat=args.wavelengths_mat,
            wavelengths_csv=args.wavelengths_csv,
            title=args.title,
            ylim=ylim,
            save=args.save,
            show=show,
        )
