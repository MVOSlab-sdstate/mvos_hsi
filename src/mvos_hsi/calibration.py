from __future__ import annotations
import os, glob, re
from typing import Dict, Optional, List
import numpy as np

from spectral import envi, io
from skimage.filters import threshold_multiotsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from scipy.io import savemat

# ----- camera / pipeline defaults (mirrors your MATLAB) -----
N_LINE_SAMPLE = 250
N_LINE_DARK   = 20
N_SAMPLE      = 810
N_BAND        = 348

DEFAULT_SPECTRAL_BIN = 3
DEFAULT_SPATIAL_BIN  = 3

# Index windows (MATLAB -> Python slices)
INDEX_WHITE   = slice(20, 40)      # 21:40
INDEX_IFOV    = slice(10, 260)     # 11:260 (columns)
INDEX_PLANT_R = slice(46, 246)     # 47:246 (rows)
INDEX_PLANT_F_SHIFT = -2

# Bands used for mask/visuals after calibration
B815 = 80    # reflectance mask band
B685 = 56    # example F band (not required in outputs)

# -------- helpers --------
def _check_factor(k: int, name: str):
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"{name} must be a positive integer (got {k}).")

def _bin_spectral(arr: np.ndarray, k: int) -> np.ndarray:
    b = arr.shape[-1]
    if b % k != 0:
        arr = arr[..., : (b // k) * k]
        b = arr.shape[-1]
    return arr.reshape(*arr.shape[:-1], b // k, k).mean(axis=-1)

def _bin_spatial_along_samples(arr: np.ndarray, k: int) -> np.ndarray:
    s = arr.shape[1]
    if s % k != 0:
        arr = arr[:, : (s // k) * k, :]
        s = arr.shape[1]
    return arr.reshape(arr.shape[0], s // k, k, arr.shape[2]).mean(axis=2)

def _rotate(arr: np.ndarray, k: int) -> np.ndarray:
    # numpy.rot90 CCW; MATLAB used +90 for R and -90 for F
    if k == 1:
        return np.rot90(arr, k=1, axes=(0, 1))
    elif k == -1:
        return np.rot90(arr, k=3, axes=(0, 1))
    raise ValueError("k must be 1 or -1.")

def _imquantize_mask(img: np.ndarray) -> np.ndarray:
    ths = threshold_multiotsu(img, classes=2)
    mask = (img >= ths[0])
    mask = remove_small_objects(mask, 100, connectivity=2)
    mask = binary_fill_holes(mask)
    return mask

def _extract_masked_spectra(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w, b = cube.shape
    flat = cube.reshape(h * w, b)
    flat_mask = mask.reshape(h * w)
    return flat[flat_mask]

# case-insensitive _R/_F + .hdr/.HDR + .img/.IMG
def _read_envi_cube(base: str, suffix: str) -> np.ndarray:
    hdr_candidates = [f"{base}_{suffix}.hdr", f"{base}_{suffix}.HDR"]
    for hdr in hdr_candidates:
        if os.path.exists(hdr):
            ds = envi.open(hdr)
            return np.asarray(ds.load())

    img_candidates = [f"{base}_{suffix}.img", f"{base}_{suffix}.IMG"]
    for img in img_candidates:
        if os.path.exists(img):
            shape = (N_LINE_SAMPLE if suffix.upper() == 'R' else N_LINE_DARK, N_SAMPLE, N_BAND)
            arr = io.bilopen(img, shape=shape, dtype=np.int16, byteorder='<').load()
            return np.asarray(arr)

    raise FileNotFoundError(f"Could not find any of {hdr_candidates + img_candidates}")

_BASE_RE = re.compile(r'_(R|F)\.(hdr|img)$', flags=re.IGNORECASE)
def _strip_base(path: str) -> str:
    return _BASE_RE.sub('', path)

# -------- core calibration --------
def _calibrate_pair(
    sample_base: str,
    dark_base: str,
    spectral_bin: int = DEFAULT_SPECTRAL_BIN,
    spatial_bin: int  = DEFAULT_SPATIAL_BIN,
) -> Dict[str, np.ndarray]:
    _check_factor(spectral_bin, "spectral_bin")
    _check_factor(spatial_bin,  "spatial_bin")

    # Read cubes
    IR_sample = _read_envi_cube(sample_base, "R")
    IF_sample = _read_envi_cube(sample_base, "F")
    IR_dark   = _read_envi_cube(dark_base,   "R")
    IF_dark   = _read_envi_cube(dark_base,   "F")

    # Binning
    IR_sample = _bin_spectral(IR_sample, spectral_bin)
    IF_sample = _bin_spectral(IF_sample, spectral_bin)
    IR_dark   = _bin_spectral(IR_dark,   spectral_bin)
    IF_dark   = _bin_spectral(IF_dark,   spectral_bin)

    IR_sample = _bin_spatial_along_samples(IR_sample, spatial_bin)
    IF_sample = _bin_spatial_along_samples(IF_sample, spatial_bin)
    IR_dark   = _bin_spatial_along_samples(IR_dark,   spatial_bin)
    IF_dark   = _bin_spatial_along_samples(IF_dark,   spatial_bin)

    # IFOV crop (cols)
    IR_sample = IR_sample[:, INDEX_IFOV, :]
    IF_sample = IF_sample[:, INDEX_IFOV, :]
    IR_dark   = IR_dark[:,   INDEX_IFOV, :]
    IF_dark   = IF_dark[:,   INDEX_IFOV, :]

    # dark averages over lines
    IR_dark_mean = IR_dark.mean(axis=0, keepdims=True).repeat(IR_sample.shape[0], axis=0)
    IF_dark_mean = IF_dark.mean(axis=0, keepdims=True).repeat(IF_sample.shape[0], axis=0)

    # white (lines 21–40) from sample R
    I_white = IR_sample[INDEX_WHITE, :, :].mean(axis=0, keepdims=True).repeat(IR_sample.shape[0], axis=0)

    # flat-field reflectance + fluorescence
    with np.errstate(divide='ignore', invalid='ignore'):
        R_sample = (IR_sample - IR_dark_mean) / (I_white - IR_dark_mean)
        R_sample = np.nan_to_num(R_sample, nan=0.0, posinf=0.0, neginf=0.0)
    F_sample = (IF_sample - IF_dark_mean)

    # rotations
    R_sample_rot = _rotate(R_sample, k=1)
    F_sample_rot = _rotate(F_sample, k=-1)

    # plant crops
    R_plant = R_sample_rot[:, INDEX_PLANT_R, :]
    idx_f_start = INDEX_PLANT_R.start + INDEX_PLANT_F_SHIFT
    idx_f_stop  = INDEX_PLANT_R.stop  + INDEX_PLANT_F_SHIFT
    F_plant = F_sample_rot[:, slice(idx_f_start, idx_f_stop), :]

    # mask from R@815
    R815 = R_plant[:, :, B815]
    M_plant = _imquantize_mask(R815)

    # abnormal reflectance removal
    bad = (R_plant.min(axis=2) <= 0) | (R815 < 0.2)
    R_mask = M_plant & (~bad)
    F_mask = M_plant  # keep same as MATLAB-style (F abnormal was commented)

    # spectra
    RS_plant = _extract_masked_spectra(R_plant, R_mask)
    FS_plant = _extract_masked_spectra(F_plant, F_mask)

    return {
        "R_plant": R_plant.astype(np.float32),
        "F_plant": F_plant.astype(np.float32),
        "M_plant": M_plant.astype(bool),
        "RS_plant": RS_plant.astype(np.float32),
        "FS_plant": FS_plant.astype(np.float32),
    }

# -------- public api --------
def calibrate(
    sample_base: str,
    dark_base: str,
    spectral_bin: int = DEFAULT_SPECTRAL_BIN,
    spatial_bin: int  = DEFAULT_SPATIAL_BIN,
) -> Dict[str, np.ndarray]:
    return _calibrate_pair(sample_base, dark_base, spectral_bin, spatial_bin)

def calibrate_folder(
    folder: str,
    dark_base: str,
    spectral_bin: int = DEFAULT_SPECTRAL_BIN,
    spatial_bin: int  = DEFAULT_SPATIAL_BIN,
    pattern: str = "*_R.*",
) -> List[Dict[str, np.ndarray]]:
    _check_factor(spectral_bin, "spectral_bin")
    _check_factor(spatial_bin,  "spatial_bin")

    # gather R candidates, case-insensitive
    candidates = set()
    for p in glob.glob(os.path.join(folder, "*_R.*")): candidates.add(p)
    for p in glob.glob(os.path.join(folder, "*_r.*")): candidates.add(p)

    norm_dark = os.path.normcase(_strip_base(dark_base))
    results = []

    for rfile in sorted(candidates):
        base = _strip_base(rfile)
        if os.path.normcase(base) == norm_dark:
            continue  # skip dark

        out = _calibrate_pair(base, dark_base, spectral_bin, spatial_bin)

        # save calibrated cubes
        savemat(base + "_R.mat", {"R_plant": out["R_plant"]})
        savemat(base + "_F.mat", {"F_plant": out["F_plant"]})

        results.append(out)
    return results

# -------- cli --------
def main(argv: Optional[List[str]] = None):
    import argparse
    p = argparse.ArgumentParser(prog="mvos-hsi", description="MVOS HSI calibration")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("calibrate", help="Calibrate one sample")
    s1.add_argument("--sample", required=True, help="base path without _R/_F")
    s1.add_argument("--dark",   required=True, help="dark base without _R/_F")
    s1.add_argument("--k", type=int, default=DEFAULT_SPECTRAL_BIN, help="spectral bin factor (>=1)")
    s1.add_argument("--spatial", type=int, default=DEFAULT_SPATIAL_BIN, help="spatial bin factor (>=1)")

    s2 = sub.add_parser("calibrate-folder", help="Calibrate all samples in a folder")
    s2.add_argument("--folder", required=True)
    s2.add_argument("--dark",   required=True)
    s2.add_argument("--k", type=int, default=DEFAULT_SPECTRAL_BIN)
    s2.add_argument("--spatial", type=int, default=DEFAULT_SPATIAL_BIN)
    s2.add_argument("--pattern", default="*_R.*", help="glob to find R files (case-insensitive handled)")

    args = p.parse_args(argv)

    if args.cmd == "calibrate":
        if args.k == 0:
            raise ValueError("spectral bin 0 is not allowed.")
        out = calibrate(args.sample, args.dark, spectral_bin=args.k, spatial_bin=args.spatial)
        # save outputs like folder mode
        savemat(args.sample + "_R.mat", {"R_plant": out["R_plant"]})
        savemat(args.sample + "_F.mat", {"F_plant": out["F_plant"]})
        print(f"Calibrated and saved: {args.sample}_R.mat, {args.sample}_F.mat")

    elif args.cmd == "calibrate-folder":
        if args.k == 0:
            raise ValueError("spectral bin 0 is not allowed.")
        outs = calibrate_folder(args.folder, args.dark, spectral_bin=args.k, spatial_bin=args.spatial, pattern=args.pattern)
        print(f"Calibrated {len(outs)} sample(s) and saved *_R.mat / *_F.mat")
