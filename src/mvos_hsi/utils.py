from __future__ import annotations

import os
import csv
from typing import Optional, Dict, List

import numpy as np
import scipy.io as sio


def ensure_outdir(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    If `path` looks like a file (has an extension), we create the parent dir.
    If `path` looks like a directory, we create that directory.
    """
    if not path:
        return

    # If it has an extension, treat as file path
    if os.path.splitext(path)[1]:
        d = os.path.dirname(path)
    else:
        d = path

    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def load_wavelengths(
    *,
    wavelengths_mat: Optional[str] = None,
    wavelengths_csv: Optional[str] = None,
    envi_header_meta: Optional[Dict] = None,
) -> Optional[np.ndarray]:
    """
    Load wavelengths from:
      1) MAT file containing key 'wavelength' (preferred)
      2) CSV with one wavelength per row (or a column containing 'wavelength')
      3) ENVI header metadata 'wavelength' field (if provided)

    Returns
    -------
    np.ndarray or None
        Float array of shape (bands,), or None if nothing found.

    Raises
    ------
    ValueError if a MAT/CSV file is given but wavelengths cannot be parsed.
    """
    # 1) MAT file
    if wavelengths_mat:
        if not os.path.exists(wavelengths_mat):
            raise FileNotFoundError(wavelengths_mat)
        m = sio.loadmat(wavelengths_mat)
        wl = m.get("wavelength", None)
        if wl is None:
            raise ValueError(f"No 'wavelength' key in {wavelengths_mat}")
        return np.asarray(wl).ravel().astype(float)

    # 2) CSV file
    if wavelengths_csv:
        if not os.path.exists(wavelengths_csv):
            raise FileNotFoundError(wavelengths_csv)
        vals: List[float] = []
        with open(wavelengths_csv, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and any("wavelength" in h.lower() for h in header):
                idx = [i for i, h in enumerate(header) if "wavelength" in h.lower()][0]
                for row in reader:
                    if row and row[idx]:
                        vals.append(float(row[idx]))
            else:
                # no header or unknown -> treat first column as numeric
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

    # 3) ENVI metadata
    if envi_header_meta and "wavelength" in envi_header_meta:
        wl = np.asarray(envi_header_meta["wavelength"], dtype=float).ravel()
        return wl

    # Nothing given
    return None
