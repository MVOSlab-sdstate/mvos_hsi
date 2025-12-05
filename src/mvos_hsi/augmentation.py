from __future__ import annotations

import os
import glob
from typing import Optional, List, Dict

import numpy as np
import scipy.io as sio
from spectral import envi, open_image
from imgaug import augmenters as iaa

from .utils import ensure_outdir


# -------- helpers --------
def _build_aug_seq(
    flip: bool = False,
    rotate: Optional[tuple] = None,
    shear: Optional[tuple] = None,
    scale: Optional[tuple] = None,
) -> iaa.Sequential:
    """
    Build augmentation pipeline based on user choices.

    Parameters
    ----------
    flip : bool
        Apply horizontal flip with p=0.5 if True.
    rotate : (min_deg, max_deg), optional
        Rotation range in degrees.
    shear : (min_deg, max_deg), optional
        Shear range in degrees.
    scale : (min_factor, max_factor), optional
        Scaling factors.

    Returns
    -------
    iaa.Sequential
        Composed augmentation sequence.
    """
    augs = []
    if flip:
        augs.append(iaa.Fliplr(0.5))
    if rotate is not None:
        augs.append(iaa.Affine(rotate=rotate))
    if shear is not None:
        augs.append(iaa.Affine(shear=shear))
    if scale is not None:
        augs.append(iaa.Affine(scale=scale))
    if not augs:
        raise ValueError("No augmentation selected. Please specify at least one.")
    return iaa.Sequential(augs)


def _augment_cube(cube: np.ndarray, aug_seq) -> np.ndarray:
    # imgaug expects (H,W,C)
    return aug_seq.augment_image(cube)


def _save_augmented_mat(cube: np.ndarray, out_path: str):
    sio.savemat(out_path, {"R_Leaf": cube})


def _save_augmented_envi(cube: np.ndarray, out_stem: str):
    meta = {
        "description": "Augmented hyperspectral cube",
        "samples": cube.shape[1],
        "lines":   cube.shape[0],
        "bands":   cube.shape[2],
        "data type": 4,  # float32
        "interleave": "bil",
    }
    envi.save_image(out_stem + ".hdr", cube.astype(np.float32), metadata=meta)


# -------- core functions --------
def augment_image(
    path: str,
    num_aug: int,
    outdir: Optional[str] = None,
    *,
    flip: bool = False,
    rotate: Optional[tuple] = None,
    shear: Optional[tuple] = None,
    scale: Optional[tuple] = None,
) -> List[str]:
    """
    Augment one hyperspectral cube (.mat or ENVI .hdr).

    Parameters
    ----------
    path : str
        Path to .mat or .hdr file.
    num_aug : int
        Number of augmented variants to create.
    outdir : str, optional
        Output directory (default: <folder>/augmented_hypercubes).
    flip, rotate, shear, scale :
        Augmentation options.

    Returns
    -------
    list of str
        List of saved stems (without extension).
    """
    aug_seq = _build_aug_seq(flip=flip, rotate=rotate, shear=shear, scale=scale)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if outdir is None:
        outdir = os.path.join(os.path.dirname(path), "augmented_hypercubes")
    ensure_outdir(outdir)

    base = os.path.splitext(os.path.basename(path))[0]

    # load cube
    if path.lower().endswith(".mat"):
        m = sio.loadmat(path)
        if "R_plant" in m:
            cube = m["R_plant"]
        elif "R_Leaf" in m:
            cube = m["R_Leaf"]
        else:
            raise KeyError("MAT must contain R_plant or R_Leaf.")
        save_fn = lambda arr, stem: _save_augmented_mat(arr, os.path.join(outdir, stem + ".mat"))
    elif path.lower().endswith(".hdr"):
        cube = open_image(path).load().astype(np.float32)
        save_fn = lambda arr, stem: _save_augmented_envi(arr, os.path.join(outdir, stem))
    else:
        raise ValueError("Unsupported file type: must be .mat or .hdr")

    saved: List[str] = []
    for i in range(num_aug):
        aug = _augment_cube(cube.copy(), aug_seq)
        stem = f"{base}_aug{i+1}"
        save_fn(aug, stem)
        saved.append(stem)
    return saved


def augment_folder(
    folder: str,
    num_aug: int,
    outdir: Optional[str] = None,
    **kwargs,
) -> Dict[str, List[str]]:
    """
    Augment all .mat/.hdr cubes in a folder with chosen augmentations.

    Parameters
    ----------
    folder : str
        Folder containing .mat/.hdr cubes.
    num_aug : int
        Number of augmented variants to create per image.
    outdir : str, optional
        Output folder (default: <folder>/augmented_hypercubes).
    kwargs :
        Passed through to `augment_image` (flip, rotate, shear, scale).

    Returns
    -------
    dict
        Mapping from input file path to list of saved stems.
    """
    files = glob.glob(os.path.join(folder, "*.mat")) + glob.glob(os.path.join(folder, "*.hdr"))
    results: Dict[str, List[str]] = {}
    for f in files:
        results[f] = augment_image(f, num_aug, outdir=outdir, **kwargs)
    return results


def augment_classes(
    root: str,
    num_aug: int,
    outdir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Augment each class subfolder separately (root/classX/*.mat or *.hdr).

    Parameters
    ----------
    root : str
        Root folder, with subfolders for each class.
    num_aug : int
        Number of augmentations per image.
    outdir : str, optional
        If given, all class outputs go here; otherwise each class gets its own
        `augmented_hypercubes` subfolder.
    kwargs :
        Passed through to `augment_folder`.

    Returns
    -------
    dict
        Mapping class name -> (mapping file -> list of saved stems).
    """
    results: Dict[str, Dict[str, List[str]]] = {}
    for sub in sorted(os.listdir(root)):
        subpath = os.path.join(root, sub)
        if os.path.isdir(subpath):
            results[sub] = augment_folder(
                subpath,
                num_aug,
                outdir=(outdir or os.path.join(subpath, "augmented_hypercubes")),
                **kwargs,
            )
    return results


# -------- CLI --------
def main(argv: Optional[List[str]] = None):
    """
    CLI entry for augmentation (legacy). For unified CLI, use `mvos-hsi augmentation ...`.
    """
    import argparse
    import json

    p = argparse.ArgumentParser(prog="mvos-hsi-augment", description="Augment hyperspectral cubes")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_aug_flags(sp):
        sp.add_argument("--flip", action="store_true", help="Enable horizontal flip (p=0.5)")
        sp.add_argument("--rotate", nargs=2, type=int, help="Rotation range, e.g. --rotate -10 10")
        sp.add_argument("--shear", nargs=2, type=int, help="Shear range, e.g. --shear -16 16")
        sp.add_argument("--scale", nargs=2, type=float, help="Scale range, e.g. --scale 0.9 1.1")

    s1 = sub.add_parser("image", help="Augment one image")
    s1.add_argument("--path", required=True)
    s1.add_argument("--num", type=int, required=True)
    s1.add_argument("--outdir")
    add_aug_flags(s1)

    s2 = sub.add_parser("folder", help="Augment all cubes in a folder")
    s2.add_argument("--folder", required=True)
    s2.add_argument("--num", type=int, required=True)
    s2.add_argument("--outdir")
    add_aug_flags(s2)

    s3 = sub.add_parser("classes", help="Augment each class subfolder")
    s3.add_argument("--root", required=True)
    s3.add_argument("--num", type=int, required=True)
    s3.add_argument("--outdir")
    add_aug_flags(s3)

    args = p.parse_args(argv)

    kwargs = {
        "flip": args.flip,
        "rotate": tuple(args.rotate) if args.rotate else None,
        "shear": tuple(args.shear) if args.shear else None,
        "scale": tuple(args.scale) if args.scale else None,
    }

    if args.cmd == "image":
        res = augment_image(args.path, args.num, outdir=args.outdir, **kwargs)
        print(json.dumps({"saved": res}, indent=2))
    elif args.cmd == "folder":
        res = augment_folder(args.folder, args.num, outdir=args.outdir, **kwargs)
        print(json.dumps(res, indent=2))
    elif args.cmd == "classes":
        res = augment_classes(args.root, args.num, outdir=args.outdir, **kwargs)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
