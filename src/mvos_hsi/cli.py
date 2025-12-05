from __future__ import annotations

import argparse
import sys
from typing import Optional, List

from . import calibration as _cal
from . import clipping as _clip
from . import augmentation as _aug
from . import plotting as _plot


def _add_calibration(sub):
    p = sub.add_parser("calibration", help="Calibrate raw ENVI cubes")
    sp = p.add_subparsers(dest="mode", required=True)

    s1 = sp.add_parser("single", help="Calibrate a single sample")
    s1.add_argument("--sample", required=True)
    s1.add_argument("--dark", required=True)
    s1.add_argument("--k", type=int, default=_cal.DEFAULT_SPECTRAL_BIN, help="Spectral bin factor (>=1)")
    s1.add_argument("--spatial", type=int, default=_cal.DEFAULT_SPATIAL_BIN, help="Spatial bin factor (>=1)")
    s1.set_defaults(
        func=lambda a: _cal.calibrate(
            a.sample,
            a.dark,
            spectral_bin=a.k,
            spatial_bin=a.spatial,
        )
    )

    s2 = sp.add_parser("folder", help="Calibrate a whole folder")
    s2.add_argument("--folder", required=True)
    s2.add_argument("--dark", required=True)
    s2.add_argument("--k", type=int, default=_cal.DEFAULT_SPECTRAL_BIN)
    s2.add_argument("--spatial", type=int, default=_cal.DEFAULT_SPATIAL_BIN)
    s2.set_defaults(
        func=lambda a: _cal.calibrate_folder(
            a.folder,
            dark_base=a.dark,
            spectral_bin=a.k,
            spatial_bin=a.spatial,
        )
    )


def _add_clipping(sub):
    p = sub.add_parser("clipping", help="Clip leaves using NDVI/CI-RedEdge/GCI")
    sp = p.add_subparsers(dest="mode", required=True)

    s1 = sp.add_parser("folder", help="Clip all calibrated samples in a folder")
    s1.add_argument("--folder", required=True)
    s1.add_argument("--index", choices=["ndvi", "ciredge", "gci"], default="ndvi")
    s1.add_argument("--wavelengths-mat")
    s1.add_argument("--wavelengths-csv")
    s1.add_argument("--threshold-mode", choices=["auto", "manual"], default="auto")
    s1.add_argument("--threshold-value", type=float)
    s1.add_argument("--min-area", type=int, default=100)
    s1.add_argument("--crop-mode", choices=["square", "tight"], default="square")
    s1.add_argument("--crop-size", type=int, default=30)
    s1.add_argument("--pad", type=int, default=0)
    s1.add_argument("--outdir")
    s1.set_defaults(
        func=lambda a: _clip.clip_folder(
            folder=a.folder,
            index=a.index,
            wavelengths_mat=a.wavelengths_mat,
            wavelengths_csv=a.wavelengths_csv,
            threshold_mode=a.threshold_mode,
            threshold_value=a.threshold_value,
            min_area=a.min_area,
            crop_mode=a.crop_mode,
            crop_size=a.crop_size,
            pad=a.pad,
            outdir=a.outdir,
        )
    )


def _add_augmentation(sub):
    p = sub.add_parser("augmentation", help="Augment clipped cubes")
    sp = p.add_subparsers(dest="mode", required=True)

    def flags(spx):
        spx.add_argument("--flip", action="store_true")
        spx.add_argument("--rotate", nargs=2, type=int)
        spx.add_argument("--shear", nargs=2, type=int)
        spx.add_argument("--scale", nargs=2, type=float)

    s1 = sp.add_parser("image", help="Augment a single image (.mat or .hdr)")
    s1.add_argument("--path", required=True)
    s1.add_argument("--num", type=int, default=3)
    s1.add_argument("--outdir")
    flags(s1)
    s1.set_defaults(
        func=lambda a: _aug.augment_image(
            a.path,
            a.num,
            outdir=a.outdir,
            flip=a.flip,
            rotate=tuple(a.rotate) if a.rotate else None,
            shear=tuple(a.shear) if a.shear else None,
            scale=tuple(a.scale) if a.scale else None,
        )
    )

    s2 = sp.add_parser("folder", help="Augment all cubes in a folder")
    s2.add_argument("--folder", required=True)
    s2.add_argument("--num", type=int, default=3)
    s2.add_argument("--outdir")
    flags(s2)
    s2.set_defaults(
        func=lambda a: _aug.augment_folder(
            a.folder,
            a.num,
            outdir=a.outdir,
            flip=a.flip,
            rotate=tuple(a.rotate) if a.rotate else None,
            shear=tuple(a.shear) if a.shear else None,
            scale=tuple(a.scale) if a.scale else None,
        )
    )

    s3 = sp.add_parser("classes", help="Augment each class subfolder under root")
    s3.add_argument("--root", required=True)
    s3.add_argument("--num", type=int, default=3)
    s3.add_argument("--outdir")
    flags(s3)
    s3.set_defaults(
        func=lambda a: _aug.augment_classes(
            a.root,
            a.num,
            outdir=a.outdir,
            flip=a.flip,
            rotate=tuple(a.rotate) if a.rotate else None,
            shear=tuple(a.shear) if a.shear else None,
            scale=tuple(a.scale) if a.scale else None,
        )
    )


def _add_plotting(sub):
    p = sub.add_parser("plotting", help="Plot spectral profiles")
    sp = p.add_subparsers(dest="mode", required=True)

    # leaf center(s), single stem
    s1 = sp.add_parser("leaf", help="Leaf center spectrum(s) for one stem")
    s1.add_argument("--clipped-dir", required=True)
    s1.add_argument("--stem", required=True)
    s1.add_argument("--leaf", nargs="+", type=int, required=True)
    s1.add_argument("--wavelengths-mat")
    s1.add_argument("--wavelengths-csv")
    s1.add_argument("--title")
    s1.add_argument("--save")
    s1.add_argument("--no-show", action="store_true")
    s1.add_argument("--ylim", nargs=2, type=float)
    s1.set_defaults(
        func=lambda a: _plot.plot_leaf_center(
            a.clipped_dir,
            stem=a.stem,
            leaves=a.leaf,
            wavelengths_mat=a.wavelengths_mat,
            wavelengths_csv=a.wavelengths_csv,
            title=a.title,
            ylim=(tuple(a.ylim) if a.ylim else None),
            save=a.save,
            show=(not a.no_show),
        )
    )

    # leaf centers across stems
    s2 = sp.add_parser("leaf-multi", help="Leaf center spectra across stems")
    s2.add_argument("--clipped-dir", required=True)
    s2.add_argument(
        "--item",
        action="append",
        required=True,
        help="Pair as STEM:LEAF (repeat). Example: --item H_P1_V4_B:1",
    )
    s2.add_argument("--wavelengths-mat")
    s2.add_argument("--wavelengths-csv")
    s2.add_argument("--title")
    s2.add_argument("--save")
    s2.add_argument("--no-show", action="store_true")
    s2.add_argument("--ylim", nargs=2, type=float)

    def _items(a):
        out = []
        for s in a.item:
            stem, leaf = s.rsplit(":", 1)
            out.append((stem, int(leaf)))
        return out

    s2.set_defaults(
        func=lambda a: _plot.plot_leaf_centers_multi(
            a.clipped_dir,
            _items(a),
            wavelengths_mat=a.wavelengths_mat,
            wavelengths_csv=a.wavelengths_csv,
            title=a.title,
            ylim=(tuple(a.ylim) if a.ylim else None),
            save=a.save,
            show=(not a.no_show),
        )
    )

    # single pixel
    s3 = sp.add_parser("pixel", help="Pixel spectrum")
    s3.add_argument("--image", required=True)
    s3.add_argument("--x", type=int, required=True)
    s3.add_argument("--y", type=int, required=True)
    s3.add_argument("--key")
    s3.add_argument("--wavelengths-mat")
    s3.add_argument("--wavelengths-csv")
    s3.add_argument("--title")
    s3.add_argument("--save")
    s3.add_argument("--no-show", action="store_true")
    s3.add_argument("--ylim", nargs=2, type=float)
    s3.set_defaults(
        func=lambda a: _plot.plot_pixel(
            a.image,
            x=a.x,
            y=a.y,
            key=a.key,
            wavelengths_mat=a.wavelengths_mat,
            wavelengths_csv=a.wavelengths_csv,
            title=a.title,
            ylim=(tuple(a.ylim) if a.ylim else None),
            save=a.save,
            show=(not a.no_show),
        )
    )

    # ROI
    s4 = sp.add_parser("roi", help="ROI mean spectrum")
    s4.add_argument("--image", required=True)
    s4.add_argument("--x0", type=int, required=True)
    s4.add_argument("--y0", type=int, required=True)
    s4.add_argument("--x1", type=int, required=True)
    s4.add_argument("--y1", type=int, required=True)
    s4.add_argument("--key")
    s4.add_argument("--wavelengths-mat")
    s4.add_argument("--wavelengths-csv")
    s4.add_argument("--title")
    s4.add_argument("--save")
    s4.add_argument("--no-show", action="store_true")
    s4.add_argument("--ylim", nargs=2, type=float)
    s4.set_defaults(
        func=lambda a: _plot.plot_roi(
            a.image,
            x0=a.x0,
            y0=a.y0,
            x1=a.x1,
            y1=a.y1,
            key=a.key,
            wavelengths_mat=a.wavelengths_mat,
            wavelengths_csv=a.wavelengths_csv,
            title=a.title,
            ylim=(tuple(a.ylim) if a.ylim else None),
            save=a.save,
            show=(not a.no_show),
        )
    )


def main(argv: Optional[List[str]] = None):
    """
    Unified CLI entry point for MVOS HSI.

    Example:
        mvos-hsi calibration folder --folder ... --dark ...
        mvos-hsi clipping folder --folder ... --index ndvi ...
        mvos-hsi augmentation folder --folder ... --num 3 --flip
        mvos-hsi plotting leaf --clipped-dir ... --stem H_P1_V4_B --leaf 1 3
    """
    if argv is None:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(prog="mvos-hsi", description="Unified MVOS HSI CLI")
    sub = p.add_subparsers(dest="command", required=True)

    _add_calibration(sub)
    _add_clipping(sub)
    _add_augmentation(sub)
    _add_plotting(sub)

    args = p.parse_args(argv)
    res = args.func(args)

    # If a function returns something, dump it in a basic way
    if res is not None:
        try:
            import json

            print(json.dumps(res, indent=2, default=lambda o: str(o)))
        except Exception:
            print(res)


if __name__ == "__main__":
    main()
