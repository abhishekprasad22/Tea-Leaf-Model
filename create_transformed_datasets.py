#!/usr/bin/env python3
"""
create_transformed_datasets.py

Given a dataset root (e.g. "Tea_Leaf_Disease Dataset"), create four sibling dataset folders:
  - "<orig_name> (Histogram Equalized Image)"
  - "<orig_name> (Grayscale Image)"
  - "<orig_name> (Negative Image)"
  - "<orig_name> (False-Colored (mapped) Image)"

Each new dataset will preserve the directory tree of the original dataset.
All image files (by extension) are transformed and saved with the same filenames.

Usage:
    python create_transformed_datasets.py --dataset_root "/path/to/Tea_Leaf_Disease Dataset" \
        --colormap JET --extensions jpg jpeg png bmp tif tiff

Dependencies:
    pip install opencv-python tqdm numpy
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Map of supported colormap names to cv2 constants
_COLORMAP_MAP = {
    "AUTUMN": cv2.COLORMAP_AUTUMN,
    "BONE": cv2.COLORMAP_BONE,
    "JET": cv2.COLORMAP_JET,
    "WINTER": cv2.COLORMAP_WINTER,
    "RAINBOW": cv2.COLORMAP_RAINBOW,
    "OCEAN": cv2.COLORMAP_OCEAN,
    "SUMMER": cv2.COLORMAP_SUMMER,
    "SPRING": cv2.COLORMAP_SPRING,
    "COOL": cv2.COLORMAP_COOL,
    "HSV": cv2.COLORMAP_HSV,
    "PINK": cv2.COLORMAP_PINK,
    "HOT": cv2.COLORMAP_HOT,
}


def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    # If image has alpha channel, drop alpha for processing
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def histogram_equalize_color(img_bgr: np.ndarray):
    """Equalize luminance channel (Y) in YCrCb to preserve colors."""
    if img_bgr.ndim == 2:
        return cv2.equalizeHist(img_bgr)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


def to_grayscale(img_bgr: np.ndarray):
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def negative_image(img: np.ndarray):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return 255 - img


def false_color_map(gray_img: np.ndarray, colormap_name: str = "JET"):
    if gray_img.ndim != 2:
        gray = to_grayscale(gray_img)
    else:
        gray = gray_img
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cmap = _COLORMAP_MAP.get(colormap_name.upper(), cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(gray, cmap)
    return colored


def make_dirs_for_file(src_file: Path, src_root: Path, dest_root: Path):
    """Given a source file path and the src_root, return and create the destination parent dir under dest_root."""
    rel = src_file.relative_to(src_root)
    dest_parent = dest_root.joinpath(rel.parent)
    dest_parent.mkdir(parents=True, exist_ok=True)
    return dest_parent


def is_image_file(p: Path, extensions_set):
    return p.is_file() and p.suffix.lower().lstrip(".") in extensions_set


def main():
    p = argparse.ArgumentParser(description="Create transformed dataset copies for a dataset root.")
    p.add_argument("--dataset_root", required=True, help="Path to the original dataset root folder")
    p.add_argument("--colormap", default="JET", help="Colormap for false color (default JET).")
    p.add_argument("--extensions", nargs="+", default=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                   help="Image file extensions to process (no dots).")
    args = p.parse_args()

    src_root = Path(args.dataset_root).expanduser().resolve()
    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(f"dataset_root does not exist or is not a directory: {src_root}")

    base_parent = src_root.parent
    base_name = src_root.name

    # names requested by user
    out_names = {
        "hist_eq": f"{base_name} (Histogram Equalized Image)",
        "grayscale": f"{base_name} (Grayscale Image)",
        "negative": f"{base_name} (Negative Image)",
        "false_color": f"{base_name} (False-Colored (mapped) Image)",
    }

    # create root dirs
    out_roots = {k: (base_parent / name) for k, name in out_names.items()}
    for d in out_roots.values():
        d.mkdir(parents=True, exist_ok=True)

    extensions_set = set(e.lower().lstrip(".") for e in args.extensions)

    # Collect all image files recursively under src_root
    image_paths = [p for p in src_root.rglob("*") if is_image_file(p, extensions_set)]

    print(f"Found {len(image_paths)} image files under {src_root}")
    if len(image_paths) == 0:
        print("Nothing to do. Check extensions or dataset_root.")
        return

    # Transform functions map
    transforms = {
        "hist_eq": histogram_equalize_color,
        "grayscale": to_grayscale,
        "negative": negative_image,
        "false_color": lambda img: false_color_map(img, args.colormap),
    }

    # Process with progress bar
    for src_path in tqdm(image_paths, desc="Processing images"):
        try:
            img = read_image(src_path)
        except Exception as e:
            tqdm.write(f"Warning: skipping {src_path} (read error: {e})")
            continue

        for tkey, tfunc in transforms.items():
            try:
                out_img = tfunc(img)
                dest_root = out_roots[tkey]
                dest_parent = make_dirs_for_file(src_path, src_root, dest_root)
                dest_path = dest_parent / src_path.name

                # For single-channel images (grayscale) ensure saving is correct
                # OpenCV imwrite supports both single- and 3-channel images.
                cv2.imwrite(str(dest_path), out_img)
            except Exception as e:
                tqdm.write(f"Failed to process {src_path} -> {tkey}: {e}")

    print("All done.")
    print("Created:")
    for k, root in out_roots.items():
        print(f"  - {root}")

if __name__ == "__main__":
    main()
