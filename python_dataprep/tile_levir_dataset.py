#!/usr/bin/env python3
"""
LEVIR-CD Image Tiling and Dataset Preparation Pipeline
LandWatch AI - Project 50

Splits every LEVIR-CD triplet (A = earlier image, B = later image,
label = ground-truth change mask) into non-overlapping square patches
(default 256 x 256, so one 1024 x 1024 triplet gives 16 patch triplets).

Safety and correctness guarantees:
- Dry run by default. Patches are written only when --execute is given.
- Every source image is fully decoded (img.load()), so truncated or corrupted
  files are detected before anything is written.
- A missing split folder, a missing A/B/label folder or an incomplete triplet
  is a validation error, and execution refuses to start if any error exists.
- A, B and label are cropped with the identical box. The official
  train/val/test splits are kept separate.
- Masks are cropped from the untouched original image (no colour conversion),
  so every pixel value (or palette index for 'P' masks) is preserved exactly.
  Every saved patch is re-opened and compared pixel-for-pixel with its crop.
- The exact set of mask values is taken from the full histogram of every mask
  (not only its minimum and maximum), so non-binary values are reported.
- Patches are written to '<output>.incomplete' and the folder is renamed to its
  final name only after every file has been written and verified. If anything
  fails, only that newly created '.incomplete' folder is removed. Source data
  is never modified.
- Refuses to run if the output folder (or a leftover '.incomplete') exists.
- The preparation report counts the files actually present on disk.
- Paths are resolved relative to this script, so it works on any machine.

Usage (from the project root):
    python3 python_dataprep/tile_levir_dataset.py             # dry run only
    python3 python_dataprep/tile_levir_dataset.py --execute   # generate patches
"""

import argparse
import csv
import math
import os
import re
import shutil
import sys
import time
from collections import Counter

try:
    from PIL import Image
except ImportError:
    Image = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_LEVIR_DIR = os.path.join(PROJECT_ROOT, "datasets", "LEVIR-CD")

SPLITS = ("train", "val", "test")
SUBDIRS = ("A", "B", "label")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
SUPPORTED_MASK_MODES = ("L", "P", "1", "LA", "RGB", "RGBA")
STAGING_SUFFIX = ".incomplete"
SIZE_OVERHEAD_FACTOR = 1.08   # small tiles compress slightly worse than the full image
DISK_SAFETY_FACTOR = 1.5      # require 50% more free space than the estimate

CSV_FIELDNAMES = [
    "patch_filename", "relative_path_A", "relative_path_B", "relative_path_label",
    "original_filename", "split", "patch_idx", "row", "col",
    "x1", "y1", "x2", "y2", "patch_width", "patch_height",
    "has_change", "change_pixel_count", "change_ratio", "mask_values",
]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def natural_key(name):
    """Sort 'train_2.png' before 'train_10.png'."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def display_path(path):
    """Path relative to the project root when possible, so reports are portable."""
    try:
        rel = os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return path
    return path if rel.startswith("..") else rel.replace(os.sep, "/")


def list_image_files(folder):
    """Image filenames in a folder. Hidden/system files are ignored, including
    the macOS '._*' AppleDouble files that appear on exFAT/FAT USB drives."""
    return {
        f for f in os.listdir(folder)
        if not f.startswith(".")
        and f.lower().endswith(IMAGE_EXTENSIONS)
        and os.path.isfile(os.path.join(folder, f))
    }


def mask_value_image(mask_img):
    """Single-band view of a mask whose pixel values are the real mask values.

    'L' masks are used as they are. Palette ('P') and 1-bit masks are converted
    to 'L' so that values are the displayed grey levels, not palette indices.
    For LA/RGB/RGBA masks the first band is used. This view is only used for
    statistics; saved patches are always cropped from the untouched original.
    """
    if mask_img.mode == "L":
        return mask_img
    if mask_img.mode in ("P", "1"):
        return mask_img.convert("L")
    if mask_img.mode in ("LA", "RGB", "RGBA"):
        return mask_img.getchannel(0)
    raise ValueError(f"unsupported mask mode '{mask_img.mode}'")


def histogram_values(hist):
    """{pixel value: pixel count} for the non-empty bins of a 256-bin histogram."""
    return {value: count for value, count in enumerate(hist) if count}


def format_values(value_counts):
    return ";".join(str(v) for v in sorted(value_counts))


def nearest_existing_dir(path):
    path = os.path.abspath(path)
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def is_inside(path, folder):
    path, folder = os.path.realpath(path), os.path.realpath(folder)
    return path == folder or path.startswith(folder + os.sep)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def get_image_info(filepath, mask_patch_size=None):
    """Fully decode one image and return its properties.

    img.load() forces a full decode, so truncated or corrupted files are caught
    here. When mask_patch_size is given the file is treated as a ground-truth
    mask, and its exact pixel values and number of patches containing change
    (value > 0) are also returned.
    """
    if not os.path.isfile(filepath):
        return {"valid": False, "error": "File does not exist"}
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return {"valid": False, "error": "File is empty (0 bytes)"}

    try:
        with Image.open(filepath) as img:
            img.load()  # full decode
            info = {
                "valid": True,
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
                "file_size": file_size,
                "error": None,
            }
            if mask_patch_size:
                if img.mode not in SUPPORTED_MASK_MODES:
                    return {"valid": False, "error": f"Unsupported mask mode '{img.mode}'"}
                values = mask_value_image(img)
                info["value_counts"] = histogram_values(values.histogram())
                if img.width % mask_patch_size == 0 and img.height % mask_patch_size == 0:
                    tile_pixels = mask_patch_size * mask_patch_size
                    positive = 0
                    for y in range(0, img.height, mask_patch_size):
                        for x in range(0, img.width, mask_patch_size):
                            tile_hist = values.crop((x, y, x + mask_patch_size, y + mask_patch_size)).histogram()
                            if tile_hist[0] < tile_pixels:
                                positive += 1
                    info["positive_patches"] = positive
            return info
    except Exception as e:
        return {"valid": False, "error": f"decode error: {e}"}


def validate_split_integrity(split_dir, patch_size=256):
    """Validate one split folder (train, val or test).

    Compares the filename sets of A, B and label, fully decodes every file of
    every complete triplet, checks dimensions and divisibility by the patch
    size, and audits the exact pixel values of every mask.
    """
    split = os.path.basename(os.path.normpath(split_dir))
    result = {
        "exists": os.path.isdir(split_dir),
        "count_A": 0, "count_B": 0, "count_label": 0,
        "total_unique_filenames": 0,
        "complete_triplets": 0,
        "validated_triplets": [],
        "missing_in_A": [], "missing_in_B": [], "missing_in_label": [],
        "validation_errors": [],
        "warnings": [],
        "dimension_counts": Counter(),
        "mode_counts": Counter(),
        "total_input_bytes": 0,
        "source_files": [],          # (file size, patches per file), for the storage estimate
        "mask_value_pixels": Counter(),
        "nonbinary_masks": [],       # (filename, {value: pixel count}) for values other than 0/255
        "expected_patches_per_type": 0,
        "positive_patches": 0,
    }
    errors = result["validation_errors"]

    if not result["exists"]:
        errors.append(f"MISSING SPLIT DIRECTORY: {display_path(split_dir)}/")
        return result

    files_by_sub = {}
    missing_dirs = set()
    for sub in SUBDIRS:
        sub_dir = os.path.join(split_dir, sub)
        if os.path.isdir(sub_dir):
            files_by_sub[sub] = list_image_files(sub_dir)
        else:
            errors.append(f"MISSING DIRECTORY: {split}/{sub}/")
            missing_dirs.add(sub)
            files_by_sub[sub] = set()

    set_A, set_B, set_L = files_by_sub["A"], files_by_sub["B"], files_by_sub["label"]
    result["count_A"], result["count_B"], result["count_label"] = len(set_A), len(set_B), len(set_L)
    result["total_unique_filenames"] = len(set_A | set_B | set_L)
    complete = sorted(set_A & set_B & set_L, key=natural_key)
    result["complete_triplets"] = len(complete)

    # Files without a counterpart make a triplet incomplete: these are errors.
    for sub, own in (("A", set_A), ("B", set_B), ("label", set_L)):
        missing = sorted((set_A | set_B | set_L) - own, key=natural_key)
        result[f"missing_in_{sub}"] = missing
        if sub in missing_dirs:
            continue  # already reported as a missing directory
        for fname in missing:
            errors.append(f"MISSING in {split}/{sub}/: {fname} (present in another folder of this triplet)")

    for filename in complete:
        info_A = get_image_info(os.path.join(split_dir, "A", filename))
        info_B = get_image_info(os.path.join(split_dir, "B", filename))
        info_L = get_image_info(os.path.join(split_dir, "label", filename), mask_patch_size=patch_size)

        bad = [(sub, info) for sub, info in (("A", info_A), ("B", info_B), ("label", info_L)) if not info["valid"]]
        if bad:
            for sub, info in bad:
                errors.append(f"{split}/{sub}/{filename}: {info['error']}")
            continue

        dim_A = (info_A["width"], info_A["height"])
        dim_B = (info_B["width"], info_B["height"])
        dim_L = (info_L["width"], info_L["height"])
        if dim_A != dim_B or dim_A != dim_L:
            errors.append(f"{split}/{filename}: dimension mismatch A={dim_A} B={dim_B} label={dim_L}")
            continue
        if dim_A[0] % patch_size != 0 or dim_A[1] % patch_size != 0:
            errors.append(f"{split}/{filename}: dimensions {dim_A} are not divisible by patch size {patch_size}")
            continue
        if info_A["mode"] != info_B["mode"]:
            result["warnings"].append(
                f"{split}/{filename}: A is '{info_A['mode']}' but B is '{info_B['mode']}' (copied unchanged)")

        modes = (info_A["mode"], info_B["mode"], info_L["mode"])
        patches = (dim_A[0] // patch_size) * (dim_A[1] // patch_size)
        result["dimension_counts"][dim_A] += 1
        result["mode_counts"][modes] += 1
        result["total_input_bytes"] += info_A["file_size"] + info_B["file_size"] + info_L["file_size"]
        for info in (info_A, info_B, info_L):
            result["source_files"].append((info["file_size"], patches))
        result["mask_value_pixels"].update(info_L["value_counts"])
        odd = {v: n for v, n in info_L["value_counts"].items() if v not in (0, 255)}
        if odd:
            result["nonbinary_masks"].append((filename, odd))
        result["expected_patches_per_type"] += patches
        result["positive_patches"] += info_L.get("positive_patches", 0)
        result["validated_triplets"].append((filename, dim_A, modes))

    result["expected_total_files"] = result["expected_patches_per_type"] * len(SUBDIRS)
    return result


def estimate_output_bytes(source_files, allocation_unit):
    """Each patch is estimated as (source file size / patches) * 1.08, rounded
    up to the disk allocation unit (32 KB clusters on a typical exFAT USB stick,
    4 KB on APFS), plus a small allowance for the metadata CSVs."""
    unit = max(allocation_unit, 1)
    total = 0
    patch_count = 0
    for size, patches in source_files:
        per_patch = size / patches * SIZE_OVERHEAD_FACTOR
        total += patches * math.ceil(per_patch / unit) * unit
        patch_count += patches
    total += (patch_count // len(SUBDIRS)) * 2 * 250  # split CSV + all_patches.csv, ~250 bytes per row
    return total


def perform_dry_run(dataset_dir, output_dir, patch_size=256):
    """Validate everything and estimate the output, without writing any patches."""
    start_time = time.time()
    staging_dir = output_dir + STAGING_SUFFIX
    results = {
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "staging_dir": staging_dir,
        "patch_size": patch_size,
        "output_dir_already_exists": os.path.exists(output_dir),
        "staging_dir_already_exists": os.path.exists(staging_dir),
        "splits": {},
        "overall": {
            "total_validated_pairs": 0,
            "total_expected_patches": 0,
            "total_expected_files": 0,
            "total_input_bytes": 0,
            "estimated_output_bytes": 0,
            "free_disk_bytes": -1,
            "allocation_unit": 4096,
            "disk_has_sufficient_space": False,
            "all_errors": [],
            "all_warnings": [],
            "mask_value_pixels": Counter(),
            "nonbinary_masks": [],
            "masks_inspected": 0,
        },
    }
    overall = results["overall"]

    disk_dir = nearest_existing_dir(output_dir)
    try:
        overall["free_disk_bytes"] = shutil.disk_usage(disk_dir).free
        overall["allocation_unit"] = os.statvfs(disk_dir).f_frsize or 4096
    except (OSError, AttributeError):
        pass

    source_files = []
    for split in SPLITS:
        audit = validate_split_integrity(os.path.join(dataset_dir, split), patch_size)
        results["splits"][split] = audit
        overall["total_validated_pairs"] += len(audit["validated_triplets"])
        overall["total_expected_patches"] += audit["expected_patches_per_type"]
        overall["total_expected_files"] += audit.get("expected_total_files", 0)
        overall["total_input_bytes"] += audit["total_input_bytes"]
        overall["all_errors"].extend(audit["validation_errors"])
        overall["all_warnings"].extend(audit["warnings"])
        overall["mask_value_pixels"].update(audit["mask_value_pixels"])
        overall["nonbinary_masks"].extend((f"{split}/label/{f}", odd) for f, odd in audit["nonbinary_masks"])
        overall["masks_inspected"] += len(audit["validated_triplets"])
        source_files.extend(audit["source_files"])

    overall["estimated_output_bytes"] = estimate_output_bytes(source_files, overall["allocation_unit"])
    if overall["free_disk_bytes"] > 0:
        overall["disk_has_sufficient_space"] = (
            overall["free_disk_bytes"] > overall["estimated_output_bytes"] * DISK_SAFETY_FACTOR)

    results["duration_sec"] = time.time() - start_time
    return results


def format_dry_run_report(results):
    overall = results["overall"]
    out_name = os.path.basename(results["output_dir"])
    lines = []
    lines.append("=" * 80)
    lines.append("           LEVIR-CD DATASET TILING & VALIDATION (DRY-RUN AUDIT)")
    lines.append("                         LandWatch AI - Project 50")
    lines.append("=" * 80)
    lines.append(f"Timestamp:             {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Source Directory:      {display_path(results['dataset_dir'])}")
    lines.append(f"Target Output:         {display_path(results['output_dir'])}")
    lines.append(f"Target Patch Size:     {results['patch_size']} x {results['patch_size']} px (non-overlapping, no padding)")
    lines.append("Execution Engine:      PIL (Pillow), every file fully decoded")
    lines.append(f"Audit Elapsed Time:    {results['duration_sec']:.2f} seconds")
    lines.append("")

    lines.append("-" * 80)
    lines.append("1. OUTPUT SAFETY & DIRECTORY STATUS")
    lines.append("-" * 80)
    if results["output_dir_already_exists"]:
        lines.append(f"  [SAFETY LOCK]: Output directory '{out_name}' ALREADY EXISTS.")
        lines.append("  Execution will REFUSE to run, so existing data cannot be overwritten.")
    else:
        lines.append(f"  • Output directory '{out_name}' does not exist yet.")
    if results["staging_dir_already_exists"]:
        lines.append(f"  [SAFETY LOCK]: Leftover '{os.path.basename(results['staging_dir'])}' folder from an")
        lines.append("  interrupted run exists. Inspect and delete it manually before executing.")
    else:
        lines.append(f"  • Patches are written to '{os.path.basename(results['staging_dir'])}' first and renamed to")
        lines.append(f"    '{out_name}' only after every file is written and verified. On failure,")
        lines.append("    only that new folder is removed. Source images are never modified.")
    lines.append("")

    lines.append("-" * 80)
    lines.append("2. DISK SPACE & STORAGE REQUIREMENT AUDIT")
    lines.append("-" * 80)
    mb = 1024 * 1024
    input_mb = overall["total_input_bytes"] / mb
    est_mb = overall["estimated_output_bytes"] / mb
    free_mb = overall["free_disk_bytes"] / mb if overall["free_disk_bytes"] >= 0 else 0
    lines.append(f"• Total Source Images (validated):  {input_mb:.2f} MB ({input_mb / 1024:.2f} GB)")
    lines.append(f"• Estimated Output Storage:         {est_mb:.2f} MB ({est_mb / 1024:.2f} GB)")
    lines.append(f"  (Method: per source file, size / patches x {SIZE_OVERHEAD_FACTOR}, rounded up to the")
    lines.append(f"   {overall['allocation_unit'] // 1024} KB disk allocation unit, plus metadata CSVs)")
    lines.append(f"• Actual Free Disk Space:           {free_mb:.2f} MB ({free_mb / 1024:.2f} GB)")
    needed_mb = est_mb * DISK_SAFETY_FACTOR
    if overall["disk_has_sufficient_space"]:
        lines.append(f"• Storage Verification:             PASSED (needs {needed_mb / 1024:.2f} GB incl. 50% safety margin)")
    else:
        lines.append(f"• Storage Verification:             FAILED - needs {needed_mb / 1024:.2f} GB free incl. 50% margin")
    lines.append("")

    lines.append("-" * 80)
    lines.append("3. VALIDATED PAIRS & EXPECTED PATCH COUNTS BY SPLIT")
    lines.append("-" * 80)
    lines.append(f"{'Split':<6} | {'A':<5} | {'B':<5} | {'Mask':<5} | {'Valid Triplets':<14} | {'Patches/Modality':<16} | {'Total Files':<11}")
    lines.append("-" * 80)
    for split in SPLITS:
        s = results["splits"].get(split, {})
        lines.append(
            f"{split:<6} | {s.get('count_A', 0):<5} | {s.get('count_B', 0):<5} | {s.get('count_label', 0):<5} | "
            f"{len(s.get('validated_triplets', [])):<14} | {s.get('expected_patches_per_type', 0):<16} | "
            f"{s.get('expected_total_files', 0):<11}")
        if not s.get("exists", False):
            lines.append("         [!] split directory is missing")
        miss = [len(s.get(k, [])) for k in ("missing_in_A", "missing_in_B", "missing_in_label")]
        if any(miss):
            lines.append(f"         [!] files without counterparts - missing in A:{miss[0]}  B:{miss[1]}  label:{miss[2]}")
    lines.append("-" * 80)
    lines.append(f"{'TOTAL':<6} | {'':<5} | {'':<5} | {'':<5} | {overall['total_validated_pairs']:<14} | "
                 f"{overall['total_expected_patches']:<16} | {overall['total_expected_files']:<11}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("4. DIMENSIONS, DIVISIBILITY & IMAGE MODES")
    lines.append("-" * 80)
    p = results["patch_size"]
    for split in SPLITS:
        s = results["splits"].get(split, {})
        for (w, h), count in s.get("dimension_counts", {}).items():
            lines.append(f"  • [{split}] {w}x{h} px: {count} triplets, divisible by {p} -> "
                         f"{w // p}x{h // p} = {(w // p) * (h // p)} patches per triplet")
        for modes, count in s.get("mode_counts", {}).items():
            lines.append(f"    - image modes (A, B, label) = {modes}: {count} triplets")
    lines.append("")

    lines.append("-" * 80)
    lines.append("5. GROUND-TRUTH MASK AUDIT (full histogram of every mask)")
    lines.append("-" * 80)
    value_pixels = overall["mask_value_pixels"]
    total_px = sum(value_pixels.values()) or 1
    lines.append(f"• Masks inspected:          {overall['masks_inspected']} (all validated masks, not a sample)")
    lines.append("• Pixel values found:")
    for value in sorted(value_pixels):
        lines.append(f"    value {value:>3}: {value_pixels[value]:>12,} px ({value_pixels[value] * 100.0 / total_px:.4f}%)")
    nonbinary = overall["nonbinary_masks"]
    lines.append(f"• Masks with values other than 0/255: {len(nonbinary)} of {overall['masks_inspected']}")
    for name, odd in nonbinary[:10]:
        lines.append(f"    - {name}: " + ", ".join(f"value {v} x {n} px" for v, n in sorted(odd.items())))
    if len(nonbinary) > 10:
        lines.append(f"    ... and {len(nonbinary) - 10} more")
    lines.append("• Preservation policy:      mask pixels are copied unchanged (no thresholding).")
    lines.append("• Change definition:        0 = no change; any value > 0 = change (LEVIR-CD uses 255).")
    lines.append("                            Used for has_change / change_pixel_count in the metadata.")
    for split in SPLITS:
        s = results["splits"].get(split, {})
        n = s.get("expected_patches_per_type", 0)
        pos = s.get("positive_patches", 0)
        if n:
            lines.append(f"• [{split}] patches containing change: {pos} of {n} ({pos * 100.0 / n:.1f}%); "
                         f"background-only: {n - pos} ({(n - pos) * 100.0 / n:.1f}%)")
    lines.append("")

    lines.append("-" * 80)
    lines.append("6. ERROR & WARNING AUDIT")
    lines.append("-" * 80)
    all_errors = overall["all_errors"]
    lines.append(f"• Validation errors: {len(all_errors)}")
    if all_errors:
        for err in all_errors[:20]:
            lines.append(f"  [ERROR]: {err}")
        if len(all_errors) > 20:
            lines.append(f"  ... and {len(all_errors) - 20} more")
    else:
        lines.append("  • None. Every triplet is complete, fully decodable, dimension-matched and divisible.")
    warnings = overall["all_warnings"]
    lines.append(f"• Warnings: {len(warnings)}")
    for warn in warnings[:10]:
        lines.append(f"  [WARNING]: {warn}")
    lines.append("=" * 80)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def save_and_verify(patch, path):
    """Save one patch as PNG (lossless), re-open it and confirm it is identical."""
    patch.save(path, format="PNG")
    with Image.open(path) as saved:
        saved.load()
        same = (saved.size == patch.size and saved.mode == patch.mode
                and saved.tobytes() == patch.tobytes())
        if same and patch.mode == "P":
            same = mask_value_image(saved).tobytes() == mask_value_image(patch).tobytes()
    if not same:
        raise RuntimeError(f"saved patch does not match its source crop: {path}")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_patches(dataset_dir, staging_dir, split_triplets, patch_size):
    """Crop, save and verify every patch triplet, and write the metadata CSVs."""
    metadata_dir = os.path.join(staging_dir, "metadata")
    os.makedirs(metadata_dir)
    tile_pixels = patch_size * patch_size
    summaries = {}
    all_rows = []

    for split in SPLITS:
        src = os.path.join(dataset_dir, split)
        dst = os.path.join(staging_dir, split)
        for sub in SUBDIRS:
            os.makedirs(os.path.join(dst, sub))

        triplets = split_triplets[split]
        rows = []
        value_pixels = Counter()
        verified_files = 0
        print(f"\nTiling split '{split}' ({len(triplets)} image pairs)...")

        for idx, (filename, (width, height), _modes) in enumerate(triplets):
            stem = os.path.splitext(filename)[0]
            with Image.open(os.path.join(src, "A", filename)) as img_A, \
                    Image.open(os.path.join(src, "B", filename)) as img_B, \
                    Image.open(os.path.join(src, "label", filename)) as img_L:
                img_A.load()
                img_B.load()
                img_L.load()
                values_L = mask_value_image(img_L)
                patch_idx = 0
                for r in range(height // patch_size):
                    for c in range(width // patch_size):
                        x1, y1 = c * patch_size, r * patch_size
                        x2, y2 = x1 + patch_size, y1 + patch_size
                        box = (x1, y1, x2, y2)
                        patch_name = f"{stem}_patch_{patch_idx:02d}.png"

                        # Identical crop box for before, after and mask. The mask is
                        # cropped from the original image, so its values are unchanged.
                        for sub, img in (("A", img_A), ("B", img_B), ("label", img_L)):
                            save_and_verify(img.crop(box), os.path.join(dst, sub, patch_name))
                            verified_files += 1

                        counts = histogram_values(values_L.crop(box).histogram())
                        value_pixels.update(counts)
                        change_pixels = tile_pixels - counts.get(0, 0)
                        rows.append({
                            "patch_filename": patch_name,
                            "relative_path_A": f"{split}/A/{patch_name}",
                            "relative_path_B": f"{split}/B/{patch_name}",
                            "relative_path_label": f"{split}/label/{patch_name}",
                            "original_filename": filename,
                            "split": split,
                            "patch_idx": patch_idx,
                            "row": r,
                            "col": c,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "patch_width": patch_size,
                            "patch_height": patch_size,
                            "has_change": 1 if change_pixels > 0 else 0,
                            "change_pixel_count": change_pixels,
                            "change_ratio": f"{change_pixels / tile_pixels:.6f}",
                            "mask_values": format_values(counts),
                        })
                        patch_idx += 1

            if (idx + 1) % 50 == 0 or (idx + 1) == len(triplets):
                print(f"  [{split}] {idx + 1}/{len(triplets)} pairs done "
                      f"({len(rows)} patch triplets written and verified)")

        write_csv(os.path.join(metadata_dir, f"{split}_patches.csv"), rows)
        all_rows.extend(rows)
        summaries[split] = {
            "source_pairs": len(triplets),
            "rows": rows,
            "value_pixels": value_pixels,
            "verified_files": verified_files,
        }

    write_csv(os.path.join(metadata_dir, "all_patches.csv"), all_rows)
    return summaries


def count_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def verify_output_on_disk(staging_dir, summaries):
    """Count the files actually on disk and check they match the metadata exactly."""
    disk = {"splits": {}, "total_patch_files": 0, "total_bytes": 0}
    problems = []
    for split in SPLITS:
        expected = {row["patch_filename"] for row in summaries[split]["rows"]}
        counts = {}
        for sub in SUBDIRS:
            folder = os.path.join(staging_dir, split, sub)
            on_disk = list_image_files(folder)
            counts[sub] = len(on_disk)
            if on_disk != expected:
                problems.append(f"{split}/{sub}: {len(expected - on_disk)} missing, "
                                f"{len(on_disk - expected)} unexpected files")
            disk["total_bytes"] += sum(os.path.getsize(os.path.join(folder, f)) for f in on_disk)
        csv_rows = count_csv_rows(os.path.join(staging_dir, "metadata", f"{split}_patches.csv"))
        if csv_rows != len(expected):
            problems.append(f"{split}_patches.csv has {csv_rows} rows, expected {len(expected)}")
        counts["csv_rows"] = csv_rows
        disk["splits"][split] = counts
        disk["total_patch_files"] += counts["A"] + counts["B"] + counts["label"]

    all_rows = count_csv_rows(os.path.join(staging_dir, "metadata", "all_patches.csv"))
    expected_all = sum(len(s["rows"]) for s in summaries.values())
    if all_rows != expected_all:
        problems.append(f"all_patches.csv has {all_rows} rows, expected {expected_all}")
    disk["all_csv_rows"] = all_rows

    if problems:
        raise RuntimeError("on-disk verification failed: " + "; ".join(problems))
    return disk


def write_preparation_report(staging_dir, output_dir, dataset_dir, patch_size, summaries, disk, elapsed):
    mb = 1024 * 1024
    lines = []
    lines.append("=" * 80)
    lines.append("              LEVIR-CD DATASET TILING - PREPARATION REPORT")
    lines.append("                         LandWatch AI - Project 50")
    lines.append("=" * 80)
    lines.append(f"Completed:          {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Source directory:   {display_path(dataset_dir)}")
    lines.append(f"Output directory:   {display_path(output_dir)}")
    lines.append(f"Patch size:         {patch_size} x {patch_size} px, non-overlapping, no padding")
    lines.append(f"Elapsed time:       {elapsed:.1f} seconds")
    lines.append("")
    lines.append("FILES COUNTED ON DISK AFTER GENERATION")
    lines.append("-" * 80)
    lines.append(f"{'Split':<6} | {'Source pairs':<12} | {'A':<6} | {'B':<6} | {'label':<6} | {'CSV rows':<8} | Patches with change")
    lines.append("-" * 80)
    tot_pairs = tot_pos = tot_rows = 0
    for split in SPLITS:
        s, d = summaries[split], disk["splits"][split]
        pos = sum(row["has_change"] for row in s["rows"])
        n = len(s["rows"])
        tot_pairs += s["source_pairs"]
        tot_pos += pos
        tot_rows += n
        pct = pos * 100.0 / n if n else 0.0
        lines.append(f"{split:<6} | {s['source_pairs']:<12} | {d['A']:<6} | {d['B']:<6} | {d['label']:<6} | "
                     f"{d['csv_rows']:<8} | {pos} ({pct:.1f}%)")
    lines.append("-" * 80)
    pct = tot_pos * 100.0 / tot_rows if tot_rows else 0.0
    lines.append(f"TOTAL: {tot_pairs} source pairs -> {tot_rows} patch triplets -> "
                 f"{disk['total_patch_files']} patch image files on disk")
    lines.append(f"Patches containing change: {tot_pos} ({pct:.1f}%)")
    lines.append(f"Metadata: train_patches.csv, val_patches.csv, test_patches.csv, "
                 f"all_patches.csv ({disk['all_csv_rows']} rows)")
    lines.append(f"Patch image size on disk: {disk['total_bytes'] / mb:.1f} MB")
    lines.append("")
    lines.append("VERIFICATION")
    lines.append("-" * 80)
    verified = sum(s["verified_files"] for s in summaries.values())
    lines.append(f"• Every patch was re-opened after saving and compared pixel-for-pixel with its")
    lines.append(f"  crop: {verified} of {disk['total_patch_files']} files matched.")
    lines.append("• The filenames on disk match the metadata CSVs exactly (none missing, none extra).")
    lines.append("• A, B and label patches were cut with identical crop boxes; splits kept separate.")
    lines.append("")
    lines.append("MASK VALUES IN THE GENERATED LABEL PATCHES")
    lines.append("-" * 80)
    value_pixels = Counter()
    for s in summaries.values():
        value_pixels.update(s["value_pixels"])
    total_px = sum(value_pixels.values()) or 1
    for value in sorted(value_pixels):
        lines.append(f"  value {value:>3}: {value_pixels[value]:>12,} px ({value_pixels[value] * 100.0 / total_px:.4f}%)")
    lines.append("• Mask values were copied unchanged. In the metadata, any value > 0 counts as change.")
    lines.append("=" * 80)
    with open(os.path.join(staging_dir, "preparation_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def remove_partial_output(staging_dir, dataset_dir):
    """Delete ONLY the '.incomplete' folder created by this run. Never touches source data."""
    real = os.path.realpath(staging_dir)
    if not real.endswith(STAGING_SUFFIX):
        print(f"Refusing to delete '{staging_dir}': it is not a '{STAGING_SUFFIX}' folder.", file=sys.stderr)
        return
    for split in SPLITS:
        if is_inside(os.path.join(dataset_dir, split), real):
            print(f"Refusing to delete '{staging_dir}': it contains source data.", file=sys.stderr)
            return
    if os.path.isdir(real):
        shutil.rmtree(real)
        print(f"Removed partial output folder: {display_path(staging_dir)} (source data untouched).",
              file=sys.stderr)


def execute_tiling(dataset_dir, output_dir, patch_size=256):
    """Generate all patches with full pre-validation, verification and cleanup on failure."""
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required for dataset tiling.")
    if os.path.exists(output_dir):
        raise FileExistsError(f"Output directory '{output_dir}' already exists. Refusing to overwrite it.")
    staging_dir = output_dir + STAGING_SUFFIX
    if os.path.exists(staging_dir):
        raise FileExistsError(f"Leftover folder '{staging_dir}' from an interrupted run exists. "
                              "Inspect it and delete it manually before retrying.")

    print("=== LEVIR-CD DATASET TILING EXECUTION ===")
    print(f"Source:  {display_path(dataset_dir)}")
    print(f"Target:  {display_path(output_dir)}  (written to '{os.path.basename(staging_dir)}' first)")

    print("Phase 1: Re-validating all triplets across all splits...")
    split_triplets = {}
    for split in SPLITS:
        audit = validate_split_integrity(os.path.join(dataset_dir, split), patch_size)
        if audit["validation_errors"]:
            raise ValueError(f"validation errors in split '{split}': {audit['validation_errors'][:3]}")
        split_triplets[split] = audit["validated_triplets"]
    total = sum(len(t) for t in split_triplets.values())
    print(f"Validation successful: {total} triplets fully decoded and verified.")

    os.makedirs(staging_dir, exist_ok=False)  # from here on, this run owns the staging folder
    start_time = time.time()
    try:
        print("Phase 2: Writing and verifying patches...")
        summaries = write_patches(dataset_dir, staging_dir, split_triplets, patch_size)
        print("\nPhase 3: Counting files on disk and cross-checking the metadata...")
        disk = verify_output_on_disk(staging_dir, summaries)
        report = write_preparation_report(staging_dir, output_dir, dataset_dir, patch_size,
                                          summaries, disk, time.time() - start_time)
        os.rename(staging_dir, output_dir)
    except BaseException as exc:  # includes Ctrl+C
        print(f"\n[FAILED]: {type(exc).__name__}: {exc}", file=sys.stderr)
        remove_partial_output(staging_dir, dataset_dir)
        raise

    print("\n" + report)
    print(f"\nDataset tiling completed in {time.time() - start_time:.1f} seconds.")
    print(f"Report saved to: {display_path(os.path.join(output_dir, 'preparation_report.txt'))}")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LEVIR-CD Dataset Tiling Pipeline (LandWatch AI - Project 50)")
    parser.add_argument("--execute", action="store_true", default=False,
                        help="Generate the patches and CSVs (default: dry run only).")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Square patch size in pixels (default: 256).")
    parser.add_argument("--dataset-dir", default=DEFAULT_LEVIR_DIR,
                        help="LEVIR-CD folder containing train/, val/ and test/ "
                             "(default: <project>/datasets/LEVIR-CD).")
    parser.add_argument("--output-dir", default=None,
                        help="Output folder (default: <dataset-dir>/processed_<patch-size>).")
    args = parser.parse_args()

    if Image is None:
        print("Error: Pillow is required. Install it with:  python3 -m pip install Pillow")
        sys.exit(1)
    if args.patch_size <= 0:
        print("Error: --patch-size must be a positive integer.")
        sys.exit(1)

    levir_dir = os.path.abspath(args.dataset_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(levir_dir, f"processed_{args.patch_size}"))

    if not os.path.isdir(levir_dir):
        print(f"Error: LEVIR-CD directory not found at {levir_dir}")
        sys.exit(1)
    for split in SPLITS:
        if is_inside(output_dir, os.path.join(levir_dir, split)):
            print(f"Error: the output folder must not be inside the source '{split}' folder.")
            sys.exit(1)
    if not os.path.isdir(os.path.dirname(output_dir)):
        print(f"Error: the parent folder of the output does not exist: {os.path.dirname(output_dir)}")
        sys.exit(1)

    print(f"Running dry-run audit on: {display_path(levir_dir)} "
          f"(patch size {args.patch_size}x{args.patch_size}). This decodes every image and takes a few minutes...")
    results = perform_dry_run(levir_dir, output_dir, patch_size=args.patch_size)
    report_text = format_dry_run_report(results)
    print("\n" + report_text)

    report_path = os.path.join(levir_dir, "dataset_tiling_dryrun_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nDry-run report saved to: {display_path(report_path)}")

    if not args.execute:
        print("\n[DRY RUN COMPLETE]: No patches or output folders were created; "
              "only the report file above was written.")
        print("To generate the patches after reviewing this report, run again with '--execute'.")
        return

    blockers = []
    if results["overall"]["all_errors"]:
        blockers.append(f"{len(results['overall']['all_errors'])} validation error(s)")
    if results["output_dir_already_exists"]:
        blockers.append(f"output folder already exists: {display_path(output_dir)}")
    if results["staging_dir_already_exists"]:
        blockers.append(f"leftover folder from an interrupted run: {display_path(results['staging_dir'])}")
    if not results["overall"]["disk_has_sufficient_space"]:
        blockers.append("insufficient free disk space")
    if blockers:
        print("\n[ABORT]: Execution halted - " + "; ".join(blockers) + ".")
        sys.exit(1)

    print("\nAll pre-execution checks PASSED. Starting patch generation...")
    try:
        execute_tiling(levir_dir, output_dir, patch_size=args.patch_size)
    except KeyboardInterrupt:
        print("\n[ABORT]: Interrupted by user. Partial output was removed.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ABORT]: Tiling failed and partial output was removed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
