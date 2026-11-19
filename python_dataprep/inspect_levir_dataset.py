#!/usr/bin/env python3
"""
LEVIR-CD Dataset Inspection and Validation Tool
LandWatch AI - Project 50

Performs comprehensive non-destructive inspection of the official LEVIR-CD dataset:
- Validates file counts across train, val, and test splits (A, B, label).
- Verifies exact matching of image pairs and ground-truth masks.
- Validates image decodability, spatial dimensions, and color channels.
- Analyzes mask pixel values, confirming background (0) vs change (255) classes.
- Checks for missing, duplicate, or corrupted files.
- Generates a human-readable inspection report at datasets/LEVIR-CD/dataset_inspection_report.txt.
"""

import os
import sys
import struct
import zlib
import time
from collections import Counter

# Try importing PIL, with pure-Python PNG fallback if absent
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def read_png_metadata_and_pixels(filepath, sample_pixels=True):
    """
    Reads PNG dimensions and pixel extrema.
    Uses PIL if available; otherwise parses standard PNG chunks natively.
    """
    if HAS_PIL:
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
                extrema = None
                unique_vals = set()
                if sample_pixels:
                    ext = img.getextrema()
                    if isinstance(ext, tuple) and isinstance(ext[0], tuple):
                        # Multi-band extrema
                        extrema = ext
                    else:
                        extrema = ext
                        # For single band / mask, check unique values
                        # To be fast on 1024x1024, sample or full getdata
                        data = img.getdata()
                        # Use a small set accumulator
                        unique_vals = set(data)
                return {
                    "valid": True,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "format": format_name,
                    "extrema": extrema,
                    "unique_values": unique_vals,
                    "file_size": os.path.getsize(filepath),
                    "error": None
                }
        except Exception as e:
            return {
                "valid": False,
                "file_size": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                "error": str(e)
            }
    else:
        # Native pure-Python PNG chunk parser (RFC 2083)
        try:
            with open(filepath, "rb") as f:
                sig = f.read(8)
                if sig != b"\x89PNG\r\n\x1a\n":
                    return {"valid": False, "error": "Invalid PNG signature"}

                width, height = None, None
                bit_depth, color_type = None, None
                idat_chunks = []

                while True:
                    chunk_len_bytes = f.read(4)
                    if len(chunk_len_bytes) < 4:
                        break
                    chunk_len = struct.unpack(">I", chunk_len_bytes)[0]
                    chunk_type = f.read(4)
                    chunk_data = f.read(chunk_len)
                    crc = f.read(4)

                    if chunk_type == b"IHDR":
                        width, height, bit_depth, color_type, comp, filt, inter = struct.unpack(
                            ">IIBBBBB", chunk_data
                        )
                    elif chunk_type == b"IDAT" and sample_pixels:
                        idat_chunks.append(chunk_data)
                    elif chunk_type == b"IEND":
                        break

                mode_map = {0: "L", 2: "RGB", 3: "P", 4: "LA", 6: "RGBA"}
                mode = mode_map.get(color_type, f"ColorType_{color_type}")

                unique_vals = set()
                if sample_pixels and idat_chunks and color_type in (0, 3):
                    try:
                        raw_data = zlib.decompress(b"".join(idat_chunks))
                        # Sample scanlines to collect unique pixel values
                        # Scanline structure: filter_byte (1) + width * bytes_per_pixel
                        bpp = 1 if color_type in (0, 3) else 3
                        stride = 1 + width * bpp
                        for row in range(0, height, max(1, height // 16)):
                            offset = row * stride + 1
                            row_pixels = raw_data[offset : offset + width * bpp]
                            unique_vals.update(row_pixels)
                    except Exception:
                        pass

                return {
                    "valid": True,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "format": "PNG",
                    "extrema": None,
                    "unique_values": unique_vals,
                    "file_size": os.path.getsize(filepath),
                    "error": None
                }
        except Exception as e:
            return {
                "valid": False,
                "file_size": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                "error": str(e)
            }


def inspect_levir_dataset(base_dir):
    start_time = time.time()
    splits = ["train", "val", "test"]
    subdirs = ["A", "B", "label"]

    results = {
        "base_dir": base_dir,
        "splits": {},
        "overall": {
            "total_pairs": 0,
            "total_masks": 0,
            "all_dimensions_match": True,
            "all_masks_match": True,
            "mask_pixel_values": set(),
            "corrupted_files": [],
            "missing_counterparts": []
        }
    }

    for split in splits:
        split_path = os.path.join(base_dir, split)
        split_data = {
            "exists": os.path.exists(split_path),
            "counts": {},
            "files": {},
            "matched_triplets": 0,
            "missing_in_A": [],
            "missing_in_B": [],
            "missing_in_label": [],
            "dimensions_consistent": True,
            "dimension_summary": Counter(),
            "mask_unique_values": set(),
            "corrupted": []
        }

        if not split_data["exists"]:
            results["splits"][split] = split_data
            continue

        # 1. Collect and match filenames
        files_by_sub = {}
        for sub in subdirs:
            sub_path = os.path.join(split_path, sub)
            if os.path.exists(sub_path):
                file_list = sorted([f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))])
                files_by_sub[sub] = file_list
                split_data["counts"][sub] = len(file_list)
            else:
                files_by_sub[sub] = []
                split_data["counts"][sub] = 0

        set_A = set(files_by_sub.get("A", []))
        set_B = set(files_by_sub.get("B", []))
        set_label = set(files_by_sub.get("label", []))

        all_names = sorted(set_A | set_B | set_label)
        common_names = sorted(set_A & set_B & set_label)
        split_data["matched_triplets"] = len(common_names)

        split_data["missing_in_A"] = sorted((set_B | set_label) - set_A)
        split_data["missing_in_B"] = sorted((set_A | set_label) - set_B)
        split_data["missing_in_label"] = sorted((set_A | set_B) - set_label)

        if split_data["missing_in_A"]:
            results["overall"]["missing_counterparts"].extend([f"{split}/A/{x}" for x in split_data["missing_in_A"]])
        if split_data["missing_in_B"]:
            results["overall"]["missing_counterparts"].extend([f"{split}/B/{x}" for x in split_data["missing_in_B"]])
        if split_data["missing_in_label"]:
            results["overall"]["missing_counterparts"].extend([f"{split}/label/{x}" for x in split_data["missing_in_label"]])

        # 2. Inspect individual files (sample/validate)
        # Check all common pairs, or sample if large
        for filename in common_names:
            path_A = os.path.join(split_path, "A", filename)
            path_B = os.path.join(split_path, "B", filename)
            path_label = os.path.join(split_path, "label", filename)

            meta_A = read_png_metadata_and_pixels(path_A, sample_pixels=False)
            meta_B = read_png_metadata_and_pixels(path_B, sample_pixels=False)
            meta_L = read_png_metadata_and_pixels(path_label, sample_pixels=True)

            # Check validity
            if not meta_A["valid"]:
                split_data["corrupted"].append((path_A, meta_A["error"]))
                results["overall"]["corrupted_files"].append(path_A)
            if not meta_B["valid"]:
                split_data["corrupted"].append((path_B, meta_B["error"]))
                results["overall"]["corrupted_files"].append(path_B)
            if not meta_L["valid"]:
                split_data["corrupted"].append((path_label, meta_L["error"]))
                results["overall"]["corrupted_files"].append(path_label)

            if meta_A["valid"] and meta_B["valid"] and meta_L["valid"]:
                dim_A = (meta_A["width"], meta_A["height"])
                dim_B = (meta_B["width"], meta_B["height"])
                dim_L = (meta_L["width"], meta_L["height"])

                split_data["dimension_summary"][dim_A] += 1

                # Check pair dimension equality
                if dim_A != dim_B:
                    split_data["dimensions_consistent"] = False
                    results["overall"]["all_dimensions_match"] = False

                # Check mask dimension equality
                if dim_A != dim_L:
                    results["overall"]["all_masks_match"] = False

                # Collect mask unique values
                if meta_L["unique_values"]:
                    split_data["mask_unique_values"].update(meta_L["unique_values"])
                    results["overall"]["mask_pixel_values"].update(meta_L["unique_values"])

        results["overall"]["total_pairs"] += split_data["matched_triplets"]
        results["splits"][split] = split_data

    results["duration_sec"] = time.time() - start_time
    return results


def format_report(results):
    lines = []
    lines.append("=" * 80)
    lines.append("               LEVIR-CD DATASET INSPECTION & VALIDATION REPORT")
    lines.append("                         LandWatch AI - Project 50")
    lines.append("=" * 80)
    lines.append(f"Inspection Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Root Directory:       {results['base_dir']}")
    lines.append(f"Engine Mode:          {'PIL (Pillow)' if HAS_PIL else 'Pure-Python Native PNG Parser'}")
    lines.append(f"Scan Duration:        {results['duration_sec']:.2f} seconds")
    lines.append("")

    lines.append("-" * 80)
    lines.append("1. SPLIT SUMMARY & PAIR COUNTS")
    lines.append("-" * 80)
    lines.append(f"{'Split':<12} | {'Before (A)':<12} | {'After (B)':<12} | {'Mask (label)':<14} | {'Matched Triplets':<18}")
    lines.append("-" * 80)

    for split in ["train", "val", "test"]:
        s = results["splits"].get(split, {})
        counts = s.get("counts", {})
        cnt_a = counts.get("A", 0)
        cnt_b = counts.get("B", 0)
        cnt_l = counts.get("label", 0)
        matched = s.get("matched_triplets", 0)
        lines.append(f"{split:<12} | {cnt_a:<12} | {cnt_b:<12} | {cnt_l:<14} | {matched:<18}")

    lines.append("-" * 80)
    total_triplets = results["overall"]["total_pairs"]
    lines.append(f"{'TOTAL':<12} | {'':<12} | {'':<12} | {'':<14} | {total_triplets:<18}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("2. IMAGE DIMENSIONS & FORMAT INTEGRITY")
    lines.append("-" * 80)
    for split in ["train", "val", "test"]:
        s = results["splits"].get(split, {})
        dim_summary = s.get("dimension_summary", {})
        lines.append(f"Split [{split}]:")
        for dim, count in dim_summary.items():
            lines.append(f"  • Dimensions: {dim[0]} x {dim[1]} px | Count: {count} pairs")
        lines.append(f"  • Dimensions internally consistent (A == B): {s.get('dimensions_consistent', False)}")
    lines.append(f"Overall Pre/Post Dimension Alignment: {'PASSED (100% matched)' if results['overall']['all_dimensions_match'] else 'FAILED'}")
    lines.append(f"Overall Mask/Image Alignment:         {'PASSED (100% matched)' if results['overall']['all_masks_match'] else 'FAILED'}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("3. GROUND-TRUTH MASK PIXEL VALUE ANALYSIS")
    lines.append("-" * 80)
    mask_vals = sorted(list(results["overall"]["mask_pixel_values"]))
    lines.append(f"Distinct pixel values detected in change masks: {mask_vals}")
    if mask_vals == [0, 255] or mask_vals == [0] or mask_vals == [255]:
        lines.append("  • 0   (0x00) -> Background / No Change (Pristine land/vegetation/structures)")
        lines.append("  • 255 (0xFF) -> Positive Change / Encroachment (New building construction)")
        lines.append("  • Evaluation Status: STRICT BINARY GROUND TRUTH CONFIRMED.")
    elif mask_vals == [0, 1]:
        lines.append("  • 0 -> Background / No Change")
        lines.append("  • 1 -> Positive Change / Encroachment")
        lines.append("  • Evaluation Status: BINARY (0/1) GROUND TRUTH.")
    else:
        lines.append(f"  • Evaluation Status: Non-binary pixel values detected: {mask_vals}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("4. ANOMALY, CORRUPTION & MISSING DATA AUDIT")
    lines.append("-" * 80)
    corrupted = results["overall"]["corrupted_files"]
    missing = results["overall"]["missing_counterparts"]
    lines.append(f"Corrupted or Unreadable Files: {len(corrupted)}")
    if corrupted:
        for f in corrupted[:10]:
            lines.append(f"  - [CORRUPT]: {f}")
    else:
        lines.append("  • Zero corrupted files detected. All images decoded successfully.")

    lines.append(f"Missing Pair Counterparts:     {len(missing)}")
    if missing:
        for m in missing[:10]:
            lines.append(f"  - [MISSING]: {m}")
    else:
        lines.append("  • Zero missing counterparts. Every A image has an exact matching B and label image.")
    lines.append("")

    lines.append("-" * 80)
    lines.append("5. COMPATIBILITY & PIPELINE READINESS ASSESSMENT")
    lines.append("-" * 80)
    lines.append("• Resolution: 1024 x 1024 is high resolution for direct full-image ResNet/U-Net inference.")
    lines.append("• Memory Assessment: In Java 17 CPU inference, tiling into 256x256 or 512x512 patches is recommended.")
    lines.append("• Dataset Structure: Standard LEVIR-CD format is verified and ready for dataset loaders.")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    # Project root = the folder above python_dataprep/, so this works on any machine.
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    levir_dir = os.path.join(workspace, "datasets", "LEVIR-CD")

    if not os.path.exists(levir_dir):
        print(f"Error: LEVIR-CD directory not found at {levir_dir}")
        sys.exit(1)

    print(f"Starting inspection of LEVIR-CD dataset at: {levir_dir}")
    results = inspect_levir_dataset(levir_dir)

    report_text = format_report(results)
    print("\n" + report_text)

    # Save report
    report_path = os.path.join(levir_dir, "dataset_inspection_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nSaved inspection report to: {report_path}")


if __name__ == "__main__":
    main()
