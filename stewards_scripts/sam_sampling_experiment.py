"""SAM-guided vs random sampling experiment for sidewalk fix training.

Compares two tile selection strategies:
  Part A: SAM-guided — iterate per category in descending area order,
          select tiles that need fixing (rIoU < threshold).
  Part B: Random — iterate tiles in random order with same rIoU check.

Both train on the same number of tiles (N_TILES_PER_CAT * 7) and are
evaluated on a shared validation set stratified by rIoU hardness buckets.

Usage
-----
python sam_sampling_experiment.py \
    --tiles_dir .../tiles \
    --t2n_dir .../masks_tile2net_polygons \
    --conf_dir .../masks_confidence \
    --gt_dir .../masks_groundtruth_polygons \
    --sam_csv .../sam3_descriptors.csv \
    --n_tiles_per_cat 3 \
    --riou_threshold 0.75 \
    --epochs 100 \
    --enable_remove
"""

import argparse
import csv
import os
import sys
import random
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "helper_scripts"))
from tile2net_gt_utils import get_all_tile_ids, load_tile_data
from tile2net_training_utils import (
    get_non_empty_tile_ids, stratified_split,
    ResidualFixNet, train_model, evaluate_tiles,
    compute_refined_metrics, display_predictions,
)


SAM3_CATEGORIES = [
    "fragmented_sidewalk", "meandering_sidewalk", "isolated_walkway",
    "paved_plaza_surface", "isolated_roadway", "vegetated_ground_surface",
    "road_marking_surface",
]


def compute_tile_riou(tid, tiles_dir, t2n_dir, conf_dir, gt_dir):
    """Compute rIoU for a single tile. Returns 1.0 for NaN (empty tiles)."""
    try:
        _, _, t2n, gt = load_tile_data(
            tid, tiles_dir=tiles_dir, t2n_dir=t2n_dir,
            conf_dir=conf_dir, gt_dir=gt_dir,
        )
        riou, _, _, _ = compute_refined_metrics(t2n, gt)
        if np.isnan(riou):
            return 1.0
        return riou
    except Exception:
        return 1.0  # treat missing/broken tiles as not needing fixing


def evaluate_per_bucket(model, val_ids, bucket_info, device, **kwargs):
    """Run evaluate_tiles per complexity bucket."""
    bucket_labels = list(bucket_info.keys())
    bucket_sizes = [bucket_info[b]["val"] for b in bucket_labels]
    bucket_starts = [0] + list(np.cumsum(bucket_sizes[:-1]))

    result = {}
    for label, start, size in zip(bucket_labels, bucket_starts, bucket_sizes):
        ids = val_ids[start : start + size]
        result[label] = evaluate_tiles(model, ids, device, **kwargs)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="SAM-guided vs random sampling experiment."
    )
    parser.add_argument("--tiles_dir", default=os.getenv("TILES_DIR"))
    parser.add_argument("--t2n_dir", default=os.getenv("T2N_DIR"))
    parser.add_argument("--conf_dir", default=os.getenv("CONF_DIR"))
    parser.add_argument("--gt_dir", default=os.getenv("GT_DIR"))
    parser.add_argument("--sam_csv", default=os.getenv("SAM_CSV"),
                        help="Path to sam3_descriptors.csv")
    parser.add_argument("--n_tiles_per_cat", type=int, default=3,
                        help="Tiles to select per SAM category (default: 3)")
    parser.add_argument("--riou_threshold", type=float, default=0.75,
                        help="rIoU threshold — tiles below this need fixing (default: 0.75)")
    parser.add_argument("--val_frac", type=float, default=0.2,
                        help="Validation fraction (default: 0.2)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--enable_remove", action="store_true",
                        help="Enable add+remove head")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Validate required paths
    for arg_name in ["tiles_dir", "t2n_dir", "conf_dir", "gt_dir", "sam_csv"]:
        if getattr(args, arg_name) is None:
            parser.error(f"--{arg_name} is required (set via CLI or .env)")

    dir_kwargs = dict(
        tiles_dir=args.tiles_dir, t2n_dir=args.t2n_dir,
        conf_dir=args.conf_dir, gt_dir=args.gt_dir,
    )

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── All tile IDs (intersection of all 4 folders) ──
    all_ids = get_all_tile_ids(**dir_kwargs)
    print(f"All tiles: {len(all_ids)}")

    # ── Stratified train/val split on non-empty tiles ──
    # Val set uses non-empty tiles (need GT to evaluate)
    non_empty_ids = get_non_empty_tile_ids(tile_ids=all_ids, **dir_kwargs)
    print(f"Non-empty tiles: {len(non_empty_ids)}")

    train_ids, val_ids, bucket_info, n_dropped = stratified_split(
        non_empty_ids, val_frac=args.val_frac, seed=args.seed, **dir_kwargs,
    )
    print(f"Val tiles: {len(val_ids)}, Dropped (NaN): {n_dropped}")
    for b, info in bucket_info.items():
        print(f"  {b}: train={info['train']}, val={info['val']}")

    # Training pool: all tiles except val
    all_ids_without_val = [t for t in all_ids if t not in set(val_ids)]
    print(f"Training pool (all minus val): {len(all_ids_without_val)}")

    # ── Load SAM3 descriptors ──
    sam3_data = {}
    with open(args.sam_csv) as f:
        for row in csv.DictReader(f):
            tid = row["tile_name"]
            sam3_data[tid] = {cat: float(row[f"{cat}_area"]) for cat in SAM3_CATEGORIES}

    n_total = args.n_tiles_per_cat * len(SAM3_CATEGORIES)
    print(f"\nTarget: {args.n_tiles_per_cat} tiles/category x "
          f"{len(SAM3_CATEGORIES)} categories = {n_total} tiles")
    print(f"rIoU threshold: {args.riou_threshold}")

    # ══════════════════════════════════════════════════════════════════════
    # Part A: SAM-guided sampling
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PART A: SAM-guided sampling")
    print(f"{'='*60}")

    training_pool_set = set(all_ids_without_val)
    checked_tiles = set()  # tiles already checked across categories
    sam_training_set = []
    sam_stats = {}  # per-category stats

    for cat in SAM3_CATEGORIES:
        # Sort pool tiles by this category's area, descending
        ranked = []
        for tid in training_pool_set:
            if tid in sam3_data and sam3_data[tid][cat] > 0:
                ranked.append((tid, sam3_data[tid][cat]))
        ranked.sort(key=lambda x: x[1], reverse=True)

        found = 0
        checked_this_cat = 0
        for tid, area_val in ranked:
            if tid in checked_tiles:
                continue  # already checked in a previous category
            checked_tiles.add(tid)
            checked_this_cat += 1

            riou = compute_tile_riou(tid, **dir_kwargs)
            if riou < args.riou_threshold:
                sam_training_set.append(tid)
                found += 1
                if found >= args.n_tiles_per_cat:
                    break

        sam_stats[cat] = {
            "found": found,
            "checked": checked_this_cat,
            "pool_size": len(ranked),
        }
        print(f"  {cat:<30}  checked={checked_this_cat:>4}  "
              f"found={found}/{args.n_tiles_per_cat}  pool={len(ranked)}")

    total_checked_sam = sum(s["checked"] for s in sam_stats.values())
    print(f"\nSAM total: {len(sam_training_set)} tiles selected, "
          f"{total_checked_sam} tiles checked")

    # ══════════════════════════════════════════════════════════════════════
    # Part B: Random sampling
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PART B: Random sampling")
    print(f"{'='*60}")

    random.seed(args.seed)
    random_pool = list(all_ids_without_val)
    random.shuffle(random_pool)

    random_training_set = []
    checked_random = 0

    for tid in random_pool:
        checked_random += 1
        riou = compute_tile_riou(tid, **dir_kwargs)
        if riou < args.riou_threshold:
            random_training_set.append(tid)
            if len(random_training_set) >= n_total:
                break

    print(f"Random: {len(random_training_set)} tiles selected, "
          f"{checked_random} tiles checked")

    # ══════════════════════════════════════════════════════════════════════
    # Comparison: iteration efficiency
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SAMPLING EFFICIENCY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Method':<30}  {'Tiles checked':>15}  {'Tiles selected':>15}")
    print(f"{'-'*62}")
    for cat in SAM3_CATEGORIES:
        s = sam_stats[cat]
        print(f"  SAM: {cat:<24}  {s['checked']:>15}  {s['found']:>15}")
    print(f"  {'SAM total':<28}  {total_checked_sam:>15}  {len(sam_training_set):>15}")
    print(f"  {'Random total':<28}  {checked_random:>15}  {len(random_training_set):>15}")

    # ══════════════════════════════════════════════════════════════════════
    # Train Part A: SAM model
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"TRAINING: SAM-guided ({len(sam_training_set)} tiles)")
    print(f"{'='*60}")

    model_sam = ResidualFixNet(
        warm_start_t2n=False, enable_remove=args.enable_remove
    ).to(device)

    hist_sam, tb_sam, vb_sam, vf_sam = train_model(
        model_sam, sam_training_set, val_ids, bucket_info, device,
        n_epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, eval_every=args.eval_every, **dir_kwargs,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Train Part B: Random model
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"TRAINING: Random ({len(random_training_set)} tiles)")
    print(f"{'='*60}")

    model_random = ResidualFixNet(
        warm_start_t2n=False, enable_remove=args.enable_remove
    ).to(device)

    hist_random, tb_random, vb_random, vf_random = train_model(
        model_random, random_training_set, val_ids, bucket_info, device,
        n_epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, eval_every=args.eval_every, **dir_kwargs,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Results: Overall
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("RESULTS: Overall validation set")
    print(f"{'='*60}")

    metrics = ["fix_riou", "fix_rrec", "fix_iou", "fix_rec"]
    scenarios = {
        "SAM-guided":  vf_sam,
        "Random":      vf_random,
    }

    rows = []
    # T2N baseline
    t2n_row = {"Scenario": "T2N baseline"}
    for m in metrics:
        t2n_key = m.replace("fix_", "t2n_")
        t2n_row[m] = f"{vf_sam[t2n_key]:.4f}"
    rows.append(t2n_row)

    for name, vf in scenarios.items():
        row = {"Scenario": name}
        for m in metrics:
            t2n_key = m.replace("fix_", "t2n_")
            delta = vf[m] - vf[t2n_key]
            row[m] = f"{vf[m]:.4f} ({delta:+.4f})"
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df[["Scenario"] + metrics].to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════
    # Results: Per bucket
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("RESULTS: Per complexity bucket")
    print(f"{'='*60}")

    eval_kwargs = dict(**dir_kwargs)
    bm_sam = evaluate_per_bucket(model_sam, val_ids, bucket_info, device, **eval_kwargs)
    bm_random = evaluate_per_bucket(model_random, val_ids, bucket_info, device, **eval_kwargs)

    bucket_scenarios = {
        "SAM-guided": bm_sam,
        "Random":     bm_random,
    }

    for bucket in bucket_info.keys():
        print(f"\n  Bucket: {bucket}  ({bucket_info[bucket]['val']} val tiles)")
        print(f"  {'-'*55}")
        rows = []
        t2n_row = {"Scenario": "T2N baseline"}
        for m in metrics:
            t2n_key = m.replace("fix_", "t2n_")
            t2n_row[m] = f"{bm_sam[bucket][t2n_key]:.4f}"
        rows.append(t2n_row)

        for name, bm in bucket_scenarios.items():
            row = {"Scenario": name}
            for m in metrics:
                t2n_key = m.replace("fix_", "t2n_")
                delta = bm[bucket][m] - bm[bucket][t2n_key]
                row[m] = f"{bm[bucket][m]:.4f} ({delta:+.4f})"
            rows.append(row)

        df = pd.DataFrame(rows)
        print(df[["Scenario"] + metrics].to_string(index=False))

    print(f"\nDone!")


if __name__ == "__main__":
    main()
