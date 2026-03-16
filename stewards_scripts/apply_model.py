"""Apply a trained sidewalk fix model and produce network + polygon GeoJSON.

Takes zoom-18 tile IDs (or a GeoJSON to derive them from), loads a trained
model, runs inference on zoom-19 children, re-polygonizes, runs Tile2Net
topology to produce a sidewalk network, and saves both as GeoJSON.

Usage
-----
python apply_model.py \
    --tile_ids 79325_97025 79322_97010 79303_97006 \
    --model_path ./outputs/suggestion_model.pt \
    --tiles_dir .../tiles \
    --t2n_dir .../masks_tile2net_polygons \
    --conf_dir .../masks_confidence \
    --output_polygons ./outputs/polygons.geojson \
    --output_network ./outputs/network.geojson \
    --head fix

Or derive tile IDs from a GeoJSON:

python apply_model.py \
    --geojson ./suggestion_sample.geojson \
    --model_path ./outputs/suggestion_model.pt \
    ...
"""

import argparse
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent / "helper_scripts"))
sys.path.insert(0, str(Path(__file__).parent))
from polygon_fixing import get_tile_bounds
from tile2net_training_utils import ResidualFixNet
from train_from_suggestions import (
    mask_to_polygons,
    run_inference,
)

# Tile2Net topology imports (from pip install -e .)
from tile2net.raster.project import Project
from tile2net.raster.raster import Raster
from tile2net.raster.pednet import PedNet


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def zoom18_to_zoom19(tile_ids_18):
    """Expand zoom-18 tile IDs to their 4 zoom-19 children."""
    tile_ids_19 = []
    for tid in tile_ids_18:
        x, y = map(int, tid.split("_"))
        for dx in range(2):
            for dy in range(2):
                tile_ids_19.append(f"{x * 2 + dx}_{y * 2 + dy}")
    return tile_ids_19


def filter_existing_tiles(tile_ids, tiles_dir, t2n_dir, conf_dir):
    """Keep only tile IDs where RGB, T2N, and confidence files exist."""
    tiles_dir = Path(tiles_dir)
    t2n_dir = Path(t2n_dir)
    conf_dir = Path(conf_dir)

    valid = []
    for tid in tile_ids:
        if ((tiles_dir / f"{tid}.jpg").exists()
                and (t2n_dir / f"{tid}.png").exists()
                and (conf_dir / f"{tid}.png").exists()):
            valid.append(tid)
    return valid


def compute_bbox_from_tiles(tile_ids_18):
    """Compute lon/lat bounding box from zoom-18 tile IDs."""
    all_w, all_s, all_e, all_n = [], [], [], []
    for tid in tile_ids_18:
        x, y = map(int, tid.split("_"))
        w, s, e, n = get_tile_bounds(x, y, zoom=18)
        all_w.append(w)
        all_s.append(s)
        all_e.append(e)
        all_n.append(n)
    return (min(all_w), min(all_s), max(all_e), max(all_n))


def polygonize_predictions(predictions):
    """Convert predicted masks to GeoDataFrame with zoom-18 tile_id."""
    all_polys = []
    all_tile_ids = []

    for tid, mask in tqdm(predictions.items(), desc="Re-polygonizing"):
        if mask.max() == 0:
            continue
        xtile, ytile = map(int, tid.split("_"))
        polys = mask_to_polygons(mask, xtile, ytile)
        tile_id_18 = f"{xtile // 2}_{ytile // 2}"
        for p in polys:
            all_polys.append(p)
            all_tile_ids.append(tile_id_18)

    if not all_polys:
        return gpd.GeoDataFrame(columns=["tile_id", "geometry"], crs="EPSG:4326")

    return gpd.GeoDataFrame(
        {"tile_id": all_tile_ids},
        geometry=all_polys,
        crs="EPSG:4326",
    )


def run_topology(gdf_polygons, bbox):
    """Run Tile2Net topology engine on polygons to produce network lines.

    Parameters
    ----------
    gdf_polygons : GeoDataFrame
        Polygon geometries with CRS EPSG:4326.
    bbox : tuple
        (min_lon, min_lat, max_lon, max_lat).

    Returns
    -------
    GeoDataFrame with network line geometries.
    """
    gdf = gdf_polygons.copy()
    if "f_type" not in gdf.columns:
        gdf["f_type"] = "sidewalk"

    print("\nCreating raster + project for topology...")
    raster = Raster(
        location=list(bbox),
        name="polygon_injection",
        zoom=19,
        base_tilesize=256,
    )

    project = Project("polygon_network_output", "outputs", raster)

    print("Running topology engine...")
    pednet = PedNet(gdf, project)
    pednet.convert_whole_poly2line()

    network = pednet.complete_net
    print(f"  {len(network)} network edges produced")
    return network


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply trained model and produce network + polygon GeoJSON."
    )

    # Tile ID source (one of these required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tile_ids", nargs="+",
                       help="Zoom-18 tile IDs (e.g., 79325_97025 79322_97010)")
    group.add_argument("--geojson",
                       help="GeoJSON file — tile IDs extracted from 'tile_id' column")

    # Model + data paths
    parser.add_argument("--model_path", required=True, help="Path to trained model .pt")
    parser.add_argument("--tiles_dir", required=True, help="RGB satellite tiles directory")
    parser.add_argument("--t2n_dir", required=True, help="T2N rasterized polygon masks directory")
    parser.add_argument("--conf_dir", required=True, help="T2N confidence masks directory")

    # Outputs
    parser.add_argument("--output_polygons", required=True, help="Output polygon GeoJSON path")
    parser.add_argument("--output_network", required=True, help="Output network GeoJSON path")

    # Options
    parser.add_argument("--head", choices=["fix", "full"], default="fix",
                        help="Model head to use (default: fix)")

    args = parser.parse_args()

    # ── Step 1: Get zoom-18 tile IDs ──
    if args.geojson:
        print(f"Loading GeoJSON: {args.geojson}")
        gdf_input = gpd.read_file(args.geojson)
        tile_ids_18 = gdf_input["tile_id"].unique().tolist()
    else:
        tile_ids_18 = args.tile_ids

    print(f"  {len(tile_ids_18)} zoom-18 tiles")

    # ── Step 2: Expand to zoom-19 children ──
    tile_ids_19 = zoom18_to_zoom19(tile_ids_18)
    print(f"  {len(tile_ids_19)} zoom-19 children")

    # ── Step 3: Filter to tiles with existing data ──
    tile_ids_19 = filter_existing_tiles(
        tile_ids_19, args.tiles_dir, args.t2n_dir, args.conf_dir
    )
    print(f"  {len(tile_ids_19)} zoom-19 tiles with data")

    if not tile_ids_19:
        print("No valid tiles found. Check paths and tile IDs.")
        sys.exit(1)

    # ── Step 4: Load model ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nUsing device: {device}")

    print(f"Loading model: {args.model_path}")
    model = ResidualFixNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # ── Step 5: Run inference ──
    print(f"\nRunning inference ({args.head} head) on {len(tile_ids_19)} tiles...")
    predictions = run_inference(
        model, tile_ids_19, device, args.head,
        tiles_dir=args.tiles_dir,
        t2n_dir=args.t2n_dir,
        conf_dir=args.conf_dir,
    )
    print(f"  {len(predictions)} masks produced")

    # ── Step 6: Re-polygonize ──
    print("\nRe-polygonizing predicted masks...")
    gdf_polygons = polygonize_predictions(predictions)
    print(f"  {len(gdf_polygons)} polygons from {gdf_polygons['tile_id'].nunique()} zoom-18 tiles")

    if len(gdf_polygons) == 0:
        print("No polygons produced. Exiting.")
        sys.exit(1)

    # Save polygon GeoJSON
    out_poly = Path(args.output_polygons)
    out_poly.parent.mkdir(parents=True, exist_ok=True)
    gdf_polygons.to_file(out_poly, driver="GeoJSON")
    print(f"  Polygon GeoJSON saved to: {out_poly}")

    # ── Step 7: Compute bbox from zoom-18 tiles ──
    bbox = compute_bbox_from_tiles(tile_ids_18)
    print(f"\nBounding box: {bbox}")

    # ── Step 8: Run Tile2Net topology ──
    network = run_topology(gdf_polygons, bbox)

    # Save network GeoJSON
    out_net = Path(args.output_network)
    out_net.parent.mkdir(parents=True, exist_ok=True)
    network.to_file(out_net, driver="GeoJSON")
    print(f"  Network GeoJSON saved to: {out_net}")

    print(f"\nDone! {len(gdf_polygons)} polygons, {len(network)} network edges.")


if __name__ == "__main__":
    main()
