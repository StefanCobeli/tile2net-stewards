# Stewards Scripts

Three scripts for generating suggestions, training, and applying sidewalk polygon fix models.

## Scripts

### 1. `train_from_suggestions.py` — Train a model from polygon suggestions

Takes a GeoJSON of polygon suggestions, trains a ResidualFixNet model, runs inference, and outputs corrected polygons as GeoJSON.

```bash
python stewards_scripts/train_from_suggestions.py \
    --geojson ./suggestion_sample.geojson \
    --tiles_dir /path/to/tiles \
    --t2n_dir /path/to/masks_tile2net_polygons \
    --conf_dir /path/to/masks_confidence \
    --output ./outputs/corrected_polygons.geojson \
    --model_output ./outputs/suggestion_model.pt \
    --epochs 200 \
    --head fix
```

Optional: add `--enable_remove` to train with the add+remove head (default is add-only).

### 2. `apply_model.py` — Apply a trained model and produce global outputs

Loads a trained model, runs inference on specified tiles, re-polygonizes, runs Tile2Net topology on the new polygons, then merges the results into the original global polygon and network files (replacing old data for the input tiles). New network endpoints are snapped to the existing network within a configurable tolerance.

```bash
python stewards_scripts/apply_model.py \
    --tile_ids 79337_97000 79315_97000 \
    --model_path ./outputs/suggestion_model.pt \
    --tiles_dir /path/to/tiles \
    --t2n_dir /path/to/masks_tile2net_polygons \
    --conf_dir /path/to/masks_confidence \
    --original_polygons /path/to/polygon_suggestions_zoom18.geojson \
    --original_network /path/to/network_original.geojson \
    --output_polygons ./outputs/corrected_polygons_global.geojson \
    --output_network ./outputs/corrected_network_global.geojson \
    --head fix
```

Tile IDs can also be derived from a GeoJSON file using `--geojson` instead of `--tile_ids`.

Optional flags:
- `--enable_remove` — use if the model was trained with the remove head
- `--snap_tolerance 3.0` — snap distance in meters for connecting new network to existing (default: 3.0)

### 0. `generate_suggestions.py` — Generate polygon suggestions

Takes an input polygon file (GeoJSON or SHP), clips polygons to zoom-18 tiles, generates elongation suggestions, and saves the result with `tile_id` and `n_suggestion` attributes. Filters to sidewalks only if the input has an `f_type` column.

```bash
python stewards_scripts/generate_suggestions.py \
    --input /path/to/polygons.shp \
    --tiles_dir /path/to/tiles \
    --output ./outputs/polygon_suggestions_zoom18.geojson
```

Optional flags:
- `--elongation_dist 50` — extension distance in meters (default: 50)
- `--convexity_threshold 0.8` — skip non-convex polygons (default: 0.8)
- `--max_elongate N` — limit elongation suggestions per tile (default: unlimited)
- `--epsg_utm 32619` — UTM projection code (default: 32619 = Boston)


## Setup

### 1. Clone tile2net

```bash
git clone https://github.com/VIDA-NYU/tile2net.git
cd tile2net
```

### 2. Download tile data

Download the satellite RGB tiles, T2N polygon rasters, and confidence masks from:

https://uofi.box.com/s/q2028l50qdgt871lbc0z01wi4oxd0bfu

Extract the contents into a folder (e.g., `./data/`). The scripts expect three subdirectories: `tiles/`, `masks_tile2net_polygons/`, and `masks_confidence/`.

### 3. Create conda environment

```bash
conda create -n tile2net-env python=3.11 -y
```

### 4. Install tile2net

```bash
conda run -n tile2net-env pip install -e .
```

### 5. Install ML dependencies

```bash
conda run -n tile2net-env pip install torch torchvision segmentation-models-pytorch rasterio opencv-python-headless tqdm Pillow
conda run -n tile2net-env pip install "numpy<2.0"
```

### 6. Run scripts

All scripts should be run from the tile2net repository root:

```bash
conda activate tile2net-env
python stewards_scripts/generate_suggestions.py --input ...
python stewards_scripts/train_from_suggestions.py --geojson ...
python stewards_scripts/apply_model.py --tile_ids ...
```
