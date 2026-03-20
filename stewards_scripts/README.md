# Stewards Scripts

Scripts for generating suggestions, training, and applying sidewalk polygon fix models.

## Configuration

All scripts read shared data paths from a `.env` file in the repository root. This avoids repeating long paths in every command.

```bash
cp .env.example .env
# Edit .env with your actual data paths
```

`.env` variables:
```
TILES_DIR          — RGB satellite tile images (zoom-19)
T2N_DIR            — T2N rasterized polygon masks
CONF_DIR           — T2N confidence masks
GT_DIR             — Ground truth masks
ORIGINAL_POLYGONS  — Global polygon GeoJSON (with tile_id and n_suggestion)
ORIGINAL_NETWORK   — Global network GeoJSON
SUGGESTION_GEOJSON — Polygon suggestions sample for training
MODEL_PATH         — Trained model path (shared between train and apply)
```

All paths can still be overridden via CLI arguments.

## Scripts

### 1. `train_from_suggestions.py` — Train a model from polygon suggestions

Trains a ResidualFixNet model, runs inference, outputs corrected polygons. Saves model to `MODEL_PATH`.

```bash
python stewards_scripts/train_from_suggestions.py
```

Optional flags:
- `--epochs 200` — training epochs (default: 200)
- `--head fix` — model head to use (default: fix)
- `--enable_remove` — train with add+remove head (default: add-only)

### 2. `apply_model.py` — Apply a trained model and produce global outputs

Runs inference, re-polygonizes, generates network, merges into global files. Snaps new network endpoints to existing network. Loads model from `MODEL_PATH`.

```bash
python stewards_scripts/apply_model.py --tile_ids 79337_97000 79315_97000
```

Optional flags:
- `--head fix` — model head to use: `fix` or `full` (default: fix)
- `--enable_remove` — use if model was trained with remove head
- `--snap_tolerance 3.0` — snap distance in meters (default: 3.0)



### 0. `generate_suggestions.py` — Generate polygon suggestions (optional)

Clips input polygons to zoom-18 tiles, generates elongation suggestions, outputs GeoJSON with `tile_id` and `n_suggestion`.

```bash
python stewards_scripts/generate_suggestions.py
```

Optional flags:
- `--elongation_dist 50` — extension distance in meters (default: 50)
- `--convexity_threshold 0.8` — skip non-convex polygons (default: 0.8)
- `--max_elongate N` — limit elongation suggestions per tile
- `--filter_across_roads` — filter suggestions that cross road polygons

## Setup

### 1. Clone tile2net

```bash
git clone https://github.com/VIDA-NYU/tile2net.git
cd tile2net
```

### 2. Download tile data

Download RGB tiles, T2N polygon rasters, and confidence masks from:

https://uofi.box.com/s/q2028l50qdgt871lbc0z01wi4oxd0bfu

### 3. Create conda environment

```bash
conda create -n tile2net-env python=3.11 -y
```

### 4. Install dependencies

```bash
conda run -n tile2net-env pip install -e .
conda run -n tile2net-env pip install torch torchvision segmentation-models-pytorch rasterio opencv-python-headless tqdm Pillow python-dotenv
conda run -n tile2net-env pip install "numpy<2.0"
```

### 5. Configure and run

```bash
cp .env.example .env
# Edit .env with your actual data paths

conda activate tile2net-env
python stewards_scripts/generate_suggestions.py
python stewards_scripts/train_from_suggestions.py
python stewards_scripts/apply_model.py --tile_ids 79337_97000
```
