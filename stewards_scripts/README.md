# Stewards Scripts

Three scripts for training and applying sidewalk polygon fix models.

## Scripts

### 1. `train_from_suggestions.py` — Train a model from polygon suggestions

Takes a GeoJSON of polygon suggestions, trains a ResidualFixNet model, runs inference, and outputs corrected polygons as GeoJSON.

```bash
python stewards_scripts/train_from_suggestions.py \
    --geojson ./suggestion_sample.geojson \
    --tiles_dir ./path/to/tiles \
    --t2n_dir ./path/to/masks_tile2net_polygons \
    --conf_dir ./path/to/masks_confidence \
    --output ./outputs/corrected_polygons.geojson \
    --model_output ./outputs/suggestion_model.pt \
    --epochs 200 \
    --head fix
```

### 2. `apply_model.py` — Apply a trained model and generate network

Loads a trained model, runs inference on specified tiles, re-polygonizes the predictions, and runs Tile2Net topology to produce both polygon and network GeoJSON outputs.

```bash
python stewards_scripts/apply_model.py \
    --tile_ids 79337_97000 79315_97000 \
    --model_path ./outputs/suggestion_model.pt \
    --tiles_dir ./path/to/tiles \
    --t2n_dir ./path/to/masks_tile2net_polygons \
    --conf_dir ./path/to/masks_confidence \
    --output_polygons ./outputs/corrected_polygons.geojson \
    --output_network ./outputs/corrected_network.geojson \
    --head fix
```

Tile IDs can also be derived from a GeoJSON file using `--geojson` instead of `--tile_ids`.
'

### This script can be ignored: 3. `polygon_to_network.py` — Convert polygons to network (standalone)

Runs the Tile2Net topology engine on a polygon shapefile to produce a sidewalk network.

```bash
mkdir -p outputs && \
python stewards_scripts/polygon_to_network.py \
    --input ./path/to/polygons.shp \
    --output dorchester_network \
    --bbox -71.099395752 42.2808650053 -71.0307312012 42.3275853885
```

Output is saved to `outputs/<name>.shp`.

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
python stewards_scripts/train_from_suggestions.py --geojson ...
python stewards_scripts/apply_model.py --tile_ids ...
python stewards_scripts/polygon_to_network.py --input ...
```
