# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

This repository converts an architectural floor plan exported as a DXF (for example from SketchUp) into a simulation-ready occupancy grid for indoor Wi‑Fi planning.

The main outputs are:
- `grid.npy`: binary 2D array where 1 = wall/obstacle and 0 = free space
- `grid_meta.json`: metadata describing grid scale, bounds, and cropping
- `grid_preview.png`: visualization of the wall grid for quick inspection

There are two main ways to go from DXF to a grid:
- **3DFACE-based pipeline** (two-step, recommended for 3D wall geometry)
- **LINE/LWPOLYLINE-based pipeline** (single script, works when walls are drawn as lines/polylines)

All scripts expect the DXF file (default `house.dxf`) to be in the repository root.

## Commands and workflows

### Environment

This is a small, script-based Python project. There is no packaging, build system, or test runner configured.

Key Python dependencies (inferred from imports):
- `ezdxf`
- `numpy`
- `matplotlib`

A typical setup in a virtual environment might look like:

```bash
python -m venv .venv
source .venv/bin/activate
pip install ezdxf numpy matplotlib
```

(Adjust to your Python version and environment tooling as needed.)

### 3DFACE-based DXF → grid pipeline

This pipeline is used when the DXF encodes walls as `3DFACE` entities.

1. **Extract wall faces and flatten to 2D**
   - Script: `flatten_3dface.py`
   - Default input: `DXF_FILE = "house.dxf"`
   - Output: `wall_faces_xy.npy` (list-like structure of 4-point polygons in XY)

   Run:
   ```bash
   python flatten_3dface.py
   ```

2. **Rasterize wall faces, clean noise, crop, and write outputs**
   - Script: `rasterize_to_grid.py`
   - Reads: `wall_faces_xy.npy`
   - Writes:
     - `grid.npy`
     - `grid_meta.json`
     - `grid_preview.png`

   Run:
   ```bash
   python rasterize_to_grid.py
   ```

   If you need to adjust resolution or sampling, edit in `rasterize_to_grid.py`:
   - `cell_size` argument to `rasterize_faces_to_grid` (grid resolution)
   - `face_sample_step` (edge sampling density)

### LINE/LWPOLYLINE-based DXF → grid pipeline

When walls are drawn as `LINE` or `LWPOLYLINE` entities in modelspace, you can use the simpler, direct rasterization script.

- Script: `dxf_to_grid.py`
- Default input: `DXF_FILE = "house.dxf"`
- Output: `grid.npy`

Run:
```bash
python dxf_to_grid.py
```

This script:
- Scans `LINE` and `LWPOLYLINE` entities in modelspace
- Samples points along each segment at a configurable density
- Marks those samples into a 2D numpy grid with cell size `CELL_SIZE`

### Inspecting DXF contents

Use this helper script to understand how geometry is stored in a DXF before choosing a pipeline or tuning parameters.

- Script: `inspect_dxf.py`
- Default input: `DXF_FILE = "house.dxf"`

Run:
```bash
python inspect_dxf.py
```

This will:
- Load the DXF
- Print entity counts for modelspace and paperspace
- Summarize entity types (e.g., counts of `LINE`, `LWPOLYLINE`, `3DFACE`, etc.)
- List block names (useful when geometry is stored via `INSERT`ed blocks)

### Tests, linting, and formatting

As of this version, there are **no automated tests** and **no linting or formatting tooling** configured (no `pytest`, `tox`, `ruff`, `flake8`, or similar config files are present).

If you add tests or tooling later, update this section with the exact commands (e.g., `pytest`, `pytest path/to/test_file.py::TestClass::test_case`, or linting commands).

## Code structure and data flow

The repository is intentionally small and script-oriented. There is no package layout (no `src/` or module hierarchy); instead, each top-level script encapsulates a stage of the pipeline.

### `flatten_3dface.py`

Purpose:
- Extract 3D wall faces from a DXF and flatten them into 2D XY polygons for later rasterization.

Key pieces:
- Constant `DXF_FILE = "house.dxf"` controls the input drawing.
- `extract_3dface_polygons(dxf_path)`:
  - Opens the DXF with `ezdxf.readfile`
  - Iterates through modelspace entities
  - Collects any entity with `dxftype() == "3DFACE"`
  - For each face, stores the XY coordinates of vertices `vtx0..vtx3` as a 4-point polygon
- In the `__main__` block:
  - Calls `extract_3dface_polygons(DXF_FILE)`
  - Prints the number of extracted faces
  - Saves the polygons to `wall_faces_xy.npy` via `numpy.save`

This file is the **first stage** of the 3DFACE-based pipeline and defines the schema used by `rasterize_to_grid.py`.

### `rasterize_to_grid.py`

Purpose:
- Convert the flattened wall polygons into a numpy grid, remove speck noise, crop to the building footprint, and emit artifacts for later simulation/visualization.

Core flow (in order of use):
- `load_faces(path)`: loads `wall_faces_xy.npy` with `allow_pickle=True` and returns the list-like structure of polygons.
- `compute_bounds(faces)`: computes global min/max X and Y over all vertices to define the spatial bounding box.
- `rasterize_faces_to_grid(faces, cell_size, face_sample_step)`:
  - Derives grid width/height from bounds and `cell_size`
  - Allocates a `uint8` grid initialized to zero
  - For each 4-point polygon, loops over its 4 edges
  - Samples points along each edge at intervals of roughly `face_sample_step`
  - Converts each sample to grid indices and marks the corresponding cell as a wall (1)
  - Returns `(grid, meta)` where `meta` includes:
    - `cell_size`
    - `bounds`
    - `grid_shape`
    - simple code legend and notes
- `keep_largest_wall_component(grid)`:
  - Treats `grid > 0` as wall pixels
  - Runs an 8-connected component search (BFS) over the grid
  - Keeps only the largest connected component, zeroing out all smaller specks
- `crop_grid_to_walls(grid, margin)`:
  - Finds min/max `x`/`y` indices where `grid > 0`
  - Expands that box by `margin` cells in all directions (clamped to grid bounds)
  - Returns the cropped grid and a `crop_box` metadata structure
- `save_outputs(grid, meta)`:
  - Writes `grid.npy`
  - Writes `grid_meta.json` with metadata plus cleanup/cropping info
  - Generates `grid_preview.png` via `matplotlib`, where walls are rendered as white pixels

In the main execution path, the intended sequence is:
1. Load faces from `wall_faces_xy.npy` via `load_faces`.
2. Call `rasterize_faces_to_grid` with chosen `cell_size` and `face_sample_step`.
3. Clean up the resulting grid with `keep_largest_wall_component`.
4. Crop to the bounding box of walls with `crop_grid_to_walls`.
5. Attach cleanup/crop metadata to `meta`.
6. Persist outputs with `save_outputs`.

All later consumers of the grid (e.g., Wi‑Fi simulators) should rely on `grid.npy` and `grid_meta.json` produced here.

### `dxf_to_grid.py`

Purpose:
- Direct DXF → grid conversion based on `LINE` and `LWPOLYLINE` entities, without going through `3DFACE` polygons.

Key behavior:
- Reads `DXF_FILE` with `ezdxf.readfile` and iterates over all modelspace entities.
- For each entity:
  - If `dxftype() == "LINE"`, extracts `(x1, y1)` and `(x2, y2)` and appends to a `lines` list.
  - If `dxftype() == "LWPOLYLINE"`, flattens consecutive vertex pairs into segments and appends those.
- Computes global min/max X/Y across all segment endpoints to define bounds.
- Determines grid width/height using `CELL_SIZE` and allocates a `uint8` numpy grid.
- `mark_line(p1, p2)` samples points along each segment at a step proportional to `CELL_SIZE` and marks each sample’s cell as a wall.
- Iterates through all collected segments, calling `mark_line` for each, then saves `grid.npy`.

This script is useful when the DXF has simple 2D wall lines and you do not need 3DFACE processing, or as a debugging baseline.

### `inspect_dxf.py`

Purpose:
- Help you understand the structure and content of a DXF so you can decide which pipeline to use and how to configure it.

Behavior:
- Loads `DXF_FILE` with `ezdxf.readfile`.
- Prints counts of entities in modelspace and paperspace.
- Uses `collections.Counter` to summarize entity types in each space.
- Lists block names (`doc.blocks`) to reveal how geometry is organized.

This script is a diagnostic/inspection tool; it does not produce any `.npy` or grid outputs.

## Conventions and implementation notes

- **Input file naming**: All scripts use a hard-coded `DXF_FILE = "house.dxf"`. If you work with multiple drawings, either edit this constant per run or refactor scripts to accept a DXF path via command-line arguments.
- **Units and resolution**:
  - `CELL_SIZE` in `dxf_to_grid.py` and `cell_size` in `rasterize_to_grid.py` are in the DXF’s native units (often meters or feet depending on export).
  - Changing these values directly affects grid resolution and memory usage.
- **Output semantics**:
  - Grids are `uint8` arrays where `0` means free space and `1` means wall/obstacle.
  - `grid_meta.json` is the authoritative source for grid shape, bounds, and cell size; any consumer should read this file rather than assuming defaults.
- **Noise removal and cropping** (3DFACE pipeline):
  - Speck noise is removed by keeping only the largest connected wall component.
  - The grid is cropped to the bounding box of walls plus a configurable `margin`. Update `margin` in `crop_grid_to_walls` if you need more or less surrounding free space.
