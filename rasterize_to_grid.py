import numpy as np
import json
import math
import matplotlib.pyplot as plt

WALL_FACES_FILE = "wall_faces_xy.npy"

def load_faces(path: str):
    faces = np.load(path, allow_pickle=True)
    # faces is a list-like of 4-point polygons: [[(x,y),...], ...]
    return faces

def compute_bounds(faces):
    xs = []
    ys = []
    for poly in faces:
        for (x, y) in poly:
            xs.append(float(x))
            ys.append(float(y))
    return min(xs), min(ys), max(xs), max(ys)

def rasterize_faces_to_grid(faces, cell_size=0.25, face_sample_step=0.05):
    """
    cell_size: grid cell size in DXF units (often meters, or inches/feet depending on export).
    face_sample_step: sampling resolution along edges (smaller = more accurate but slower).
    """
    minx, miny, maxx, maxy = compute_bounds(faces)

    width = maxx - minx
    height = maxy - miny

    gw = int(math.ceil(width / cell_size)) + 1
    gh = int(math.ceil(height / cell_size)) + 1

    grid = np.zeros((gh, gw), dtype=np.uint8)

    # Helper: mark a point in grid
    def mark_point(x, y):
        gx = int((x - minx) / cell_size)
        gy = int((y - miny) / cell_size)
        if 0 <= gx < gw and 0 <= gy < gh:
            grid[gy, gx] = 1

    # Sample edges of each face (good enough for wall footprint)
    for poly in faces:
        pts = [(float(x), float(y)) for (x, y) in poly]

        # close polygon loop
        for i in range(4):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 4]

            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)

            steps = max(1, int(dist / face_sample_step))
            for s in range(steps + 1):
                t = s / steps
                x = x1 + t * dx
                y = y1 + t * dy
                mark_point(x, y)

    meta = {
        "cell_size": cell_size,
        "bounds": {"minx": float(minx), "miny": float(miny), "maxx": float(maxx), "maxy": float(maxy)},
        "grid_shape": [int(gh), int(gw)],
        "codes": {"FREE": 0, "WALL": 1},
        "notes": "Grid derived from 3DFACE edges (2D projection)."
    }

    return grid, meta

def save_outputs(grid, meta):
    # Save numpy grid (this is the main deliverable)
    np.save("grid.npy", grid)

    # Save metadata (cell size, bounds, codes)
    with open("grid_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Preview image disabled because grid is too large and causes RAM crash
    print("Saved grid.npy and grid_meta.json (preview disabled).")


if __name__ == "__main__":
    faces = load_faces(WALL_FACES_FILE)
    print("Loaded faces:", len(faces))

    # Start with a coarse-ish cell size so it runs fast.
    # We can refine later (0.5, 0.25, 0.1...)
    grid, meta = rasterize_faces_to_grid(faces, cell_size=0.25, face_sample_step=0.05)

    print("Grid shape:", grid.shape)
    print("Wall cells:", int(grid.sum()))

    save_outputs(grid, meta)
    print("Saved: grid.npy, grid_meta.json, grid_preview.png")
