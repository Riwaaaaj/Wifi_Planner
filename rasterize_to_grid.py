import numpy as np
import json
import math
import matplotlib.pyplot as plt

def crop_grid_to_walls(grid, margin=20):
    """
    Crops grid to the bounding box of wall cells (grid==1),
    with extra margin (in grid cells).
    Returns cropped_grid and crop metadata.
    """
   

    ys, xs = np.where(grid > 0)
    if len(xs) == 0 or len(ys) == 0:
        return grid, {"cropped": False, "reason": "no wall cells found"}

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    y_min = max(0, y_min - margin)
    y_max = min(grid.shape[0] - 1, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(grid.shape[1] - 1, x_max + margin)

    cropped = grid[y_min:y_max + 1, x_min:x_max + 1]

    info = {
        "cropped": True,
        "crop_box": {"y_min": int(y_min), "y_max": int(y_max), "x_min": int(x_min), "x_max": int(x_max)},
        "original_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "cropped_shape": [int(cropped.shape[0]), int(cropped.shape[1])]
    }
    return cropped, info


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
    import numpy as np
    import json
    import matplotlib.pyplot as plt

    np.save("grid.npy", grid)

    with open("grid_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Human-friendly preview: show walls as white pixels
    plt.figure(figsize=(8, 8))
    plt.imshow(grid == 1, origin="lower")  # boolean image: walls only
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("grid_preview.png", dpi=200)
    plt.close()

    print("Saved grid.npy, grid_meta.json, grid_preview.png")
    print("Grid shape:", grid.shape)

def keep_largest_wall_component(grid):
    """
    Keeps only the largest connected component of wall cells.
    Removes tiny specks scattered across the drawing.
    Uses 8-connectivity. Returns cleaned_grid.
    """
    from collections import deque

    H, W = grid.shape
    wall = (grid > 0)

    visited = np.zeros((H, W), dtype=bool)
    best_count = 0
    best_cells = None

    # 8 directions
    dirs = [(-1,-1), (-1,0), (-1,1),
            (0,-1),          (0,1),
            (1,-1),  (1,0),  (1,1)]

    for r in range(H):
        for c in range(W):
            if wall[r, c] and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                cells = [(r, c)]
                cnt = 1

                while q:
                    cr, cc = q.popleft()
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if wall[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                                cells.append((nr, nc))
                                cnt += 1

                if cnt > best_count:
                    best_count = cnt
                    best_cells = cells

    cleaned = np.zeros_like(grid)
    if best_cells is None:
        return cleaned

    for r, c in best_cells:
        cleaned[r, c] = 1  # keep as wall

    return cleaned


if __name__ == "__main__":
    faces = load_faces(WALL_FACES_FILE)
    print("Loaded faces:", len(faces))

    # Start with a coarse-ish cell size so it runs fast.
    # We can refine later (0.5, 0.25, 0.1...)
    grid, meta = rasterize_faces_to_grid(
    faces,
    cell_size=1.5,
    face_sample_step=0.3
)


print("Grid shape BEFORE cleanup:", grid.shape)
print("Wall cells BEFORE cleanup:", int(grid.sum()))

# ✅ Remove speck noise
grid_clean = keep_largest_wall_component(grid)

print("Wall cells AFTER cleanup:", int(grid_clean.sum()))

# ✅ Now crop
grid_cropped, crop_info = crop_grid_to_walls(grid_clean, margin=20)

print("Grid shape AFTER crop:", grid_cropped.shape)
print("Wall cells AFTER crop:", int(grid_cropped.sum()))

meta["crop"] = crop_info
meta["cleanup"] = {"method": "largest_connected_component"}

save_outputs(grid_cropped, meta)
print("Saved CLEANED+CROPPED: grid.npy, grid_meta.json, grid_preview.png")


