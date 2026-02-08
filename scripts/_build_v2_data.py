"""Build enriched scene catalog + NDVI stack for NB02 v2.

Produces:
  data/eo_scene_catalog.csv   – 24 rows, 4 tiles × 6 dates, two platforms
  data/eo_ndvi_stack.npz      – ndvi (24,32,32), cloud (24,32,32), scene_id, datetime arrays
"""
from pathlib import Path
import numpy as np, csv, itertools

np.random.seed(42)

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

tiles = ["T32UQD", "T32UPD", "T33UUP", "T33UVP"]
platforms = ["S2A", "S2B"]
dates = ["2024-03-15", "2024-04-15", "2024-05-15",
         "2024-06-15", "2024-07-15", "2024-08-15"]

rows = []
for tile in tiles:
    for dt in dates:
        plat = np.random.choice(platforms)
        cloud = round(float(np.random.beta(2, 5) * 100), 1)  # skew toward low
        sid = f"{plat}_{tile}_{dt}_cloud{int(cloud)}"
        rows.append({
            "scene_id": sid,
            "tile": tile,
            "platform": plat,
            "datetime": f"{dt}T10:30:00Z",
            "cloud_cover": cloud,
        })

csv_path = DATA / "eo_scene_catalog.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["scene_id","tile","platform","datetime","cloud_cover"])
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {len(rows)} rows -> {csv_path}")

# NDVI stack: (24, 32, 32) float32 + cloud mask bool
n = len(rows)
ny, nx = 32, 32
ndvi = np.empty((n, ny, nx), dtype=np.float32)
cloud_mask = np.zeros((n, ny, nx), dtype=bool)

for i, r in enumerate(rows):
    base = 0.3 + 0.3 * np.sin(2 * np.pi * (int(r["datetime"][5:7]) - 1) / 12)
    ndvi[i] = np.random.normal(base, 0.08, (ny, nx)).astype(np.float32)
    cloud_frac = r["cloud_cover"] / 100
    cloud_mask[i] = np.random.random((ny, nx)) < cloud_frac

ndvi = np.clip(ndvi, -0.2, 1.0)
scene_ids = np.array([r["scene_id"] for r in rows])
datetimes = np.array([r["datetime"] for r in rows])

npz_path = DATA / "eo_ndvi_stack.npz"
np.savez_compressed(npz_path, ndvi=ndvi, cloud=cloud_mask,
                    scene_id=scene_ids, datetime=datetimes)
print(f"Wrote ndvi {ndvi.shape} + cloud {cloud_mask.shape} -> {npz_path}")
