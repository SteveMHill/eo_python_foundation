from __future__ import annotations

import json
from pathlib import Path


def _lines(text: str) -> list[str]:
    text = text.strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.splitlines()]


def md(cell_id: str, text: str) -> dict:
    return {
        "id": cell_id,
        "cell_type": "markdown",
        "metadata": {"language": "markdown"},
        "source": _lines(text),
    }


def code(cell_id: str, text: str) -> dict:
    return {
        "id": cell_id,
        "cell_type": "code",
        "metadata": {"language": "python"},
        "execution_count": None,
        "outputs": [],
        "source": _lines(text),
    }


def build_notebook() -> dict:
    cells: list[dict] = []

    cells.append(
        md(
            "title",
            """
![Course header](../assets/img/header.png)

# 03 — xarray for Earth Observation (Course-Ready Student Notebook)

**Duration:** 2-3 hours

Question: How do we move from raw EO arrays to clear, labeled analysis?
Tool: xarray + pandas + numpy + matplotlib.
Result: A processing shortlist with exports and plots.
""",
        )
    )

    cells.append(
        md(
            "howto",
            """
## 0) How to Use This Notebook

- Run cells from top to bottom.
- If output looks strange: **Kernel -> Restart Kernel and Run All Cells**.
- Keep all paths repo-local (`data/`, `outputs/`).
- Keep notes short and focus on hands-on practice.
""",
        )
    )

    cells.append(
        md(
            "setup_head",
            """
## 1) Setup & Data Loading

Question: Do we have the EO files we need, and can we load them cleanly?
Tool: `Path`, `numpy`, `pandas`, `xarray`.
Result: One labeled `Dataset` ready for analysis.

**Objective:**
- Use repo-local paths that work on student machines.
- Load the shipped scene catalog and NDVI stack.
- Build a compact xarray dataset with EO coordinates.
""",
        )
    )

    cells.append(
        code(
            "imports",
            """
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
""",
        )
    )

    cells.append(
        code(
            "paths",
            """
if Path("data").exists():
    DATA_DIR = Path("data")
    OUT_DIR = Path("outputs")
elif Path("../data").exists():
    DATA_DIR = Path("../data")
    OUT_DIR = Path("../outputs")
else:
    raise FileNotFoundError("Could not find repo-local data/ directory.")

OUT_DIR.mkdir(exist_ok=True)

print("Current working directory:", Path.cwd())
print("DATA_DIR:", DATA_DIR.resolve())
print("OUT_DIR:", OUT_DIR.resolve())
""",
        )
    )

    cells.append(
        code(
            "sanity",
            """
scene_catalog_path = DATA_DIR / "eo_scene_catalog.csv"
ndvi_stack_path = DATA_DIR / "eo_ndvi_stack.npz"

missing_files = [
    path.name for path in [scene_catalog_path, ndvi_stack_path] if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "Missing required data files in data/: " + ", ".join(missing_files)
    )

print("Sanity check passed. Files found:")
print("-", scene_catalog_path)
print("-", ndvi_stack_path)
""",
        )
    )

    cells.append(
        code(
            "load_raw",
            """
stack = np.load(ndvi_stack_path, allow_pickle=True)
scene_catalog = pd.read_csv(scene_catalog_path)

ndvi = stack["ndvi"].astype("float32")
cloud = stack["cloud"].astype(bool)
scene_ids = stack["scene_id"].astype(str)

scene_catalog["datetime"] = pd.to_datetime(scene_catalog["datetime"], utc=True).dt.tz_convert(None)
scene_catalog = scene_catalog.set_index("scene_id").loc[scene_ids].reset_index()

print("ndvi shape:", ndvi.shape, "dtype:", ndvi.dtype)
print("cloud shape:", cloud.shape, "dtype:", cloud.dtype)
print("catalog rows:", len(scene_catalog))
scene_catalog.head(3)
""",
        )
    )

    cells.append(
        code(
            "build_ds",
            """
ds = xr.Dataset(
    data_vars={
        "ndvi": (("time", "y", "x"), ndvi),
        "cloud": (("time", "y", "x"), cloud),
    },
    coords={
        "time": scene_catalog["datetime"].to_numpy(),
        "y": np.arange(ndvi.shape[1]),
        "x": np.arange(ndvi.shape[2]),
        "scene_id": ("time", scene_catalog["scene_id"].to_numpy()),
        "tile": ("time", scene_catalog["tile"].to_numpy()),
        "platform": ("time", scene_catalog["platform"].to_numpy()),
        "cloud_cover": ("time", scene_catalog["cloud_cover"].to_numpy()),
    },
    attrs={"title": "EO mini NDVI stack", "source": "data/eo_ndvi_stack.npz"},
)

ds["ndvi"].attrs.update({"long_name": "NDVI", "units": "1"})
ds["cloud"].attrs.update({"long_name": "Cloud mask", "description": "True = cloudy"})
ds = ds.sortby("time")

ds
""",
        )
    )

    cells.append(
        md(
            "setup_try",
            """
✅ Try it
1. Change `scene_index` and print a different scene ID.
2. Print `ds.dims` and identify the spatial axes.
3. Print two coordinate names that are EO metadata (not spatial).
""",
        )
    )

    cells.append(
        code(
            "setup_try_code",
            """
# TODO: pick another scene index between 0 and len(ds.time)-1
scene_index = 0

print("scene_id:", ds["scene_id"].isel(time=scene_index).item())
print("dims:", ds.dims)
print("metadata coords:", ["scene_id", "tile", "platform", "cloud_cover"])
""",
        )
    )

    cells.append(
        md(
            "setup_ck",
            """
🧠 Checkpoint
Use both indexing and metadata:
- Get the platform value for the third scene.
- Get the mean NDVI of that same scene using `isel` and `mean`.
""",
        )
    )

    cells.append(
        code(
            "setup_ck_code",
            """
third_platform = ds["platform"].isel(time=2).item()
third_mean_ndvi = float(ds["ndvi"].isel(time=2).mean().item())

print("third scene platform:", third_platform)
print("third scene mean NDVI:", round(third_mean_ndvi, 3))
""",
        )
    )

    cells.append(
        md(
            "xr_head",
            """
## 2) xarray Essentials: Indexing, Selection, and Labels

Question: How do we pull exactly the pixels and scenes we need?
Tool: `isel`, `sel`, slicing, and coordinates.
Result: Confident access to EO cubes without guessing axis order.

**Objective:**
- Use positional and label-based indexing.
- Extract a single scene and a single-pixel time series.
- Practice date slicing with readable code.
""",
        )
    )

    cells.append(
        code(
            "xr_demo_dims",
            """
print("ndvi dims:", ds["ndvi"].dims)
print("ndvi shape:", ds["ndvi"].shape)
print("first 3 times:", ds["time"].values[:3])
""",
        )
    )

    cells.append(
        code(
            "xr_demo_isel",
            """
scene_demo = ds["ndvi"].isel(time=0)

fig, ax = plt.subplots(figsize=(4.5, 4))
img = ax.imshow(scene_demo, cmap="YlGn", vmin=0, vmax=1)
ax.set_title("NDVI scene at time index 0")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(img, ax=ax, label="NDVI")
plt.tight_layout()
plt.show()
""",
        )
    )

    cells.append(
        md(
            "xr_try",
            """
✅ Try it
1. Change `scene_index` and display a different NDVI scene.
2. Change `(pixel_y, pixel_x)` and plot a different pixel time series.
3. Slice a date range with `sel(time=slice(...))` and report how many scenes are included.
""",
        )
    )

    cells.append(
        code(
            "xr_try_scene",
            """
# TODO: try another scene index
scene_index = 5
scene_try = ds["ndvi"].isel(time=scene_index)

print("scene id:", ds["scene_id"].isel(time=scene_index).item())
scene_try.mean().item()
""",
        )
    )

    cells.append(
        code(
            "xr_try_pixel",
            """
# TODO: change pixel location
pixel_y, pixel_x = 12, 20
pixel_ts = ds["ndvi"].isel(y=pixel_y, x=pixel_x)

fig, ax = plt.subplots(figsize=(6, 3.5))
pixel_ts.plot(ax=ax, marker="o")
ax.set_title(f"Pixel time series at y={pixel_y}, x={pixel_x}")
ax.set_xlabel("time")
ax.set_ylabel("NDVI")
plt.tight_layout()
plt.show()
""",
        )
    )

    cells.append(
        code(
            "xr_try_date",
            """
# TODO: adjust the date window
time_slice = ds["ndvi"].sel(time=slice("2024-04-01", "2024-06-30"))
print("Scenes in slice:", time_slice.sizes["time"])
time_slice
""",
        )
    )

    cells.append(
        md(
            "xr_ck",
            """
🧠 Checkpoint
Combine indexing + reduction:
- Select scenes from May through July.
- Compute one scalar mean NDVI across time and space.
""",
        )
    )

    cells.append(
        code(
            "xr_ck_code",
            """
may_to_july = ds["ndvi"].sel(time=slice("2024-05-01", "2024-07-31"))
mean_may_to_july = float(may_to_july.mean().item())

print("May-July scene count:", may_to_july.sizes["time"])
print("Mean NDVI (May-Jul):", round(mean_may_to_july, 3))
""",
        )
    )

    cells.append(
        md(
            "mask_head",
            """
## 3) Masking and Reductions

Question: How do we keep clear pixels and summarize scenes quickly?
Tool: `where`, boolean masks, reductions over named dimensions.
Result: Per-scene metrics we can use for quality filtering.

**Objective:**
- Mask cloudy pixels correctly.
- Compute mean NDVI per scene.
- Build a first pass of "good scenes".
""",
        )
    )

    cells.append(
        code(
            "mask_demo",
            """
clear_ndvi = ds["ndvi"].where(~ds["cloud"])
scene_mean_ndvi = clear_ndvi.mean(dim=("y", "x"), skipna=True)
clear_fraction = (~ds["cloud"]).mean(dim=("y", "x"))

ds = ds.assign(scene_mean_ndvi=scene_mean_ndvi, clear_fraction=clear_fraction)

ds[["scene_mean_ndvi", "clear_fraction"]]
""",
        )
    )

    cells.append(
        code(
            "mask_demo_table",
            """
scene_quality = pd.DataFrame(
    {
        "scene_id": ds["scene_id"].values,
        "tile": ds["tile"].values,
        "cloud_cover": ds["cloud_cover"].values,
        "mean_ndvi": ds["scene_mean_ndvi"].values,
        "clear_fraction": ds["clear_fraction"].values,
    }
)
scene_quality.head(6)
""",
        )
    )

    cells.append(
        md(
            "mask_try",
            """
✅ Try it
1. Change `max_cloud` and count how many scenes pass.
2. Add a NDVI threshold and keep only scenes above it.
3. Find the best scene per tile by lowest cloud cover.
""",
        )
    )

    cells.append(
        code(
            "mask_try_filter",
            """
# TODO: try other thresholds
max_cloud = 30
min_mean_ndvi = 0.40

good_mask = (scene_quality["cloud_cover"] <= max_cloud) & (
    scene_quality["mean_ndvi"] >= min_mean_ndvi
)

good_scenes = scene_quality.loc[good_mask].copy()
print("Good scenes count:", len(good_scenes))
good_scenes.head()
""",
        )
    )

    cells.append(
        code(
            "mask_try_best_tile",
            """
best_per_tile = (
    scene_quality.sort_values(["tile", "cloud_cover", "mean_ndvi"], ascending=[True, True, False])
    .groupby("tile", as_index=False)
    .head(1)
)

best_per_tile[["tile", "scene_id", "cloud_cover", "mean_ndvi"]]
""",
        )
    )

    cells.append(
        md(
            "mask_ck",
            """
🧠 Checkpoint
Use two concepts together:
- Pick scenes with `clear_fraction >= 0.7`.
- Among those, show the top 5 by mean NDVI.
""",
        )
    )

    cells.append(
        code(
            "mask_ck_code",
            """
high_quality = scene_quality.loc[scene_quality["clear_fraction"] >= 0.7]
top5_high_quality = high_quality.sort_values("mean_ndvi", ascending=False).head(5)

top5_high_quality[["scene_id", "tile", "clear_fraction", "mean_ndvi"]]
""",
        )
    )

    cells.append(
        md(
            "time_head",
            """
## 4) Temporal Patterns with xarray

Question: How does NDVI behavior change through time?
Tool: datetime coordinates + `groupby` + rolling window.
Result: Seasonal summary signals for EO interpretation.

**Objective:**
- Create a month feature from time.
- Compute monthly NDVI patterns.
- Smooth per-scene signals with rolling means.
""",
        )
    )

    cells.append(
        code(
            "time_demo_month",
            """
ds = ds.assign_coords(month=("time", ds["time"].dt.month.data))
monthly_ndvi = ds["scene_mean_ndvi"].groupby("month").mean()

monthly_ndvi
""",
        )
    )

    cells.append(
        code(
            "time_demo_plot",
            """
fig, ax = plt.subplots(figsize=(6, 3.5))
monthly_ndvi.plot(ax=ax, marker="o")
ax.set_title("Mean scene NDVI by month")
ax.set_xlabel("month")
ax.set_ylabel("mean NDVI")
plt.tight_layout()
plt.show()
""",
        )
    )

    cells.append(
        md(
            "time_try",
            """
✅ Try it
1. Compute a 3-scene rolling mean of `scene_mean_ndvi`.
2. Compute each scene anomaly vs the global mean NDVI.
3. Build a small month summary table with count and mean cloud cover.
""",
        )
    )

    cells.append(
        code(
            "time_try_roll",
            """
rolling_ndvi = ds["scene_mean_ndvi"].rolling(time=3, center=True).mean()
rolling_ndvi
""",
        )
    )

    cells.append(
        code(
            "time_try_anom",
            """
global_mean_ndvi = float(ds["scene_mean_ndvi"].mean().item())
scene_anomaly = ds["scene_mean_ndvi"] - global_mean_ndvi

scene_anomaly
""",
        )
    )

    cells.append(
        code(
            "time_try_table",
            """
month_summary = (
    pd.DataFrame({
        "month": ds["month"].values,
        "cloud_cover": ds["cloud_cover"].values,
        "mean_ndvi": ds["scene_mean_ndvi"].values,
    })
    .groupby("month", as_index=False)
    .agg(scene_count=("month", "size"), mean_cloud=("cloud_cover", "mean"), mean_ndvi=("mean_ndvi", "mean"))
)

month_summary
""",
        )
    )

    cells.append(
        md(
            "time_ck",
            """
🧠 Checkpoint
Find the month with:
- highest mean NDVI, and
- lowest mean cloud cover.
Are they the same month?
""",
        )
    )

    cells.append(
        code(
            "time_ck_code",
            """
month_max_ndvi = month_summary.loc[month_summary["mean_ndvi"].idxmax(), "month"]
month_min_cloud = month_summary.loc[month_summary["mean_cloud"].idxmin(), "month"]

print("Month with highest mean NDVI:", int(month_max_ndvi))
print("Month with lowest mean cloud:", int(month_min_cloud))
print("Same month?:", bool(month_max_ndvi == month_min_cloud))
""",
        )
    )

    cells.append(
        md(
            "plot_head",
            """
## 5) matplotlib Essentials for EO Outputs

Question: Which core visual checks should we always make?
Tool: histogram, scatter, and raster quicklook.
Result: Fast quality checks plus saved figures for reporting.

**Objective:**
- Plot cloud distribution.
- Plot cloud cover vs mean NDVI.
- Plot one NDVI scene and save figures to `outputs/`.
""",
        )
    )

    cells.append(
        code(
            "plot_hist",
            """
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(ds["cloud_cover"].values, bins=10, color="#4C72B0", edgecolor="white")
ax.set_title("Cloud cover distribution")
ax.set_xlabel("cloud cover (%)")
ax.set_ylabel("scene count")
plt.tight_layout()
plt.show()
""",
        )
    )

    cells.append(
        code(
            "plot_scatter",
            """
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(ds["cloud_cover"].values, ds["scene_mean_ndvi"].values, alpha=0.85, color="#2A9D8F")
ax.set_title("Cloud cover vs mean NDVI")
ax.set_xlabel("cloud cover (%)")
ax.set_ylabel("mean NDVI")
plt.tight_layout()

scatter_path = OUT_DIR / "xarray_cloud_vs_ndvi.png"
fig.savefig(scatter_path, dpi=150)
plt.show()

print("Saved:", scatter_path.resolve())
""",
        )
    )

    cells.append(
        code(
            "plot_raster",
            """
raster_index = 3
raster_scene = ds["ndvi"].isel(time=raster_index)

fig, ax = plt.subplots(figsize=(4.8, 4.2))
img = ax.imshow(raster_scene, cmap="RdYlGn", vmin=0, vmax=1)
ax.set_title(f"NDVI quicklook: {ds['scene_id'].isel(time=raster_index).item()}")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(img, ax=ax, label="NDVI")
plt.tight_layout()
plt.show()
""",
        )
    )

    cells.append(
        md(
            "plot_try",
            """
✅ Try it
1. Change `raster_index` to another scene and rerun.
2. Save a second quicklook image with a different filename.
3. Try a different colormap and compare readability.
""",
        )
    )

    cells.append(
        code(
            "plot_try_save",
            """
# TODO: change index and filename
custom_index = 10
custom_filename = "xarray_quicklook_custom.png"

fig, ax = plt.subplots(figsize=(4.8, 4.2))
img = ax.imshow(ds["ndvi"].isel(time=custom_index), cmap="viridis", vmin=0, vmax=1)
ax.set_title(f"Custom quicklook: {ds['scene_id'].isel(time=custom_index).item()}")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(img, ax=ax, label="NDVI")
plt.tight_layout()

custom_path = OUT_DIR / custom_filename
fig.savefig(custom_path, dpi=150)
plt.show()

print("Saved:", custom_path.resolve())
""",
        )
    )

    cells.append(
        md(
            "plot_ck",
            """
🧠 Checkpoint
Confirm exports from this section exist in `outputs/`.
""",
        )
    )

    cells.append(
        code(
            "plot_ck_code",
            """
sorted([p.name for p in OUT_DIR.glob("xarray_*.png")])
""",
        )
    )

    cells.append(
        md(
            "cap_head",
            """
## 6) Mini Capstone: Build a Processing Shortlist

Question: Which scenes should we process first for an AOI?
Tool: quality thresholds + tile distance + optional "most recent per tile" rule.
Result: Exported shortlist table and summary plot.

**Student prompt (20-30 min):**
- Input: AOI point (`aoi_lon`, `aoi_lat`), `max_cloud`, `min_mean_ndvi`.
- Optional: keep only the most recent good scene per tile.
- Output: `tile, datetime, cloud_cover, mean_ndvi, distance_to_aoi_km`.
- Export: `outputs/processing_shortlist_xarray.csv`.
""",
        )
    )

    cells.append(
        code(
            "cap_inputs",
            """
aoi_lon, aoi_lat = 11.75, 48.25
max_cloud = 35.0
min_mean_ndvi = 0.40
keep_most_recent_per_tile = True

# Approximate tile centroids for this teaching dataset.
tile_centroids = pd.DataFrame(
    {
        "tile": ["T32UPD", "T32UQD", "T33UUP", "T33UVP"],
        "tile_lon": [10.5, 11.2, 12.6, 13.1],
        "tile_lat": [47.8, 48.3, 47.4, 48.0],
    }
)


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c
""",
        )
    )

    cells.append(
        code(
            "cap_table",
            """
scene_table = pd.DataFrame(
    {
        "scene_id": ds["scene_id"].values,
        "tile": ds["tile"].values,
        "datetime": pd.to_datetime(ds["time"].values),
        "cloud_cover": ds["cloud_cover"].values,
        "mean_ndvi": ds["scene_mean_ndvi"].values,
    }
)

scene_table = scene_table.merge(tile_centroids, on="tile", how="left")
scene_table["distance_to_aoi_km"] = haversine_km(
    scene_table["tile_lon"], scene_table["tile_lat"], aoi_lon, aoi_lat
)

scene_table.head()
""",
        )
    )

    cells.append(
        code(
            "cap_filter",
            """
shortlist = scene_table.loc[
    (scene_table["cloud_cover"] <= max_cloud)
    & (scene_table["mean_ndvi"] >= min_mean_ndvi)
].copy()

if keep_most_recent_per_tile:
    shortlist = (
        shortlist.sort_values("datetime")
        .groupby("tile", as_index=False)
        .tail(1)
        .sort_values("tile")
    )

shortlist = shortlist[["tile", "datetime", "cloud_cover", "mean_ndvi", "distance_to_aoi_km", "scene_id"]]
shortlist
""",
        )
    )

    cells.append(
        code(
            "cap_export",
            """
shortlist_path = OUT_DIR / "processing_shortlist_xarray.csv"
shortlist.to_csv(shortlist_path, index=False)

print("Rows exported:", len(shortlist))
print("Saved:", shortlist_path.resolve())
shortlist.head(10)
""",
        )
    )

    cells.append(
        code(
            "cap_plot",
            """
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(shortlist["cloud_cover"], shortlist["mean_ndvi"], s=70, color="#E76F51")

for _, row in shortlist.iterrows():
    ax.annotate(row["tile"], (row["cloud_cover"], row["mean_ndvi"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

ax.set_title("Processing shortlist: cloud vs mean NDVI")
ax.set_xlabel("cloud cover (%)")
ax.set_ylabel("mean NDVI")
plt.tight_layout()

shortlist_plot_path = OUT_DIR / "processing_shortlist_xarray.png"
fig.savefig(shortlist_plot_path, dpi=150)
plt.show()

print("Saved:", shortlist_plot_path.resolve())
""",
        )
    )

    cells.append(
        md(
            "cap_try",
            """
✅ Try it
1. Tighten `max_cloud` and compare shortlist size.
2. Raise `min_mean_ndvi` and inspect which tile drops out first.
3. Set `keep_most_recent_per_tile = False` and compare outputs.
""",
        )
    )

    cells.append(
        code(
            "cap_try_code",
            """
# TODO: experiment with stricter thresholds
max_cloud_try = 25.0
min_mean_ndvi_try = 0.50

shortlist_try = scene_table.loc[
    (scene_table["cloud_cover"] <= max_cloud_try)
    & (scene_table["mean_ndvi"] >= min_mean_ndvi_try)
].copy()

print("Trial shortlist count:", len(shortlist_try))
shortlist_try[["tile", "datetime", "cloud_cover", "mean_ndvi"]].head()
""",
        )
    )

    cells.append(
        md(
            "cap_ck",
            """
🧠 Checkpoint
Summarize your shortlist in one sentence:
- How many scenes were selected?
- Which tile is closest to the AOI?
""",
        )
    )

    cells.append(
        code(
            "cap_ck_code",
            """
if len(shortlist) > 0:
    closest_tile = shortlist.sort_values("distance_to_aoi_km").iloc[0]["tile"]
else:
    closest_tile = "No scenes selected"

print("Selected scenes:", len(shortlist))
print("Closest selected tile:", closest_tile)
""",
        )
    )

    cells.append(
        md(
            "wrap",
            """
## 7) Wrap-Up: What You Can Do Now

You can now:
- Build a labeled EO cube from shipped arrays and scene metadata.
- Use `isel`, `sel`, masks, and reductions for practical scene filtering.
- Generate core plots and save outputs reproducibly.
- Produce an EO processing shortlist with explicit quality rules.

💡 Tip
Before extending this workflow to larger datasets, keep this same sequence:
1) sanity-check files, 2) build labeled data, 3) compute quality metrics, 4) export shortlist.
""",
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "notebooks" / "03_xarray_for_earth_observation_student.ipynb"
    notebook = build_notebook()
    out_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
