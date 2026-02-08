#!/usr/bin/env python3
"""Build appendix notebooks A1 (Cloud-Native Formats) and A2 (Dask Parallel)."""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def _new_nb():
    return {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def _md(nb, source: str):
    lines = source.split("\n")
    for i in range(len(lines) - 1):
        lines[i] += "\n"
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": lines})


def _code(nb, source: str):
    lines = source.split("\n")
    for i in range(len(lines) - 1):
        lines[i] += "\n"
    nb["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines,
        }
    )


def _write(nb, path: Path):
    if path.exists():
        path.unlink()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Written {len(nb['cells'])} cells to {path.name}")


# ============================================================================
# A1 — Cloud-Native Geospatial Formats
# ============================================================================
def build_a1():
    nb = _new_nb()
    md = lambda s: _md(nb, s)
    code = lambda s: _code(nb, s)

    # ------------------------------------------------------------------
    md(
        "![Course header](../assets/img/header.png)\n"
        "\n"
        "# A1 \u2014 Cloud-Native Geospatial Formats\n"
        "Optional reference \u2014 understand the file formats behind modern EO data\n"
        "\n"
        "This appendix explains **why** the data you loaded in Notebooks 04 and 05 "
        "works so efficiently in the cloud.\n"
        "\n"
        "## Learning Objectives\n"
        "\n"
        "By the end of this notebook you will be able to:\n"
        "\n"
        "- Explain what Analysis Ready Data (ARD) means\n"
        "- Describe the key cloud-native raster and vector formats (COG, Zarr, GeoParquet)\n"
        "- Read a Cloud-Optimised GeoTIFF (COG) with partial reads and overviews\n"
        "- Open a Zarr store as an xarray Dataset\n"
        "- Compare formats and choose the right one for a given task\n"
        "\n"
        "Tooling:\n"
        "- rasterio / rioxarray\n"
        "- xarray (Zarr backend)\n"
        "- geopandas (optional \u2014 for vector examples)\n"
        "\n"
        "\u23f1\ufe0f Estimated time: **30 \u2013 45 minutes**"
    )

    md(
        "---\n"
        "\n"
        "## Table of contents\n"
        "\n"
        "1. Analysis Ready Data (ARD)\n"
        "2. Cloud-native format overview\n"
        "3. Cloud-Optimised GeoTIFF (COG)\n"
        "4. Zarr\n"
        "5. GeoParquet & vector formats\n"
        "6. Format comparison\n"
        "7. Exercises\n"
        "8. Recap & further reading"
    )

    # --- 1. ARD -----------------------------------------------------------
    md(
        "---\n"
        "\n"
        "## 1) Analysis Ready Data (ARD)\n"
        "\n"
        "> *\u201cIt is often said that 80 % of data analysis is spent on the process "
        "of cleaning and preparing the data.\u201d* \u2014 Hadley Wickham\n"
        "\n"
        "**Analysis Ready Data** means satellite imagery that has already been:\n"
        "\n"
        "| Step | What it does |\n"
        "|------|-------------|\n"
        "| Radiometric calibration | Raw DN \u2192 physical units (reflectance) |\n"
        "| Atmospheric correction | Remove haze / aerosols |\n"
        "| Geometric correction | Orthorectify to a known CRS |\n"
        "| Co-registration | Align multi-date / multi-sensor images |\n"
        "| Cloud / shadow masking | Provide quality flags for unusable pixels |\n"
        "\n"
        "Sentinel-2 **Level-2A** (the collection we use in NB 04 / 05) is ARD \u2014 "
        "surface reflectance with a Scene Classification Layer (SCL).\n"
        "\n"
        "See [CEOS ARD for Land (CARD4L)](https://ceos.org/ard/) for the formal spec."
    )

    # --- 2. Overview -------------------------------------------------------
    md(
        "---\n"
        "\n"
        "## 2) Cloud-native format overview\n"
        "\n"
        "Traditional workflows download whole files, then process locally.\n"
        "**Cloud-optimised formats** flip this:\n"
        "\n"
        "- **Partial reads** \u2014 fetch only the bytes you need (HTTP range requests)\n"
        "- **Internal tiling / chunking** \u2014 data is already split for parallel access\n"
        "- **Overviews / pyramids** \u2014 coarse previews without reading full resolution\n"
        "\n"
        "| Format | Data type | Key feature |\n"
        "|--------|-----------|-------------|\n"
        "| **COG** (Cloud-Optimised GeoTIFF) | Raster | Internal tiles + overviews |\n"
        "| **Zarr** | N-D arrays | Chunked + compressed, cloud-native I/O |\n"
        "| **GeoParquet** | Vector / tabular | Columnar, fast filtering |\n"
        "| **FlatGeobuf** | Vector | Spatial index, streaming |\n"
        "| **COPC** | Point cloud | Octree LOD |\n"
        "\n"
        "\U0001f4da [Cloud-Native Geospatial Guide](https://guide.cloudnativegeo.org/)"
    )

    # --- 3. COG ------------------------------------------------------------
    md(
        "---\n"
        "\n"
        "## 3) Cloud-Optimised GeoTIFF (COG)\n"
        "\n"
        "A COG is a regular GeoTIFF organised so that a client can read just the tiles "
        "and overview levels it needs via HTTP range requests.\n"
        "\n"
        "Key properties:\n"
        "- **Internal tiling** (typically 256\u00d7256 or 512\u00d7512)\n"
        "- **Overviews** (pre-computed pyramids for quick preview)\n"
        "- **Compression** (DEFLATE, LZW, ZSTD, \u2026)"
    )

    code(
        "import rasterio\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "# A Sentinel-2 Red band stored as a COG on AWS\n"
        "cog_url = (\n"
        "    'https://sentinel-cogs.s3.us-west-2.amazonaws.com/'\n"
        "    'sentinel-s2-l2a-cogs/32/T/PS/2024/12/'\n"
        "    'S2B_32TPS_20241228_0_L2A/B04.tif'\n"
        ")\n"
        "\n"
        "with rasterio.open(cog_url) as src:\n"
        "    print('COG properties')\n"
        "    print(f'  Size:       {src.width} \u00d7 {src.height}')\n"
        "    print(f'  CRS:        {src.crs}')\n"
        "    print(f'  Block size: {src.block_shapes}')\n"
        "    print(f'  Overviews:  {src.overviews(1)}')"
    )

    md("### Partial reads\nYou can request just a small window \u2014 the server sends only those bytes.")

    code(
        "from rasterio.windows import Window\n"
        "\n"
        "with rasterio.open(cog_url) as src:\n"
        "    window = Window(col_off=5000, row_off=5000, width=500, height=500)\n"
        "    subset = src.read(1, window=window)\n"
        "    full_mb = src.width * src.height * 2 / 1024 / 1024\n"
        "    print(f'Read {subset.nbytes / 1024:.1f} KB instead of {full_mb:.1f} MB')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 6))\n"
        "ax.imshow(subset, cmap='Reds', vmin=0, vmax=3000)\n"
        "ax.set_title('COG subset (500\u00d7500 px)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )

    md("### Overview reads\nOverviews let you get a quick thumbnail without touching full-resolution tiles.")

    code(
        "with rasterio.open(cog_url) as src:\n"
        "    ovr = src.overviews(1)[2]  # 3rd overview level\n"
        "    thumb = src.read(1, out_shape=(src.height // ovr, src.width // ovr))\n"
        "    print(f'Overview shape: {thumb.shape}  (1/{ovr} of full res)')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 8))\n"
        "ax.imshow(thumb, cmap='Reds', vmin=0, vmax=3000)\n"
        "ax.set_title(f'Quick preview (overview level 2, 1/{ovr})')\n"
        "ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )

    # --- 4. Zarr -----------------------------------------------------------
    md(
        "---\n"
        "\n"
        "## 4) Zarr\n"
        "\n"
        "Zarr stores N-dimensional arrays in **chunks**, each as a separate object "
        "in cloud storage (S3, Azure Blob, GCS).\n"
        "\n"
        "- **Chunked** \u2014 read only the chunks that overlap your query\n"
        "- **Compressed** \u2014 multiple codecs supported\n"
        "- **Parallel I/O** \u2014 each chunk is an independent object\n"
        "- **xarray native** \u2014 `xr.open_zarr()` returns a lazy Dataset"
    )

    code(
        "import xarray as xr\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# Daymet daily weather data for Hawaii (hosted on Azure)\n"
        "zarr_url = 'https://daymeteuwest.blob.core.windows.net/daymet-zarr/daily/hi.zarr'\n"
        "\n"
        "ds = xr.open_zarr(zarr_url)\n"
        "print('Zarr Dataset:')\n"
        "ds"
    )

    code(
        "# Efficient temporal slice \u2014 only the 2020 chunks are read\n"
        "tmax_2020 = ds['tmax'].sel(time='2020').mean(dim='time')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 6))\n"
        "tmax_2020.plot(ax=ax, cmap='RdYlBu_r')\n"
        "ax.set_title('Mean max temperature \u2014 Hawaii 2020')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )

    # --- 5. Vector formats -------------------------------------------------
    md(
        "---\n"
        "\n"
        "## 5) GeoParquet & vector formats\n"
        "\n"
        "### GeoParquet\n"
        "\n"
        "Apache Parquet extended with geometry columns:\n"
        "\n"
        "- **Columnar** \u2014 read only the columns you need\n"
        "- **Row-group filtering** \u2014 skip irrelevant data blocks\n"
        "- **High compression** \u2014 much smaller than Shapefile or GeoJSON\n"
        "- **Ecosystem** \u2014 DuckDB, Spark, pandas/geopandas\n"
        "\n"
        "### FlatGeobuf\n"
        "\n"
        "A compact binary vector format with a built-in spatial index.\n"
        "Great for streaming large feature collections over HTTP.\n"
        "\n"
        "> \U0001f4a1 **Tip:** Both formats can be read by `geopandas.read_file()` / "
        "`geopandas.read_parquet()` with optional `bbox` filtering."
    )

    code(
        "# GeoParquet example \u2014 Overture Maps buildings for Andorra\n"
        "# (only ~18 k rows, downloads fast)\n"
        "import geopandas as gpd\n"
        "\n"
        "gpq_url = 'https://data.source.coop/cholmes/overture/geoparquet-country-quad-2/AD.parquet'\n"
        "%time buildings = gpd.read_parquet(gpq_url)\n"
        "\n"
        "print(f'Loaded {len(buildings):,} buildings')\n"
        "buildings.head(3)"
    )

    code(
        "fig, ax = plt.subplots(figsize=(8, 8))\n"
        "buildings.plot(ax=ax, facecolor='lightblue', edgecolor='navy',\n"
        "              linewidth=0.3, alpha=0.7)\n"
        "ax.set_title('Building footprints \u2014 Andorra (GeoParquet)')\n"
        "ax.set_xlabel('Longitude')\n"
        "ax.set_ylabel('Latitude')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )

    # --- 6. Comparison table -----------------------------------------------
    md(
        "---\n"
        "\n"
        "## 6) Format comparison\n"
        "\n"
        "| Format | Best for | Cloud-native? | xarray support |\n"
        "|--------|----------|---------------|----------------|\n"
        "| **COG** | Single raster images | \u2705 Yes | via `rioxarray` |\n"
        "| **Zarr** | Large N-D arrays / time series | \u2705 Yes | Native |\n"
        "| **NetCDF** | Climate / atmospheric data | \u26a0\ufe0f With Kerchunk | Native |\n"
        "| **GeoParquet** | Large vector datasets | \u2705 Yes | via `geopandas` |\n"
        "| **FlatGeobuf** | Streaming vector with spatial queries | \u2705 Yes | via `geopandas` |\n"
        "\n"
        "**Rule of thumb:** if you can choose, use **COG** for rasters and **GeoParquet** for vectors."
    )

    # --- 7. Exercises -------------------------------------------------------
    md("---\n\n## 7) Exercises")

    md(
        "### \u2705 Try it \u2014 Partial COG read\n"
        "\n"
        "1. Change the `Window` parameters to read a **1000\u00d71000** pixel patch "
        "from the top-left corner of the COG.\n"
        "2. Display it with `imshow`."
    )

    code("# TODO: read a 1000\u00d71000 patch starting at (0, 0)\n")

    md(
        "### \u2705 Try it \u2014 Zarr temporal query\n"
        "\n"
        "Using the Daymet dataset opened above, compute and plot the **mean minimum temperature** "
        "(`tmin`) for **2015**."
    )

    code("# TODO: ds['tmin'].sel(time='2015').mean(dim='time').plot()\n")

    md(
        "### \U0001f9e0 Checkpoint\n"
        "\n"
        "**Q1.** What is a COG\u2019s main advantage over a regular GeoTIFF?\n"
        "\n"
        "- A) Smaller file size\n"
        "- B) Internal tiling + overviews allow partial HTTP reads\n"
        "- C) It stores vector data\n"
        "\n"
        "**Q2.** Which format is best for storing a large 4-D (time, band, y, x) data cube in the cloud?\n"
        "\n"
        "- A) Shapefile\n"
        "- B) COG\n"
        "- C) Zarr\n"
        "\n"
        "**Q3.** What does `xr.open_zarr()` return?\n"
        "\n"
        "- A) A NumPy array loaded into memory\n"
        "- B) A lazy xarray Dataset backed by Dask arrays\n"
        "- C) A pandas DataFrame"
    )

    # --- 8. Recap ----------------------------------------------------------
    md(
        "---\n"
        "\n"
        "## 8) Recap & further reading\n"
        "\n"
        "| Concept | Key takeaway |\n"
        "|---------|-------------|\n"
        "| ARD | Pre-processed data ready for analysis (e.g., Sentinel-2 L2A) |\n"
        "| COG | GeoTIFF with internal tiles + overviews \u2192 partial HTTP reads |\n"
        "| Zarr | Chunked N-D arrays \u2192 cloud-native parallel I/O |\n"
        "| GeoParquet | Columnar vector format \u2192 fast filtering |\n"
        "\n"
        "\U0001f4a1 **Don\u2019t download everything** \u2014 cloud-native formats let you read only what you need.\n"
        "\n"
        "### Further reading\n"
        "\n"
        "- [Cloud-Native Geospatial Guide](https://guide.cloudnativegeo.org/)\n"
        "- [COG specification](https://www.cogeo.org/)\n"
        "- [Zarr docs](https://zarr.readthedocs.io/)\n"
        "- [CEOS ARD (CARD4L)](https://ceos.org/ard/)\n"
        "- [STAC specification](https://stacspec.org/)"
    )

    _write(nb, NB_DIR / "A1_cloud_data_formats.ipynb")


# ============================================================================
# A2 — Parallel Computing with Dask
# ============================================================================
def build_a2():
    nb = _new_nb()
    md = lambda s: _md(nb, s)
    code = lambda s: _code(nb, s)

    # ------------------------------------------------------------------
    md(
        "![Course header](../assets/img/header.png)\n"
        "\n"
        "# A2 \u2014 Parallel Computing with Dask\n"
        "Optional reference \u2014 scale your EO workflows beyond laptop RAM\n"
        "\n"
        "This appendix introduces **Dask**, a library that adds chunked, parallel "
        "execution to NumPy and xarray.\n"
        "You will create a local cluster, compute NDVI in parallel, and learn the "
        "lazy-evaluation pattern.\n"
        "\n"
        "## Learning Objectives\n"
        "\n"
        "By the end of this notebook you will be able to:\n"
        "\n"
        "- Explain why Dask is needed for large-scale EO analysis\n"
        "- Create a local Dask cluster and open the Dashboard\n"
        "- Open raster data with `chunks=` for lazy, parallel processing\n"
        "- Understand the difference between lazy graphs and `.compute()`\n"
        "- Follow best practices for chunk sizing\n"
        "\n"
        "Tooling:\n"
        "- dask.distributed\n"
        "- xarray + rioxarray\n"
        "- matplotlib\n"
        "\n"
        "\u23f1\ufe0f Estimated time: **30 \u2013 45 minutes**"
    )

    md(
        "---\n"
        "\n"
        "## Table of contents\n"
        "\n"
        "1. Why Dask?\n"
        "2. Setup \u2014 create a local cluster\n"
        "3. Lazy evaluation with xarray + Dask\n"
        "4. Parallel NDVI example\n"
        "5. Best practices\n"
        "6. Scaling beyond your laptop\n"
        "7. Cleanup\n"
        "8. Exercises\n"
        "9. Recap"
    )

    # --- 1. Why Dask? -----------------------------------------------------
    md(
        "---\n"
        "\n"
        "## 1) Why Dask?\n"
        "\n"
        "A single Sentinel-2 tile at 10 m resolution is ~1 GB.  \n"
        "A year of data over a country can easily exceed **hundreds of GB**.\n"
        "\n"
        "| Challenge | Dask\u2019s solution |\n"
        "|-----------|------------------|\n"
        "| Data doesn\u2019t fit in RAM | **Chunking** \u2014 process piece by piece |\n"
        "| Loading data you don\u2019t need | **Lazy evaluation** \u2014 load on demand |\n"
        "| Single-core bottleneck | **Parallelisation** \u2014 use all cores |\n"
        "\n"
        "Dask integrates seamlessly with xarray:  \n"
        "open data with `chunks=` and every subsequent xarray operation runs in parallel automatically."
    )

    # --- 2. Setup ---------------------------------------------------------
    md("---\n\n## 2) Setup \u2014 create a local cluster")

    code(
        "import numpy as np\n"
        "import xarray as xr\n"
        "import rioxarray\n"
        "import matplotlib.pyplot as plt"
    )

    code(
        "from dask.distributed import Client, LocalCluster\n"
        "\n"
        "cluster = LocalCluster(\n"
        "    n_workers=4,\n"
        "    threads_per_worker=1,\n"
        "    memory_limit='2GB',\n"
        ")\n"
        "client = Client(cluster)\n"
        "client  # click the Dashboard link to monitor workers"
    )

    md(
        "### The Dask Dashboard\n"
        "\n"
        "Click the **Dashboard link** above. It shows:\n"
        "\n"
        "- **Task Stream** \u2014 what each worker is doing over time\n"
        "- **Progress** \u2014 overall bar for the current computation\n"
        "- **Memory** \u2014 RAM usage per worker\n"
        "\n"
        "> \U0001f4a1 **Tip:** In JupyterLab the Dask sidebar (orange icon) lets you embed dashboard panels."
    )

    # --- 3. Lazy evaluation -----------------------------------------------
    md(
        "---\n"
        "\n"
        "## 3) Lazy evaluation with xarray + Dask\n"
        "\n"
        "When you open data with `chunks=`, xarray stores **Dask arrays** instead of NumPy arrays.  \n"
        "Operations build a task graph but **don\u2019t execute** until you call `.compute()`."
    )

    code(
        "# Open a COG with chunking enabled \u2014 nothing is loaded yet\n"
        "cog_url = (\n"
        "    'https://sentinel-cogs.s3.us-west-2.amazonaws.com/'\n"
        "    'sentinel-s2-l2a-cogs/32/T/PS/2024/12/'\n"
        "    'S2B_32TPS_20241228_0_L2A/B04.tif'\n"
        ")\n"
        "\n"
        "red = rioxarray.open_rasterio(cog_url, chunks='auto')\n"
        "\n"
        "print(f'Shape:  {red.shape}')\n"
        "print(f'Chunks: {red.chunks}')\n"
        "print(f'Backed by: {type(red.data).__name__}')  # dask.array\n"
        "red"
    )

    code(
        "# Chain lazy operations \u2014 NO data is loaded\n"
        "red_sq = red.squeeze('band', drop=True)\n"
        "patch  = red_sq.isel(x=slice(2000, 4000), y=slice(2000, 4000))\n"
        "refl   = patch / 10_000.0\n"
        "mean_r = refl.mean()\n"
        "\n"
        "print(f'Mean (lazy): {mean_r}')  # still a Dask scalar"
    )

    md(
        "### Triggering computation: `.compute()`\n"
        "\n"
        "Call `.compute()` to execute the graph.  Watch the Dashboard!"
    )

    code(
        "%%time\n"
        "result = mean_r.compute()\n"
        "print(f'Mean reflectance: {result:.4f}')"
    )

    # --- 4. Parallel NDVI -------------------------------------------------
    md(
        "---\n"
        "\n"
        "## 4) Parallel NDVI example\n"
        "\n"
        "Load Red and NIR bands with chunking, compute NDVI, and plot \u2014 all in parallel."
    )

    code(
        "base_url = (\n"
        "    'https://sentinel-cogs.s3.us-west-2.amazonaws.com/'\n"
        "    'sentinel-s2-l2a-cogs/32/T/PS/2024/12/'\n"
        "    'S2B_32TPS_20241228_0_L2A'\n"
        ")\n"
        "\n"
        "red = rioxarray.open_rasterio(\n"
        "    f'{base_url}/B04.tif', chunks={'x': 1024, 'y': 1024}\n"
        ").squeeze('band', drop=True)\n"
        "\n"
        "nir = rioxarray.open_rasterio(\n"
        "    f'{base_url}/B08.tif', chunks={'x': 1024, 'y': 1024}\n"
        ").squeeze('band', drop=True)\n"
        "\n"
        "print(f'Red chunks: {red.chunks}')\n"
        "print(f'NIR chunks: {nir.chunks}')"
    )

    code(
        "# NDVI \u2014 still lazy\n"
        "ndvi = (nir - red) / (nir + red)\n"
        "ndvi.name = 'ndvi'\n"
        "\n"
        "print(f'Backed by: {type(ndvi.data).__name__}')\n"
        "print(f'Chunks:    {len(ndvi.chunks[0])} \u00d7 {len(ndvi.chunks[1])}')\n"
        "ndvi"
    )

    code(
        "%%time\n"
        "# Coarsen for faster demo, then compute\n"
        "ndvi_small = ndvi.coarsen(x=4, y=4, boundary='trim').mean()\n"
        "ndvi_computed = ndvi_small.compute()\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 8))\n"
        "ndvi_computed.plot(ax=ax, cmap='RdYlGn', vmin=-0.5, vmax=0.8)\n"
        "ax.set_title('NDVI (computed with Dask)')\n"
        "ax.set_aspect('equal')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )

    # --- 5. Best practices ------------------------------------------------
    md(
        "---\n"
        "\n"
        "## 5) Best practices\n"
        "\n"
        "### When to call `.compute()`\n"
        "\n"
        "| Situation | Approach |\n"
        "|-----------|----------|\n"
        "| Building a processing chain | Stay **lazy** \u2014 chain operations |\n"
        "| Checking an intermediate value | `.compute()` on a small subset |\n"
        "| Final result | `.compute()` |\n"
        "| Saving to file | `.to_netcdf()` / `.to_zarr()` trigger compute |\n"
        "| Plotting | `.plot()` triggers compute |\n"
        "\n"
        "### Chunk-size guidelines\n"
        "\n"
        "- Target **100 MB \u2013 1 GB** per chunk (rule of thumb)\n"
        "- Too small \u2192 scheduling overhead\n"
        "- Too large \u2192 memory pressure\n"
        "- Align chunks with COG tile size or Zarr chunk layout"
    )

    code(
        "# Check actual chunk size in MB\n"
        "chunk_mb = (\n"
        "    red.data.chunksize[0] * red.data.chunksize[1]\n"
        "    * red.dtype.itemsize / 1e6\n"
        ")\n"
        "print(f'Chunk size: {chunk_mb:.1f} MB')"
    )

    md(
        "### Common mistakes\n"
        "\n"
        "| Mistake | Why it\u2019s bad |\n"
        "|---------|-------------|\n"
        "| Calling `.values` on a huge array | Loads everything into memory |\n"
        "| Rechunking repeatedly | Expensive shuffle |\n"
        "| Millions of tiny tasks | Scheduler overhead |\n"
        "| Forgetting to close the client | Resource leak |"
    )

    # --- 6. Scaling -------------------------------------------------------
    md(
        "---\n"
        "\n"
        "## 6) Scaling beyond your laptop\n"
        "\n"
        "The same code works on bigger systems \u2014 only the cluster definition changes:\n"
        "\n"
        "| Environment | Cluster type | Package |\n"
        "|-------------|--------------|----------|\n"
        "| Laptop | `LocalCluster` | dask.distributed |\n"
        "| HPC (SLURM / PBS) | `SLURMCluster` | dask-jobqueue |\n"
        "| Kubernetes | `KubeCluster` | dask-kubernetes |\n"
        "| Managed cloud | `Gateway` | Dask Gateway / Coiled |\n"
        "\n"
        "```python\n"
        "# Example: SLURM cluster on an HPC\n"
        "from dask_jobqueue import SLURMCluster\n"
        "\n"
        "cluster = SLURMCluster(cores=4, memory='16GB', walltime='01:00:00')\n"
        "cluster.scale(jobs=10)\n"
        "client = Client(cluster)\n"
        "```"
    )

    # --- 7. Cleanup -------------------------------------------------------
    md("---\n\n## 7) Cleanup")

    code(
        "client.close()\n"
        "cluster.close()\n"
        "print('\u2705 Dask cluster closed')"
    )

    # --- 8. Exercises -----------------------------------------------------
    md("---\n\n## 8) Exercises")

    md(
        "### \u2705 Try it \u2014 Chunk comparison\n"
        "\n"
        "1. Re-open the Red band COG with `chunks={'x': 512, 'y': 512}`.\n"
        "2. Print the number of chunks (`len(red.chunks[0]) * len(red.chunks[1])`).\n"
        "3. Compare with `chunks={'x': 2048, 'y': 2048}` \u2014 how many chunks now?\n"
        "4. Which would you expect to be faster? Why?"
    )

    code("# TODO: experiment with different chunk sizes\n")

    md(
        "### \u2705 Try it \u2014 Parallel statistics\n"
        "\n"
        "Compute the **min, max, and standard deviation** of the NDVI array "
        "(before coarsening). Use `.compute()` once on all three."
    )

    code(
        "# TODO: compute min, max, std in one go\n"
        "# hint: ndvi_stats = xr.Dataset({\n"
        "#     'min': ndvi.min(),\n"
        "#     'max': ndvi.max(),\n"
        "#     'std': ndvi.std(),\n"
        "# }).compute()"
    )

    md(
        "### \U0001f9e0 Checkpoint\n"
        "\n"
        "**Q1.** What does `chunks='auto'` do when opening a raster?\n"
        "\n"
        "- A) Loads the entire file into a NumPy array\n"
        "- B) Lets Dask choose chunk sizes that match the file\u2019s internal tiling\n"
        "- C) Disables chunking\n"
        "\n"
        "**Q2.** When does a Dask computation actually run?\n"
        "\n"
        "- A) As soon as you write the expression\n"
        "- B) When you call `.compute()` (or plot / save)\n"
        "- C) It never runs automatically\n"
        "\n"
        "**Q3.** What is a good target chunk size?\n"
        "\n"
        "- A) 1 KB\n"
        "- B) 100 MB \u2013 1 GB\n"
        "- C) 50 GB"
    )

    # --- 9. Recap ---------------------------------------------------------
    md(
        "---\n"
        "\n"
        "## 9) Recap\n"
        "\n"
        "| Concept | Key takeaway |\n"
        "|---------|-------------|\n"
        "| Dask | Chunked parallel execution for NumPy / xarray |\n"
        "| `LocalCluster` | Run workers on your own machine |\n"
        "| `chunks=` | Turns xarray arrays into lazy Dask graphs |\n"
        "| `.compute()` | Triggers actual execution |\n"
        "| Dashboard | Monitor workers, tasks, and memory |\n"
        "\n"
        "\U0001f4a1 **Stay lazy as long as possible** \u2014 chain operations before computing.\n"
        "\n"
        "### Further reading\n"
        "\n"
        "- [Dask documentation](https://docs.dask.org/)\n"
        "- [xarray with Dask](https://docs.xarray.dev/en/stable/user-guide/dask.html)\n"
        "- [Dask best practices](https://docs.dask.org/en/stable/best-practices.html)\n"
        "- [Pangeo](https://pangeo.io/) \u2014 community using Dask for geoscience"
    )

    _write(nb, NB_DIR / "A2_optional_dask_parallel.ipynb")


# ============================================================================
if __name__ == "__main__":
    print("Building appendix notebooks\u2026")
    build_a1()
    build_a2()
    print("Done.")
