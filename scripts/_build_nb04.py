#!/usr/bin/env python3
"""Build notebook 04 – STAC Fundamentals."""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
OUT = NB_DIR / "04_stac_fundamentals.ipynb"

nb = {
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


def md(source: str):
    lines = source.split("\n")
    for i in range(len(lines) - 1):
        lines[i] += "\n"
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": lines})


def code(source: str):
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


# =============================================================================
# Cell 1 – Title & learning objectives
# =============================================================================
md(
    "![Course header](../assets/img/header.png)\n"
    "\n"
    "# 04 \u2014 STAC Fundamentals\n"
    "Search, filter, and preview satellite data using a real STAC API\n"
    "\n"
    "This notebook introduces the **SpatioTemporal Asset Catalog (STAC)** standard.\n"
    "You will use the Microsoft Planetary Computer STAC API to find Sentinel\u20112 scenes.\n"
    "\n"
    "## Learning Objectives\n"
    "\n"
    "This notebook serves as a **quick reference** for STAC discovery. "
    "If you\u2019re already comfortable with STAC, feel free to skim or skip ahead to Notebook 05.\n"
    "\n"
    "By the end of this notebook, you will be able to:\n"
    "\n"
    "- Explain what STAC is and why it matters for EO data discovery\n"
    "- Connect to a STAC API and list available collections\n"
    "- Search for satellite imagery by area, time, and cloud cover\n"
    "- Inspect Item metadata and assets\n"
    "- Build a results table with pandas\n"
    "- Preview a band from a STAC Item\n"
    "- Export results for reuse\n"
    "\n"
    "Tooling in this notebook:\n"
    "- pystac-client\n"
    "- planetary-computer\n"
    "- pandas\n"
    "- matplotlib\n"
    "\n"
    "\u23f1\ufe0f Estimated time: **1 \u2013 1.5 hours**\n"
    "\n"
    "We keep the AOI small and the number of Items low to stay fast."
)

# =============================================================================
# Cell 2 – How to use this notebook
# =============================================================================
md(
    "---\n"
    "\n"
    "## How to use this notebook\n"
    "\n"
    "1. Run cells in order.\n"
    "2. Keep bbox small.\n"
    "3. If you get empty results, loosen filters (wider time range or higher cloud threshold).\n"
    "4. If something breaks, restart kernel and run all."
)

# =============================================================================
# Cell 3 – Table of contents
# =============================================================================
md(
    "---\n"
    "\n"
    "## Table of contents\n"
    "\n"
    "1. Setup\n"
    "2. What is STAC?\n"
    "3. Connect to a STAC API\n"
    "4. Define a search area (bbox)\n"
    "5. Search for Sentinel-2 images\n"
    "6. Inspect Item metadata and assets\n"
    "7. Build a results table with pandas\n"
    "8. Preview a band from a STAC Item\n"
    "9. Export results\n"
    "10. Exercises\n"
    "11. Recap"
)

# =============================================================================
# Section 1 – Setup
# =============================================================================
md("---\n\n## 1) Setup\n\n### Imports")

code(
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "import pystac_client\n"
    "import planetary_computer"
)

md("### Paths\nOutputs go to `../outputs/`.")

code(
    "OUT_DIR = Path('..') / 'outputs'\n"
    "OUT_DIR.mkdir(exist_ok=True)\n"
    "OUT_DIR.resolve()"
)

# =============================================================================
# Section 2 – What is STAC?
# =============================================================================
md(
    "---\n"
    "\n"
    "## 2) What is STAC?\n"
    "\n"
    "**STAC** (SpatioTemporal Asset Catalog) is a specification for describing geospatial data.\n"
    "It has become the standard for organising and discovering satellite imagery in the cloud.\n"
    "\n"
    "### STAC Components\n"
    "\n"
    "| Component      | Description                          | Example                                |\n"
    "|----------------|--------------------------------------|----------------------------------------|\n"
    "| **Catalog**    | The root entry point                 | Planetary Computer, Earth Search       |\n"
    "| **Collection** | A group of related Items             | Sentinel-2 L2A, Landsat 8             |\n"
    "| **Item**       | A single observation (one scene)     | One Sentinel-2 granule on 2024-06-15  |\n"
    "| **Asset**      | A file associated with an Item       | Red band GeoTIFF, thumbnail PNG       |\n"
    "\n"
    "### Why STAC?\n"
    "\n"
    "Before STAC, every data provider had their own API and query syntax.\n"
    "With STAC:\n"
    "\n"
    "- **Standardised** \u2014 the same query works across providers\n"
    "- **Cloud-native** \u2014 direct links to COGs (Cloud-Optimised GeoTIFFs), no bulk download needed\n"
    "- **Rich metadata** \u2014 spatial extent, temporal coverage, cloud cover, band info, etc.\n"
    "\n"
    "Think of STAC as a **library catalogue for satellite images**: you search the catalogue first, "
    "then load only the data you need."
)

# =============================================================================
# Section 3 – Connect to a STAC API
# =============================================================================
md(
    "---\n"
    "\n"
    "## 3) Connect to a STAC API\n"
    "\n"
    "We use **Microsoft Planetary Computer**. It hosts Sentinel-2, Landsat, and many other collections.\n"
    "\n"
    "Planetary Computer requires **signed URLs** \u2014 the `planetary_computer.sign_inplace` modifier "
    "handles that automatically."
)

code(
    "STAC_API_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1'\n"
    "\n"
    "catalog = pystac_client.Client.open(\n"
    "    STAC_API_URL,\n"
    "    modifier=planetary_computer.sign_inplace,\n"
    ")\n"
    "print(f'Connected to: {catalog.title}')"
)

md("### 3.1 List available collections\n\nA STAC Catalog can host many Collections. Let\u2019s see a selection.")

code(
    "# Print collections whose names contain common EO keywords\n"
    "print('Selected Collections:')\n"
    "print('-' * 60)\n"
    "keywords = ['sentinel', 'landsat', 'modis', 'dem']\n"
    "for collection in catalog.get_collections():\n"
    "    if any(kw in collection.id.lower() for kw in keywords):\n"
    "        print(f'  {collection.id}: {collection.title}')"
)

# =============================================================================
# Section 4 – Define a search area (bbox)
# =============================================================================
md(
    "---\n"
    "\n"
    "## 4) Define a search area (bbox)\n"
    "\n"
    "We define an **area of interest (AOI)** as a bounding box:\n"
    "\n"
    "```\n"
    "(west, south, east, north)\n"
    "```\n"
    "\n"
    "in geographic coordinates (longitude / latitude).\n"
    "\n"
    "Keep it small \u2014 a few kilometres is enough for this exercise."
)

code(
    "# AOI: small area near W\u00fcrzburg, Germany\n"
    "AOI_BBOX = (9.95, 49.78, 10.05, 49.83)\n"
    "\n"
    "print(f'West:  {AOI_BBOX[0]}')\n"
    "print(f'South: {AOI_BBOX[1]}')\n"
    "print(f'East:  {AOI_BBOX[2]}')\n"
    "print(f'North: {AOI_BBOX[3]}')"
)

# =============================================================================
# Section 5 – Search for Sentinel-2 images
# =============================================================================
md(
    "---\n"
    "\n"
    "## 5) Search for Sentinel-2 images\n"
    "\n"
    "We search the `sentinel-2-l2a` collection (Level-2A = surface reflectance) with filters:\n"
    "\n"
    "- **bbox** \u2014 spatial extent\n"
    "- **datetime** \u2014 time range\n"
    "- **query** \u2014 cloud cover threshold"
)

code(
    "# Search parameters\n"
    "DATE_RANGE = '2024-06-01/2024-06-30'\n"
    "MAX_CLOUD = 20  # percent\n"
    "\n"
    "search = catalog.search(\n"
    "    collections=['sentinel-2-l2a'],\n"
    "    bbox=AOI_BBOX,\n"
    "    datetime=DATE_RANGE,\n"
    "    query={'eo:cloud_cover': {'lt': MAX_CLOUD}},\n"
    ")\n"
    "\n"
    "items = list(search.items())\n"
    "print(f'Found {len(items)} items with cloud cover < {MAX_CLOUD}%')"
)

md("### 5.1 Quick list of results")

code(
    "print('Found Items:')\n"
    "print('-' * 70)\n"
    "for item in items:\n"
    "    cloud = item.properties.get('eo:cloud_cover', 'N/A')\n"
    "    dt = item.properties.get('datetime', 'N/A')\n"
    "    print(f'  {item.id}')\n"
    "    print(f'    Date: {dt}    Cloud cover: {cloud:.1f}%')\n"
    "    print()"
)

md("> **\U0001f4a1 Tip:** If `Found 0 items`, try widening the time range or raising `MAX_CLOUD`.")

# =============================================================================
# Section 6 – Inspect Item metadata and assets
# =============================================================================
md(
    "---\n"
    "\n"
    "## 6) Inspect Item metadata and assets\n"
    "\n"
    "Each STAC **Item** carries rich metadata. Let\u2019s examine the first result."
)

code(
    "item = items[0]\n"
    "\n"
    "print('Item Details')\n"
    "print('=' * 50)\n"
    "print(f'ID:       {item.id}')\n"
    "print(f'Datetime: {item.datetime}')\n"
    'print(f\'Geometry: {item.geometry["type"]}\')\n'
    "print(f'Bbox:     {item.bbox}')"
)

md("### 6.1 Key properties\n\nProperties are the metadata dictionary of the Item.")

code(
    "print('Key Properties:')\n"
    "print('-' * 50)\n"
    "\n"
    "props_of_interest = [\n"
    "    'eo:cloud_cover',\n"
    "    'proj:epsg',\n"
    "    's2:granule_id',\n"
    "    'platform',\n"
    "    'constellation',\n"
    "]\n"
    "\n"
    "for prop in props_of_interest:\n"
    "    value = item.properties.get(prop, 'N/A')\n"
    "    print(f'  {prop}: {value}')"
)

md("### 6.2 Available assets\n\nAssets are the actual data files (GeoTIFFs, thumbnails, metadata) linked to an Item.")

code(
    "print('Available Assets:')\n"
    "print('-' * 50)\n"
    "for key, asset in item.assets.items():\n"
    "    title = asset.title if asset.title else key\n"
    "    print(f'  {key}: {title}')"
)

md(
    "### 6.3 Get URLs for specific bands\n"
    "\n"
    "Each spectral band is an Asset with a URL pointing to a Cloud-Optimised GeoTIFF (COG).\n"
    "These URLs let you stream pixel data directly \u2014 no download required."
)

code(
    "print('Band URLs:')\n"
    "print('-' * 50)\n"
    "\n"
    "bands_to_check = ['B02', 'B03', 'B04', 'B08']  # blue, green, red, NIR\n"
    "for band in bands_to_check:\n"
    "    if band in item.assets:\n"
    "        url = item.assets[band].href\n"
    "        short_url = '...' + url[-60:] if len(url) > 60 else url\n"
    "        print(f'  {band}: {short_url}')\n"
    "    else:\n"
    "        print(f'  {band}: not found in assets')"
)

# =============================================================================
# Section 7 – Build a results table with pandas
# =============================================================================
md(
    "---\n"
    "\n"
    "## 7) Build a results table with pandas\n"
    "\n"
    "For more than a handful of Items, a DataFrame is easier to work with than looping and printing."
)

code(
    "rows = []\n"
    "for it in items:\n"
    "    props = it.properties\n"
    "    rows.append({\n"
    "        'id': it.id,\n"
    "        'datetime': props.get('datetime'),\n"
    "        'cloud_cover': props.get('eo:cloud_cover', np.nan),\n"
    "        'platform': props.get('platform', 'N/A'),\n"
    "        'epsg': props.get('proj:epsg', 'N/A'),\n"
    "    })\n"
    "\n"
    "df = pd.DataFrame(rows)\n"
    "df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')\n"
    "df['cloud_cover'] = pd.to_numeric(df['cloud_cover'], errors='coerce')\n"
    "df.sort_values('cloud_cover').head(10)"
)

md("### 7.1 Quick stats")

code(
    "print(f'Total items:       {len(df)}')\n"
    'print(f\'Date range:        {df["datetime"].min()} to {df["datetime"].max()}\')\n'
    'print(f\'Mean cloud cover:  {df["cloud_cover"].mean():.1f}%\')\n'
    'print(f\'Platforms:         {df["platform"].unique().tolist()}\')'
)

md("### 7.2 Cloud-cover histogram")

code(
    "fig, ax = plt.subplots(figsize=(6, 3))\n"
    "ax.hist(df['cloud_cover'].dropna(), bins=10, edgecolor='white')\n"
    "ax.set_xlabel('Cloud cover (%)')\n"
    "ax.set_ylabel('Number of scenes')\n"
    "ax.set_title('Cloud-cover distribution of search results')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# =============================================================================
# Section 8 – Preview a band from a STAC Item
# =============================================================================
md(
    "---\n"
    "\n"
    "## 8) Preview a band from a STAC Item\n"
    "\n"
    "We can load a single band directly from its COG URL using `rioxarray`.\n"
    "We use `overview_level=3` (a coarse overview pyramid) so it downloads fast.\n"
    "\n"
    "> **Note:** `rioxarray` extends xarray with rasterio-backed I/O.\n"
    "> Install with `pip install rioxarray` if needed."
)

code(
    "import rioxarray\n"
    "\n"
    "# Pick the best (lowest-cloud) item\n"
    "best_item = items[0]\n"
    "red_url = best_item.assets['B04'].href\n"
    "\n"
    "print(f'Loading red band (B04) from: {best_item.id}')\n"
    "%time red_band = rioxarray.open_rasterio(red_url, overview_level=3)\n"
    "\n"
    "print(f'Shape:  {red_band.shape}')\n"
    "print(f'CRS:    {red_band.rio.crs}')\n"
    "print(f'Bounds: {red_band.rio.bounds()}')"
)

code(
    "fig, ax = plt.subplots(figsize=(8, 8))\n"
    "red_band.squeeze().plot(ax=ax, cmap='Reds', vmin=0, vmax=3000)\n"
    "ax.set_title(f'Red band (B04) \\u2014 {best_item.id}\\n{best_item.datetime}')\n"
    "ax.set_aspect('equal')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

md(
    "### 8.1 Display a thumbnail (if available)\n"
    "\n"
    "Many STAC Items include a pre-rendered thumbnail or preview image as an Asset."
)

code(
    "# Check for a rendered preview or thumbnail asset\n"
    "preview_key = 'rendered_preview' if 'rendered_preview' in best_item.assets else 'thumbnail'\n"
    "\n"
    "if preview_key in best_item.assets:\n"
    "    from IPython.display import Image, display\n"
    "    thumb_url = best_item.assets[preview_key].href\n"
    "    print(f'Displaying: {preview_key}')\n"
    "    display(Image(url=thumb_url, width=400))\n"
    "else:\n"
    "    print('No thumbnail or rendered_preview asset available for this item.')"
)

# =============================================================================
# Section 9 – Export results
# =============================================================================
md(
    "---\n"
    "\n"
    "## 9) Export results\n"
    "\n"
    "Save the results table to CSV so you can reuse it later (e.g., in Notebook 05)."
)

code(
    "out_csv = OUT_DIR / 'stac_search_results.csv'\n"
    "df.to_csv(out_csv, index=False)\n"
    "print(f'Saved {len(df)} rows to {out_csv.resolve()}')"
)

# =============================================================================
# Section 10 – Exercises
# =============================================================================
md("---\n\n## 10) Exercises")

md(
    "### \u2705 Try it \u2014 Change the search parameters\n"
    "\n"
    "1. Pick a **different AOI** \u2014 try a place you know.\n"
    "   Look up approximate coordinates on Google Maps or [bboxfinder.com](http://bboxfinder.com).\n"
    "2. Change the **time range** to a different month.\n"
    "3. Tighten `MAX_CLOUD` to `10` \u2014 how many results do you get?"
)

code(
    "# TODO: fill in your own AOI and time range\n"
    "my_bbox = (___,  ___,  ___,  ___)  # (west, south, east, north)\n"
    "my_date_range = '____-__-__/____-__-__'\n"
    "my_max_cloud = 10\n"
    "\n"
    "my_search = catalog.search(\n"
    "    collections=['sentinel-2-l2a'],\n"
    "    bbox=my_bbox,\n"
    "    datetime=my_date_range,\n"
    "    query={'eo:cloud_cover': {'lt': my_max_cloud}},\n"
    ")\n"
    "my_items = list(my_search.items())\n"
    "print(f'Found {len(my_items)} items')"
)

code(
    "# TODO: Print the dates and cloud cover of your first 5 results\n"
    "for it in my_items[:5]:\n"
    "    dt = it.properties.get('datetime', 'N/A')\n"
    "    cc = it.properties.get('eo:cloud_cover', 'N/A')\n"
    "    print(f'  {dt}  \\u2014  cloud: {cc}%')"
)

md(
    "### \u2705 Try it \u2014 Explore a different collection\n"
    "\n"
    "Search for **Landsat** imagery instead of Sentinel-2.\n"
    "The collection ID on Planetary Computer is `landsat-c2-l2`."
)

code("# TODO: search for Landsat images over the same AOI\n")

md(
    '### \U0001f9e0 Checkpoint\n'
    "\n"
    '**Q1.** What does a STAC "Collection" represent?\n'
    "\n"
    "- A) A single satellite image\n"
    "- B) A group of related Items (e.g., all Sentinel-2 L2A scenes)\n"
    "- C) A file like a GeoTIFF\n"
    "\n"
    "**Q2.** How do you get the download URL for the red band of a STAC Item?\n"
    "\n"
    "- A) `item.red.url`\n"
    '- B) `item.assets["B04"].href`\n'
    '- C) `item.properties["red"]`\n'
    "\n"
    "**Q3.** What does the `bbox` parameter in a STAC search represent?\n"
    "\n"
    "- A) The pixel dimensions of the image\n"
    "- B) The geographic bounding box `(west, south, east, north)` of the area of interest\n"
    "- C) The cloud cover threshold\n"
    "\n"
    "**Q4.** Why do we use `planetary_computer.sign_inplace` as a modifier?\n"
    "\n"
    "- A) It converts images to PNG\n"
    "- B) It signs the asset URLs so we can access the data without authentication tokens\n"
    "- C) It compresses the search results"
)

# =============================================================================
# Section 11 – Recap
# =============================================================================
md(
    "---\n"
    "\n"
    "## 11) Recap\n"
    "\n"
    "You now know how to:\n"
    "\n"
    "| Skill | Tool / Code |\n"
    "|-------|-------------|\n"
    "| Connect to a STAC API | `pystac_client.Client.open(url)` |\n"
    "| Search by area, time, cloud | `catalog.search(collections, bbox, datetime, query)` |\n"
    "| Get Items from search | `list(search.items())` |\n"
    "| Read Item metadata | `item.properties`, `item.datetime`, `item.bbox` |\n"
    '| Access asset URLs | `item.assets["B04"].href` |\n'
    "| Build a table from Items | Loop \u2192 list of dicts \u2192 `pd.DataFrame()` |\n"
    "| Preview a band | `rioxarray.open_rasterio(url, overview_level=3)` |\n"
    "\n"
    "### STAC hierarchy reminder\n"
    "\n"
    "```\n"
    "Catalog\n"
    " \u2514\u2500\u2500 Collection  (e.g., sentinel-2-l2a)\n"
    "      \u2514\u2500\u2500 Item   (one scene on one date)\n"
    "           \u2514\u2500\u2500 Asset  (one file: B04.tif, thumbnail.png, \u2026)\n"
    "```\n"
    "\n"
    "### Next steps\n"
    "\n"
    "In **Notebook 05** you will stack multiple bands and scenes into an **xarray cube** using "
    "`stackstac`, compute NDVI, and export results."
)

# =============================================================================
# Write
# =============================================================================
if OUT.exists():
    OUT.unlink()

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written {len(nb['cells'])} cells to {OUT}")
