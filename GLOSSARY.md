# Glossary

Key terms used throughout the EO Python Foundations course.

---

## Python & Jupyter

**Cell**
A block in a Jupyter notebook. Code cells contain runnable Python; Markdown cells contain formatted text.

**Kernel**
The Python process that executes code in a notebook. Variables persist until you restart the kernel.

**f-string**
A string prefixed with `f` that embeds expressions in `{curly braces}`:
`f'Cloud cover: {cloud:.1f}%'`

**List comprehension**
A compact way to build a list from a loop:
`[x * 2 for x in range(5)]`

**Dictionary (dict)**
A key → value mapping: `{'tile': 'T32UQD', 'cloud_cover': 12}`. JSON and STAC metadata use the same structure.

**pathlib.Path**
A cross-platform way to work with file paths in Python:
`Path('data') / 'scene.csv'`

---

## Data-analysis libraries

**pandas**
Python library for tabular data. A **DataFrame** is a table with named columns; a **Series** is a single column.

**NumPy**
Library for fast numerical arrays. Every satellite raster is stored as a NumPy array under the hood.

**Matplotlib**
The standard Python plotting library. Key plot types for EO: `hist`, `scatter`, `imshow`.

**xarray**
Multi-dimensional labelled array library. Extends NumPy with named dimensions, coordinates, and metadata — ideal for data cubes.

**Dask**
Parallel computing library that enables out-of-core computation on datasets larger than memory. Integrates with xarray via chunked arrays.

---

## Earth observation (EO)

**NDVI (Normalised Difference Vegetation Index)**
`(NIR − Red) / (NIR + Red)` — ranges from −1 to 1, with higher values indicating healthier vegetation.

**Scene**
A single satellite acquisition covering a specific area at a specific time.

**Tile**
A fixed grid cell in a satellite tiling scheme (e.g., Sentinel-2 MGRS tile `T32UQD`).

**Cloud cover**
The percentage of a scene obscured by clouds. Low cloud cover means clearer imagery.

**Cloud mask**
A boolean array (True = cloud, False = clear) used to exclude cloudy pixels from analysis.

**Masking**
Setting pixels to `NaN` or filtering them out based on quality flags (clouds, shadows, snow) before analysis.

**Raster**
A grid of pixels representing spatial data — e.g., an NDVI image stored as a 2-D NumPy array.

**Band**
A single channel of a satellite image capturing a specific wavelength range (e.g., Red, NIR, SWIR).

**Composite**
An image formed by stacking or combining multiple bands or scenes — e.g., an RGB composite from Red, Green, Blue bands.

**Reflectance**
The fraction of incoming sunlight reflected by a surface. Satellite sensors measure reflectance at different wavelengths.

---

## Coordinate systems & geometry

**CRS (Coordinate Reference System)**
Defines how coordinates map to locations on Earth. Common examples: WGS 84 (EPSG:4326), UTM Zone 32N (EPSG:32632).

**Bounding box (bbox)**
A rectangle defined by `[min_lon, min_lat, max_lon, max_lat]` used to specify a geographic search area.

**AOI (Area of Interest)**
The geographic region you want to analyse.

**Reprojection**
Transforming data from one CRS to another.

---

## Data formats

**CSV (Comma-Separated Values)**
A plain-text tabular format. Loaded with `pandas.read_csv()`.

**JSON (JavaScript Object Notation)**
A text format for structured data using key-value pairs and arrays. STAC metadata is JSON.

**GeoTIFF**
A raster image format that includes geographic metadata (CRS, extent).

**COG (Cloud Optimised GeoTIFF)**
A GeoTIFF structured with internal tiling and overviews so that remote clients can read only the part they need via HTTP range requests.

**NetCDF (Network Common Data Form)**
A self-describing format for multi-dimensional arrays, common in climate science.

**Zarr**
A chunked, compressed array format designed for cloud storage. Supports parallel access.

**NPZ**
NumPy's compressed archive format for saving multiple arrays to a single file.

---

## STAC (SpatioTemporal Asset Catalog)

**STAC**
An open specification for describing geospatial data using standardised JSON.

**Catalog**
Top-level container that organises STAC Collections.

**Collection**
A group of related Items sharing common properties — e.g., all Sentinel-2 Level-2A scenes.

**Item**
A single spatiotemporal unit (one scene) with metadata and links to Assets.

**Asset**
An actual data file linked from an Item — e.g., a COG for band B04.

**pystac-client**
Python library for searching STAC APIs.

**Planetary Computer**
Microsoft's platform providing STAC-catalogued EO data with a free Jupyter computing environment.

**stackstac**
Python library that loads STAC items directly into an xarray DataArray backed by Dask.

---

## Processing concepts

**Vectorised operation**
Applying a function to an entire array at once (no Python loop), leveraging NumPy's C backend for speed.

**Boolean indexing**
Filtering rows or pixels using a True/False array:
`df[df['cloud_cover'] < 20]`

**GroupBy**
Splitting data by a categorical column, applying a function to each group, and combining the results.

**Lazy evaluation**
Deferring computation until results are explicitly requested (e.g., `.compute()` in Dask).

**Resampling**
Changing the temporal resolution of a time series — e.g., daily → monthly.

---

*Last updated: February 2026*
