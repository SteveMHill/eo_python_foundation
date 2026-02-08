# EO Python Foundations

A beginner-friendly, notebook-based course covering the Python skills you need
for Earth-observation (EO) data analysis — from core syntax to satellite data
cubes in the cloud.

## Who this is for

- You have **little or no Python experience** and want to work with satellite data.
- You are comfortable using a laptop, a browser, and an editor / terminal.
- You want to build toward **xarray data cubes** and **STAC catalogs**.

## Prerequisites

| Requirement | Details |
|---|---|
| Python | ≥ 3.10 |
| Editor | VS Code with the Jupyter extension, **or** JupyterLab |
| OS | macOS, Linux, or Windows (WSL recommended) |

No prior EO experience is required — key terms are explained in the
[Glossary](GLOSSARY.md).

## Contents

| # | Notebook | Topic | Key skills |
|---|---|---|---|
| 01 | `01_python_basics.ipynb` | Python & Jupyter fundamentals | Cells, variables, types, f-strings, strings, lists, dicts, if/for, functions, paths, JSON |
| 02 | `02_data_analysis_basics.ipynb` | pandas + NumPy + Matplotlib | `read_csv`, filter, groupby, merge, arrays, shapes, masking, raster thinking, histogram, scatter, imshow |
| 03 | `03_xarray_eo.ipynb` | xarray for EO data cubes | Labelled dimensions, coordinates, selection, computation, plotting |
| 04 | `04_stac_fundamentals.ipynb` | STAC catalogs & search | STAC concepts, pystac-client, Planetary Computer, browsing items & assets |
| 05 | `05_stac_xarray_satellite_data.ipynb` | From STAC to analysis-ready cubes | stackstac, loading Sentinel-2, cloud masking, NDVI time series |

### Appendices (optional)

| # | Notebook | Topic |
|---|---|---|
| A1 | `A1_cloud_data_formats.ipynb` | COG, rasterio, cloud-native formats |
| A2 | `A2_optional_dask_parallel.ipynb` | Dask for parallel / out-of-core processing |

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd eo_python_foundations
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Launch

**VS Code:** Open the folder and select the `.venv` kernel in a notebook.

**JupyterLab:**

```bash
jupyter lab
```

## Repository structure

```
eo_python_foundations/
├── README.md                ← You are here
├── GLOSSARY.md              ← Key EO & Python terms
├── requirements.txt         ← All Python dependencies
├── .gitignore
├── assets/img/              ← Images used by notebooks
├── data/                    ← Small bundled datasets
│   ├── eo_scene_catalog.csv
│   └── eo_ndvi_stack.npz
├── notebooks/               ← The course notebooks
│   ├── 01_python_basics.ipynb
│   ├── 02_data_analysis_basics.ipynb
│   ├── 03_xarray_eo.ipynb
│   ├── 04_stac_fundamentals.ipynb
│   ├── 05_stac_xarray_satellite_data.ipynb
│   ├── A1_cloud_data_formats.ipynb
│   └── A2_optional_dask_parallel.ipynb
├── outputs/                 ← Generated at runtime (git-ignored)
└── scripts/                 ← Build / maintenance scripts (internal)
```

## Teaching / learning design

- **Short explanations → runnable code → exercises** in every section.
- **✅ Try it** exercises with `<details>` solutions (click to reveal).
- **🧠 Checkpoint** quizzes to consolidate understanding.
- **⚠️ Common mistakes** called out inline.
- Notebooks are designed to be run **top-to-bottom** in order.

## License

© EORC — University of Würzburg. All rights reserved.
