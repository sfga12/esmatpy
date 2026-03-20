# ESMAT Python Data Library (esmatpy)

`esmatpy` is a specialized Python library designed as a component of the ESMAT project. It automates the retrieval, extraction, and processing of large spatial data environments, such as 3D solar wind variables derived from NOAA WSA-Enlil models.

## Installation

You can install this package via pip from GitHub:
```bash
pip install git+https://github.com/sfga12/esmatpy.git
```

Or for a local editable installation (for developers):
```bash
git clone https://github.com/sfga12/esmatpy.git
cd esmatpy
pip install -e .
```

## Usage Example

```python
from esmatpy import get_enlil_data, load_enlil_dataset

# Fetch netCDF files automatically for a date range
nc_files = get_enlil_data("2026-02-02", "2026-02-05", cache_dir="./enlil_cache")

# Load 3D solar wind variables as an xarray Dataset
dataset = load_enlil_dataset(nc_files)

print(dataset)
```
