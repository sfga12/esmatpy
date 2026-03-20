import os
import requests
import tarfile
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://data.ngdc.noaa.gov/earth-science-services/models/space-weather/wsa-enlil"

def fetch_enlil_data_for_date(date: datetime, run_time: str = "0000", cache_dir: str = "enlil_cache"):
    """
    Downloads WSA-Enlil solar wind data for a specific date and run time.
    """
    year_str = date.strftime("%Y")
    month_str = date.strftime("%m")
    date_str = date.strftime("%Y%m%d")
    
    filename = f"swpc_wsaenlil_bkg_{date_str}_{run_time}.tar.gz"
    url = f"{BASE_URL}/{year_str}/{month_str}/{filename}"
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    tar_path = cache_path / filename
    extract_dir = cache_path / f"extracted_{date_str}_{run_time}"
    
    # Download file if not cached
    if not tar_path.exists():
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        else:
            print(f"Failed to download {url} (Status {response.status_code})")
            return None
            
    # Extract file if not extracted
    if not extract_dir.exists() or not list(extract_dir.rglob("*.nc")):
        print(f"Extracting {filename}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
            print("Extraction complete.")
        except Exception as e:
            print(f"Failed to extract {filename}: {e}")
            return None
            
    # Find and return all .nc files
    nc_files = list(extract_dir.rglob("*.nc"))
    return nc_files

def get_enlil_data(start_date: str, end_date: str, run_time: str = "0000", cache_dir: str = "enlil_cache"):
    """
    Fetches ENLIL .nc data files for a given date range.
    Dates should be in 'YYYY-MM-DD' format.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current = start
    all_nc_files = []
    
    while current <= end:
        nc_files = fetch_enlil_data_for_date(current, run_time, cache_dir)
        if nc_files:
            all_nc_files.extend(nc_files)
            
        current += timedelta(days=1)
        
    return all_nc_files

def load_enlil_dataset(nc_files: list):
    """
    Attempts to load a list of .nc files using xarray.
    Returns the parsed xarray Dataset containing 3D solar wind variables.
    """
    if not nc_files:
        return None
        
    try:
        # Load multiple NetCDF files into an xarray sequence or single dataset
        # 'combine="nested"' is typical for combining sequential timesteps.
        ds = xr.open_mfdataset(nc_files, engine='netcdf4')
        return ds
    except Exception as e:
        print(f"Error loading datasets as multi-file dataset: {e}")
        # fallback to returning list of opened Datasets
        print("Falling back to reading files individually...")
        return [xr.open_dataset(f, engine='netcdf4') for f in nc_files]

