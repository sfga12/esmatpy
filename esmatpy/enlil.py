import os
import requests
import tarfile
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://data.ngdc.noaa.gov/earth-science-services/models/space-weather/wsa-enlil"

def fetch_enlil_data_for_date(date: datetime, run_time: str = "0000", cache_dir: str = "enlil_cache"):
    year_str = date.strftime("%Y")
    month_str = date.strftime("%m")
    date_str = date.strftime("%Y%m%d")
    
    filename = f"swpc_wsaenlil_bkg_{date_str}_{run_time}.tar.gz"
    url = f"{BASE_URL}/{year_str}/{month_str}/{filename}"
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    tar_path = cache_path / filename
    extract_dir = cache_path / f"extracted_{date_str}_{run_time}"
    
    if not tar_path.exists():
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            return None
            
    if not extract_dir.exists() or not list(extract_dir.rglob("*.nc")):
        print(f"Extracting {filename}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
        except Exception:
            return None
            
    if tar_path.exists() and list(extract_dir.rglob("*.nc")):
        try:
            tar_path.unlink()
        except OSError:
            pass
            
    return list(extract_dir.rglob("*.nc"))

def get_enlil_data(start_date: str, end_date: str, run_time: str = "0000", cache_dir: str = "enlil_cache"):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current_request_date = start
    all_nc_files = []
    
    while current_request_date <= end:
        nc_files = fetch_enlil_data_for_date(current_request_date, run_time, cache_dir)
        
        if not nc_files:
            current_request_date += timedelta(days=1)
            continue
            
        for f in nc_files:
            if f not in all_nc_files:
                all_nc_files.append(f)
        
        try:
            ds = xr.open_mfdataset(nc_files, engine='netcdf4')
            
            max_ns = None
            if 'time' in ds.coords or 'time' in ds.data_vars:
                max_ns = pd.Series(ds.time.values).max()
            elif 'Earth_TIME' in ds.coords or 'Earth_TIME' in ds.data_vars:
                max_ns = pd.Series(ds.Earth_TIME.values).max()
                
            if max_ns is not None:
                ref_date_str = ds.attrs.get('REFDATE_CAL', current_request_date.strftime('%Y-%m-%dT00:00:00'))
                ref_date = pd.to_datetime(ref_date_str)
                
                max_time_dt = ref_date + max_ns
                max_date = datetime(max_time_dt.year, max_time_dt.month, max_time_dt.day)
                
                if max_date >= end:
                    ds.close()
                    break
                else:
                    if max_date > current_request_date:
                        current_request_date = max_date
                    else:
                        current_request_date += timedelta(days=1)
            else:
                current_request_date += timedelta(days=1)
                
            ds.close()
            
        except Exception:
            current_request_date += timedelta(days=1)
            
    return all_nc_files

def load_enlil_dataset(nc_files: list):
    if not nc_files:
        return None
    try:
        ds = xr.open_mfdataset(nc_files, engine='netcdf4')
        return ds
    except Exception:
        return [xr.open_dataset(f, engine='netcdf4') for f in nc_files]
