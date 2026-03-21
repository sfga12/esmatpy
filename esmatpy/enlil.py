import os
import requests
import tarfile
import numpy as np
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
                
            for item in extract_dir.rglob("*"):
                if item.is_file() and item.suffix != '.nc':
                    try:
                        item.unlink()
                    except OSError:
                        pass
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
            ds = xr.open_mfdataset(nc_files, engine='netcdf4', decode_timedelta=True)
            
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
        ds = xr.open_mfdataset(nc_files, engine='netcdf4', decode_timedelta=True)
        return ds
    except Exception:
        return [xr.open_dataset(f, engine='netcdf4', decode_timedelta=True) for f in nc_files]

def create_cropped_enlil_dataset(start_date: str, end_date: str, output_path: str, run_time: str = "0000", cache_dir: str = "enlil_cache"):
    """
    Downloads data for the given date range, crops it strictly within start_date and end_date,
    and saves it to a single NetCDF (.nc) file.
    """
    nc_files = get_enlil_data(start_date, end_date, run_time, cache_dir)
    if not nc_files:
        print("No files found for the given dates.")
        return None
        
    ds = load_enlil_dataset(nc_files)
    if isinstance(ds, list):
        print("Could not merge datasets automatically. Cropping is not supported for lists.")
        return None

    start_dt = pd.to_datetime(start_date)
    # Include up to the exact end of the requested end_date
    end_dt = pd.to_datetime(end_date) + timedelta(days=1) - pd.Timedelta(seconds=1)
    
    time_var = None
    if 'time' in ds.coords or 'time' in ds.data_vars:
        time_var = 'time'
    elif 'Earth_TIME' in ds.coords or 'Earth_TIME' in ds.data_vars:
        time_var = 'Earth_TIME'
        
    if time_var is not None:
        ref_date_str = ds.attrs.get('REFDATE_CAL', start_dt.strftime('%Y-%m-%dT00:00:00'))
        ref_date = pd.to_datetime(ref_date_str)
        
        start_td = start_dt - ref_date
        end_td = end_dt - ref_date
        
        dim_name = time_var if time_var in ds.dims else ds[time_var].dims[0]
        
        # Convert pandas Timedelta to numpy timedelta64[ns] to avoid Dask Array comparison TypeError
        start_np = np.timedelta64(start_td.value, 'ns')
        end_np = np.timedelta64(end_td.value, 'ns')
        
        mask = ((ds[time_var] >= start_np) & (ds[time_var] <= end_np)).values
        ds_cropped = ds.isel({dim_name: mask})
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ds_cropped.to_netcdf(output_path)
        print(f"Data successfully cropped and saved to {output_path}")
        return output_path
    else:
        print("Could not determine time variable to crop data.")
        return None
