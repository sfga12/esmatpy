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
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    run_times_to_try = [run_time]
    if run_time == "0000":
        run_times_to_try.extend(["1200", "0600", "1800"])
        
    for rt in run_times_to_try:
        filename = f"swpc_wsaenlil_bkg_{date_str}_{rt}.tar.gz"
        url = f"{BASE_URL}/{year_str}/{month_str}/{filename}"
        
        tar_path = cache_path / filename
        extract_dir = cache_path / f"extracted_{date_str}_{rt}"
        
        if not tar_path.exists():
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(tar_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                continue
                
        if not extract_dir.exists() or not list(extract_dir.rglob("*.nc")):
            print(f"Extracting {filename}...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    nc_members = [m for m in tar.getmembers() if m.name.endswith('.nc')]
                    if not nc_members:
                        continue
                    tar.extractall(path=extract_dir, members=nc_members)
            except Exception as e:
                print(f"Extraction failed: {e}")
                continue
                
        if tar_path.exists() and list(extract_dir.rglob("*.nc")):
            try:
                tar_path.unlink()
            except OSError:
                pass
                
        nc_files = list(extract_dir.rglob("*.nc"))
        if nc_files:
            return nc_files
            
    return []

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
            if str(f) not in [str(af) for af in all_nc_files]:
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
    
    datasets = []
    for f in sorted(nc_files):  # Sort to enforce chronological ordering of runs
        try:
            ds = xr.open_dataset(f, engine='netcdf4', decode_timedelta=True)
            ref_date_str = ds.attrs.get('REFDATE_CAL')
            if ref_date_str:
                ref_date = pd.to_datetime(ref_date_str)
                # Convert timedeltas to absolute datetimes
                for t_var in ['time', 'Earth_TIME']:
                    if t_var in ds.variables:
                        ds.coords[t_var] = ref_date + ds[t_var]
                
                # Swap index dimensions so xarray can logically merge disparate files
                dims_to_swap = {}
                if 'time' in ds.coords and ds.coords['time'].dims == ('t',):
                    dims_to_swap['t'] = 'time'
                if 'Earth_TIME' in ds.coords and ds.coords['Earth_TIME'].dims == ('earth_t',):
                    dims_to_swap['earth_t'] = 'Earth_TIME'
                    
                if dims_to_swap:
                    ds = ds.swap_dims(dims_to_swap)
                
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: Could not optimally load {f}: {e}")
            
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
        
    try:
        # ds.combine_first() iteratively overwrites overlapping coordinates
        # Since datasets chronologically step forward, iterating in reverse lets
        # the newer forecasts supersede older runs over shared forecast dates.
        ds_combined = datasets[-1]
        for ds in reversed(datasets[:-1]):
            ds_combined = ds_combined.combine_first(ds)
                
        return ds_combined
    except Exception as e:
        print(f"combine_first failed: {e}. Returning unmerged dataset list.")
        return datasets

def create_cropped_enlil_dataset(start_date: str, end_date: str, output_path: str, run_time: str = "0000", cache_dir: str = "enlil_cache", vars_to_keep: list = None):
    """
    Downloads data for the given date range, crops it strictly within start_date and end_date,
    and saves it to a single NetCDF (.nc) file.
    """
    if vars_to_keep is None:
        vars_to_keep = ['dd13_3d', 'vv13_3d', 'x_coord', 'y_coord', 'z_coord', 'time']

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
    
    start_dt_np = np.datetime64(start_dt)
    end_dt_np = np.datetime64(end_dt)
    
    isel_args = {}
    
    for t_var in ['time', 'Earth_TIME']:
        if t_var in ds.coords or t_var in ds.data_vars:
            # Check if variable is absolute datetime
            if np.issubdtype(ds[t_var].dtype, np.datetime64):
                mask = ((ds[t_var] >= start_dt_np) & (ds[t_var] <= end_dt_np)).values
                dim_name = t_var if t_var in ds.dims else ds[t_var].dims[0]
                isel_args[dim_name] = mask
            else:
                # Legacy fallback to relative logic
                ref_date_str = ds.attrs.get('REFDATE_CAL', start_dt.strftime('%Y-%m-%dT00:00:00'))
                ref_date = pd.to_datetime(ref_date_str)
                
                start_td = start_dt - ref_date
                end_td = end_dt - ref_date
                
                # Convert pandas Timedelta to numpy timedelta64[ns]
                start_np = np.timedelta64(start_td.value, 'ns')
                end_np = np.timedelta64(end_td.value, 'ns')
                
                mask = ((ds[t_var] >= start_np) & (ds[t_var] <= end_np)).values
                dim_name = t_var if t_var in ds.dims else ds[t_var].dims[0]
                
                isel_args[dim_name] = mask

    if not isel_args:
        print("Could not determine any time variable to crop data.")
        return None
        
    ds_cropped = ds.isel(isel_args)
    
    if vars_to_keep:
        vars_to_drop = [v for v in ds_cropped.variables if v not in vars_to_keep]
        ds_cropped = ds_cropped.drop_vars(vars_to_drop)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds_cropped.to_netcdf(output_path)
    
    # Close datasets to release file locks (crucial before deletion)
    ds_cropped.close()
    ds.close()
    
    # Clean up massive uncropped original files to save disk/cache space
    for nc_file in nc_files:
        try:
            nc_path = Path(nc_file)
            if nc_path.exists():
                nc_path.unlink()
            # Remove the extracted date folder if it became empty
            if nc_path.parent.exists() and nc_path.parent.is_dir() and not any(nc_path.parent.iterdir()):
                nc_path.parent.rmdir()
        except Exception as e:
            pass # Keep silent if deletion fails due to OS locks

    print(f"Data successfully cropped and saved to {output_path}")
    print(f"Removed {len(nc_files)} original uncropped cache file(s) to save space.")
    return output_path
