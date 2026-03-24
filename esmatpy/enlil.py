import os
import requests
import tarfile
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://data.ngdc.noaa.gov/earth-science-services/models/space-weather/wsa-enlil"

def _has_cme_on_date(date: datetime) -> bool:
    """Check (listing only, no download) whether a CME run exists for the given date."""
    import re
    date_str = date.strftime("%Y%m%d")
    dir_url = f"{BASE_URL}/{date.strftime('%Y')}/{date.strftime('%m')}/"
    try:
        content = requests.get(dir_url, timeout=10).text
    except Exception:
        return False
    pattern = r'swpc_wsaenlil_(?!bkg)([a-zA-Z]+)_' + date_str + r'_\d{4}\.tar\.gz'
    return bool(re.search(pattern, content))


def fetch_enlil_data_for_date(date: datetime, default_run_time: str = "0000", cache_dir: str = "enlil_cache") -> list:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    date_str = date.strftime("%Y%m%d")
    year_str = date.strftime("%Y")
    month_str = date.strftime("%m")
    
    dir_url = f"{BASE_URL}/{year_str}/{month_str}/"
    try:
        content = requests.get(dir_url).text
    except Exception:
        content = ""
        
    import re
    pattern = r'swpc_wsaenlil_([a-zA-Z]+)_' + date_str + r'_(\d{4})\.tar\.gz'
    matches = re.findall(pattern, content)
    
    if not matches:
        return []

    cme_matches = [m for m in matches if m[0] != 'bkg']
    bkg_matches = [m for m in matches if m[0] == 'bkg']
    
    if cme_matches:
        cme_matches.sort(key=lambda x: x[1], reverse=True)
        best_mode, best_rt = cme_matches[0]
    elif bkg_matches:
        bkg_matches.sort(key=lambda x: x[1], reverse=True)
        best_mode, best_rt = bkg_matches[0]
    else:
        return []
        
    filename = f"swpc_wsaenlil_{best_mode}_{date_str}_{best_rt}.tar.gz"
    url = f"{dir_url}{filename}"
    tar_path = cache_path / filename
    extract_dir = cache_path / f"extracted_{date_str}_{best_rt}_{best_mode}"
    
    if not tar_path.exists() and not extract_dir.exists():
        print(f"Downloading {filename} (Mode: {best_mode.upper()})...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
    if tar_path.exists() and not extract_dir.exists():
        print(f"Extracting {filename}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            import sys
            with tarfile.open(tar_path, "r:gz") as tar:
                if sys.version_info.major == 3 and sys.version_info.minor >= 12:
                    tar.extractall(path=extract_dir, filter='data')
                else:
                    tar.extractall(path=extract_dir)
            tar_path.unlink()
        except OSError:
            pass
            
    if extract_dir.exists():
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
        
        jumped = False
        if nc_files:
            for f in nc_files:
                if str(f) not in [str(af) for af in all_nc_files]:
                    all_nc_files.append(f)
                    
            try:
                with xr.open_dataset(nc_files[0], engine='netcdf4', decode_timedelta=True) as ds:
                    ref_date_str = ds.attrs.get('REFDATE_CAL')
                    if ref_date_str:
                        ref_date = pd.to_datetime(ref_date_str)
                        max_time = None
                        
                        for t_var in ['time', 'Earth_TIME']:
                            if t_var in ds.variables:
                                max_val = ds[t_var].max().values
                                t = None
                                
                                try:
                                    if hasattr(max_val, 'dtype'):
                                        dtype_str = str(max_val.dtype)
                                    else:
                                        dtype_str = type(max_val).__name__
                                        
                                    if 'datetime64' in dtype_str:
                                        t = pd.to_datetime(max_val)
                                    elif 'timedelta' in dtype_str:
                                        t = ref_date + pd.to_timedelta(max_val)
                                except Exception:
                                    pass
                                    
                                if t is not None and (max_time is None or t > max_time):
                                    max_time = t
                                    
                        if max_time is not None:
                            max_date = max_time.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
                            if max_date > current_request_date:
                                # If we downloaded a BKG run, scan intermediate days for a newer CME run.
                                # A CME run published mid-window is more accurate and should replace the BKG file.
                                is_bkg = 'bkg' in str(nc_files[0]).lower()
                                if is_bkg:
                                    scan_date = current_request_date + timedelta(days=1)
                                    while scan_date <= min(max_date, end):
                                        # Check listing only first — no download needed to confirm CME existence
                                        if _has_cme_on_date(scan_date):
                                            cme_files = fetch_enlil_data_for_date(scan_date, run_time, cache_dir)
                                            if cme_files:
                                                print(f"Newer CME run found on {scan_date.strftime('%Y-%m-%d')}. Replacing BKG run.")
                                                for old in nc_files:
                                                    if str(old) in [str(af) for af in all_nc_files]:
                                                        all_nc_files.remove(old)
                                                for f in cme_files:
                                                    if str(f) not in [str(af) for af in all_nc_files]:
                                                        all_nc_files.append(f)
                                                nc_files = cme_files
                                                break
                                        scan_date += timedelta(days=1)

                                current_request_date = max_date
                                jumped = True
            except Exception:
                pass
                    
        if not jumped:
            current_request_date += timedelta(days=1)
            
    return all_nc_files

# Default Earth-point variables (1D time series, very small)
EARTH_VARS = [
    'Earth_TIME', 'Earth_Density', 'Earth_Temperature',
    'Earth_V1', 'Earth_V2', 'Earth_V3',
    'Earth_B1', 'Earth_B2', 'Earth_B3',
    'Earth_DP_CME', 'Earth_BP_POLARITY',
    'Earth_X1', 'Earth_X2', 'Earth_X3',
]

def load_enlil_dataset(nc_files: list):
    if not nc_files:
        return None
    
    datasets = []
    for f in sorted(nc_files):
        try:
            ds = xr.open_dataset(f, engine='netcdf4', decode_timedelta=True)
            ref_date_str = ds.attrs.get('REFDATE_CAL')
            if ref_date_str:
                ref_date = pd.to_datetime(ref_date_str)
                for t_var in ['time', 'Earth_TIME']:
                    if t_var in ds.variables:
                        ds.coords[t_var] = ref_date + ds[t_var]
                
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
        merged_datasets = []
        for i in range(len(datasets)):
            ds = datasets[i]
            if i < len(datasets) - 1:
                next_ds = datasets[i+1]
                limit_time = next_ds.indexes['time'][0] if 'time' in next_ds.indexes else None
                limit_earth = next_ds.indexes['Earth_TIME'][0] if 'Earth_TIME' in next_ds.indexes else None
                
                if limit_time is not None and 'time' in ds.indexes:
                    ds = ds.isel(time=(ds.indexes['time'] < limit_time))
                if limit_earth is not None and 'Earth_TIME' in ds.indexes:
                    ds = ds.isel(Earth_TIME=(ds.indexes['Earth_TIME'] < limit_earth))
                    
            merged_datasets.append(ds)
            
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds_combined = xr.combine_by_coords(merged_datasets, combine_attrs='override', join='outer', data_vars='all', compat='override')
        return ds_combined
    except Exception as e:
        print(f"combine_by_coords failed: {e}. Returning unmerged dataset list.")
        return datasets

def create_cropped_enlil_dataset(start_date: str, end_date: str, output_path: str,
                                  run_time: str = "0000", cache_dir: str = "enlil_cache",
                                  vars_to_keep: list = None):
    """
    Downloads ENLIL data for the given date range, crops to the window, and saves
    a single NetCDF file.

    Dimension names are preserved as-is from the source files:
      - 3D time  → dim 't',       variable 'time'  (timedelta64 relative to REFDATE_CAL)
      - Earth ts → dim 'earth_t', variable 'Earth_TIME' (timedelta64 relative to REFDATE_CAL)

    Visualise with:
        ds = xr.open_dataset(output_path)
        ref = pd.to_datetime(ds.attrs['REFDATE_CAL'])
        t_abs = ref + pd.to_timedelta(ds['time'].values)   # absolute datetimes
        raw = ds['dd13_3d'].isel(t=time_idx).values
    """
    if vars_to_keep is None:
        vars_to_keep = EARTH_VARS

    nc_files = get_enlil_data(start_date, end_date, run_time, cache_dir)
    if not nc_files:
        print("No files found for the given dates.")
        return None

    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    slices_t  = []   # pieces along 't'       (3-D time)
    slices_et = []   # pieces along 'earth_t'  (Earth point time)
    global_attrs = {}
    seen_t_max  = None
    seen_et_max = None

    for f in sorted(nc_files):
        try:
            with xr.open_dataset(f, engine='netcdf4', decode_timedelta=True) as ds_raw:
                ref_date_str = ds_raw.attrs.get('REFDATE_CAL')
                if not ref_date_str:
                    continue
                ref_date = pd.Timestamp(ref_date_str)
                if not global_attrs:
                    global_attrs = dict(ds_raw.attrs)

                # --- Select requested variables ---
                available = [v for v in vars_to_keep if v in ds_raw.variables]
                if not available:
                    continue

                # Always load time variables for cropping (original timedelta form)
                time_extra = [v for v in ['time', 'Earth_TIME']
                              if v in ds_raw.variables and v not in available]

                # Auto-include 1-D spatial coordinate arrays (x_coord, z_coord, …)
                selected_dims: set = set()
                for v in available:
                    selected_dims.update(ds_raw[v].dims)
                coord_extra = [
                    v for v in ds_raw.variables
                    if v not in available and v not in time_extra
                    and v not in ('time', 'Earth_TIME')
                    and len(ds_raw[v].dims) == 1
                    and ds_raw[v].dims[0] in selected_dims
                ]

                ds = ds_raw[available + time_extra + coord_extra].load()

                # --- Crop 3-D time (dim 't') ---
                if 'time' in ds.variables:
                    t_dim  = ds['time'].dims[0]          # usually 't'
                    t_abs  = ref_date + pd.to_timedelta(ds['time'].values)
                    mask_t = (t_abs >= start_dt) & (t_abs <= end_dt)
                    if seen_t_max is not None:
                        mask_t &= (t_abs > seen_t_max)
                    if mask_t.any():
                        s_t = ds.isel({t_dim: mask_t})
                        seen_t_max = t_abs[mask_t][-1]
                        slices_t.append(s_t)

                # --- Crop Earth_TIME (dim 'earth_t') ---
                if 'Earth_TIME' in ds.variables:
                    et_dim  = ds['Earth_TIME'].dims[0]   # usually 'earth_t'
                    et_abs  = ref_date + pd.to_timedelta(ds['Earth_TIME'].values)
                    mask_et = (et_abs >= start_dt) & (et_abs <= end_dt)
                    if seen_et_max is not None:
                        mask_et &= (et_abs > seen_et_max)
                    if mask_et.any():
                        s_et = ds.isel({et_dim: mask_et})
                        seen_et_max = et_abs[mask_et][-1]
                        slices_et.append(s_et)

        except Exception as e:
            print(f"Warning: Could not process {f}: {e}")

    if not slices_t and not slices_et:
        print("No data remained after cropping.")
        return None

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        parts = []
        if slices_t:
            t_dim_name  = slices_t[0]['time'].dims[0]        # 't'
            et_dim_name = (slices_et[0]['Earth_TIME'].dims[0]
                           if slices_et else 'earth_t')       # 'earth_t'
            # Drop earth_t-based vars from 3D slices before concat along 't'
            def _keep_t(s):
                drop = [v for v in s.data_vars if et_dim_name in s[v].dims]
                return s.drop_vars(drop, errors='ignore')
            ds_t = xr.concat([_keep_t(s) for s in slices_t],
                             dim=t_dim_name, combine_attrs='override')
            parts.append(ds_t)
        if slices_et:
            et_dim_name = slices_et[0]['Earth_TIME'].dims[0]  # 'earth_t'
            t_dim_name2 = (slices_t[0]['time'].dims[0]
                           if slices_t else 't')               # 't'
            # Drop t-based vars from Earth slices before concat along 'earth_t'
            def _keep_et(s):
                drop = [v for v in s.data_vars if t_dim_name2 in s[v].dims]
                return s.drop_vars(drop, errors='ignore')
            ds_et = xr.concat([_keep_et(s) for s in slices_et],
                              dim=et_dim_name, combine_attrs='override')
            parts.append(ds_et)

        if len(parts) == 2:
            ds_final = xr.merge(parts, combine_attrs='override')
        else:
            ds_final = parts[0]

    ds_final.attrs.update(global_attrs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds_final.to_netcdf(output_path)
    ds_final.close()

    # Clean up extracted tar directories
    import shutil
    cleaned: set = set()
    for nc_file in nc_files:
        try:
            extract_dir = Path(nc_file).parent
            if extract_dir not in cleaned and extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
                cleaned.add(extract_dir)
        except Exception:
            pass

    print(f"Saved cropped dataset to {output_path}")
    return output_path

