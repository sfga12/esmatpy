import os
import requests
import tarfile
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://data.ngdc.noaa.gov/earth-science-services/models/space-weather/wsa-enlil"

def fetch_available_runs(start_date: datetime, end_date: datetime) -> list:
    """Finds all available runs that could provide data for [start_date, end_date]."""
    # Runs cover [Date - 2 days, Date + 5 days].
    # To cover any day in [start, end], runs could be published between start - 5 days and end + 2 days.
    search_start = start_date - timedelta(days=5)
    search_end = end_date + timedelta(days=2)
    
    months_to_check = set()
    curr = search_start
    while curr <= search_end:
        months_to_check.add((curr.year, curr.month))
        curr += timedelta(days=1)
        
    runs = []
    import re
    pattern = r'swpc_wsaenlil_([a-zA-Z]+)_(\d{8})_(\d{4})\.tar\.gz'
    
    for year, month in sorted(months_to_check):
        url = f"{BASE_URL}/{year:04d}/{month:02d}/"
        try:
            content = requests.get(url, timeout=10).text
        except Exception:
            continue
            
        for match in re.finditer(pattern, content):
            mode = match.group(1).lower()
            date_str = match.group(2)
            time_str = match.group(3)
            
            run_dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
            run_start = run_dt - timedelta(days=2)
            run_end = run_dt + timedelta(days=5)
            
            if run_end >= start_date and run_start <= end_date:
                filename = match.group(0)
                runs.append({
                    "mode": mode,
                    "date": run_dt,
                    "time": time_str,
                    "filename": filename,
                    "url": f"{url}{filename}",
                    "valid_start": pd.Timestamp(run_start),
                    "valid_end": pd.Timestamp(run_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                })
    return runs

def get_authoritative_timeline(start_date: datetime, end_date: datetime, runs: list) -> list:
    """Build non-overlapping intervals prioritizing CME > BKG, then newest."""
    target_start = pd.Timestamp(start_date)
    target_end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    curr_day = target_start.normalize()
    end_day = target_end.normalize()
    
    daily_mapping = []
    while curr_day <= end_day:
        covering_runs = [r for r in runs if r["valid_start"] <= curr_day and r["valid_end"] >= curr_day]
        if not covering_runs:
            daily_mapping.append((curr_day, None))
        else:
            covering_runs.sort(key=lambda x: (
                1 if x["mode"] == "cme" else 0,
                x["date"],
                x["time"]
            ), reverse=True)
            daily_mapping.append((curr_day, covering_runs[0]))
        curr_day += pd.Timedelta(days=1)
        
    intervals = []
    if not daily_mapping: return intervals
        
    curr_run = daily_mapping[0][1]
    int_start = daily_mapping[0][0]
    
    for i in range(1, len(daily_mapping)):
        day, run = daily_mapping[i]
        if run != curr_run:
            if curr_run is not None:
                intervals.append({
                    "run": curr_run,
                    "interval_start": int_start,
                    "interval_end": day - pd.Timedelta(seconds=1)
                })
            curr_run = run
            int_start = day
            
    if curr_run is not None:
        last_day = daily_mapping[-1][0]
        intervals.append({
            "run": curr_run,
            "interval_start": int_start,
            "interval_end": last_day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        })
        
    for interval in intervals:
        interval["interval_start"] = max(interval["interval_start"], target_start)
        interval["interval_end"] = min(interval["interval_end"], target_end)
        
    return intervals

def __download_extract_run(run: dict, cache_path: Path) -> list:
    filename = run["filename"]
    url = run["url"]
    date_str = run["date"].strftime("%Y%m%d")
    extract_dir = cache_path / f"extracted_{date_str}_{run['time']}_{run['mode']}"
    tar_path = cache_path / filename
    
    if not tar_path.exists() and not extract_dir.exists():
        print(f"Downloading {filename} (Mode: {run['mode'].upper()})...")
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
        
        # Prioritize single summary files over hundreds of individual 3D timestep files
        # to prevent memory bloat and duplicate coordinate zigzags during concat
        summary_file = next((f for f in nc_files if "suball" in f.name.lower()), None)
        if not summary_file:
            summary_file = next((f for f in nc_files if "latest.nc" in f.name.lower()), None)
            
        if summary_file:
            return [summary_file]
            
        return nc_files
    return []

def get_enlil_data_intervals(start_date: str, end_date: str, cache_dir: str = "enlil_cache") -> list:
    """Returns a list of dicts: {'nc_files': [...], 'start': ..., 'end': ...}"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    runs = fetch_available_runs(start, end)
    intervals = get_authoritative_timeline(start, end, runs)
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    result = []
    for interval in intervals:
        nc_files = __download_extract_run(interval["run"], cache_path)
        if nc_files:
            result.append({
                "nc_files": nc_files,
                "interval_start": interval["interval_start"],
                "interval_end": interval["interval_end"]
            })
            
    return result

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

    intervals_info = get_enlil_data_intervals(start_date, end_date, cache_dir)
    if not intervals_info:
        print("No files found for the given dates.")
        return None

    # We do not use these globals; we rely strictly on authoritative intervals
    # but we still bound by the maximal request size.
    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    slices_t  = []   # pieces along 't'       (3-D time)
    slices_et = []   # pieces along 'earth_t'  (Earth point time)
    global_attrs = {}

    for interval in intervals_info:
        nc_files = interval["nc_files"]
        int_start = interval["interval_start"]
        int_end = interval["interval_end"]
        
        for f in sorted(nc_files):
            try:
                with xr.open_dataset(f, engine='netcdf4', decode_timedelta=True) as ds_raw:
                    ref_date_str = ds_raw.attrs.get('REFDATE_CAL')
                if not ref_date_str:
                    continue
                ref_date = pd.Timestamp(ref_date_str)
                if not global_attrs:
                    global_attrs = dict(ds_raw.attrs)
                    global_ref_date = ref_date

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
                
                # Make time and Earth_TIME proper indexed dimensions
                dims_to_swap = {}
                if 'time' in ds.variables and ds.variables['time'].dims == ('t',):
                    dims_to_swap['t'] = 'time'
                if 'Earth_TIME' in ds.variables and ds.variables['Earth_TIME'].dims == ('earth_t',):
                    dims_to_swap['earth_t'] = 'Earth_TIME'
                if dims_to_swap:
                    ds = ds.swap_dims(dims_to_swap)

                # --- Crop 3-D time (dim 't') ---
                if 'time' in ds.variables:
                    t_dim  = ds['time'].dims[0]
                    t_abs  = ref_date + pd.to_timedelta(ds['time'].values)
                    mask_t = (t_abs >= int_start) & (t_abs <= int_end)
                    if mask_t.any():
                        s_t = ds.isel({t_dim: mask_t})
                        new_t = t_abs[mask_t] - global_ref_date
                        s_t = s_t.assign_coords({'time': new_t.values})
                        slices_t.append(s_t)

                # --- Crop Earth_TIME (dim 'earth_t') ---
                if 'Earth_TIME' in ds.variables:
                    et_dim  = ds['Earth_TIME'].dims[0]
                    et_abs  = ref_date + pd.to_timedelta(ds['Earth_TIME'].values)
                    mask_et = (et_abs >= int_start) & (et_abs <= int_end)
                    if mask_et.any():
                        s_et = ds.isel({et_dim: mask_et})
                        new_et = et_abs[mask_et] - global_ref_date
                        s_et = s_et.assign_coords({'Earth_TIME': new_et.values})
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
            t_dim_name  = slices_t[0]['time'].dims[0]        # 'time'
            et_dim_name = (slices_et[0]['Earth_TIME'].dims[0]
                           if slices_et else 'Earth_TIME')       # 'Earth_TIME'
            def _prep_t(s):
                # Drop earth_t-based vars from 3D slices
                drop = [v for v in s.data_vars if et_dim_name in s[v].dims]
                if et_dim_name in s.coords:
                    drop.append(et_dim_name)
                s = s.drop_vars(drop, errors='ignore')
                # Promote spatial vars to coords so xr.concat doesn't stack them
                spatial = [v for v in s.data_vars
                           if len(s[v].dims) == 1
                           and s[v].dims[0] not in (t_dim_name, et_dim_name)]
                return s.set_coords(spatial)

            ds_t = xr.concat([_prep_t(s) for s in slices_t],
                             dim=t_dim_name, combine_attrs='override',
                             join='outer')
            parts.append(ds_t)

        if slices_et:
            et_dim_name = slices_et[0]['Earth_TIME'].dims[0]  # 'Earth_TIME'
            t_dim_name2 = (slices_t[0]['time'].dims[0]
                           if slices_t else 'time')               # 'time'
            def _prep_et(s):
                # Drop t-based vars from Earth slices
                drop = [v for v in s.data_vars if t_dim_name2 in s[v].dims]
                if t_dim_name2 in s.coords:
                    drop.append(t_dim_name2)
                s = s.drop_vars(drop, errors='ignore')
                spatial = [v for v in s.data_vars
                           if len(s[v].dims) == 1
                           and s[v].dims[0] not in (t_dim_name2, et_dim_name)]
                return s.set_coords(spatial)

            ds_et = xr.concat([_prep_et(s) for s in slices_et],
                              dim=et_dim_name, combine_attrs='override',
                              join='outer')
            parts.append(ds_et)

        if len(parts) == 2:
            ds_final = xr.merge(parts, combine_attrs='override', join='outer')
        else:
            ds_final = parts[0]

        # Failsafe sort to prevent plotting zigzags
        if 'time' in ds_final.coords:
            ds_final = ds_final.sortby('time')
        if 'Earth_TIME' in ds_final.coords:
            ds_final = ds_final.sortby('Earth_TIME')

    ds_final.attrs.update(global_attrs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds_final.to_netcdf(output_path)
    ds_final.close()

    # Clean up extracted tar directories
    import shutil
    cleaned = set()
    for interval in intervals_info:
        for nc_file in interval["nc_files"]:
            try:
                extract_dir = Path(nc_file).parent
                if extract_dir not in cleaned and extract_dir.exists():
                    shutil.rmtree(extract_dir, ignore_errors=True)
                    cleaned.add(extract_dir)
            except Exception:
                pass

    print(f"Saved cropped dataset to {output_path}")
    return output_path

