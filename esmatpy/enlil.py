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
                    "run_start": run_start,
                    "run_end": run_end,
                    "valid_start": pd.Timestamp(run_start).normalize(),
                    "valid_end": pd.Timestamp(run_end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                })
    return runs

def get_authoritative_timeline(start_date: datetime, end_date: datetime, runs: list, mode: str = "hybrid", minimize_jumps: bool = False) -> list:
    """Build non-overlapping intervals prioritizing CME > BKG, then newest."""
    target_start = pd.Timestamp(start_date)
    target_end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    curr_time = target_start.normalize()
    end_time = target_end
    
    hourly_mapping = []
    active_run = None
    
    while curr_time <= end_time:
        # exact chronological overlap
        covering_runs = [r for r in runs if pd.Timestamp(r["run_start"]) <= curr_time <= pd.Timestamp(r["run_end"])]
        
        if mode == "cme":
            covering_runs = [r for r in covering_runs if r["mode"] == "cme"]
        elif mode == "bkg":
            covering_runs = [r for r in covering_runs if r["mode"] == "bkg"]
            
        if not covering_runs:
            hourly_mapping.append((curr_time, None))
            active_run = None
        else:
            if minimize_jumps and active_run is not None:
                # If we're already on a run and minimize_jumps is True, 
                # we prefer the run that is closest in time to our current one
                # to minimize "initialization shock" from boundary changes.
                covering_runs.sort(key=lambda x: (
                    0 if x["mode"] == "cme" else 1, # CME first
                    abs((x["date"] - active_run["date"]).total_seconds())
                ))
            else:
                # Standard behavior: newest simulation is best.
                covering_runs.sort(key=lambda x: (
                    1 if x["mode"] == "cme" else 0,
                    x["date"],
                    x["time"]
                ), reverse=True)
            
            chosen_run = covering_runs[0]
            
            if minimize_jumps and active_run is not None:
                # Is the active run still a valid choice for this hour?
                if any(r["filename"] == active_run["filename"] for r in covering_runs):
                    # In hybrid mode, if active is BKG but a CME appears, switch to the CME!
                    if mode == "hybrid" and active_run["mode"] == "bkg" and chosen_run["mode"] == "cme":
                        pass # allow jump to superior data
                    else:
                        chosen_run = active_run # stick to active run
                        
            hourly_mapping.append((curr_time, chosen_run))
            active_run = chosen_run
            
        curr_time += pd.Timedelta(hours=1)
        
    intervals = []
    if not hourly_mapping: return intervals
        
    curr_run = hourly_mapping[0][1]
    int_start = hourly_mapping[0][0]
    
    for i in range(1, len(hourly_mapping)):
        t_time, run = hourly_mapping[i]
        if run != curr_run:
            if curr_run is not None:
                intervals.append({
                    "run": curr_run,
                    "interval_start": int_start,
                    "interval_end": t_time - pd.Timedelta(seconds=1)
                })
            curr_run = run
            int_start = t_time
            
    if curr_run is not None:
        last_time = hourly_mapping[-1][0]
        intervals.append({
            "run": curr_run,
            "interval_start": int_start,
            "interval_end": last_time + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
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

def get_enlil_data_intervals(start_date: str, end_date: str, cache_dir: str = "enlil_cache", mode: str = "hybrid", minimize_jumps: bool = False) -> list:
    """Returns a list of dicts: {'nc_files': [...], 'start': ..., 'end': ...}"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    runs = fetch_available_runs(start, end)
    intervals = get_authoritative_timeline(start, end, runs, mode, minimize_jumps)
    
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
                                  vars_to_keep: list = None, mode: str = "hybrid",
                                  minimize_jumps: bool = False, blend_hours: int = 12):
    """
    Downloads ENLIL data for the given date range, crops to the window, and saves
    a single NetCDF file with linear blending between simulation transitions.

    Args:
        mode: 'hybrid' (CME prioritized, BKG fallback), 'cme' (strict CME), or 'bkg' (strict BKG)
        minimize_jumps: If True, sticks to the current simulation continuously for as long as possible.
        blend_hours: Number of hours for linear crossfade between simulation runs.

    Dimension names are preserved as-is from the source files:
      - 3D time  → dim 't',       variable 'time'  (timedelta64 relative to REFDATE_CAL)
      - Earth ts → dim 'earth_t', variable 'Earth_TIME' (timedelta64 relative to REFDATE_CAL)
    """
    if vars_to_keep is None:
        vars_to_keep = EARTH_VARS

    intervals_info = get_enlil_data_intervals(start_date, end_date, cache_dir, mode, minimize_jumps)
    if not intervals_info:
        print("No files found for the given dates.")
        return None

    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    slices_t  = []   # pieces along 't'       (3-D time)
    slices_et = []   # pieces along 'earth_t'  (Earth point time)
    global_attrs = {}

    for i, interval in enumerate(intervals_info):
        nc_files = interval["nc_files"]
        int_start = interval["interval_start"]
        int_end = interval["interval_end"]
        
        # Extend the interval end for blending purposes if there is a next one
        effective_end = int_end
        if blend_hours > 0 and i < len(intervals_info) - 1:
            effective_end = int_end + pd.Timedelta(hours=blend_hours)

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

                # --- Selection ---
                available = [v for v in vars_to_keep if v in ds_raw.variables]
                if not available: continue

                time_extra = [v for v in ['time', 'Earth_TIME'] if v in ds_raw.variables and v not in available]
                selected_dims = {d for v in available for d in ds_raw[v].dims}
                coord_extra = [v for v in ds_raw.variables if v not in (available + time_extra + ['time', 'Earth_TIME'])
                               and len(ds_raw[v].dims) == 1 and ds_raw[v].dims[0] in selected_dims]

                ds_crop = ds_raw[available + time_extra + coord_extra].load()
                
                # Swap dims
                dims_to_swap = {}
                if 'time' in ds_crop.variables and ds_crop.variables['time'].dims == ('t',):
                    dims_to_swap['t'] = 'time'
                if 'Earth_TIME' in ds_crop.variables and ds_crop.variables['Earth_TIME'].dims == ('earth_t',):
                    dims_to_swap['earth_t'] = 'Earth_TIME'
                if dims_to_swap:
                    ds_crop = ds_crop.swap_dims(dims_to_swap)

                # --- Crop 3-D time (with potential overlap extension) ---
                if 'time' in ds_crop.variables:
                    t_dim  = ds_crop['time'].dims[0]
                    t_abs  = ref_date + pd.to_timedelta(ds_crop['time'].values)
                    mask_t = (t_abs >= int_start) & (t_abs <= effective_end)
                    if mask_t.any():
                        s_t = ds_crop.isel({t_dim: mask_t})
                        new_t = t_abs[mask_t] - global_ref_date
                        s_t = s_t.assign_coords({'time': new_t.values})
                        slices_t.append(s_t)

                # --- Crop Earth_TIME (with potential overlap extension) ---
                if 'Earth_TIME' in ds_crop.variables:
                    et_dim  = ds_crop['Earth_TIME'].dims[0]
                    et_abs  = ref_date + pd.to_timedelta(ds_crop['Earth_TIME'].values)
                    mask_et = (et_abs >= int_start) & (et_abs <= effective_end)
                    if mask_et.any():
                        s_et = ds_crop.isel({et_dim: mask_et})
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

        # Function to merge or blend slices along a time dimension
        def _process_slices(slices, time_dim, other_time_dim, blend_h):
            if not slices: return None
            
            def _prep(s):
                drop = [v for v in s.data_vars if other_time_dim in s[v].dims]
                if other_time_dim in s.coords: drop.append(other_time_dim)
                s = s.drop_vars(drop, errors='ignore')
                spatial = [v for v in s.data_vars if len(s[v].dims) == 1 and s[v].dims[0] not in (time_dim, other_time_dim)]
                return s.set_coords(spatial)

            prepped = [_prep(s) for s in slices]
            
            # Identify overlaps and blend
            # Since slices are sorted by their start time in intervals_info, 
            # we can identify contiguous/overlapping regions.
            res = prepped[0]
            for i in range(1, len(prepped)):
                next_ds = prepped[i]
                
                # Common time range
                t_common = np.intersect1d(res[time_dim].values, next_ds[time_dim].values)
                
                if len(t_common) > 0 and blend_h > 0:
                    t0, t1 = t_common[0], t_common[-1]
                    total_span = t1 - t0
                    
                    if total_span > 0:
                        # Weights: res fades out 1->0, next_ds fades in 0->1
                        # convert to nanoseconds for calculation
                        weights = (next_ds.sel({time_dim: t_common})[time_dim].values.astype(float) - float(t0)) / float(total_span)
                        w2 = xr.DataArray(weights, coords={time_dim: t_common}, dims=[time_dim])
                        w1 = 1.0 - w2
                        
                        # Apply weights to all data variables in the overlap
                        overlap_vars = [v for v in res.data_vars if v in next_ds.data_vars and time_dim in res[v].dims]
                        for v in overlap_vars:
                            # We must handle dtypes (int16 vs float)
                            v1 = res[v].sel({time_dim: t_common})
                            v2 = next_ds[v].sel({time_dim: t_common})
                            blended = v1 * w1 + v2 * w2
                            # Preserve attrs (like dd13_max/min for int16 packed data)
                            blended.attrs.update(v1.attrs)
                            
                            # Replace in both datasets (so concat later picks up the blended values)
                            # Actually we only need it in one or we can manually concat
                            res[v].loc[{time_dim: t_common}] = blended
                    
                    # Concat remaining non-overlapping part of next_ds
                    t_new = next_ds[time_dim].values[next_ds[time_dim].values > t1]
                    if len(t_new) > 0:
                        res = xr.concat([res, next_ds.sel({time_dim: t_new})], dim=time_dim, combine_attrs='override')
                else:
                    # Generic concat
                    res = xr.concat([res, next_ds], dim=time_dim, combine_attrs='override', join='outer')
            
            return res.sortby(time_dim)

        ds_t = _process_slices(slices_t, 'time', 'Earth_TIME', blend_hours)
        ds_et = _process_slices(slices_et, 'Earth_TIME', 'time', blend_hours)

        if ds_t and ds_et:
            ds_final = xr.merge([ds_t, ds_et], combine_attrs='override', join='outer')
        else:
            ds_final = ds_t if ds_t else ds_et

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

