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
    Downloads ENLIL data for the given date range, processes each file individually
    (keeping only the requested variables and cropping to the date window), then
    concatenates and saves a single compact NetCDF file.

    This approach never loads full 3D grids into memory simultaneously, making it
    safe to use in memory-constrained environments like Google Colab.

    Parameters
    ----------
    start_date : str  e.g. '2026-01-31'
    end_date   : str  e.g. '2026-02-07'
    output_path: str  Path for the output .nc file, e.g. 'output/earth_data.nc'
    vars_to_keep: list  Variables to retain. Defaults to all Earth_* point variables.
    
    Returns
    -------
    str  Path to the saved .nc file, or None on failure.
    """
    if vars_to_keep is None:
        vars_to_keep = EARTH_VARS

    nc_files = get_enlil_data(start_date, end_date, run_time, cache_dir)
    if not nc_files:
        print("No files found for the given dates.")
        return None

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    slices = []
    # Find which time dimension names correspond to Earth_TIME and 3D time
    for f in sorted(nc_files):
        try:
            with xr.open_dataset(f, engine='netcdf4', decode_timedelta=True) as ds_raw:
                ref_date_str = ds_raw.attrs.get('REFDATE_CAL')
                if not ref_date_str:
                    continue
                ref_date = pd.to_datetime(ref_date_str)

                # Only load the requested variables (+ their dimension coords)
                available = [v for v in vars_to_keep if v in ds_raw.variables]
                if not available:
                    continue
                # Always include time variables so that axis conversion and
                # cropping work regardless of what vars_to_keep contains.
                time_vars = [v for v in ['time', 'Earth_TIME']
                             if v in ds_raw.variables and v not in available]
                selected_dims = set()
                for v in available:
                    selected_dims.update(ds_raw[v].dims)
                coord_vars = [
                    v for v in ds_raw.variables
                    if v not in available and v not in time_vars
                    and len(ds_raw[v].dims) == 1
                    and ds_raw[v].dims[0] in selected_dims
                    and v not in ('time', 'Earth_TIME')
                ]
                ds = ds_raw[available + time_vars + coord_vars].load()

                # Convert all time variables to absolute datetime
                for t_var in ['time', 'Earth_TIME']:
                    if t_var in ds.variables:
                        raw_vals = ds[t_var].values
                        dtype_str = str(raw_vals.dtype)
                        if 'timedelta' in dtype_str:
                            abs_times = (ref_date + pd.to_timedelta(raw_vals)).values.astype('datetime64[ns]')
                        else:
                            abs_times = pd.to_datetime(raw_vals).values.astype('datetime64[ns]')
                        ds[t_var] = xr.DataArray(abs_times, dims=ds[t_var].dims)

                        # Promote to index dimension.
                        # Case 1: variable is on a different dim (e.g. 'time' var on 't' dim) → swap
                        if ds[t_var].dims[0] in ds.dims and t_var != ds[t_var].dims[0]:
                            ds = ds.assign_coords({t_var: (ds[t_var].dims[0], abs_times)})
                            ds = ds.swap_dims({ds[t_var].dims[0]: t_var})
                        # Case 2: variable IS the dim (e.g. 'time' var on 'time' dim) → assign_coords
                        # Without this, ds.indexes won't contain 'time' and the crop is skipped.
                        elif ds[t_var].dims[0] == t_var and t_var in ds.dims:
                            ds = ds.assign_coords({t_var: abs_times})

                # Crop each time axis to [start_dt, end_dt].
                start_np = np.datetime64(start_dt, 'ns')
                end_np   = np.datetime64(end_dt,   'ns')

                all_empty = True
                for t_var in ['time', 'Earth_TIME']:
                    if t_var not in ds.indexes:
                        continue
                    idx = ds.indexes[t_var]
                    mask = (~np.isnat(idx)) & (idx >= start_np) & (idx <= end_np)
                    if mask.any():
                        ds = ds.isel({t_var: mask})
                        all_empty = False
                    else:
                        # Drop this dim entirely to avoid writing garbage values
                        ds = ds.isel({t_var: slice(0, 0)})

                if all_empty:
                    continue

                _min_valid = np.datetime64('2000-01-01', 'ns')
                has_plausible_time = False
                for t_var in ['time', 'Earth_TIME']:
                    if t_var in ds.indexes and ds.sizes.get(t_var, 0) > 0:
                        if ds.indexes[t_var].max() > _min_valid:
                            has_plausible_time = True
                            break
                if not has_plausible_time:
                    continue

                ds.attrs['REFDATE_CAL'] = str(ref_date)
                slices.append(ds)

        except Exception as e:
            print(f"Warning: Could not process {f}: {e}")

    if not slices:
        print("No data remained after cropping.")
        return None

    # --- Streaming write: append each slice directly to the output file.
    # This keeps peak RAM to ~1 file at a time rather than holding all
    # slices in memory simultaneously before the final concat+save.
    import netCDF4 as nc4
    import warnings

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Determine which unlimited dims we will use
    unlimited_dims = set()
    for s in slices:
        for dim in ['time', 'Earth_TIME']:
            if dim in s.indexes:
                unlimited_dims.add(dim)

    # Sort slices by their first time value on any time dimension
    def _first_time(s):
        for dim in ['time', 'Earth_TIME']:
            if dim in s.indexes:
                return s.indexes[dim][0]
        return None

    slices_sorted = sorted(slices, key=_first_time)

    # Track last written timestamp per dim to avoid overlaps
    seen_max = {dim: None for dim in unlimited_dims}

    written = False
    with nc4.Dataset(output_path, 'w') as out:
        for s in slices_sorted:
            # Deduplicate along each unlimited dim before writing
            for dim in list(unlimited_dims):
                if dim not in s.indexes:
                    continue
                if seen_max[dim] is not None:
                    s = s.isel({dim: s.indexes[dim] > seen_max[dim]})
                if s.sizes.get(dim, 0) == 0:
                    continue
                seen_max[dim] = s.indexes[dim][-1]

            if all(s.sizes.get(d, 0) == 0 for d in unlimited_dims if d in s.indexes):
                continue
            # Store datetime64 as int64 seconds since 1970-01-01.
            # Using i8 avoids netCDF4's automatic f8 _FillValue=9.97e36
            # which causes xarray to overflow during time decoding.
            # Read back with decode_times=False and convert manually.
            _EPOCH_S = np.datetime64('1970-01-01', 's')
            def _to_nc_values(arr):
                if np.issubdtype(arr.dtype, np.datetime64):
                    return (arr.astype('datetime64[s]') - _EPOCH_S).astype(np.int64)
                return arr

            def _nc_dtype(arr):
                if np.issubdtype(arr.dtype, np.datetime64):
                    return 'i8'
                return arr.dtype

            def _nc_attrs(var):
                attrs = {k: var.attrs[k] for k in var.attrs}
                if np.issubdtype(var.dtype, np.datetime64):
                    attrs['units']    = 'seconds since 1970-01-01'
                    attrs['calendar'] = 'proleptic_gregorian'
                return attrs

            # --- First slice: create dimensions and variables
            if not written:
                for dim, size in s.sizes.items():
                    if dim not in out.dimensions:
                        out.createDimension(dim, None if dim in unlimited_dims else size)

                for vname, var in s.variables.items():
                    if vname not in out.variables:
                        nc_dt = _nc_dtype(var.values)
                        is_dt = np.issubdtype(var.dtype, np.datetime64)
                        v = out.createVariable(vname, nc_dt, var.dims,
                                               zlib=True, complevel=4,
                                               fill_value=False if is_dt else None)
                        v.setncatts(_nc_attrs(var))

                # Global attrs
                out.setncatts({k: s.attrs[k] for k in s.attrs})
                written = True

            # --- Append data along unlimited dims
            for vname, var in s.variables.items():
                if vname not in out.variables:
                    continue
                out_var = out.variables[vname]
                data = _to_nc_values(var.values)

                # Find which dim is unlimited
                unlim_ax = None
                for ax, dim in enumerate(var.dims):
                    if dim in unlimited_dims:
                        unlim_ax = ax
                        unlim_dim = dim
                        break

                if unlim_ax is None:
                    # Non-time variable: write once
                    if out_var.size == 0:
                        out_var[:] = data
                else:
                    cur_len = out.dimensions[unlim_dim].size
                    new_len = cur_len + data.shape[unlim_ax]
                    slc = [slice(None)] * len(var.dims)
                    slc[unlim_ax] = slice(cur_len, new_len)
                    out_var[tuple(slc)] = data

        if not written:
            print("Could not write any data.")
            return None

    # --- Release all slice datasets immediately
    for s in slices:
        try:
            s.close()
        except Exception:
            pass
    del slices

    # Clean up raw downloaded files and their extracted folders
    import shutil
    cleaned = set()
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
