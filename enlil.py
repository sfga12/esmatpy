import os
import requests
import tarfile
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://data.ngdc.noaa.gov/earth-science-services/models/space-weather/wsa-enlil"

def fetch_enlil_data_for_date(date: datetime, run_time: str = "0000", cache_dir: str = "enlil_cache"):
    """
    Downloads WSA-Enlil solar wind data for a specific model run date.
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
            
    return list(extract_dir.rglob("*.nc"))

def get_enlil_data(start_date: str, end_date: str, run_time: str = "0000", cache_dir: str = "enlil_cache"):
    """
    Akıllı İndirme: Belirtilen başlangıç tarihinden dosyayı indirir. .nc dosyasını okuyup,
    içerisindeki verinin hangi tarihe kadar uzandığına (max time) bakar. Eğer kullanıcının 
    belirttiği Bitiş Tarihine (end_date) henüz ulaşılamamışsa, var olan dosyanın bittiği 
    tarihten itibaren yeni bir dosyayı indirmeye başlar. Böylece 10 aylık veriyi de hiç 
    boş yere üst üste bindirmeden en az indirmeyle çeker.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current_request_date = start
    all_nc_files = []
    
    print(f"Fetching data from {start_date} to {end_date}...")
    
    while current_request_date <= end:
        print(f"\nChecking Model Run for Date: {current_request_date.strftime('%Y-%m-%d')}")
        nc_files = fetch_enlil_data_for_date(current_request_date, run_time, cache_dir)
        
        if not nc_files:
            print(f"No run found for {current_request_date.strftime('%Y-%m-%d')}. Advancing by 1 day.")
            current_request_date += timedelta(days=1)
            continue
            
        # Listeye ekle (eğer daha önceden eklendiyse set mantığıyla engelleriz gerçi ama list yeterli)
        for f in nc_files:
            if f not in all_nc_files:
                all_nc_files.append(f)
        
        # Dosyanın içindeki en ileri tarihi (max_time) bulalım
        try:
            # decode_times=False koymadık çünkü zamanı okuyacağız.
            ds = xr.open_mfdataset(nc_files, engine='netcdf4')
            
            # Zaman aralığı koordinatını bulalım ('time' veya uydular için)
            if 'time' in ds.coords or 'time' in ds.data_vars:
                max_time_val = ds.time.max().values
            elif 'Earth_TIME' in ds.coords or 'Earth_TIME' in ds.data_vars:
                max_time_val = ds.Earth_TIME.max().values
            else:
                max_time_val = None
                
            if max_time_val is not None:
                max_time_dt = pd.to_datetime(str(max_time_val)).to_pydatetime()
                # Saati sıfırlayıp sadece güne odaklanalım
                max_date = datetime(max_time_dt.year, max_time_dt.month, max_time_dt.day)
                
                print(f"-> This file covers up to: {max_date.strftime('%Y-%m-%d')}")
                
                if max_date >= end:
                    print("-> Required date range has been successfully covered!")
                    ds.close()
                    break
                else:
                    # Ufak bir çakışma (overlap) olmaması için direkt kaldığı günden sonrasını çekebiliriz
                    # Ama Enlil run'ları her gün tam aynı saatte çıktığı için en güvenlisi max_date'den devam etmektir.
                    if max_date > current_request_date:
                        current_request_date = max_date
                    else:
                        current_request_date += timedelta(days=1)
            else:
                current_request_date += timedelta(days=1)
                
            ds.close()
            
        except Exception as e:
            print(f"Could not automatically read time coverage: {e}")
            current_request_date += timedelta(days=1)
            
    return all_nc_files

def load_enlil_dataset(nc_files: list):
    if not nc_files:
        return None
    try:
        ds = xr.open_mfdataset(nc_files, engine='netcdf4')
        return ds
    except Exception as e:
        print(f"Error loading datasets as multi-file dataset: {e}")
        print("Falling back to reading files individually...")
        return [xr.open_dataset(f, engine='netcdf4') for f in nc_files]
