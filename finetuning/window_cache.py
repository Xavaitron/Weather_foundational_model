"""Bounded, process-safe cache shared by paired training runs on one host."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import xarray as xr


def _evict(folder, max_bytes):
    files = sorted(folder.glob('*.nc'), key=lambda p:p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files if p.exists())
    for path in files:
        if total <= max_bytes:
            break
        with path.with_suffix('.lock').open('a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                continue  # Another process is reading/writing this window.
            if path.exists():
                size=path.stat().st_size
                path.unlink()
                total-=size


def cached_window(folder, identity, producer, max_bytes=32*1024**3):
    """Serialize each window's download and atomically publish its NetCDF file.

    Identity must include source, split/date, levels, rollout and grid. No model
    variant is included: both variants must receive the same weather window.
    """
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    if max_bytes <= 0:
        raise ValueError('Cache limit must be positive')
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path=folder/(key+'.nc')
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            with xr.open_dataset(path) as saved:
                if saved.attrs.get('window_cache_key') != key:
                    raise ValueError('Window cache identity mismatch')
                data=saved.load()
            os.utime(path,None)
            return data
        data=producer()
        data.attrs['window_cache_key']=key
        # Bound persistent storage; oversized windows can still train uncached.
        if data.nbytes > max_bytes:
            return data
        with (folder/'eviction.lock').open('a') as eviction:
            fcntl.flock(eviction,fcntl.LOCK_EX)
            _evict(folder,max(0,max_bytes-data.nbytes))
        if shutil.disk_usage(folder).free < data.nbytes+2*1024**3:
            raise OSError('Insufficient free disk space for a safe window-cache write')
        temp=path.with_suffix(f'.{os.getpid()}.tmp')
        try:
            encoding={n:{'zlib':True,'complevel':1} for n in data.data_vars}
            data.to_netcdf(temp,engine='netcdf4',encoding=encoding)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    # Serialize eviction across processes, in addition to the per-window locks.
    with (folder/'eviction.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _evict(folder,max_bytes)
    return data
