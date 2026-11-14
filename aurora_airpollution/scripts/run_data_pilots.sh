#!/usr/bin/env bash
# Immediate, bounded pilot work. Submit only one ADS request at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
py="${AIRADV_PYTHON:-python}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p validation/server
trap 'printf "FAILED at line %s\n" "$LINENO" > validation/server/pilot_pipeline.status' ERR
printf 'Waiting for preceding request\n' > validation/server/pilot_pipeline.status
if [[ -n "${1:-}" ]]; then
  while kill -0 "$1" 2>/dev/null; do sleep 10; done
fi
printf 'Retrieving EAC4 pilot\n' > validation/server/pilot_pipeline.status
"$py" -u scripts/download.py --product eac4 --start 2003-01-01 --end 2003-01-03 --execute
"$py" -u scripts/prepare.py --surface 'data/raw/eac4/2003-01/2003-01-0[12]-sfc.nc' --atmos 'data/raw/eac4/2003-01/2003-01-0[12]-plev.nc' --static data/assets/aurora-0.4-air-pollution-static.pickle --trust-official-static-pickle --product eac4 --output data/processed/pilot_eac4_1p5.zarr
"$py" -u scripts/stats.py --data data/processed/pilot_eac4_1p5.zarr --start 2003-01-01 --end 2003-01-03 --output data/processed/pilot_eac4_stats.json
"$py" - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('configs/eac4.json').read_text())
c['data'].update(path='data/processed/pilot_eac4_1p5.zarr',stats='data/processed/pilot_eac4_stats.json',train_end='2003-01-03')
c['output']='runs/PILOT_ONLY_not_a_scientific_experiment'
Path('validation/server/pilot_eac4_config.json').write_text(json.dumps(c,indent=2)+'\n')
PY
printf 'Retrieving CAMS lead-zero pilot\n' > validation/server/pilot_pipeline.status
"$py" -u scripts/download.py --product cams_forecast_t0 --start 2017-10-01 --end 2017-10-02 --execute
"$py" -u scripts/prepare.py --surface 'data/raw/cams_forecast_t0/2017-10/2017-10-01-sfc.nc' --atmos 'data/raw/cams_forecast_t0/2017-10/2017-10-01-plev.nc' --static data/assets/aurora-0.4-air-pollution-static.pickle --trust-official-static-pickle --product cams_forecast_t0 --output data/processed/pilot_cams_t0_1p5.zarr
"$py" - <<'PY'
import json
from pathlib import Path
import numpy as np
import xarray as xr
from airadv.data import validate_schema
records=[]
for product,path,frames in [('eac4','data/processed/pilot_eac4_1p5.zarr',4),('cams_forecast_t0','data/processed/pilot_cams_t0_1p5.zarr',2)]:
    with xr.open_zarr(path,consolidated=True) as ds:
        validate_schema(ds)
        if ds.sizes['time']!=frames: raise ValueError('Unexpected pilot timestamps')
        for v in ds.data_vars:
            if bool((~np.isfinite(ds[v])).any().compute()): raise ValueError(f'Nonfinite {product}/{v}')
        records.append(dict(product=product,path=path,frames=frames,shape=[120,240],variables=len(ds.data_vars),finite=True,schema_valid=True))
Path('validation/server/data_pilots.json').write_text(json.dumps(records,indent=2)+'\n')
PY
printf 'PASSED: bounded pilots only; full historical data not downloaded\n' > validation/server/pilot_pipeline.status
