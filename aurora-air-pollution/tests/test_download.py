import importlib.util
from pathlib import Path
import zipfile
import pytest
from datetime import date

spec=importlib.util.spec_from_file_location("download_script",Path(__file__).parents[1]/"scripts/download.py")
download=importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)

def test_archive_recovery_replaces_partial_and_records_digest(tmp_path):
    archive=tmp_path/"raw.download"; final=tmp_path/"raw.nc"
    with zipfile.ZipFile(archive,"w") as z: z.writestr("data_sfc.nc",b"validated-test-content")
    final.with_suffix(".nc.partial").write_bytes(b"interrupted-extraction")
    download.finalise_download(archive,final)
    assert final.read_bytes()==b"validated-test-content"
    assert final.with_suffix(".sha256").read_text().strip()==download.sha256(final)
    assert not archive.exists()

def test_ambiguous_archive_never_becomes_completed_file(tmp_path):
    archive=tmp_path/"raw.download"; final=tmp_path/"raw.nc"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("one.nc",b"one"); z.writestr("two.nc",b"two")
    with pytest.raises(ValueError,match="Expected one NetCDF"):
        download.finalise_download(archive,final)
    assert archive.exists() and not final.exists()

def test_grouped_requests_cover_dates_once_and_stop_at_month_boundary():
    periods=list(download.request_periods(date(2020,2,27),date(2020,4,2),31))
    assert periods==[(date(2020,2,27),date(2020,3,1)),(date(2020,3,1),date(2020,4,1)),(date(2020,4,1),date(2020,4,2))]
    assert sum((b-a).days for a,b in periods)==35
