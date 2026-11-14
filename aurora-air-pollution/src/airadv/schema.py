"""Physical-unit data contract. No implicit mixing-ratio/concentration conversion."""
SURF_MET = ("2t", "10u", "10v", "msl")
SURF_POLL = ("pm1", "pm2p5", "pm10", "tcco", "tc_no", "tcno2", "gtco3", "tcso2")
ATMOS_MET = ("z", "u", "v", "t", "q")
ATMOS_POLL = ("co", "no", "no2", "go3", "so2")
SURF = SURF_MET + SURF_POLL
ATMOS = ATMOS_MET + ATMOS_POLL
STATIC = ("lsm", "z", "slt", "static_ammonia", "static_ammonia_log", "static_co", "static_co_log", "static_nox", "static_nox_log", "static_so2", "static_so2_log")
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
ALIASES = {"t2m": "2t", "u10": "10u", "v10": "10v", "o3": "go3", "tc_no2": "tcno2", "tc_so2": "tcso2", "tco3": "gtco3"}
UNITS = {"2t": "K", "10u": "m s-1", "10v": "m s-1", "msl": "Pa", "t": "K", "u": "m s-1", "v": "m s-1", "q": "kg kg-1", "z": "m2 s-2"}
UNITS.update({k: "kg m-3" if k.startswith("pm") else "kg m-2" for k in SURF_POLL})
UNITS.update({k: "kg kg-1" for k in ATMOS_POLL})
ADS_NAMES = {
    "2t": "2m_temperature", "10u": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind", "msl": "mean_sea_level_pressure",
    "pm1": "particulate_matter_1um", "pm2p5": "particulate_matter_2.5um", "pm10": "particulate_matter_10um",
    "tcco": "total_column_carbon_monoxide", "tc_no": "total_column_nitrogen_monoxide", "tcno2": "total_column_nitrogen_dioxide", "gtco3": "total_column_ozone", "tcso2": "total_column_sulphur_dioxide",
    "z": "geopotential", "u": "u_component_of_wind", "v": "v_component_of_wind", "t": "temperature", "q": "specific_humidity",
    "co": "carbon_monoxide", "no": "nitrogen_monoxide", "no2": "nitrogen_dioxide", "go3": "ozone", "so2": "sulphur_dioxide",
}

def channel_names(levels=LEVELS, pollutants_only=False):
    surf = SURF_POLL if pollutants_only else SURF
    atmos = ATMOS_POLL if pollutants_only else ATMOS
    return list(surf) + [f"{v}@{p}" for v in atmos for p in levels]

