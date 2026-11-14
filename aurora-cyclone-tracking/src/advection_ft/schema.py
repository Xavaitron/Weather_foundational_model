"""Explicit variable interfaces. Source values must already use Aurora units."""

LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
CORE_SURF = ("2t", "10u", "10v", "msl")
CORE_ATMOS = ("z", "u", "v", "t", "q")
CORE_STATIC = ("lsm", "z", "slt")
POLL_SURF = CORE_SURF + ("pm1", "pm2p5", "pm10", "tcco", "tc_no", "tcno2", "gtco3", "tcso2")
POLL_ATMOS = CORE_ATMOS + ("co", "no", "no2", "go3", "so2")
POLL_STATIC = CORE_STATIC + ("static_ammonia", "static_ammonia_log", "static_co", "static_co_log",
                             "static_nox", "static_nox_log", "static_so2", "static_so2_log")
ALIASES = {
    "surf__2t": ["2t", "t2m", "2m_temperature"],
    "surf__10u": ["10u", "u10", "10m_u_component_of_wind"],
    "surf__10v": ["10v", "v10", "10m_v_component_of_wind"],
    "surf__msl": ["msl", "mean_sea_level_pressure"],
    "atmos__z": ["z", "geopotential"],
    "atmos__u": ["u", "u_component_of_wind"],
    "atmos__v": ["v", "v_component_of_wind"],
    "atmos__t": ["t", "temperature"],
    "atmos__q": ["q", "specific_humidity"],
    "static__z": ["surface_geopotential", "geopotential_at_surface", "z"],
    "static__lsm": ["lsm", "land_sea_mask"],
    "static__slt": ["slt", "soil_type"],
}


def interface(task):
    if task == "cyclone":
        return CORE_SURF, CORE_ATMOS, CORE_STATIC, 6
    if task == "air_pollution":
        return POLL_SURF, POLL_ATMOS, POLL_STATIC, 12
    raise ValueError(f"Unknown task: {task}")


def canonical_names(task):
    surf, atmos, static, _ = interface(task)
    return ([f"surf__{v}" for v in surf] + [f"atmos__{v}" for v in atmos]
            + [f"static__{v}" for v in static])
