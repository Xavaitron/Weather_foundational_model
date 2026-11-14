"""IBTrACS position verification and storm-cluster bootstrap."""

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    delta = np.deg2rad(np.asarray(lon2)-np.asarray(lon1))
    a = np.sin((phi2-phi1)/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(delta/2)**2
    return 6371.0 * 2*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def wrap_delta(degrees):
    return (degrees + 180) % 360 - 180


def bootstrap_mean_by_storm(frame, repetitions=1000, seed=42, value="error_km"):
    """Stratify storm resampling by basin; retain every case for a drawn storm."""
    if frame.empty:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    strata = []
    for _, basin in frame.groupby("basin"):
        aggregate = basin.groupby("sid")[value].agg(["sum", "count"])
        strata.append(aggregate.to_numpy())
    estimates = []
    for _ in range(repetitions):
        numerator, denominator = 0., 0.
        for a in strata:
            sampled = a[rng.integers(0, len(a), len(a))]
            numerator += sampled[:, 0].sum()
            denominator += sampled[:, 1].sum()
        estimates.append(numerator / denominator)
    return tuple(float(v) for v in np.percentile(estimates, [2.5, 97.5]))


def summarise_tracks(frame, leads, repetitions=1000, seed=42):
    rows = []
    for basin in ["ALL"] + sorted(frame.basin.unique().tolist()):
        for lead in leads:
            group = frame[frame.lead_hours == lead]
            if basin != "ALL":
                group = group[group.basin == basin]
            if group.empty:
                continue
            low, high = bootstrap_mean_by_storm(group, repetitions, seed)
            rows.append({"basin": basin, "lead_hours": lead, "track_mae_km": float(group.error_km.mean()),
                         "ci95_low_km": low, "ci95_high_km": high,
                         "n_cases": len(group), "n_storms": int(group.sid.nunique()),
                         "tracker_failure_rate": float(group.tracker_failed.mean())})
    return pd.DataFrame(rows)


def paired_comparison(candidate, reference, leads, repetitions=1000, seed=42):
    keys = ["sid", "init_time", "valid_time", "lead_hours", "basin"]
    if candidate.duplicated(keys).any() or reference.duplicated(keys).any():
        raise ValueError("Duplicate forecast keys cannot be compared.")
    merged = candidate.merge(reference, on=keys, suffixes=("_model", "_reference"), how="outer", indicator=True)
    if not (merged._merge == "both").all():
        raise ValueError("Evaluation cases differ. Rerun on the same complete case list; do not silently drop failures.")
    rows, rng = [], np.random.default_rng(seed)
    for basin in ["ALL"] + sorted(merged.basin.unique().tolist()):
        for lead in leads:
            group = merged[merged.lead_hours == lead]
            if basin != "ALL":
                group = group[group.basin == basin]
            if group.empty:
                continue
            def relative(a, b):
                return 100*(a-b)/b if b > 0 else np.nan
            point = relative(group.error_km_model.mean(), group.error_km_reference.mean())
            strata = []
            for _, sub in group.groupby("basin"):
                stats = sub.groupby("sid").agg(a=("error_km_model", "sum"),
                    b=("error_km_reference", "sum"), n=("error_km_model", "count"))
                strata.append(stats.to_numpy())
            samples = []
            for _ in range(repetitions):
                a, b = 0., 0.
                for stats in strata:
                    draws = stats[rng.integers(0, len(stats), len(stats))]
                    a += draws[:, 0].sum()
                    b += draws[:, 1].sum()
                samples.append(relative(a, b))
            lo, hi = np.nanpercentile(samples, [2.5, 97.5])
            rows.append({"basin": basin, "lead_hours": lead, "relative_error_percent": point,
                         "ci95_low_percent": lo, "ci95_high_percent": hi,
                         "significant_improvement": bool(hi < 0), "n_cases": len(group)})
    return pd.DataFrame(rows)
