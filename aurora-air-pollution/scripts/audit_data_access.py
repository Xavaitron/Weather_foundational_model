"""Audit every planned date against public ADS metadata; no API key needed."""
import argparse
from collections import Counter
from datetime import date, timedelta, datetime, timezone
import json
from pathlib import Path
from airadv.catalogue import fetch_catalogue, unavailable_fields
from download import requests_for_day


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="validation/server/data_access.json")
    a = p.parse_args()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    catalogues = {}
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "authenticated_retrieval_tested": False, "products": {}}
    for product, start, end in [("eac4", "2003-01-01", "2022-01-01"),
                                ("cams_analysis", "2017-10-01", "2022-12-01"),
                                ("cams_forecast_t0", "2017-10-01", "2022-12-01")]:
        missing_days = Counter()
        examples = []
        day = date.fromisoformat(start)
        while day < date.fromisoformat(end):
            for dataset, group, request in requests_for_day(day.isoformat(), product):
                if dataset not in catalogues:
                    catalogues[dataset] = fetch_catalogue(dataset, out.parent / "catalogue")
                missing = unavailable_fields(request, group, catalogues[dataset]["constraints"])
                missing_days.update(missing)
                if missing and len(examples) < 3:
                    examples.append({"date": day.isoformat(), "group": group, "unavailable": missing})
            day += timedelta(days=1)
        result = {"start": start, "end_exclusive": end,
                  "catalogue_complete": not bool(missing_days),
                  "missing_days_by_field_and_hour": dict(missing_days), "first_examples": examples}
        report["products"][product] = result
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"product": product, **result}), flush=True)


if __name__ == "__main__":
    main()
