"""CPU-only preparation of the already-frozen evaluation dates into shared caches."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from finetuning.data import open_source, read_window, model_batch
from finetuning.paired_evaluate import load_model, load_climatology
from finetuning.window_cache import cached_window

_STATE = None


def initialize(protocol_path, weather_cache, climatology_cache):
    global _STATE
    protocol = json.loads(Path(protocol_path).read_text())
    task = load_model(protocol['checkpoints']['baseline']['path']).task_config
    _STATE = (protocol, task, open_source(protocol['source']), open_source(protocol['climatology']),
              weather_cache, climatology_cache)


def prepare(date):
    protocol, task, source, climate, weather_cache, climate_cache = _STATE
    before = time.monotonic()
    init = np.datetime64(date, 'h')
    split = 'val' if date.startswith('2020') else 'test'
    identity = dict(format_version=1, source=protocol['source'], split=split,
                    initialization=str(init), steps=protocol['steps'],
                    levels=protocol['model_pressure_levels'], resolution=protocol['resolution'])
    data = cached_window(weather_cache, identity,
                         lambda: read_window(source, init, split, protocol['steps'],
                                             identity['levels'], identity['resolution']))
    _, targets, _ = model_batch(data, task, protocol['steps'])
    load_climatology(climate, init, targets, protocol, climate_cache)
    return date, time.monotonic() - before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--weather-cache', default='work/era5-1deg-window-cache')
    parser.add_argument('--climatology-cache', default='work/era5-1deg-climatology-cache')
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,
                             initargs=(args.protocol, args.weather_cache, args.climatology_cache)) as pool:
        # The evaluator is already preparing the first date itself.
        futures = {pool.submit(prepare, date): date for date in protocol['initializations'][1:]}
        for future in as_completed(futures):
            date, seconds = future.result()
            print(f'Cached {date} in {seconds:.1f}s', flush=True)


if __name__ == '__main__':
    main()
