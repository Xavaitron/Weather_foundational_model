"""Self-describing checkpoints; adapter checkpoints require this loader."""
import dataclasses
import json
import jax
from graphcast import autoregressive, casting, checkpoint, graphcast, normalization
from finetuning.advection import AdvectedGraphCast
from finetuning.geometry import CachedGraphCast

MARKER = '\nFINETUNING_CONFIG='


def save_model(path, original, config, params, variant, modes, adapter_resolution):
    metadata = dict(variant=variant, adapter_modes=modes, adapter_resolution=adapter_resolution,
                    format_version=1)
    description = original.description.split(MARKER)[0] + MARKER + json.dumps(metadata,sort_keys=True)
    model = dataclasses.replace(original, params=jax.device_get(params), model_config=config,
                                description=description)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        checkpoint.dump(f, model)
    temp.replace(path)


def configuration(model):
    has_adapter = any(k.startswith('advection_') for k in model.params)
    if MARKER not in model.description:
        if has_adapter:
            raise ValueError('Adapter weights are missing their architecture metadata')
        return dict(variant='baseline',adapter_modes=16,adapter_resolution=1.)
    config = json.loads(model.description.split(MARKER,1)[1])
    if config.get('format_version') != 1 or config.get('variant') not in ('baseline','advection'):
        raise ValueError('Unsupported checkpoint architecture metadata')
    if has_adapter != (config['variant'] == 'advection'):
        raise ValueError('Checkpoint variant disagrees with its parameter keys')
    return config


def predictor(model, stats):
    """Call inside hk.transform_with_state; apply with model.params and {} state.

    This returns physical-unit autoregressive predictions when called with
    inputs, targets_template and forcings, matching the official GraphCast API.
    """
    config = configuration(model)
    if config['variant'] == 'advection':
        network = AdvectedGraphCast(model.model_config,model.task_config,
                                   config['adapter_modes'],config['adapter_resolution'])
    else:
        network = CachedGraphCast(model.model_config,model.task_config)
    return autoregressive.Predictor(normalization.InputsAndResiduals(
        casting.Bfloat16Cast(network),**stats))
