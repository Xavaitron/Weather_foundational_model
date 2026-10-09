import dataclasses
import json
import jax
import numpy as np
import xarray as xr
from graphcast import graphcast
from finetuning.checkpoints import MARKER
from finetuning.diagnose_advection import initial_parameters, make_diagnostics, parameter_changes, materialized_scalars
from finetuning.train import transformed_loss
from test_finetuning import tiny_case


def test_diagnostic_capture_zero_lift_and_active_bf16_correction():
    config, task, stats, batch = tiny_case()
    base_params, _ = transformed_loss(config,task,stats,'baseline').init(jax.random.PRNGKey(0),*batch)
    adv_params, _ = transformed_loss(config,task,stats,'advection',2,10.).init(jax.random.PRNGKey(0),*batch)
    metadata = dict(format_version=1,variant='advection',adapter_modes=2,adapter_resolution=10.)
    original = graphcast.CheckPoint(params=base_params,model_config=config,task_config=task,description='original',license='test')
    model = graphcast.CheckPoint(params=adv_params,model_config=config,task_config=task,description='test'+MARKER+json.dumps(metadata),license='test')
    initial = initial_parameters(original,model,0)
    model = dataclasses.replace(model,params=initial)
    observe, _, _ = make_diagnostics(model,stats,initial['advection_velocity'])
    inputs, targets, forcings = batch
    (scalars, fields, boundary), _ = observe(initial,{},jax.random.PRNGKey(0),inputs,xr.zeros_like(targets),forcings)
    scalars = materialized_scalars(scalars, jax.device_get(boundary))
    assert float(scalars['correction_rms']) == 0
    assert float(scalars['changed_hidden_fraction']) == 0
    assert np.isfinite(fields['current']['distance_km']).all()
    assert float(fields['current']['distance_km'].mean()) > 0
    np.testing.assert_allclose(fields['current']['east'], fields['initial_velocity_same_features']['east'], atol=1e-6)
    active = dict(initial)
    active['advection_lift'] = {'w':np.full_like(initial['advection_lift']['w'], .1)}
    (scalars, _, boundary), _ = observe(active,{},jax.random.PRNGKey(0),inputs,xr.zeros_like(targets),forcings)
    scalars = materialized_scalars(scalars, jax.device_get(boundary))
    assert float(scalars['correction_rms']) > 0
    assert float(scalars['changed_hidden_fraction']) > 0
    audit = parameter_changes(initial,active)
    assert audit['changed_modules'] == 1
    assert audit['changed'] == initial['advection_lift']['w'].size
