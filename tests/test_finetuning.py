import dataclasses
from pathlib import Path
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import xarray as xr
from graphcast import graphcast
from finetuning.advection import sample_sphere, departure
from finetuning.data import (ATMOSPHERE, SURFACE, STATIC, valid_initializations,
                             assert_window, precipitation_six_hours, regrid, model_batch)
from finetuning.train import transformed_loss, merge_pretrained, make_update, compile_update


def test_splits():
    for split in ('train','val','test'):
        for steps in (1,12,40):
            dates = valid_initializations(split,steps)
            assert_window(dates[0],split,steps)
            assert_window(dates[-1],split,steps)
    with pytest.raises(ValueError): assert_window('2019-12-31T18','train',1)
    with pytest.raises(ValueError): assert_window('2021-01-01T12','train',1)
    with pytest.raises(ValueError): assert_window('2016-01-01T12:30','train',1)


def test_precipitation_exact_hours():
    times = np.arange(np.datetime64('2016-01-01T00'),np.datetime64('2016-01-02T00'),np.timedelta64(1,'h'))
    p = xr.DataArray(np.arange(len(times),dtype=float),dims='time',coords={'time':times})
    got = precipitation_six_hours(p,times[[6,12]])
    np.testing.assert_allclose(got,[sum(range(1,7)),sum(range(7,13))])
    with pytest.raises(KeyError): precipitation_six_hours(p.drop_sel(time=times[4]),times[[6]])


def test_regrid_seam():
    ds = xr.Dataset({'a':(('lat','lon'),np.ones((5,8),np.float32))},
                    coords={'lat':np.linspace(-90,90,5),'lon':np.arange(8)*45})
    fine = regrid(ds,10)
    assert fine.sizes == {'lat':19,'lon':36}
    np.testing.assert_allclose(fine.a,1)
    assert np.isfinite(fine.a.values).all()
    for invalid in (0, -1, float('nan')):
        with pytest.raises(ValueError): regrid(ds,invalid)
    from graphcast.losses import normalized_latitude_weights
    requested = regrid(ds,0.1)
    weights = normalized_latitude_weights(requested.a)
    assert requested.sizes == {'lat':1801,'lon':3600}
    assert np.isfinite(weights.values).all()


def test_transport_constant_seam_poles_and_gradients():
    field = jnp.ones((1,19,36,2))
    lat = jnp.array([[[-np.pi/2,-np.pi/2],[0.,0.],[np.pi/2,np.pi/2]]])
    lon = jnp.array([[[0.,2*np.pi-1e-5],[2*np.pi-1e-5,0.],[0.,2*np.pi-1e-5]]])
    np.testing.assert_allclose(sample_sphere(field,lat,lon),1,atol=3e-6)
    # At the equator a pure eastward displacement is an exact longitude shift.
    la,lo = departure(jnp.array(0.),jnp.array(1.),jnp.array(.2),jnp.array(0.))
    np.testing.assert_allclose([la,lo],[0.,.8],atol=1e-6)
    x = jnp.sin(jnp.arange(36)*2*jnp.pi/36)[None,None,:,None]*jnp.ones((1,19,1,2))
    grad = jax.grad(lambda shift:jnp.sum(sample_sphere(x,lat,lon+shift)))(jnp.array(.1))
    assert np.isfinite(grad) and abs(float(grad))>0


def tiny_case():
    task = dataclasses.replace(graphcast.TASK,pressure_levels=(500,850))
    config = graphcast.ModelConfig(resolution=10.,mesh_size=1,latent_size=8,
                                  gnn_msg_steps=1,hidden_layers=1,
                                  radius_query_fraction_edge_length=.6)
    coords = dict(time=np.array(['2016-01-01T06','2016-01-01T12','2016-01-01T18'],dtype='datetime64[ns]'),
                  lat=np.linspace(-90,90,19,dtype=np.float32),lon=np.arange(36,dtype=np.float32)*10,
                  level=[500,850])
    rng = np.random.default_rng(2)
    variables = {}
    for n in ATMOSPHERE:
        variables[n] = (('time','level','lat','lon'),rng.normal(size=(3,2,19,36)).astype(np.float32))
    for n in SURFACE+('total_precipitation_6hr',):
        variables[n] = (('time','lat','lon'),rng.normal(size=(3,19,36)).astype(np.float32))
    for n in STATIC:
        variables[n] = (('lat','lon'),rng.normal(size=(19,36)).astype(np.float32))
    ds = xr.Dataset(variables,coords=coords)
    batch = model_batch(ds,task,1)
    names = set(task.input_variables)|set(task.forcing_variables)
    means = xr.Dataset({n:xr.DataArray(np.float32(0)) for n in names})
    std = xr.Dataset({n:xr.DataArray(np.float32(1)) for n in names})
    stats = dict(mean_by_level=means,stddev_by_level=std,diffs_stddev_by_level=std)
    return config,task,stats,batch


def test_adapter_identity_gradients_and_update(tmp_path):
    config,task,stats,batch = tiny_case()
    base = transformed_loss(config,task,stats,'baseline')
    adv = transformed_loss(config,task,stats,'advection',modes=2,adapter_resolution=10.)
    rng = jax.random.PRNGKey(0)
    base_params,state = base.init(rng,*batch)
    adv_params,_ = adv.init(rng,*batch)
    params = merge_pretrained(adv_params,base_params)
    (b,_),_ = base.apply(base_params,state,rng,*batch)
    (a,_),_ = adv.apply(params,state,rng,*batch)
    np.testing.assert_allclose(a,b,rtol=0,atol=0)
    objective = lambda p:adv.apply(p,state,rng,*batch)[0][0]
    grads = jax.grad(objective)(params)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    assert float(jnp.linalg.norm(grads['advection_lift']['w']))>0
    optimizer = optax.adam(1e-3)
    update = make_update(adv,optimizer)
    compiled = compile_update(update,params,state,optimizer.init(params),rng,*batch)
    updated,_,_,value,_,norm = compiled(params,state,optimizer.init(params),rng,*batch)
    assert np.isfinite(value) and float(norm)>0
    assert not np.array_equal(updated['advection_lift']['w'],params['advection_lift']['w'])
    next_grads = jax.grad(objective)(updated)
    assert float(jnp.linalg.norm(next_grads['advection_velocity']['w']))>0
    from finetuning.checkpoints import save_model, configuration, predictor
    from graphcast import checkpoint
    original = graphcast.CheckPoint(params=base_params,model_config=config,task_config=task,
                                   description='test model',license='test')
    path = tmp_path/'adapter.npz'
    save_model(path,original,config,updated,'advection',2,10.)
    with path.open('rb') as f:
        loaded = checkpoint.load(f,graphcast.CheckPoint)
    assert configuration(loaded)['variant'] == 'advection'
    @hk.transform_with_state
    def predict(inputs,targets,forcings):
        return predictor(loaded,stats)(inputs,targets,forcings)
    forecast,_ = predict.apply(loaded.params,state,rng,*batch)
    assert set(forecast.data_vars) == set(task.target_variables)
    assert all(np.isfinite(v.values).all() for v in forecast.data_vars.values())
    with pytest.raises(ValueError):
        configuration(dataclasses.replace(loaded,description='missing metadata'))


def test_pretrained_missing_keys_rejected():
    with pytest.raises(ValueError): merge_pretrained({}, {'old':{'w':np.ones(1)}})
    legacy = 'mesh2grid_gnn/~_networks_builder/decoder_nodes_mesh_nodes_mlp/~/linear_1'
    assert merge_pretrained({}, {legacy:{'w':np.ones(1)}}) == {}


def test_donated_updates_match_and_keep_weather_reusable():
    # Exercise flattened xarray arguments and two consecutive Adam updates;
    # donating weather by mistake would invalidate the second invocation.
    @hk.transform_with_state
    def loss(inputs, targets, forcings):
        from graphcast import xarray_jax
        x = xarray_jax.unwrap_data(inputs.x, require_jax=True)
        y = xarray_jax.unwrap_data(targets.y, require_jax=True)
        w = hk.get_parameter('w', (3,), init=hk.initializers.Constant(.2))
        value = jnp.mean((x*w-y)**2)
        return value, {'mse': value}
    from graphcast import xarray_jax
    inputs = xarray_jax.Dataset({'x': (('lat',), jnp.array([1.,2.,3.]))},
                               coords={'lat': [-90.,0.,90.]})
    targets = xarray_jax.Dataset({'y': (('lat',), jnp.array([.5,.4,.3]))},
                                coords={'lat': [-90.,0.,90.]})
    batch = (inputs,targets,xr.Dataset())
    rng = jax.random.PRNGKey(0)
    params,state = loss.init(rng,*batch)
    optimizer = optax.adam(1e-3)
    update = make_update(loss,optimizer)
    reference = (params,state,optimizer.init(params))
    donated = jax.tree.map(lambda x:jnp.array(x,copy=True),reference)
    regular_step = compile_update(update,*reference,rng,*batch)
    donated_step = compile_update(update,*donated,rng,*batch,donate=True)
    for _ in range(2):
        expected = regular_step(*reference,rng,*batch)
        actual = donated_step(*donated,rng,*batch)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
            np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)
        reference,donated = expected[:3],actual[:3]
    np.testing.assert_array_equal(inputs.x.values,[1.,2.,3.])


def test_physical_rmse_axes_and_exact_alignment():
    from finetuning.evaluate import spatial_mse
    reference=xr.DataArray(np.zeros((1,2,2,3,4)),dims=('batch','time','level','lat','lon'),
                          coords={'time':[0,1],'level':[500,850],'lat':[-90.,0.,90.],'lon':[0,90,180,270]})
    prediction=reference+3
    result=spatial_mse(prediction,reference)
    assert result.dims == ('time','level')
    np.testing.assert_allclose(result,9)
    with pytest.raises(ValueError):
        spatial_mse(prediction.assign_coords(level=[600,850]),reference)


def test_checkpointing_matches_stock_forward_and_gradient():
    # Test mathematical equivalence in float32. BF16 rematerialization may
    # change rounding in the backward pass, so bitwise BF16 gradients are not
    # a valid invariant (the BF16 forward loss was checked separately).
    from graphcast import autoregressive,normalization,xarray_jax,xarray_tree
    from finetuning.geometry import CachedGraphCast
    config,task,stats,batch=tiny_case()
    def transform(cls):
        @hk.transform_with_state
        def loss(inputs,targets,forcings):
            network=autoregressive.Predictor(normalization.InputsAndResiduals(
                cls(config,task),**stats))
            return xarray_tree.map_structure(
                lambda x:xarray_jax.unwrap_data(x.mean(),require_jax=True),network.loss(inputs,targets,forcings))
        return loss
    stock,optimized=transform(graphcast.GraphCast),transform(CachedGraphCast)
    rng=jax.random.PRNGKey(0)
    params,state=stock.init(rng,*batch)
    shape,_=jax.eval_shape(optimized.init,rng,*batch)
    params=merge_pretrained(shape,params)
    def value_and_grad(network):
        return jax.value_and_grad(lambda p:network.apply(p,state,rng,*batch)[0][0])(params)
    original,original_grad=value_and_grad(stock)
    actual,actual_grad=value_and_grad(optimized)
    np.testing.assert_allclose(actual,original,rtol=1e-6,atol=1e-6)
    for expected,got in zip(jax.tree.leaves(original_grad),jax.tree.leaves(actual_grad)):
        np.testing.assert_allclose(got,expected,rtol=1e-5,atol=1e-6)
