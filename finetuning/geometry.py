"""Reuse immutable upstream graph geometry between Haiku init/apply traces."""
import dataclasses
import hashlib
import os
from pathlib import Path
import pickle
import haiku as hk
import numpy as np
from graphcast import graphcast

_CACHE = {}
_FIELDS = ('_num_mesh_nodes','_mesh_nodes_lat','_mesh_nodes_lon','_grid_lat','_grid_lon',
           '_num_grid_nodes','_grid_nodes_lat','_grid_nodes_lon',
           '_grid2mesh_graph_structure','_mesh_graph_structure','_mesh2grid_graph_structure')


class CachedGraphCast(graphcast.GraphCast):
    def __init__(self, model_config, task_config):
        super().__init__(model_config,task_config)
        self._geometry_config = repr(dataclasses.asdict(model_config))
        # Autoregressive checkpointing alone is skipped for one-step training.
        # Recompute each message-passing block during its backward pass instead
        # of retaining every processor layer's large intermediate tensors.
        for network in (self._grid2mesh_gnn,self._mesh_gnn,self._mesh2grid_gnn):
            network._embed = hk.remat(network._embed)
            network._output = hk.remat(network._output)
            original = network._process_step
            def rematerialized(processor, graph, _original=original):
                return hk.remat(lambda g: _original(processor,g))(graph)
            network._process_step = rematerialized

    def _maybe_init(self, sample_inputs):
        if self._initialized:
            return
        digest = hashlib.sha256()
        for coordinate in (sample_inputs.lat,sample_inputs.lon):
            a = np.asarray(coordinate,dtype=np.float64)
            digest.update(str(a.shape).encode()); digest.update(a.tobytes())
        key = (self._geometry_config,digest.hexdigest())
        cache_file = None
        if os.environ.get('GRAPHCAST_GEOMETRY_CACHE'):
            # Opt-in trusted local cache; never load pickle files from others.
            folder = Path(os.environ['GRAPHCAST_GEOMETRY_CACHE'])
            folder.mkdir(parents=True,exist_ok=True)
            identity = hashlib.sha256(('upstream-97d1ad50-cache-v1'+repr(key)).encode()).hexdigest()
            cache_file = folder/(identity+'.pkl')
            if key not in _CACHE and cache_file.exists():
                with cache_file.open('rb') as f:
                    geometry = pickle.load(f)
                if set(geometry) != set(_FIELDS):
                    raise ValueError('Invalid local geometry cache')
                _CACHE.clear(); _CACHE[key] = geometry
        if key not in _CACHE:
            super()._maybe_init(sample_inputs)
            # Keep only one potentially large geometry. No weather or parameters.
            _CACHE.clear()
            _CACHE[key] = {name:getattr(self,name) for name in _FIELDS}
            if cache_file:
                temp = cache_file.with_suffix(f'.{os.getpid()}.tmp')
                with temp.open('wb') as f:
                    pickle.dump(_CACHE[key],f,protocol=pickle.HIGHEST_PROTOCOL)
                temp.replace(cache_file)
        else:
            for name,value in _CACHE[key].items():
                setattr(self,name,value)
            self._initialized = True
