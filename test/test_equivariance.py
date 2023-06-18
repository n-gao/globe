import unittest
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from globe.nn.globe import Globe, _WAVE_FUNCTIONS
from globe.nn.orbitals import ProductOrbitals
from globe.utils.config import SystemConfigs


class TestEquivariance(unittest.TestCase):
    def test_equivariance(self):
        wfs = list(_WAVE_FUNCTIONS.keys())
        for wf in wfs:
            for shared in (True, False):
                net = Globe({wf: {}, 'shared': {}}, {}, shared, ProductOrbitals, {}, wf_model=wf)
                electrons = np.random.normal(size=(4, 3))
                atoms = np.random.normal(size=(2, 3))
                config = SystemConfigs(spins=((2, 2),), charges=((2, 2),))
                params = net.init(jax.random.PRNGKey(42), electrons, atoms, config)
                params = jtu.tree_map(lambda x: x + 1e-1 * np.random.normal()*np.minimum(x.std(), 1e-3), params)
                fwd = jax.jit(net.apply, static_argnums=3)
                fwd_np = lambda *x: fwd(*x).item()

                x1 = fwd_np(params, electrons, atoms, config)
                
                # Test equivariance spin up
                x2 = fwd_np(params, electrons[[1, 0, 2, 3]], atoms, config)
                assert np.allclose(x1, x2, 1e-5), f"{wf}: same spin equivariance failed with x1={x1}, x2={x2}, diff={np.abs(x1-x2)}!"
                # Test equivariance spin down
                x2 = fwd_np(params, electrons[[0, 1, 3, 2]], atoms, config)
                assert np.allclose(x1, x2, 1e-5), f"{wf}: diff spin equivariance failed with x1={x1}, x2={x2}, diff={np.abs(x1-x2)}!"
                # Test equivariance nuc exchange
                x2 = fwd_np(params, electrons, atoms[[1, 0]], config)
                assert np.allclose(x1, x2, 1e-5), f"{wf}: nuclei equivariance failed with x1={x1}, x2={x2}, diff={np.abs(x1-x2)}!"

    def test_independence(self):
        wfs = list(_WAVE_FUNCTIONS.keys())
        for wf in wfs:
            for shared in (True, False):
                net = Globe({wf: {}, 'shared': {}}, {}, shared, ProductOrbitals, {}, wf_model=wf)
                electrons = np.random.normal(size=(8, 3))
                atoms = np.random.normal(size=(2, 3))
                electrons, atoms = jtu.tree_map(jnp.array, (electrons, atoms))
                config = SystemConfigs(spins=((3, 3), (1, 1)), charges=((6,), (2,)))
                params = net.init(jax.random.PRNGKey(42), electrons, atoms, config)
                # we add some random noise to obersve change in parameters that are initialized as zero
                params = jtu.tree_map(lambda x: x + 1e-1 * np.random.normal()*np.minimum(x.std(), 1e-3), params)

                offsets = [0, np.sum(config.spins[0])]
                slices = [
                    slice(o, o+l)
                    for o, l in zip(offsets, np.sum(config.spins, -1))
                ][::-1]
                for i in range(2):
                    @jax.jit
                    def f_closure(x):
                        return net.apply(params, x, atoms, config)[i]
                    
                    grad = np.array(jax.grad(f_closure)(electrons))
                    assert np.allclose(grad[slices[i]], np.zeros_like(grad[slices[i]])), "Gradient independence failed!"
