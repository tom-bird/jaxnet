from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import os

import numpy as onp
import pytest
import jax
import jax.numpy as np
from jax import random
from jax.api import pmap, grad
from jax.lib import xla_bridge
from jax.nn import relu, log_softmax
from jax.config import config

from jaxnet import Dense, parametrized, Sequential
from jaxnet.optimizers import Sgd, Momentum, Adagrad, RmsProp, Adam, Sm3


config.parse_flags_with_absl()
prev_xla_flags = None

atol = 1e-6  # small diffs seem to occur when using pmap


def check_trees_equal(tree1, tree2):
    flat_t1, t1_def = jax.tree_flatten(tree1)
    flat_t2, t2_def = jax.tree_flatten(tree2)
    assert t1_def == t2_def
    for x, y in zip(flat_t1, flat_t2):
        assert np.allclose(x, y, atol=atol)


# Run all tests with 8 CPU devices.
def setup_module():
    print('setting up')
    global prev_xla_flags
    prev_xla_flags = os.getenv("XLA_FLAGS")
    flags_str = prev_xla_flags or ""
    # Don't override user-specified device count, or other XLA flags.
    if "xla_force_host_platform_device_count" not in flags_str:
        os.environ["XLA_FLAGS"] = (flags_str +
                               " --xla_force_host_platform_device_count=8")
    # Clear any cached backends so new CPU backend will pick up the env var.
    xla_bridge.get_backend.cache_clear()


def teardown_module():
    print('tearing down')
    if prev_xla_flags is None:
        del os.environ["XLA_FLAGS"]
    else:
        os.environ["XLA_FLAGS"] = prev_xla_flags
        xla_bridge.get_backend.cache_clear()


@parametrized
def loss(inputs, targets):
    return -np.mean(Sequential(Dense(4), relu, Dense(4), log_softmax)(inputs) * targets)


def next_batch(batch_size):
    # always return the same batch
    rng = onp.random.RandomState(0)
    return rng.randn(batch_size, 10), rng.randn(batch_size, 4)


@pytest.mark.parametrize('jit', (False, True))
def test_parametrized_apply(jit):
    num_devices = xla_bridge.device_count()
    f = Dense(5)

    shape = (num_devices, 20)
    data = onp.random.randn(*shape)
    key = random.PRNGKey(0)
    params = f.init_parameters(data[0], key=key)
    out1 = f.apply(params, data, jit=jit)
    out2 = f.apply(params, data, jit=jit, pmap=True)
    assert np.allclose(out1, out2, atol=atol)

    # out3 = pmap(lambda x: lax.psum(f.apply(params, x), axis_name='i'), axis_name='i')(data)
    # assert np.allclose(out3[0], np.sum(out2, axis=0), atol=atol)


def test_parametrized_grad():
    num_devices = xla_bridge.device_count()

    key = random.PRNGKey(0)
    params = loss.init_parameters(*next_batch(num_devices), key=key)

    gradient1 = grad(loss.apply)(params, *next_batch(num_devices))
    flat_g1, tree_def1 = jax.tree_flatten(gradient1)

    gradient2 = pmap(partial(grad(loss.apply), params))(*next_batch(num_devices))
    flat_g2, tree_def2 = jax.tree_flatten(gradient2)
    flat_g2 = [np.mean(g, axis=0) for g in flat_g2]
    for g1, g2 in zip(flat_g1, flat_g2):
        assert np.allclose(g1, g2, atol=atol)
    assert tree_def1 == tree_def2


@pytest.mark.parametrize('jit', (False, True))
@pytest.mark.parametrize('opt', (Sgd(), Momentum(.1, .1), Adagrad(), RmsProp(.1), Adam(), Sm3(.1)))
def test_optimizer(jit, opt):
    num_devices = xla_bridge.device_count()

    key = random.PRNGKey(0)
    params = loss.init_parameters(*next_batch(num_devices), key=key)

    state = opt.init(params)

    state1 = opt.update(loss.apply, state, *next_batch(num_devices), jit=jit)
    state2 = opt.update(loss.apply, state, *next_batch(num_devices), pmap=True, jit=jit)
    check_trees_equal(opt.get_parameters(state1), opt.get_parameters(state2))

    state1, loss1 = opt.update_and_get_loss(loss.apply, state, *next_batch(num_devices), jit=jit)
    state2, loss2 = opt.update_and_get_loss(loss.apply, state, *next_batch(num_devices), pmap=True, jit=jit)
    check_trees_equal(opt.get_parameters(state1), opt.get_parameters(state2))
    assert np.allclose(loss1, loss2)

    # check with batches per device
    x, y = next_batch(num_devices * 2)
    state1, loss1 = opt.update_and_get_loss(loss.apply, state, x, y, jit=jit)
    state2, loss2 = opt.update_and_get_loss(loss.apply, state,
                                            np.reshape(x, (num_devices, 2, -1)), np.reshape(y, (num_devices, 2, -1)),
                                            pmap=True, jit=jit)
    check_trees_equal(opt.get_parameters(state1), opt.get_parameters(state2))
    assert np.allclose(loss1, loss2)
