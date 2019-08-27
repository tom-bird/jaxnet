# JAXnet [![Build Status](https://travis-ci.org/JuliusKunze/jaxnet.svg?branch=master)](https://travis-ci.org/JuliusKunze/jaxnet) [![PyPI](https://img.shields.io/pypi/v/jaxnet.svg)](https://pypi.python.org/pypi/jaxnet/#history)

JAXnet is a deep learning library based on [JAX](https://github.com/google/jax).
JAXnet's functional API provides unique benefits over TensorFlow2, Keras and PyTorch,
while maintaining user-friendliness, modularity and scalability:
- More robustness through immutable weights, no global compute graph.
- GPU-compiled `numpy` code for networks, training loops, pre- and postprocessing.
- Regularization and reparametrization of any module or whole networks in one line.
- No global random state, flexible random key control.

### Modularity

```python
from jaxnet import *

net = Sequential(Dense(1024), relu, Dense(1024), relu, Dense(4), logsoftmax)
```
creates a neural net model from predefined modules.

### Extensibility

Define your own modules using `@parametrized` functions. You can reuse other modules:

```python
from jax import numpy as np

@parametrized
def loss(inputs, targets):
    return -np.mean(net(inputs) * targets)
```

All modules are composed in this way.
[`jax.numpy`](https://github.com/google/jax#whats-supported) is mirroring `numpy`,
meaning that if you know how to use `numpy`, you know most of JAXnet.
Compare this to TensorFlow2/Keras:

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda

net = Sequential([Dense(1024, 'relu'), Dense(1024, 'relu'), Dense(4), Lambda(tf.nn.log_softmax)])

def loss(inputs, targets):
    return -tf.reduce_mean(net(inputs) * targets)
```

Notice how `Lambda` layers are not needed in JAXnet.
`relu` and `logsoftmax` are plain Python functions.

### Immutable weights

Different from TensorFlow2/Keras, JAXnet has no global compute graph.
Modules like `net` and `loss` do not contain mutable weights.
Instead, weights are contained in separate, immutable objects.
They are initialized with `init_parameters`, provided a random key and example inputs:

```python
from jax.random import PRNGKey

def next_batch(): return np.zeros((3, 784)), np.zeros((3, 4))

params = loss.init_parameters(PRNGKey(0), *next_batch())

print(params.sequential.dense2.bias)  # [0.00376661 0.01038619 0.00920947 0.00792002]
```

Instead of mutating weights inline, optimizers return updated versions of weights.
They are returned as part of a new optimizer state, and can be retrieved via `get_parameters`:

```python
opt = optimizers.Adam()
state = opt.init_state(params)
for _ in range(10):
    state = opt.optimize(loss.apply, state, *next_batch()) # accelerate with jit=True

trained_params = opt.get_parameters(state)
```

`apply` evaluates a network:

```python
batch_loss = loss.apply(trained_params, inputs) # accelerate with jit=True
```

### GPU support and compilation

JAX allows any functional `numpy`/`scipy` code to be accelerated.
Make it run on GPU by replacing your `numpy` import with `jax.numpy`.
Compile a function by decorating it with [`jit`](https://github.com/google/jax#compilation-with-jit).
This will free your function from slow Python interpretation, parallelize operations where possible and optimize your compute graph.
It provides speed and scalability at the level of TensorFlow2 or PyTorch.

Due to immutable weights, whole training loops can be compiled / run on GPU ([demo](examples/mnist_vae.py#L96)).
`jit` will make your training as fast as mutating weights inline, and weights will not leave the GPU.
You can write functional code without worrying about performance.

You can easily accelerate `numpy`/`scipy` pre-/postprocessing code in the same way ([demo](examples/mnist_vae.py#L61)).

### Regularization and reparametrization

In JAXnet, regularizing a model can be done in one line ([demo](examples/wavenet.py#L167)):

```python
loss = L2Regularized(loss, scale=.1)
```

`loss` is now a module that can be used as above.
Reparametrized layers are one-liners, too (see [API](API.md#regularization-and-reparametrization)).
JAXnet allows regularizing or reparametrizing any module or subnetwork without changing its code.
This is possible because modules do not instantiate any variables.
Instead, a module's `apply` is a function with parameters as an argument.
This function can then be composed to build layers like `L2Regularized`.

In contrast, TensorFlow2/Keras/PyTorch have mutable variables baked into their model API. They therefore require:
- Regularization arguments on layer level, with separate code necessary for each layer.
- Reparametrization arguments on layer level, and separate implementations for [each](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseReparameterization) [layer](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution1DReparameterization).

### Random key control
JAXnet does not have global random state.
Random keys are provided explicitly, making code deterministic and independent of previously executed code by default.
This can help debugging and is more flexible ([demo](examples/mnist_vae.py#L89)).
Read more on random numbers in JAX [here](https://github.com/google/jax#random-numbers-are-different).

### Step-by-step debugging

JAXnet allows step-by-step debugging with concrete values like any plain Python function
(when [`jit`](https://github.com/google/jax#compilation-with-jit) compilation is not used).

## API and demos
Find more details on the API [here](API.md).

See JAXnet in action in your browser:
[Mnist Classifier](https://colab.research.google.com/drive/18kICTUbjqnfg5Lk3xFVQtUj6ahct9Vmv),
[Mnist VAE](https://colab.research.google.com/drive/19web5SnmIFglLcnpXE34phiTY03v39-g),
[OCR with RNNs (to be fixed)](https://colab.research.google.com/drive/1YuI6GUtMgnMiWtqoaPznwAiSCe9hMR1E),
[ResNet](https://colab.research.google.com/drive/1q6yoK_Zscv-57ZzPM4qNy3LgjeFzJ5xN) and
[WaveNet](https://colab.research.google.com/drive/111cKRfwYX4YFuPH3FF4V46XLfsPG1icZ).

If you are familiar with stax, read [this](STAX.md).

## Installation
**This is a preview. Expect breaking changes!** Install with

```
pip3 install jaxnet
```

To use GPU, first install the [right version of jaxlib](https://github.com/google/jax#installation).

## Questions

Please feel free to create an issue on GitHub.