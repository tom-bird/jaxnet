# Run this example in your browser: https://colab.research.google.com/drive/19web5SnmIFglLcnpXE34phiTY03v39-g

import time

from jax import jit, grad, lax, random, numpy as np
from jax.experimental import optimizers
from jax.random import PRNGKey
from matplotlib import pyplot as plt

from jaxnet import Sequential, Dense, relu, softplus, parametrized


def mnist_images():
    import tensorflow_datasets as tfds
    prep = lambda d: np.reshape(np.float32(next(tfds.as_numpy(d))['image']) / 256, (-1, 784))
    dataset = tfds.load("mnist:1.0.0")
    return (prep(dataset['train'].shuffle(50000).batch(50000)),
            prep(dataset['test'].batch(10000)))


def gaussian_kl(mu, sigmasq):
    """KL divergence from a diagonal Gaussian to the standard Gaussian."""
    return -0.5 * np.sum(1. + np.log(sigmasq) - mu ** 2. - sigmasq)


def gaussian_sample(rng, mu, sigmasq):
    """Sample a diagonal Gaussian."""
    return mu + np.sqrt(sigmasq) * random.normal(rng, mu.shape)


def bernoulli_logpdf(logits, x):
    """Bernoulli log pdf of data x given logits."""
    return -np.sum(np.logaddexp(0., np.where(x, -1., 1.) * logits))


def image_grid(nrow, ncol, imagevecs, imshape):
    """Reshape a stack of image vectors into an image grid for plotting."""
    images = iter(imagevecs.reshape((-1,) + imshape))
    return np.vstack([np.hstack([next(images).T for _ in range(ncol)][::-1])
                      for _ in range(nrow)]).T


@parametrized
def encode(input):
    input = Sequential(Dense(512), relu, Dense(512), relu)(input)
    mean = Dense(10)(input)
    variance = Sequential(Dense(10), softplus)(input)
    return mean, variance


decode = Sequential(Dense(512), relu, Dense(512), relu, Dense(28 * 28))


@parametrized
def loss(rng, images):
    """Monte Carlo estimate of the negative evidence lower bound."""
    mu_z, sigmasq_z = encode(images)
    logits_x = decode(gaussian_sample(rng, mu_z, sigmasq_z))
    return -(bernoulli_logpdf(logits_x, images) - gaussian_kl(mu_z, sigmasq_z)) / images.shape[0]


@parametrized
def image_sample_grid(rng, nrow=10, ncol=10):
    """Sample images from the generative model."""
    code_rng, img_rng = random.split(rng)
    logits = decode(random.normal(code_rng, (nrow * ncol, 10)))
    sampled_images = random.bernoulli(img_rng, np.logaddexp(0., logits))
    return image_grid(nrow, ncol, sampled_images, (28, 28))


@parametrized
def evaluate(test_rng, images):
    elbo_rng, data_rng, image_rng = random.split(test_rng, 3)
    binarized_test = random.bernoulli(data_rng, images)
    test_elbo = loss(elbo_rng, binarized_test)
    return test_elbo, image_sample_grid(image_rng)


@jit
def binarize_batch(rng, i, images):
    i = i % num_batches
    batch = lax.dynamic_slice_in_dim(images, i * batch_size, batch_size)
    return random.bernoulli(rng, batch)


if __name__ == "__main__":
    step_size = 0.001
    num_epochs = 100
    batch_size = 32
    test_rng = PRNGKey(1)  # fixed for evaluation

    train_images, test_images = mnist_images()
    num_complete_batches, leftover = divmod(train_images.shape[0], batch_size)
    num_batches = num_complete_batches + bool(leftover)
    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)


    @jit
    def run_epoch(rng, opt_state):
        def body_fun(i, opt_state):
            elbo_rng, data_rng = random.split(random.fold_in(rng, i))
            batch = binarize_batch(data_rng, i, train_images)
            g = grad(loss.apply)(get_params(opt_state), elbo_rng, batch)
            return opt_update(i, g, opt_state)

        return lax.fori_loop(0, num_batches, body_fun, opt_state)


    example_rng = PRNGKey(0)
    example_batch = binarize_batch(example_rng, 0, images=train_images)
    shaped_elbo = loss.shaped(example_rng, example_batch)
    init_params = shaped_elbo.init_params(PRNGKey(2))
    opt_state = opt_init(init_params)

    for epoch in range(num_epochs):
        tic = time.time()
        opt_state = run_epoch(PRNGKey(epoch), opt_state)
        params = get_params(opt_state)
        test_elbo, samples = evaluate.apply_from({shaped_elbo: params}, test_rng, test_images,
                                                 jit=True)
        print(f'Epoch {epoch: 3d} {test_elbo:.3f} ({time.time() - tic:.3f} sec)')
        plt.imshow(samples, cmap=plt.cm.gray)
        plt.show()
