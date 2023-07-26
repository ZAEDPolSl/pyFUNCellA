import io

import pickle
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_blobs

from enrichment_auc.metrics.vae import VAE


def dataset():
    """
    prepare exemplary test data
    """
    data = make_blobs(n_samples=10000, n_features=20, centers=2, random_state=42)[0]
    data = data - np.min(data)
    return data


def test_VAE_compresses_data():
    data = dataset()
    vae = VAE(
        intermediate_dim=2,
        latent_dim=5,
        epochs=2,
        learning_rate=1e-6,
        normalize=True,
        random_state=42,
        verbose=0,
    )
    vae.fit(data)
    encoded = vae.transform(data)
    assert encoded.shape == (10000, 5)


def test_VAE_reconstructs_data_shape():
    data = dataset()
    vae = VAE(
        intermediate_dim=2,
        latent_dim=5,
        epochs=2,
        learning_rate=1e-6,
        normalize=True,
        random_state=42,
        verbose=0,
    )
    vae.fit(data)
    recon = vae.vae_.predict(data)
    assert recon.shape == (10000, 20)


def test_VAE_yields_stable_results_without_training():
    data = dataset()
    vae1 = VAE(
        intermediate_dim=2,
        latent_dim=5,
        epochs=0,
        learning_rate=1e-6,
        normalize=True,
        random_state=42,
        verbose=0,
    )
    vae1.fit(data)
    embed1 = vae1.transform(data)
    recon1 = vae1.vae_.predict(data)

    vae2 = VAE(
        intermediate_dim=2,
        latent_dim=5,
        epochs=0,
        learning_rate=1e-6,
        normalize=True,
        random_state=42,
        verbose=0,
    )
    vae2.fit(data)
    embed2 = vae2.transform(data)
    recon2 = vae2.vae_.predict(data)

    np.testing.assert_array_equal(embed1, embed2)
    np.testing.assert_array_equal(recon1, recon2)


def test_VAE_yields_stable_results_with_training():
    data = dataset()
    vae1 = VAE(
        intermediate_dim=2,
        latent_dim=5,
        epochs=2,
        learning_rate=1e-6,
        normalize=True,
        random_state=42,
        verbose=0,
    )
    vae1.fit(data)
    embed1 = vae1.transform(data)
    recon1 = vae1.vae_.predict(data)

    vae2 = VAE(
        intermediate_dim=2,
        latent_dim=5,
        epochs=2,
        learning_rate=1e-6,
        normalize=True,
        random_state=42,
        verbose=0,
    )
    vae2.fit(data)
    embed2 = vae2.transform(data)
    recon2 = vae2.vae_.predict(data)

    np.testing.assert_array_equal(embed1, embed2)
    np.testing.assert_array_equal(recon1, recon2)


def test_VAE_is_clonable():
    data = dataset()
    vae1 = VAE(
        intermediate_dim=2,
        latent_dim=5,
        epochs=2,
        learning_rate=1e-6,
        normalize=True,
        random_state=42,
        verbose=0,
    )
    vae1.fit(data)
    embed1 = vae1.transform(data)
    recon1 = vae1.vae_.predict(data)

    vae2 = clone(vae1)
    vae2.fit(data)
    embed2 = vae2.transform(data)
    recon2 = vae2.vae_.predict(data)

    np.testing.assert_array_equal(embed1, embed2)
    np.testing.assert_array_equal(recon1, recon2)
