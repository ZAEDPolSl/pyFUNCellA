import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from enrichment_auc.metrics.vae_bn import VAE_BN


class VAE(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        intermediate_dim=512,
        latent_dim=5,
        epochs=100,
        normalize=True,
        batch_size=128,
        shuffle="batch",
        learning_rate=1e-3,
        learning_rate_schedule=None,
        verbose="auto",
        random_state=None,
    ):
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.normalize = normalize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.learning_rate_schedule = learning_rate_schedule
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None):
        div = X.sum(axis=1, keepdims=True)
        X = np.divide(X, div, out=np.zeros_like(X), where=div != 0)
        vae_bn = VAE_BN(
            nSpecFeatures=X.shape[1],
            intermediate_dim=self.intermediate_dim,
            latent_dim=self.latent_dim,
            seed=self.random_state,
        )
        self.vae_, self.encoder_ = vae_bn.get_architecture(
            verbose=self.verbose, lr=self.learning_rate
        )
        self.history_ = self.vae_.fit(
            X,
            verbose=self.verbose,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        return self

    def transform(self, X, y=None):
        div = X.sum(axis=1, keepdims=True)
        X = np.divide(X, div, out=np.zeros_like(X), where=div != 0)
        return self.encoder_.predict(X, verbose=self.verbose)[2]


def _vae_pas(geneset, data, genes, gs_name=""):
    genes_in_ds = [gene in geneset for gene in genes]
    in_gs = data[:, genes_in_ds]
    if in_gs.shape[0] == 0 or in_gs.shape[1] == 0:
        print("Incorrect geneset format:", gs_name)
        return np.zeros(data.shape[0])
    # train small vae
    return np.zeros(data.shape[0])


def VAE_PAS(genesets, data, genes):
    pas = np.empty((len(genesets), data.shape[1]))
    # train main VAE
    # transpose the result
    data = data.T
    # get the embedding as well
    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        pas[i] = _vae_pas(geneset_genes, data, genes, gs_name)
    return pas
