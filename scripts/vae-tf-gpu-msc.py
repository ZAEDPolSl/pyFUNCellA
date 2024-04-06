import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import tensorflow as tf

import enrichment_auc
from enrichment_auc.metrics.vae import VAE

tissues = [
    # "Pancreas",
    "PBMC",
    "COVID",
    # "BreastCancer",
    "BM",
    "Liver",
]
norm = "seurat"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--intermediate_dim", nargs="+", type=int)
    parser.add_argument("--lr", default=10e-3, type=float)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--reg", default=0.0, type=float)
    parser.add_argument("--GPU", default=1, type=str)

    args = parser.parse_args()
    intermediate_dim = tuple(args.intermediate_dim)
    learning_rate = args.lr
    epochs = args.epochs
    verbose = 0
    batch_size = 2048
    reg = args.reg

    with tf.device("/device:GPU:{}".format(args.GPU)):
        for tissue in tissues:
            print(tissue)
            datafolder = "/mnt/pmanas/Ania/scrna-seq/data/" + tissue + "/"
            resfolder = "results/" + tissue + "/vae/"

            gene_expr = pd.read_csv(
                datafolder + norm + "_filtered_data.csv", index_col=0
            )
            labels = pd.read_csv(datafolder + "true_labels.csv", index_col=0)
            labels = labels["CellType"].to_numpy()

            if not os.path.isdir(resfolder):
                os.makedirs(resfolder)
            gene_expr = gene_expr.to_numpy().astype(float)

            # VAE + t-SNE
            max_sil = -1.1
            for i in tqdm(range(10)):
                vae = VAE(
                    latent_dim=50,
                    verbose=verbose,
                    intermediate_dim=intermediate_dim,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    # random_state=42,
                    batch_size=batch_size,
                ).fit(gene_expr.T)
                embed = vae.transform(gene_expr.T)
                tsne = TSNE(n_components=2, perplexity=35, n_iter=1500, angle=0.5)
                embed1 = tsne.fit_transform(embed, labels)
                s = silhouette_score(embed1, labels)
                if s > max_sil:
                    max_sil = s
                    fig = px.scatter(
                        x=embed1[:, 0],
                        y=embed1[:, 1],
                        color=labels,
                        color_discrete_sequence=px.colors.qualitative.Alphabet,
                        title="Visualization of "
                        + tissue
                        + " dataset using VAE and t-SNE"
                        + "<br>Silhouette score: "
                        + str(round(s, 3)),
                        labels={"x": "t-SNE 1", "y": "t-SNE 2", "color": "Cell Type"},
                        width=1200,
                        height=700,
                    )
                    fig.update_layout(template="plotly_white")
                    fig.write_html(resfolder + "vae{}.html".format(intermediate_dim))
                    dataset = pd.DataFrame(
                        {"tsne1": embed1[:, 0], "tsne2": embed1[:, 1]}
                    )
                    dataset.to_csv(resfolder + "vae{}.csv".format(intermediate_dim))
