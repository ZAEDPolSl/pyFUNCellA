import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from tqdm import tqdm
import tensorflow as tf

import enrichment_auc
from enrichment_auc.metrics.vae import VAE
from enrichment_auc.preprocess.filter import filter

tissues = ["Pancreas", "PBMC", "COVID", "BreastCancer", "BM", "Liver"]
norm = "seurat"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_case", default="layer_size", type=str)
    parser.add_argument("--intermediate_dim", nargs="+", type=int)
    parser.add_argument("--lr", default=10e-3, type=float)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--reg", default=0.0, type=float)
    parser.add_argument("--GPU", default=0, type=str)

    args = parser.parse_args()
    test_case = args.test_case
    intermediate_dim = tuple(args.intermediate_dim)
    learning_rate = args.lr
    epochs = args.epochs
    verbose = 0
    batch_size = 2048
    reg = args.reg

    with tf.device("/device:GPU:{}".format(args.GPU)):
        for tissue in tissues:
            print(tissue)
            datafolder = "data/" + tissue + "/"
            resfolder = "results/" + tissue + "/vae/" + test_case + "/"

            gene_expr = pd.read_csv(
                datafolder + norm + "_filtered_data.csv", index_col=0
            )
            labels = pd.read_csv(datafolder + "true_labels.csv", index_col=0)
            labels = labels["CellType"].to_numpy()

            if not os.path.isdir(resfolder):
                os.makedirs(resfolder)

            filtered = filter(gene_expr, 0.2)
            filtered = filtered.to_numpy().astype(float)
            gene_expr = gene_expr.to_numpy().astype(float)

            filenames = os.listdir(resfolder)
            filenames = [
                filename for filename in filenames if filename.endswith(".txt")
            ]
            trial = len(filenames)

            s1 = []
            s2 = []
            s3 = []
            s4 = []
            s5 = []
            db1 = []
            db2 = []
            db3 = []
            db4 = []
            db5 = []

            with open(resfolder + "config{}.txt".format(trial), "a") as myfile:
                myfile.write(str(intermediate_dim))
                myfile.write("\n")
                myfile.write(str(learning_rate))
                myfile.write("\n")
                myfile.write(str(epochs))
                myfile.write("\n")
                myfile.write(str(reg))

            # filtered VAE only
            for i in tqdm(range(10)):
                vae_f = VAE(
                    latent_dim=2,
                    verbose=verbose,
                    intermediate_dim=intermediate_dim,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    # random_state=42,
                    batch_size=batch_size,
                ).fit(filtered.T)
                embed_f = vae_f.transform(filtered.T)
                s = silhouette_score(embed_f, labels)
                s1.append(s)
                s = davies_bouldin_score(embed_f, labels)
                db1.append(s)

            # filtered VAE + t-SNE
            for i in tqdm(range(10)):
                vae_f_t = VAE(
                    latent_dim=50,
                    verbose=verbose,
                    intermediate_dim=intermediate_dim,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    # random_state=42,
                    batch_size=batch_size,
                ).fit(filtered.T)
                embed_f_t = vae_f_t.transform(filtered.T)
                s = silhouette_score(embed_f_t, labels)
                s2.append(s)
                s = davies_bouldin_score(embed_f_t, labels)
                db2.append(s)
                tsne = TSNE(n_components=2, perplexity=35, n_iter=1500, angle=0.5)
                embed1_f = tsne.fit_transform(embed_f_t, labels)
                s = silhouette_score(embed1_f, labels)
                s3.append(s)
                s = davies_bouldin_score(embed1_f, labels)
                db3.append(s)

            # VAE + t-SNE
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
                s = silhouette_score(embed, labels)
                s4.append(s)
                s = davies_bouldin_score(embed, labels)
                db4.append(s)
                tsne = TSNE(n_components=2, perplexity=35, n_iter=1500, angle=0.5)
                embed1 = tsne.fit_transform(embed, labels)
                s = silhouette_score(embed1, labels)
                s5.append(s)
                s = davies_bouldin_score(embed1, labels)
                db5.append(s)

            d = [s1, s2, s3, s4, s5]
            df = pd.DataFrame(
                data=d,
                index=[
                    "filtered_vae_2D",
                    "filtered_vae_50D",
                    "filtered_vae_tsne",
                    "vae_50D",
                    "vae_tsne",
                ],
            )
            df.to_csv(resfolder + "trial{}_silhouette.csv".format(trial))
            d = [db1, db2, db3, db4, db5]
            df = pd.DataFrame(
                data=d,
                index=[
                    "filtered_vae_2D",
                    "filtered_vae_50D",
                    "filtered_vae_tsne",
                    "vae_50D",
                    "vae_tsne",
                ],
            )
            df.to_csv(resfolder + "trial{}_db.csv".format(trial))
