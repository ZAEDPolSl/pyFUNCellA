import os
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import tensorflow as tf
import json
from tqdm import tqdm

from enrichment_auc.metrics.vae import VAE

tissues = ["PBMC", "COVID", "BM", "Liver"]
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
    data_folder = "/mnt/pmanas/Ania/scrna-seq/data/"
    paths = pd.read_csv(data_folder + "chosen_paths.txt", sep="\t", index_col=0)

    with tf.device("/device:GPU:{}".format(args.GPU)):
        for tissue in tissues:
            print(tissue)
            datafolder = data_folder + tissue + "/"
            resfolder = (
                "/mnt/pmanas/Ania/scrna-seq/results/"
                + tissue
                + "/pas_vae/"
                + test_case
                + "/"
            )

            gene_expr = pd.read_csv(
                datafolder + norm + "_filtered_data.csv", index_col=0
            )
            genes = gene_expr.index.tolist()
            patients_names = gene_expr.columns.to_list()
            gene_expr = gene_expr.to_numpy().astype(float)
            with open(datafolder + "filtered_genesets_genes.json") as file:
                genesets = json.load(file)
            gs_names = list(genesets.keys())

            if not os.path.isdir(resfolder):
                os.makedirs(resfolder)

            filenames = os.listdir(resfolder)
            filenames = [
                filename for filename in filenames if filename.endswith(".txt")
            ]
            trial = len(filenames)

            with open(resfolder + "config{}.txt".format(trial), "a") as myfile:
                myfile.write(str(intermediate_dim))
                myfile.write("\n")
                myfile.write(str(learning_rate))
                myfile.write("\n")
                myfile.write(str(epochs))
                myfile.write("\n")
                myfile.write(str(reg))

            for i in tqdm(range(10)):
                pas = np.empty((paths.shape[0], gene_expr.shape[1]))
                j = 0
                for gs_name, geneset_genes in genesets.items():
                    if paths["ID"].str.contains(gs_name).any():
                        genes_in_ds = [gene in geneset_genes for gene in genes]
                        in_gs = gene_expr[genes_in_ds, :]
                        vae_f = VAE(
                            latent_dim=1,
                            verbose=verbose,
                            intermediate_dim=intermediate_dim,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            batch_size=batch_size,
                        ).fit(in_gs.T)
                        pas[j] = vae_f.transform(in_gs.T).flatten()
                        j += 1
                df = pd.DataFrame(
                    data=pas, index=list(paths["ID"]), columns=patients_names
                )
                df.to_csv(resfolder + "pas_trial{}_{}.csv".format(trial, i))
