import json
import os
import sys
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from enrichment_auc.plot.plot_scatter_flow import plot_flow


def evaluate_pas(
    scores,
    thrs,
    gs_names,
    geneset_info,
    data_type,
    score_name,
    embed=None,
    labels_arr=None,
):
    print(score_name)

    for i, gs_name in tqdm(enumerate(gs_names), total=len(gs_names)):
        gs_title = geneset_info.Title.where(geneset_info.ID == gs_name, gs_name).max()
        score = scores[i, :]
        thr = [thrs.loc[gs_name]]
        if (
            embed is not None
            and labels_arr is not None
            and score.max() - score.min() > 10 ** (-20)
        ):
            plot_flow(
                embed,
                score,
                thr,
                labels_arr,
                name=score_name,
                gs_name=gs_name,
                embed_name="t-SNE",
                save_dir=save_dir + "/AUCell/flow/",
            )


score_names = [
    "z",
    "gsva",
    "auc",
    "cerno",
    "ratios",
    "vision",
    "aucell",
    "svd",
    "sparse_pca",
    "ssgsea",
    "jasmine",
    "mean",
]  # all scores to run for each data type

if __name__ == "__main__":
    data_type = sys.argv[1]
    res_folder = sys.argv[2]
    save_dir = sys.argv[3]
    data_folder = sys.argv[4]
    save_dir = save_dir + "/" + data_type
    print(data_type)
    geneset_info = pd.read_csv(
        data_folder + "filtered_genesets_modules.csv", index_col=0
    )
    embed = None
    labels_arr = None
    if os.path.isfile(data_folder + "vae.csv"):
        embed = pd.read_csv(data_folder + "vae.csv")
        embed = embed.select_dtypes(["number"]).to_numpy().astype(float)
    if os.path.isfile(data_folder + "true_labels.csv"):
        labels_arr = pd.read_csv(data_folder + "true_labels.csv", index_col=0)
        labels_arr = labels_arr["CellType"].to_numpy()

    if not os.path.isdir(save_dir + "/AUCell/flow/"):
        os.makedirs(save_dir + "/AUCell/flow/")

    for score_name in tqdm(score_names):
        # get scores
        # AUCell_thr
        thrs = pd.read_csv(
            res_folder + data_type + "/AUCell_thr/" + score_name + "_thr.csv",
            index_col=0,
        )
        scores = pd.read_csv(
            res_folder + data_type + "/" + score_name + ".csv",
            index_col=0,
        )
        gs_names = scores.index.values.tolist()
        scores = scores.to_numpy()
        evaluate_pas(
            scores,
            thrs,
            gs_names,
            geneset_info,
            data_type,
            score_name,
            embed,
            labels_arr,
        )

        if score_name in ["svd", "sparse_pca", "z", "vision"]:
            thrs = pd.read_csv(
                res_folder + data_type + "/AUCell_thr/" + score_name + "_abs_thr.csv",
                index_col=0,
            )
            scores = np.abs(scores)
            evaluate_pas(
                scores,
                thrs,
                gs_names,
                geneset_info,
                data_type,
                score_name + "_abs",
                embed,
                labels_arr,
            )
