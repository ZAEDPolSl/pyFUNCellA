import json
import os
import sys
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from enrichment_auc.gmm.distributions import find_distribution
from enrichment_auc.gmm.thresholds import correct_via_kmeans, find_thresholds
from enrichment_auc.plot.plot_distributed_data import plot_mixtures
from enrichment_auc.plot.plot_scatter_flow import plot_flow


def pipeline_for_dist(score, geneset_name, score_name, save_dir):
    score = np.nan_to_num(score)
    # get mixtures and thresholds
    distributions = find_distribution(score, geneset_name)
    thresholds_gmm = find_thresholds(distributions, score, geneset_name)
    thresholds_kmeans = correct_via_kmeans(distributions, thresholds_gmm)

    if score.max() - score.min() > 10 ** (-20):
        thr_gmm = score.max()
        thr_kmeans = score.max()
        if thresholds_gmm.shape[0] != 0:
            thr_gmm = thresholds_gmm[-1]
        if thresholds_kmeans.shape[0] != 0:
            thr_kmeans = thresholds_kmeans[-1]
        plot_mixtures(
            geneset_name,
            distributions,
            score,
            thr_gmm,
            thresholds_gmm,
            score_name,
            save_dir=save_dir + "/top1",
            file_only=True,
        )
        plot_mixtures(
            geneset_name,
            distributions,
            score,
            thr_kmeans,
            thresholds_kmeans,
            score_name,
            save_dir=save_dir + "/kmeans",
            file_only=True,
        )
    return (thresholds_gmm, thresholds_kmeans, distributions)


def evaluate_pas(
    scores,
    gs_names,
    geneset_info,
    res_folder,
    data_type,
    score_name,
    embed=None,
    labels_arr=None,
):
    print(score_name)
    scores_thr = pd.DataFrame(0, index=gs_names, columns=np.arange(1))
    scores_thr_kmeans = pd.DataFrame(0, index=gs_names, columns=np.arange(1))
    scores_dist = []
    gmm_thrs = {}
    kmeans_thrs = {}

    for i, gs_name in tqdm(enumerate(gs_names), total=len(gs_names)):
        gs_title = geneset_info.Title.where(geneset_info.ID == gs_name, gs_name).max()
        score = scores[i, :]
        (
            thresholds_gmm,
            thresholds_kmeans,
            distributions,
        ) = pipeline_for_dist(score, gs_title, score_name, save_dir)
        distributions["weights"] = (distributions["weights"]).tolist()
        distributions["mu"] = (distributions["mu"]).tolist()
        distributions["sigma"] = (distributions["sigma"]).tolist()
        if (
            embed is not None
            and labels_arr is not None
            and score.max() - score.min() > 10 ** (-20)
        ):
            plot_flow(
                embed,
                score,
                thresholds_kmeans,
                labels_arr,
                name=score_name,
                gs_name=gs_name,
                embed_name="t-SNE",
                save_dir=save_dir + "/kmeans/flow/",
            )
            plot_flow(
                embed,
                score,
                thresholds_gmm,
                labels_arr,
                name=score_name,
                gs_name=gs_name,
                embed_name="t-SNE",
                save_dir=save_dir + "/top1/flow/",
            )

        if all(thresholds_gmm.shape):
            scores_thr.loc[gs_name] = thresholds_gmm[-1]
        else:
            scores_thr.loc[gs_name] = np.nan
        if all(thresholds_kmeans.shape):
            scores_thr_kmeans.loc[gs_name] = thresholds_kmeans[-1]
        else:
            scores_thr_kmeans.loc[gs_name] = np.nan
        scores_dist.append(distributions)

        gmm_thrs[gs_name] = thresholds_gmm.tolist()
        kmeans_thrs[gs_name] = thresholds_kmeans.tolist()

    scores_thr.to_csv(res_folder + data_type + "/gmm_thr/" + score_name + "_thr.csv")
    scores_thr_kmeans.to_csv(
        res_folder + data_type + "/kmeans_thr/" + score_name + "_thr.csv"
    )

    with open(res_folder + data_type + "/" + score_name + "_dist.json", "w") as fout:
        json.dump(scores_dist, fout)

    with open(
        res_folder + data_type + "/gmm_thr/" + score_name + "_thrs.json",
        "w",
    ) as fout:
        json.dump(gmm_thrs, fout)
    with open(
        res_folder + data_type + "/kmeans_thr/" + score_name + "_thrs.json",
        "w",
    ) as fout:
        json.dump(kmeans_thrs, fout)


score_names = [
    "z",
    "gsva",
    "auc",
    "cerno",
    "bina",
    "vision",
    "aucell",
    "svd",
    "sparse_pca",
    "ssgsea",
    "jasmine",
    "mean",
    "vae",
    "vae_corr",
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

    if not os.path.isdir(save_dir + "/kmeans/flow/"):
        os.makedirs(save_dir + "/kmeans/flow/")
    if not os.path.isdir(save_dir + "/top1/flow/"):
        os.makedirs(save_dir + "/top1/flow/")

    if not os.path.isdir(res_folder + data_type + "/gmm_thr/"):
        os.makedirs(res_folder + data_type + "/gmm_thr/")
    if not os.path.isdir(res_folder + data_type + "/kmeans_thr/"):
        os.makedirs(res_folder + data_type + "/kmeans_thr/")

    for score_name in tqdm(score_names):
        # get scores
        scores = pd.read_csv(
            res_folder + data_type + "/" + score_name + ".csv",
            index_col=0,
        )
        gs_names = scores.index.values.tolist()
        scores = scores.to_numpy()

        evaluate_pas(
            scores,
            gs_names,
            geneset_info,
            res_folder,
            data_type,
            score_name,
            embed,
            labels_arr,
        )

        if score_name in ["svd", "sparse_pca", "z", "vision"]:
            scores = np.abs(scores)
            evaluate_pas(
                scores,
                gs_names,
                geneset_info,
                res_folder,
                data_type,
                score_name + "_abs",
                embed,
                labels_arr,
            )
