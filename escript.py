import json
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from enrichment_auc.distributions import (
    find_distribution,
    find_grouped_dist_thresholds,
    group_distributions,
)
from enrichment_auc.plot.plot_distributed_data import plot_mixtures


def pipeline_for_dist(score, geneset_name, score_name, save_dir):
    score = np.nan_to_num(score)
    # get mixtures and thresholds
    distributions = find_distribution(score, geneset_name)

    localizer_gmm = group_distributions(distributions, method="gmm")
    localizer_kmeans = group_distributions(distributions, method="kmeans")

    thresholds_gmm = find_grouped_dist_thresholds(
        distributions, localizer_gmm, score, geneset_name
    )

    thresholds_kmeans = find_grouped_dist_thresholds(
        distributions, localizer_kmeans, score, geneset_name
    )

    if np.var(score) != 0:
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
    return (
        thresholds_gmm,
        thresholds_kmeans,
        distributions,
        localizer_gmm,
        localizer_kmeans,
    )


def evaluate_pas(scores, gs_names, geneset_info, res_folder, data_type, score_name):
    scores_thr = pd.DataFrame(0, index=gs_names, columns=np.arange(1))
    scores_thr_kmeans = pd.DataFrame(0, index=gs_names, columns=np.arange(1))
    scores_dist = []
    gmm_thrs = {}
    kmeans_thrs = {}
    locs_gmm = []
    locs_kmeans = []

    for i, gs_name in tqdm(enumerate(gs_names), total=len(gs_names)):
        gs_title = geneset_info.Title.where(geneset_info.ID == gs_name, gs_name).max()
        score = scores[i, :]
        (
            thresholds_gmm,
            thresholds_kmeans,
            distributions,
            localizer_gmm,
            localizer_kmeans,
        ) = pipeline_for_dist(score, gs_title, score_name, save_dir)
        del distributions["TIC"], distributions["l_lik"]
        distributions["weights"] = (distributions["weights"]).tolist()
        distributions["mu"] = (distributions["mu"]).tolist()
        distributions["sigma"] = (distributions["sigma"]).tolist()

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

        localizer_gmm = localizer_gmm.tolist()
        locs_gmm.append(localizer_gmm)
        localizer_kmeans = localizer_kmeans.tolist()
        locs_kmeans.append(localizer_kmeans)

    scores_thr.to_csv(res_folder + data_type + "/gmm_thr/" + score_name + "_thr.csv")
    scores_thr_kmeans.to_csv(
        res_folder + data_type + "/kmeans_thr/" + score_name + "_thr.csv"
    )

    with open(
        res_folder + data_type + "/gmm_thr/" + score_name + "_loc.json",
        "w",
    ) as fout:
        json.dump(locs_gmm, fout)
    with open(
        res_folder + data_type + "/kmeans_thr/" + score_name + "_loc.json",
        "w",
    ) as fout:
        json.dump(locs_kmeans, fout)

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
    for score_name in tqdm(score_names):
        print(score_name)
        # get scores
        scores = pd.read_csv(
            res_folder + data_type + "/" + score_name + ".csv",
            index_col=0,
        )
        gs_names = scores.index.values.tolist()
        scores = scores.to_numpy()

        evaluate_pas(scores, gs_names, geneset_info, res_folder, data_type, score_name)

        if score_name in ["svd", "sparse_pca", "z", "vision"]:
            scores = np.abs(scores)
            evaluate_pas(
                scores,
                gs_names,
                geneset_info,
                res_folder,
                data_type,
                score_name+"_abs",
            )
