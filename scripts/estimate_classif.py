import os
import sys
import pandas as pd

from enrichment_auc.evaluate_classification import Scores


scorenames = [
    "aucell",
    "auc",
    "cerno",
    "jasmine",
    "ratios",
    "mean",
    "vision",
    "vision_abs",
    "z",
    "z_abs",
    "svd",
    "svd_abs",
    "sparse_pca",
    "sparse_pca_abs",
    "gsva",
    "ssgsea",
    "vae",
    "vae_corr",
]

names = [
    "Balanced_accuracy_",
    "ARI_",
    "F1_",
    "Recall_",
    "Matthews_",
    "Jaccard_",
    "Hamming_",
    "Precision_",
    "FDR_",
]


if __name__ == "__main__":
    tissue = sys.argv[1]
    norm = sys.argv[2]
    clustertype = sys.argv[3]
    datafolder = sys.argv[4]
    resfolder = sys.argv[5] + tissue + "/" + norm + "/"

    plottype = clustertype
    if clustertype == "gmm":
        plottype = "top1"

    if not os.path.isdir(resfolder + "confusion_matrix_" + plottype):
        os.makedirs(resfolder + "confusion_matrix_" + plottype)

    paths = pd.read_csv(datafolder + "chosen_paths.txt", sep="\t", index_col=0)
    geneset_info = pd.read_csv(
        datafolder + tissue + "/filtered_genesets_modules.csv", index_col=0
    )
    dataset_specific = paths[paths["ID"].isin(geneset_info["ID"])]
    dataset_specific.loc[:, "Celltype"] = dataset_specific["Celltype"].str.replace(
        ";", " +"
    )
    to_save = dataset_specific[["ID", "Title", "Celltype"]]

    true_labels = pd.read_csv(datafolder + tissue + "/true_labels.csv", index_col=0)
    not_pre_B = ~true_labels["CellType"].isin(["precursor B cell", "pro-B cell"])
    true_labels = true_labels[not_pre_B]
    true_labels.loc[
        true_labels["CellType"].isin(["CD4+ T cell", "Cytotoxic T cell"]), "CellType"
    ] = "T cell"
    true_labels.loc[
        true_labels["CellType"].isin(["mature B cell"]), "CellType"
    ] = "B cell"
    true_labels.loc[
        true_labels["CellType"].isin(["Natural killer cell", "natural killer cell"]),
        "CellType",
    ] = "NK cell"
    true_labels.loc[
        ~true_labels["CellType"].isin(["NK cell", "T cell", "B cell"]), "CellType"
    ] = "other"

    to_save_ = to_save[
        to_save.Celltype.str.contains("|".join(true_labels.CellType.unique()))
    ]

    for scorename in scorenames:
        if scorename.endswith("_abs"):
            scores = pd.read_csv(resfolder + scorename[:-4] + ".csv", index_col=0)
            scores = scores.abs()
        else:
            scores = pd.read_csv(resfolder + scorename + ".csv", index_col=0)
        scores = scores.loc[:, not_pre_B]
        thresholds = pd.read_csv(
            resfolder + "/" + clustertype + "_thr/" + scorename + "_thr.csv",
            index_col=0,
        )
        eval = Scores()
        for index, row in to_save_.iterrows():
            gs_score = scores.loc[row["ID"]]
            thr = thresholds.loc[row["ID"]].max()
            preds = gs_score > thr
            preds = preds.astype(int)
            true_labels["label"] = true_labels.CellType.isin(
                row["Celltype"].split(" + ")
            ).astype(int)
            eval.get_classification_scores(true_labels["label"], preds)
            eval.save_confusion_matrix(
                true_labels["label"], preds, resfolder, plottype, scorename, row["ID"]
            )
        for i, cls_score in enumerate(eval.scores):
            to_save_.loc[:, eval.names[i] + scorename] = cls_score

    to_save_.to_csv(resfolder + "classification_scores_" + plottype + ".csv")
