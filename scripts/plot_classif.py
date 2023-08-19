import os
from itertools import combinations

import pandas as pd

from enrichment_auc.plot.plot_boxplots import visualize_difference, visualize_methods

datafolder = "data/"
resfolder = "results/"
plotfolder = "plots/"

tissues = ["PBMC", "COVID", "BM", "Liver"]
norm = "seurat"
plottypes = ["top1", "kmeans", "AUCell"]

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
data = {}
merged = {}
cols = ["ID", "Title", "Celltype"]

for tissue in tissues:
    print(tissue)
    data[tissue] = {}
    for plottype in plottypes:
        # do violin plots of each thr method per dataset
        plot_folder = (
            plotfolder + tissue + "/" + norm + "/" + plottype + "/classification/"
        )
        res_folder = resfolder + tissue + "/" + norm + "/"
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)
        df = pd.read_csv(
            res_folder + "classification_scores_" + plottype + ".csv", index_col=0
        )
        data[tissue][plottype] = df
        groups = df.groupby(["Celltype"])["Celltype"].count()
        celltypes = groups[groups > 2].keys().tolist()
        visualize_methods(df, celltypes, names, plot_folder)
    # do violin plots of differences between thr methods per dataset
    for x, y in list(combinations(plottypes, 2)):
        plot_folder = plotfolder + tissue + "/" + norm + "/" + x + "_" + y + "/"
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)
        for celltype in celltypes:
            subfolder = plot_folder + celltype + "/"
            if not os.path.isdir(subfolder):
                os.makedirs(subfolder)
            df1 = data[tissue][x].loc[data[tissue][x]["Celltype"] == celltype]
            df2 = data[tissue][y].loc[data[tissue][y]["Celltype"] == celltype]
            visualize_difference(
                df1.drop(columns=cols), df2.drop(columns=cols), names, subfolder, x, y
            )
        visualize_difference(
            data[tissue][x].drop(columns=cols),
            data[tissue][y].drop(columns=cols),
            names,
            plot_folder,
            x,
            y)

print("merged")
# do violin plots of each thr method per whole
for plottype in plottypes:
    plot_folder = plotfolder + "/merged/" + norm + "/" + plottype + "/"
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    merged[plottype] = pd.concat([data[tissue][plottype] for tissue in tissues])
    groups = merged[plottype].groupby(["Celltype"])["Celltype"].count()
    celltypes = groups[groups > 2].keys().tolist()
    visualize_methods(merged[plottype], celltypes, names, plot_folder)
    df = merged[plottype]
    df["Celltype"] = "merged"
    visualize_methods(df, ["merged"], names, plot_folder)

# do violin plots of differences between thr methods per whole
for x, y in list(combinations(plottypes, 2)):
    plot_folder = plotfolder + "/merged/" + norm + "/" + x + "_" + y + "/"
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    for celltype in celltypes:
        subfolder = plot_folder + celltype + "/"
        if not os.path.isdir(subfolder):
            os.makedirs(subfolder)
        df1 = merged[x].loc[merged[x]["Celltype"] == celltype]
        df2 = merged[y].loc[merged[y]["Celltype"] == celltype]
        visualize_difference(
            df1.drop(columns=cols), df2.drop(columns=cols), names, subfolder, x, y
        )
    visualize_difference(
        merged[x].drop(columns=cols),
        merged[y].drop(columns=cols),
        names,
        plot_folder,
        x,
        y,
    )
