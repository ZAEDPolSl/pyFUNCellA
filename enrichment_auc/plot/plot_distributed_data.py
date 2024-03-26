import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from enrichment_auc.gmm.thresholds import categorize_by_thresholds

from .legend_handler import move_legend


def plot_mixtures(
    geneset_name,
    distributions,
    score,
    thr,
    thrs,
    name,
    save_dir="plots",
    file_only=True,
):
    comp_colors_density = "#808080"
    comp_colors = {"Non significant": "#0C5AA6", "Significant": "#FF9700"}
    label_meaning = ["Non significant", "Significant"]
    f, ax = plt.subplots(figsize=(7, 5))
    # get binary labels
    binary_labels = (score > thr).astype(int)
    ns_count = len(binary_labels) - sum(binary_labels)
    s_count = sum(binary_labels)
    binary_labels = [label_meaning[label] for label in binary_labels.tolist()]
    df = pd.DataFrame([score, binary_labels]).T
    df.columns = ["score", "labels"]
    # label plot
    plt.ylabel("Frequency")
    plt.xlabel(name)
    plt.title(geneset_name, fontsize=18)
    # add binary rugplot
    _ = sns.rugplot(
        df, x="score", height=-0.02, clip_on=False, hue="labels", palette=comp_colors
    )
    # show density plot and histogram
    try:
        _ = sns.histplot(score, color="#B0B0B0", alpha=0.25, kde=False, stat="density")
    except (ValueError, np.core._exceptions._ArrayMemoryError, OverflowError):
        sns.histplot(
            x=score,
            element="step",
            bins="sturges",
            color="#B0B0B0",
            alpha=0.25,
            kde=False,
            stat="density",
        )
    # _ = sns.kdeplot(score, linewidth=2, color='k')
    move_legend(ax, "upper right", [ns_count, s_count])
    # add the distributions
    for i in range(distributions["mu"].shape[0]):
        mu = distributions["mu"][i]
        sigma = distributions["sigma"][i]
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        vals = stats.norm.pdf(x, mu, sigma) * distributions["weights"][i]
        plt.plot(x, vals, linestyle="dashed", color=comp_colors_density)
    # add the rest of thresholds - rugplot
    if thrs.shape[0] > 1:
        labels = categorize_by_thresholds(score, thrs)
        df = pd.DataFrame([score, labels]).T
        df.columns = ["score", "labels"]
        _ = sns.rugplot(df, x="score", clip_on=False, hue="labels", legend=False)
    # finishing touches
    ax.set_ylim(bottom=0)
    f.set_facecolor("w")
    geneset_name = geneset_name.replace("/", "_")
    geneset_name = geneset_name.replace(":", "_")
    plt.savefig(save_dir + "/dens_" + geneset_name + "_" + name + ".png")
    if file_only:
        plt.close()
