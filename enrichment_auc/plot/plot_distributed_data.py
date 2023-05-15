import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import jaccard_score

from enrichment_auc.distributions import categorize_by_thresholds

from .legend_handler import move_legend


def compare_mixture_cat(
    geneset_name,
    distributions,
    score1,
    thr1,
    thrs1,
    name1,
    score2,
    thr2,
    name2,
    save_dir="plots",
    file_only=True,
):
    comp_colors_density = "#808080"
    comp_colors = {"Non significant": "#0C5AA6", "Significant": "#FF9700"}
    label_meaning = ["Non significant", "Significant"]
    f = plt.figure(figsize=(7, 10))

    # get binary labels for score 1
    binary_labels = (score1 > thr1).astype(int)
    ns_count1 = len(binary_labels) - sum(binary_labels)
    s_count1 = sum(binary_labels)
    # get binary labels for score 2
    score_2 = (score2 > thr2).astype(int)
    ns_count2 = len(score_2) - sum(score_2)
    s_count2 = sum(score_2)
    # get jaccard index and code the labels
    jaccard_index = jaccard_score(score_2, binary_labels)
    binary_labels = [label_meaning[label] for label in binary_labels.tolist()]
    score_2 = [label_meaning[label] for label in score_2.tolist()]

    # SCORE 1 visualization
    # label subplot
    ax1 = plt.subplot(211)
    ax1.set_xlabel(name1)
    ax1.set_ylabel("Frequency")
    # add binary rugplot
    df = pd.DataFrame([score1, binary_labels]).T
    df.columns = ["score", "labels"]
    _ = sns.rugplot(
        df, x="score", height=-0.02, clip_on=False, hue="labels", palette=comp_colors
    )
    # show density plot and histogram
    _ = sns.histplot(score1, color="#B0B0B0", alpha=0.25, kde=False, stat="density")
    # _ = sns.kdeplot(score1, linewidth=2, color='k')
    move_legend(ax1, "upper right", [ns_count1, s_count1])
    # add the distributions
    if not np.isnan(np.sum(distributions["mu"])):
        for i in range(distributions["mu"].shape[0]):
            mu = distributions["mu"][i]
            sigma = distributions["sigma"][i]
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            vals = stats.norm.pdf(x, mu, sigma) * distributions["weights"][i]
            plt.plot(x, vals, linestyle="dashed", color=comp_colors_density)
    # add the rest of thresholds - rugplot
    if thrs1.shape[0] > 1:
        labels = categorize_by_thresholds(score1, thrs1)
        df = pd.DataFrame([score1, labels]).T
        df.columns = ["score", "labels"]
        _ = sns.rugplot(df, x="score", clip_on=False, hue="labels", legend=False)
    ax1.set_ylim(bottom=0)

    # SCORE 2 visualization
    # label subplot
    ax2 = plt.subplot(212)
    ax2.set_xlabel(name2)
    ax2.set_ylabel("Frequency")
    # mark threshold of score 2
    plt.axvline(x=thr2, c="r", linestyle="dashed")
    # add density plot, histogram and rugplot
    _ = sns.histplot(
        score2, color="#B0B0B0", alpha=0.25, ax=ax2, kde=False, stat="density"
    )
    _ = sns.kdeplot(score2, linewidth=2, color=comp_colors_density, ax=ax2)
    df = pd.DataFrame([score2, score_2]).T
    df.columns = ["score", "labels"]
    _ = sns.rugplot(
        df,
        x="score",
        height=-0.02,
        clip_on=False,
        hue="labels",
        palette=comp_colors,
        ax=ax2,
    )

    # Finishing touches
    move_legend(ax2, "upper right", [ns_count2, s_count2])

    f.set_facecolor("w")
    plt.suptitle(geneset_name, fontsize=18)
    plt.title("Jaccard index: " + str(jaccard_index), fontsize=10)
    geneset_name = geneset_name.replace("/", "_")
    geneset_name = geneset_name.replace(":", "_")
    plt.savefig(save_dir + "/dens_" + geneset_name + "_" + name1 + "_" + name2 + ".png")
    if file_only:
        plt.close()


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
    _ = sns.histplot(score, color="#B0B0B0", alpha=0.25, kde=False, stat="density")
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
