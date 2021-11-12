import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def show_labels(true_labels, predicted, binary_predicted, tsne, geneset_name, score_name="", save_dir="plots"):
    binary_colors ={"Non significant": "#0C5AA6", "Significant": "#FF9700"}
    if predicted is None:
        f, ax = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(10,10))
    else:
        f, ax = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(10,15))
        sns.scatterplot(ax=ax[1], x=tsne[0,:], y=tsne[1,:], hue=predicted)
        ax[1].set_title('Underlying groups')
        ax[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    # True labels   
    sns.scatterplot(ax=ax[0], x=tsne[0,:], y=tsne[1,:], hue=true_labels)
    ax[0].set_title('Original labels')
    ax[0].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    # binary labels
    binary_predicted = ["Non significant" if i == 0 else "Significant" for i in binary_predicted]
    sns.scatterplot(ax=ax[-1], x=tsne[0,:], y=tsne[1,:], hue=binary_predicted, palette=binary_colors)
    ax[-1].set_title('Significance')
    ax[-1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.xlabel('tsne-1')
    plt.ylabel('tsne-2')
    f.suptitle(geneset_name)
    plt.savefig(save_dir+"/"+geneset_name + "_" + score_name + ".png")
    
def show_classification(predicted, binary_predicted, tsne, geneset_name, score_name="", save_dir="plots"):
    # binary labels
    binary_predicted = ["Non significant" if i == 0 else "Significant" for i in binary_predicted]
    binary_colors ={"Non significant": "#0C5AA6", "Significant": "#FF9700"}
    if predicted is None:
        f, ax = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(10,5))
        sns.scatterplot(ax=ax, x=tsne[0,:], y=tsne[1,:], hue=binary_predicted, palette=binary_colors)
        ax.set_title('Significance')
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        
    else:
        f, ax = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(10,10))
        sns.scatterplot(ax=ax[0], x=tsne[0,:], y=tsne[1,:], hue=predicted)
        ax[0].set_title('Underlying groups')
        ax[0].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        sns.scatterplot(ax=ax[1], x=tsne[0,:], y=tsne[1,:], hue=binary_predicted, palette=binary_colors)
        ax[1].set_title('Significance')
        ax[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.xlabel('tsne-1')
    plt.ylabel('tsne-2')
    f.suptitle(geneset_name)   
    plt.savefig(save_dir+"/"+geneset_name + "_" + score_name + ".png")
    
def plot_results(binary_predicted, tsne, geneset_name, predicted=None, true_labels=None, score_name=""):
    if true_labels is None:
        show_classification(predicted, binary_predicted, tsne, geneset_name, score_name)
    else:
        show_labels(true_labels, predicted, binary_predicted, tsne, geneset_name, score_name)
        
def show_difference(score1_predicted, score2_predicted, tsne, geneset_name, true_labels=None, score_name1="score1", score_name2="score2", save_dir="plots"):
    both_significant = np.bitwise_and(score2_predicted, score1_predicted).astype(bool)
    score_2_sign = np.bitwise_and(score2_predicted, 1-np.asarray(score1_predicted)).astype(bool)
    score_1_sign = np.bitwise_and(1-score2_predicted, np.asarray(score1_predicted)).astype(bool)
    none_sign = np.bitwise_and(1-score2_predicted, 1-np.asarray(score1_predicted)).astype(bool)
    
    if true_labels is None:
        nrows = 2
        titles = ["", ""]
    else:
        nrows=3
        titles = ["Original", "", ""]
    fig, big_axes = plt.subplots( figsize=(12, nrows*4) , nrows=nrows, ncols=1, sharey=True) 

    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title(titles[row-1], fontsize=16)

        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        big_ax._frameon = False
    
    gs = fig.add_gridspec(nrows,2)
        
    if nrows == 3:
        ax0 = fig.add_subplot(gs[0, :])
        sns.scatterplot(ax=ax0, x=tsne[0,:], y=tsne[1,:], hue=true_labels)
        ax0.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        ax1 = fig.add_subplot(gs[nrows-2, 0], sharey = ax0, sharex = ax0)
    else:
        ax1 = fig.add_subplot(gs[nrows-2, 0])
        ax1.set_xlim(np.min(tsne, axis=1)[0]-1, np.max(tsne, axis=1)[0]+1)
        ax1.set_ylim(np.min(tsne, axis=1)[1]-1, np.max(tsne, axis=1)[1]+1)
    sns.scatterplot(ax=ax1, x=tsne[0, both_significant],
                    y=tsne[1, both_significant], color="green")


    ax2 = fig.add_subplot(gs[nrows-2, 1], sharey = ax1, sharex = ax1)
    sns.scatterplot(ax=ax2, x=tsne[0, score_1_sign],
                y=tsne[1, score_1_sign], color="red")

    ax3 = fig.add_subplot(gs[nrows-1, 0], sharey = ax1, sharex = ax1)
    sns.scatterplot(ax=ax3, x=tsne[0, score_2_sign],
                y=tsne[1, score_2_sign], color="red")


    ax4 = fig.add_subplot(gs[nrows-1, 1], sharey = ax1, sharex = ax1)
    sns.scatterplot(ax=ax4, x=tsne[0, none_sign],
                y=tsne[1, none_sign], color="green")

    ax3.set_title("Significant " + score_name2)
    ax1.set_title("Both significant")
    ax2.set_title("Significant " + score_name1)
    ax4.set_title("Neither")

    fig.set_facecolor('w')
    plt.suptitle("GS", fontsize=25)
    plt.tight_layout()
    plt.savefig(save_dir+"/"+geneset_name + "_" + score_name1 + "_"+score_name2 + ".png")

def show_significance(score1_predicted, score2_predicted, tsne, geneset_name, true_labels=None, score_name1="score1", score_name2="score2", save_dir="plots"):
    s1_sign = np.asarray(score1_predicted).astype(bool)
    s2_sign = np.asarray(score2_predicted).astype(bool)
    s1_nsign = (1-np.asarray(score1_predicted)).astype(bool)
    s2_nsign = (1-np.asarray(score2_predicted)).astype(bool)
    
    if true_labels is None:
        nrows = 2
        titles = [score_name1, score_name2]
    else:
        nrows=3
        titles = ["Original", score_name1, score_name2]
        
    fig, big_axes = plt.subplots( figsize=(12, nrows*4) , nrows=nrows, ncols=1, sharey=True) 

    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title(titles[row-1], fontsize=16)

        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        big_ax._frameon = False
    
    gs = fig.add_gridspec(nrows,2)
        
    if nrows == 3:
        ax0 = fig.add_subplot(gs[0, :])
        sns.scatterplot(ax=ax0, x=tsne[0,:], y=tsne[1,:], hue=true_labels)
        ax0.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        ax1 = fig.add_subplot(gs[nrows-2, 0], sharey = ax0, sharex = ax0)
    else:
        ax1 = fig.add_subplot(gs[nrows-2, 0])
        ax1.set_xlim(np.min(tsne, axis=1)[0]-1, np.max(tsne, axis=1)[0]+1)
        ax1.set_ylim(np.min(tsne, axis=1)[1]-1, np.max(tsne, axis=1)[1]+1)
    sns.scatterplot(ax=ax1, x=tsne[0, s1_sign],
                    y=tsne[1, s1_sign], color="#FF9700")


    ax2 = fig.add_subplot(gs[nrows-2, 1], sharey = ax1, sharex = ax1)
    sns.scatterplot(ax=ax2, x=tsne[0, s1_nsign],
                y=tsne[1, s1_nsign], color="#0C5AA6")

    ax3 = fig.add_subplot(gs[nrows-1, 0], sharey = ax1, sharex = ax1)
    sns.scatterplot(ax=ax3, x=tsne[0, s2_sign],
                y=tsne[1, s2_sign], color="#FF9700")


    ax4 = fig.add_subplot(gs[nrows-1, 1], sharey = ax1, sharex = ax1)
    sns.scatterplot(ax=ax4, x=tsne[0, s2_nsign],
                y=tsne[1, s2_nsign], color="#0C5AA6")

    ax3.set_title("Significant")
    ax1.set_title("Significant")
    ax2.set_title("Non significant")
    ax4.set_title("Non significant")

    fig.set_facecolor('w')
    plt.suptitle("GS", fontsize=25)
    plt.tight_layout()
    plt.savefig(save_dir+"/"+geneset_name + "_" + score_name1 + "_"+score_name2 + ".png")