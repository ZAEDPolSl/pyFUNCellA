import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os.path
from .legend_handler import move_legend

def visualize_scatter_for_cat(gs_name, score1, score2, thr1, thr2, name1, name2, true_labels, save_dir):
    if thr2 is not None:
        compare_2_categorical_scatter(score1, thr1, name1, gs_name, 
                                   score2, thr2, name2, 
                                   save_dir=save_dir, file_only=True)
    else:
        plot_cat_scatter(score1, thr1, name1, gs_name, save_dir=save_dir)
        
def get_confusion_matrix(y_pred, y_true, title, save_dir = "plots/", predicted_name='Predicted values'):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (1, 1):
        if y_pred.sum() == 0 and y_true.sum() == 0:
            cm  = np.pad(cm, (0, 1), mode='constant')
        else:
            cm  = np.pad(cm, (1,0), mode='constant')
            
    tn, fp, fn, tp = cm.ravel()
    df = pd.DataFrame(data={'TN': [tn], 'FP': [fp], 'FN': [fn], 'TP':[tp]}, index=[title])
    # this part will be useful        
    filename = "confusion_matrix_" + predicted_name + ".csv"
    if os.path.isfile(save_dir+filename):
        df.to_csv(save_dir+filename, mode='a', index=True, header=False)
    else:
        print("creating confusion matrix file")
        df.to_csv(save_dir+filename, mode='w', index=True, header=True)
        
def show_significance(score1, thr1, score2, thr2,
                      tsne, geneset_name, 
                      true_labels=None, 
                      score_name1="score1", score_name2="score2", 
                      save_dir="plots", file_only=True, take_smaller_1=True):
    
    comp_colors ={"Non significant": "#0C5AA6", "Significant": "#FF9700"}
    label_meaning = ["Non significant", "Significant"]
    
    # score 1
    if (take_smaller_1):
        score_1 = (score1 < thr1).astype(int)
    else:
        score_1 = (score1 > thr1).astype(int)

    s1_sign = np.asarray(score_1).astype(bool)
    s_count1 =  s1_sign.sum()
    s1_nsign = (1-s1_sign).astype(bool)
    ns_count1 = s1_nsign.sum()
    
    # score 2
    score_2 = (score2 > thr2).astype(int)
    
    s2_sign = np.asarray(score_2).astype(bool)
    s_count2 =  s2_sign.sum()
    s2_nsign = (1-s2_sign).astype(bool)
    ns_count2 = s2_nsign.sum()

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
        sns.scatterplot(ax=ax0, x=tsne[:,0], y=tsne[:,1], hue=true_labels)
        ax0.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        ax1 = fig.add_subplot(gs[nrows-2, 0], sharey = ax0, sharex = ax0)
    else:
        ax1 = fig.add_subplot(gs[nrows-2, 0])
        ax1.set_xlim(np.min(tsne, axis=1)[0]-1, np.max(tsne, axis=1)[0]+1)
        ax1.set_ylim(np.min(tsne, axis=1)[1]-1, np.max(tsne, axis=1)[1]+1)
    _ = sns.scatterplot(ax=ax1, x=tsne[s1_sign, 0],
                    y=tsne[s1_sign, 1], color="#FF9700")


    ax2 = fig.add_subplot(gs[nrows-2, 1], sharey = ax1, sharex = ax1)
    _ = sns.scatterplot(ax=ax2, x=tsne[s1_nsign, 0],
                y=tsne[s1_nsign, 1], color="#0C5AA6")

    ax3 = fig.add_subplot(gs[nrows-1, 0], sharey = ax1, sharex = ax1)
    _ = sns.scatterplot(ax=ax3, x=tsne[s2_sign, 0],
                y=tsne[s2_sign, 1], color="#FF9700")


    ax4 = fig.add_subplot(gs[nrows-1, 1], sharey = ax1, sharex = ax1)
    _ = sns.scatterplot(ax=ax4, x=tsne[s2_nsign, 0],
                y=tsne[s2_nsign, 1], color="#0C5AA6")

    # score 1
    ax1.set_title("Significant: " + str(s_count1))
    ax2.set_title("Non significant: "+ str(ns_count1))
    # score 2
    ax3.set_title("Significant: "+ str(s_count2))
    ax4.set_title("Non significant: "+ str(ns_count2))

    fig.set_facecolor('w')
    plt.suptitle(geneset_name, fontsize=25)
    plt.tight_layout()
    geneset_name = geneset_name.replace("/", "_")
    geneset_name = geneset_name.replace(":", "_")
    plt.savefig(save_dir+geneset_name + "_" + score_name1 + "_"+score_name2 + ".png")
    if file_only:
        plt.close()
        
def compare_cat_true_scatter(score1, thr1, score2, thr2,
                      tsne, geneset_name, 
                      true_labels, true_labels_multiclass=None,
                      score_name1="score1", score_name2="score2", 
                      save_dir="plots", file_only=True, take_smaller_1=True):
    
    comp_colors ={"Non significant": "#0C5AA6", "Significant": "#FF9700"}
    label_meaning = ["Non significant", "Significant"]
    
    # score 1
    if (take_smaller_1):
        score_1 = (score1 < thr1).astype(int)
    else:
        score_1 = (score1 > thr1).astype(int)
        
    s1_sign = np.asarray(score_1).astype(bool)
    s_count1 =  s1_sign.sum()
    s1_nsign = (1-s1_sign).astype(bool)
    ns_count1 = s1_nsign.sum()
    
    # score 2
    score_2 = (score2 > thr2).astype(int)
    
    s2_sign = np.asarray(score_2).astype(bool)
    s_count2 =  s2_sign.sum()
    s2_nsign = (1-s2_sign).astype(bool)
    ns_count2 = s2_nsign.sum()
    
    get_confusion_matrix(s1_sign, true_labels, geneset_name, save_dir = save_dir, predicted_name=score_name1)
    get_confusion_matrix(s2_sign, true_labels, geneset_name, save_dir = save_dir, predicted_name=score_name2)
    
    if true_labels_multiclass is not None:
        nrows=4
        titles = ["Original", "True classes", score_name1, score_name2]
    else:
        nrows=3
        titles = ["True classes", score_name1, score_name2]
        
    fig, big_axes = plt.subplots( figsize=(12, nrows*4) , nrows=nrows, ncols=1, sharey=True) 

    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title(titles[row-1], fontsize=16)

        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        big_ax._frameon = False
    
    gs = fig.add_gridspec(nrows,2)
        
        
    # multiclass true 
    # true labels
    ax0_1 = fig.add_subplot(gs[nrows-3, 0])
    _ = sns.scatterplot(ax=ax0_1, x=tsne[np.asarray(true_labels), 0],
                        y=tsne[np.asarray(true_labels), 1], color="#FF9700")
    
    ax0_2 = fig.add_subplot(gs[nrows-3, 1], sharey = ax0_1, sharex = ax0_1)
    _ = sns.scatterplot(ax=ax0_2, x=tsne[(1-np.asarray(true_labels)).astype(bool), 0],
                        y=tsne[(1-np.asarray(true_labels)).astype(bool), 1], color="#0C5AA6")
    
    # scores
    ax1 = fig.add_subplot(gs[nrows-2, 0], sharey = ax0_1, sharex = ax0_1)
    _ = sns.scatterplot(ax=ax1, x=tsne[s1_sign, 0],
                        y=tsne[s1_sign, 1], color="#FF9700")

    ax2 = fig.add_subplot(gs[nrows-2, 1], sharey = ax1, sharex = ax1)
    _ = sns.scatterplot(ax=ax2, x=tsne[s1_nsign, 0],
                y=tsne[s1_nsign, 1], color="#0C5AA6")

    ax3 = fig.add_subplot(gs[nrows-1, 0], sharey = ax1, sharex = ax1)
    _ = sns.scatterplot(ax=ax3, x=tsne[s2_sign, 0],
                y=tsne[s2_sign, 1], color="#FF9700")


    ax4 = fig.add_subplot(gs[nrows-1, 1], sharey = ax1, sharex = ax1)
    _ = sns.scatterplot(ax=ax4, x=tsne[s2_nsign, 0],
                y=tsne[s2_nsign, 1], color="#0C5AA6")
    
    if nrows == 4:
        ax0 = fig.add_subplot(gs[0, :], sharey = ax1, sharex = ax1)
        sns.scatterplot(ax=ax0, x=tsne[:,0], y=tsne[:,1], hue=true_labels_multiclass)
        ax0.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    # true labels
    ax0_1.set_title("Significant: " + str(sum(true_labels)))
    ax0_2.set_title("Non significant: "+ str(len(true_labels) - sum(true_labels)))
    # score 1
    ax1.set_title("Significant: " + str(s_count1))
    ax2.set_title("Non significant: "+ str(ns_count1))
    # score 2
    ax3.set_title("Significant: "+ str(s_count2))
    ax4.set_title("Non significant: "+ str(ns_count2))

    fig.set_facecolor('w')
    plt.suptitle(geneset_name, fontsize=25)
    plt.tight_layout()
    geneset_name = geneset_name.replace("/", "_")
    geneset_name = geneset_name.replace(":", "_")
    plt.savefig(save_dir+geneset_name + "_" + score_name1 + "_"+score_name2 + ".png")
    if file_only:
        plt.close()