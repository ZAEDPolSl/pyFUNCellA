import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .legend_handler import move_legend

def plot_cat_density(data, thr, score_name, geneset_name, save_dir="plots", file_only=True, take_smaller=True):
    # take_smaller = True for p-values 
    comp_colors ={"Non significant": "#0C5AA6", "Significant": "#FF9700"}
    label_meaning = ["Non significant", "Significant"]
    f, ax = plt.subplots(figsize=(7,5))
    
    if (take_smaller):
        binary_labels = (data < thr).astype(int)
    else:
        binary_labels = (data > thr).astype(int)
        
    ns_count = (len(binary_labels)-sum(binary_labels))
    s_count =  sum(binary_labels)
    binary_labels = [label_meaning[label] for label in binary_labels.tolist()]
    
    comp_colors_density = "#808080"        

    _ = sns.histplot(data, color= "#B0B0B0", alpha=0.25, kde=False, stat='density')
    _ = sns.kdeplot(data, linewidth=2, color=comp_colors_density)
        
    plt.ylabel('Frequency')
    plt.xlabel(score_name)
    plt.title(geneset_name, fontsize=18)
        
    _ = sns.rugplot(data, height=-.02, clip_on=False, hue=binary_labels,
                palette = comp_colors)
    
    move_legend(ax, "upper right", [ns_count, s_count])
    
    f.set_facecolor('w')
    geneset_name = geneset_name.replace("/", "_")
    geneset_name = geneset_name.replace(":", "_")
    plt.savefig(save_dir+"/dens_"+geneset_name + "_" + score_name + ".png")
    if file_only:
        plt.close()
        
def compare_2_categorical(score1, thr1, score_name1, geneset_name, score2, thr2, score_name2, save_dir="plots", file_only=True, take_smaller_1=True):
    comp_colors ={"Non significant": "#0C5AA6", "Significant": "#FF9700"}
    label_meaning = ["Non significant", "Significant"]
    f = plt.figure(figsize=(7,10))
    
    comp_colors_density = "#808080"     

    # score 1
    if (take_smaller_1):
        score_1 = (score1 < thr1).astype(int)
    else:
        score_1 = (score1 > thr1).astype(int)
        
    ns_count1 = (len(score_1)-sum(score_1))
    s_count1 =  sum(score_1)
    score_1 = [label_meaning[label] for label in score_1.tolist()]
    
    ax1 = plt.subplot(211)
    ax1.set_xlabel(score_name1)
    ax1.set_ylabel("Frequency")
    plt.axvline(x=thr1, c="r", linestyle="dashed")
    _ = sns.histplot(score1, color= "#B0B0B0", ax=ax1, alpha=0.25, kde=False, stat='density')
    _ = sns.kdeplot(score1, linewidth=2, color=comp_colors_density, ax=ax1)
    _ = sns.rugplot(score1, height=-.02, clip_on=False, hue=score_1,
                    palette = comp_colors, ax=ax1)

    # score 2
    score_2 = (score2 > thr2).astype(int)
    ns_count2 = (len(score_2)-sum(score_2))
    s_count2 =  sum(score_2)
    score_2 = [label_meaning[label] for label in score_2.tolist()]

    ax2 = plt.subplot(212)
    ax2.set_xlabel(score_name2)
    ax2.set_ylabel("Frequency")
    plt.axvline(x=thr2, c="r", linestyle="dashed")
    _ = sns.histplot(score2, color= "#B0B0B0", alpha=0.25, ax=ax2, kde=False, stat='density')
    _ = sns.kdeplot(score2, linewidth=2, color=comp_colors_density, ax=ax2)
    _ = sns.rugplot(score2, height=-.02, clip_on=False, hue=score_2,
                palette = comp_colors, ax=ax2)
    
    move_legend(ax1, "upper right", [ns_count1, s_count1])
    move_legend(ax2, "upper right", [ns_count2, s_count2])
    
    f.set_facecolor('w')
    plt.suptitle(geneset_name, fontsize=18)
    geneset_name = geneset_name.replace("/", "_")
    geneset_name = geneset_name.replace(":", "_")
    plt.savefig(save_dir+"/dens_"+geneset_name + "_" + score_name1 + "_"+score_name2 + ".png")
    if file_only:
        plt.close()
        
def visualize_dist_for_cat(gs_name, score1, score2, thr1, thr2, name1, name2, save_dir):
    if thr2 is not None:
        compare_2_categorical(score1, thr1, name1, gs_name, 
                                   score2, thr2, name2, 
                                   save_dir=save_dir, file_only=True)
    else:
        plot_cat_density(score1, thr1, name1, gs_name, save_dir=save_dir)