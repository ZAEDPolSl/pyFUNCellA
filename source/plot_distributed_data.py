import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import seaborn as sns
from .model import get_predictions

def plot_densities(data, model, comp_groups, score_name, geneset_name, save_dir="plots"):
    comp_colors ={"Non significant": "#0C5AA6", "Significant": "#FF9700"}
    label_meaning = ["Non significant", "Significant"]
    plt.figure(figsize=(7,5))

    labels, binary_labels = get_predictions(model, comp_groups, data)
    comp_colors_density = "#808080"
    binary_labels = [label_meaning[label] for label in binary_labels ]        
        
    sns.kdeplot(data, linewidth=2, color='k')
    
    if labels is not None:
        sns.rugplot(data, clip_on=False, hue=labels)    
        
    plt.ylabel('Frequency')
    plt.xlabel(score_name)
    plt.title(geneset_name)
    
    for i in range(model.n_components):
        mu = model.means_[i, 0]
        variance = model.covariances_[i, 0]
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        vals = stats.norm.pdf(x, mu, sigma)*model.weights_[i]
        plt.plot(x, vals, linestyle="dashed", color=comp_colors_density)
        
    sns.rugplot(data, height=-.02, clip_on=False, hue=binary_labels,
                palette = comp_colors)
    plt.savefig(save_dir+"/dens_"+geneset_name + "_" + score_name + ".png")
    
def compare_with_categorical(score1, model, comp_groups, score_name1, geneset_name, score2, thr, score_name2, save_dir="plots"):
    comp_colors ={"Non significant": "#0C5AA6", "Significant": "#FF9700"}
    label_meaning = ["Non significant", "Significant"]
    plt.figure(figsize=(7,10))
    
    
    labels, binary_labels = get_predictions(model, comp_groups, score1)
    comp_colors_density = "#808080"
    binary_labels = [label_meaning[label] for label in binary_labels ]        

        
    ax1 = plt.subplot(211)
    ax1.set_xlabel(score_name1)
    ax1.set_ylabel("Frequency")
    sns.kdeplot(score1, linewidth=2, color='k', ax=ax1)
    
    if labels is not None:
        sns.rugplot(score1, clip_on=False, hue=labels, ax=ax1)    
        
    # plt.ylabel('Frequency')
    # plt.xlabel(score_name1)

    for i in range(model.n_components):
        mu = model.means_[i, 0]
        variance = model.covariances_[i, 0]
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        vals = stats.norm.pdf(x, mu, sigma)*model.weights_[i]
        plt.plot(x, vals, linestyle="dashed", color=comp_colors_density)
        
    sns.rugplot(score1, height=-.02, clip_on=False, hue=binary_labels,
                palette = comp_colors, ax=ax1)
    
    score_2 = (score2 > thr).astype(int)
    score_2 = [label_meaning[label] for label in score_2.tolist()]

    ax2 = plt.subplot(212)
    ax2.set_xlabel(score_name2)
    ax2.set_ylabel("Frequency")
    plt.axvline(x=thr, c="r", linestyle="dashed")
    sns.kdeplot(score2, linewidth=2, color='k', ax=ax2)
    sns.rugplot(score2, height=-.02, clip_on=False, hue=score_2,
                palette = comp_colors, ax=ax2)
    
    plt.suptitle(geneset_name)
    plt.savefig(save_dir+"/dens_"+geneset_name + "_" + score_name1 + "_"+score_name2 + ".png")