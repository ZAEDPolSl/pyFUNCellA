import pandas as pd
import numpy as np
from tqdm import tqdm
from enrichment_stuff.source.distributions import find_distribution, find_thresholds, take_top_one
import json
from time import time
import scipy
import sys

def pipeline_for_dist(score, geneset_name):
    # get mixtures and thresholds
    distributions = find_distribution(score, geneset_name)
    thresholds, locarizer = find_thresholds(distributions, score, geneset_name)
    return thresholds, distributions, locarizer

score_names = ["z","gsva", "cerno_auc", "AUCell", "ratios", "vision", "svd", "sparse_pca"] # all scores to run for each data type

if __name__ == "__main__":
    data_type = sys.argv[1]
    print(data_type)
    for score_name in tqdm(score_names):
        print(score_name)
        # get scores
        scores = pd.read_csv('enrichment_stuff/data/'+data_type+'/'+score_name+'.csv', index_col=0)
        gs_names = scores.index.values.tolist()
        scores = scores.to_numpy()
        scores_thr = pd.DataFrame(0, index=gs_names, columns=np.arange(1))
        scores_dist = []
        scores_thrs = {}
        locs = []
        for i, gs_name in tqdm(enumerate(gs_names), total=len(gs_names)):
            score = scores[i, :]
            thr1_1, distributions1_1, locarizer1 = pipeline_for_dist(score, gs_name)

            del distributions1_1['TIC'], distributions1_1['l_lik']
            distributions1_1['weights'] = (distributions1_1['weights']).tolist()
            distributions1_1['mu'] = (distributions1_1['mu']).tolist()
            distributions1_1['sigma'] = (distributions1_1['sigma']).tolist()

            locarizer1 = locarizer1.tolist()      

            scores_thr.loc[gs_name] = thr1_1[-1]
            scores_dist.append(distributions1_1)

            scores_thrs[gs_name] = thr1_1.tolist()

            locs.append(locarizer1)
            break

        scores_thr.to_csv('enrichment_stuff/results/'+data_type+'/'+score_name+"_thr.csv")
        with open('enrichment_stuff/results/'+data_type+'/'+score_name+"_loc.json", 'w') as fout:
            json.dump(locs, fout)
        with open('enrichment_stuff/results/'+data_type+'/'+score_name+"_dist.json", 'w') as fout:
            json.dump(scores_dist, fout)
        with open('enrichment_stuff/results/'+data_type+'/'+score_name+"_thrs.json", 'w') as fout:
            json.dump(scores_thrs, fout)
        break
