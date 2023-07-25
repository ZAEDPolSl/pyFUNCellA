#! /bin/bash
norm='seurat'
declare -a data_types=("PBMC" "Liver" "COVID" "BM")
declare -a cluster_types=('gmm' 'kmeans' 'AUCell')

data_folder="data/"
res_folder="results/"

for data_type in ${data_types[@]}; do
    for cluster_type in ${cluster_types[@]}; do
        python scripts/estimate_classif.py $data_type $norm $cluster_type $data_folder $res_folder
        done
done
