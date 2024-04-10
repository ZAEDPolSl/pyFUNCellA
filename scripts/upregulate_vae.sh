#! /bin/bash
norm='seurat'
declare -a data_types=("PBMC" "Liver" "COVID" "BM")
data_folder="/mnt/pmanas/Ania/scrna-seq/data/"
res_folder="/mnt/pmanas/Ania/scrna-seq/results/"

for data_type in ${data_types[@]}; do
    python scripts/upregulate_vae.py $data_type $norm $data_folder $res_folder
done