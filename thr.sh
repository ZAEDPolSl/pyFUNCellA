#! /bin/bash
# poetry install
# poetry build
# declare -a data_types=('raw_counts' 'log2')
declare -a data_types=('raw_counts')
data_folder="data\COVID\\"
res_folder="results\COVID\\"
save_dir="plots\COVID"
# declare -a data_types=('raw_counts' 'log2' 'ft' 'seurat' 'dino' 'sctrans')
# Iterate the string array using for loop
for data_type in ${data_types[@]}; do
     python escript.py $data_type $res_folder $save_dir $data_folder
done
