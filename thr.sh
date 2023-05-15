#! /bin/bash
poetry install
poetry build
declare -a data_types=('raw_counts' 'log2')
res_folder="results\BM\\"
save_dir="plots\BM\\"
# declare -a data_types=('raw_counts' 'log2' 'ft' 'seurat' 'dino' 'sctrans')
# Iterate the string array using for loop
for data_type in ${data_types[@]}; do
     python escript.py $data_type $res_folder $save_dir
done
