#! /bin/bash
# poetry install
# poetry build
# declare -a data_types=('seurat' 'row')
declare -a norms=('seurat')
data_type='Pancreas'

data_folder="data/${data_type}/"
res_folder="results/${data_type}/"
plot_folder="plots/${data_type}"
# Iterate the string array using for loop
for norm in ${norms[@]}; do
     python scripts/escript.py $norm $res_folder $plot_folder $data_folder
     # zip results
    cd $"${res_folder}${norm}"
    zip "${norm}.zip" *.json
    zip -r "${norm}.zip" kmeans_thr/ gmm_thr/
    zip -r "${norm}.zip" times_thrs.csv
    cd $"../../../${plot_folder}/${norm}"
    zip -r "${norm}.zip" kmeans/ top1/
    find . -type f -iname \*.png -delete
    cd ../../../
done
