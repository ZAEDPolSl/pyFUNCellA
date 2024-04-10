#! /bin/bash
# poetry install
# poetry build
norm='seurat'
declare -a data_types=('PBMC' 'Liver' 'BM' 'COVID')

# Iterate the string array using for loop
for data_type in ${data_types[@]}; do
    data_folder="/mnt/pmanas/Ania/scrna-seq/data/${data_type}/"
    res_folder="/mnt/pmanas/Ania/scrna-seq/results/${data_type}/"
    plot_folder="/mnt/pmanas/Ania/scrna-seq/plots/${data_type}"
     python scripts/escript.py $norm $res_folder $plot_folder $data_folder
    #  zip results
    # cd $"${res_folder}${norm}"
    # zip "${norm}.zip" *.json
    # zip -r "${norm}.zip" kmeans_thr/ gmm_thr/
    # zip -r "${norm}.zip" times_thrs.csv
    # cd $"../../../${plot_folder}/${norm}"
    # zip -r "${norm}.zip" kmeans/ top1/
    # find . -type f -iname \*.png -delete
    # find . -type f -iname \*.html -delete
    # cd ../../../
done
