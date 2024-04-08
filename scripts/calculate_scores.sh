#! /bin/bash
# declare -a data_types=('seurat' 'row')
declare -a data_types=('seurat')
inpath="/mnt/pmanas/Ania/scrna-seq/data/BreastCancer"
outpath="/mnt/pmanas/Ania/scrna-seq/results/BreastCancer"
# Iterate the string array using for loop
for data_type in ${data_types[@]}; do
     python scripts/calculate_scores.py $inpath $outpath $data_type
done
