#! /bin/bash
# poetry install
# poetry build
# declare -a data_types=('seurat' 'row')
declare -a data_types=('seurat')
data_folder="data\Liver\\"
res_folder="results\Liver\\"
save_dir="plots\Liver"
# Iterate the string array using for loop
for data_type in ${data_types[@]}; do
     python escript.py $data_type $res_folder $save_dir $data_folder
done
