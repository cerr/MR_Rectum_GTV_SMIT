#!/bin/bash
#
#
# Input arguments: 
# $1 data_dir
# $2 save_folder
# $3 load_weight_name
# $4 input_nifti



#Use SMIT
use_smit=1 #Use SMIT not SMIT+

#Data folder and there need a 'data.json' file in the folder 
data_dir="$1"

#Segmentation output folder 
save_folder="$2"

#Some configrations for the model, no need to change
#Trained weight 
load_weight_name="$3"

python utils/gen_data_json.py $data_dir

python run_segmentation_rectum.py \
    --data_dir $data_dir \
    --load_weight_name $load_weight_name \
    --save_folder $save_folder \
    --out_channels 2 \
    --a_min 0 --a_max 1000 \
    --space_x 1.5 --space_y 1.5 --space_z 2.0 \
    --roi_x 128 --roi_y 128 --roi_z 128 --use_smit $use_smit

