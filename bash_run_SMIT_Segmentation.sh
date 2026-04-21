#!/bin/bash
#
#
# Input arguments: 
# $1 data_dir
# $2 save_folder
# $3 load_weight_name
# $4 input_nifti



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
    --json_list data.json \
    --datasets val \
    --model_name smit \
    --pretrained_model_path $load_weight_name \
    --results_dir $save_folder \
    --in_channels 1 \    
    --out_channels 2 \
    --norm_name instance \ 
    --a_min 0 --a_max 800 \
    --b_min 0.0 --b_max 1.0 \
    --space_x 1.0 --space_y 1.0 --space_z 1.0 \
    --roi_x 128 --roi_y 128 --roi_z 64 \
    --infer_overlap 0.5 --sw_batch_size 8\
    --use_tta --postproc standard

