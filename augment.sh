export INPUT_PATH='model_inputs/fern_3_views'
export OUTPUT_PATH='model_inputs/fern_3_views_syn_dense'
export N_NEW_IMAGES=10

python augment.py --input_path $INPUT_PATH --output_path $OUTPUT_PATH --n_new_images $N_NEW_IMAGES --stereo_fusion