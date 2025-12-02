# RUN COLMAP ON ANY FOLDER - THERE IS NO TRAIN TEST FOLDER INSIDE SOURCE, ONLY IMAGES ARE PRESENT
export SOURCE='~/Downloads/nerf-data/nerf_llff_data/horns/images_4/'
export DEST='model_inputs/horns_20'
export n_views=20

python convert.py -s $SOURCE -d $DEST --n_views $n_views \
    --train_test_split 
    # --stereo_fusion