export CUDA_VISIBLE_DEVICES=2
export SOURCE_PATH='model_inputs/fern_3_views_syn_dense/train'
export TEST_PATH='model_inputs/fern_3_views_syn_dense/test'
export MODEL_PATH='outputs_depth_prune/fern_3_views_syn_dense_pseudo_depthprune'

python train.py \
 --source_path $SOURCE_PATH \
 --test_path $TEST_PATH \
 --model_path $MODEL_PATH \
 --iterations 10000 \
 --position_lr_max_steps 10000 \
 --random_background \
 --test_iterations 1000 3000 5000 7000 10000 \
 --save_iterations 1000 3000 5000 7000 10000 \
 --port 5042 \
 --pseudo_view \
 --alt_depth_prune 

#  --alt_densification


#  --diff_max_grad_thres
#  --pseudo_view 

#  --depth_guidance 
#  --densify_grad_threshold 0.0005 



#  --checkpoint_iterations 1000 5000 10000 15000 20000 25000 30000\

# export CUDA_VISIBLE_DEVICES=1
# export SOURCE_PATH='model_inputs/horns_3_views_dense/train'
# export TEST_PATH='model_inputs/horns_3_views_dense/test'
# export MODEL_PATH='outputs_trial/horn_3_views_dense_depth_prune_chnage_th'

# python train.py \
#  --source_path $SOURCE_PATH \
#  --test_path $TEST_PATH \
#  --model_path $MODEL_PATH \
#  --iterations 10000 \
#  --test_iterations 950 1950 2950 3950 4950 5950 6950 10000\
#  --save_iterations 950 1950 2950 3950 4950 5950 6950 10000\
#  --port 5027\
#  --densify_grad_threshold 0.0001 \
#  --depth_prune \
#  --depth_prune_start 500 \
#  --depth_prune_stop 2500 \
#  --depth_prune_interval 300\
#  --densify_until_iter 3000 

#  --pseudo_view 

#  --depth_prune \
#  --densify_grad_threshold 0.0001


#  --opacity_reset_interval 1500
#  --semantic_prune \
#  --pseudo_view 

#  --densification_interval 50\
#  --densify_grad_threshold 0.0001 \