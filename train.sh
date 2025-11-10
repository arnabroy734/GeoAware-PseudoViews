# export CUDA_VISIBLE_DEVICES=2
# export SOURCE_PATH='model_inputs/horns_3_views_syn/train'
# export TEST_PATH='model_inputs/horns_3_views_syn/test'
# export MODEL_PATH='outputs/horn_3_views_syn'

# python train.py \
#  --source_path $SOURCE_PATH \
#  --test_path $TEST_PATH \
#  --model_path $MODEL_PATH \
#  --iterations 30000 \
#  --test_iterations 1000 5000 10000 15000 20000 25000 30000\
#  --save_iterations 1000 5000 10000 15000  20000 25000 30000\
#  --checkpoint_iterations 1000 5000 10000 15000 20000 25000 30000\
#  --port 5002

export CUDA_VISIBLE_DEVICES=1
export SOURCE_PATH='model_inputs/orchids_3_views_dense/train'
export TEST_PATH='model_inputs/orchids_3_views_dense/test'
export MODEL_PATH='outputs/orchids_3_views_dense_pseudo_lpips'

python train.py \
 --source_path $SOURCE_PATH \
 --test_path $TEST_PATH \
 --model_path $MODEL_PATH \
 --iterations 10000 \
 --test_iterations 500 1000 3000 5000 7000 10000\
 --save_iterations 500 1000 3000 5000 7000 10000\
 --checkpoint_iterations 500 1000 3000 5000 7000 10000\
 --port 5027\
 --pseudo_view 