for missing_ratio in "0" "0.1" "0.2"  "0.3"  "0.4"  "0.5" 
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset P19 \
    --missingratio $missing_ratio \
    --feature_removal_level sample 
done
