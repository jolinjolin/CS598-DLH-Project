python tasks/imbalanced_eval.py\
    --dataset_name iu_xray \
    --threshold 3 \
    --num_splits 8 \
    --gt_path "results/iu_xray_TIMER/ground_truths.csv"\
    --pre_path "results/iu_xray_R2Gen/preds.csv"\