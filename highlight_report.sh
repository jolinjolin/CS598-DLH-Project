python tasks/highlight_report.py\
    --dataset_name iu_xray \
    --threshold 3 \
    --infrequent_ratio 0.9 \
    --pre_path "results/iu_xray_CMN/preds.csv"\
    --pre_output_path "results/iu_xray_CMN/preds_highlighted.csv"\
    --gt_path "results/iu_xray_TIMER/ground_truths.csv"\
    --gt_output_path "results/iu_xray_TIMER/ground_truths_highlighted.csv"\