python tasks/test.py \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 100 \
    --batch_size 16 \
    --lr_ve 1e-4 \
    --lr_ed 5e-4 \
    --step_size 10 \
    --gamma 0.8 \
    --num_layers 3 \
    --topk 32 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed 7580 \
    --beam_size 3 \
    --save_dir results/iu_xray_TIMER/ \
    --log_period 50 \
    --load results/iu_xray_TIMER/best-model.pth\
    # --save_report 
