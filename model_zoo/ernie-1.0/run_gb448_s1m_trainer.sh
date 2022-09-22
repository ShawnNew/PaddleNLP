set -x
unset MLU_VISIBLE_DEVICES

# dp8 for 8 worker of data parallel
# gb512 for the global batch size is 448 = 56 * 8
# s1m for max steps is 1 million
task_name="ernie-1.0-dp8-gb448"
rm -rf output/$task_name/

python -u  -m paddle.distributed.launch \
    --mlus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name""_log" \
    run_pretrain_trainer.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0-base-zh" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 512 \
    --per_device_train_batch_size 56 \
    --per_device_eval_batch_size 56 \
    --fp16  \
    --fp16_opt_level "O2"  \
    --learning_rate 0.0001 \
    --max_steps 1143000 \
    --save_steps 50000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 20\
    --dataloader_num_workers 4 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --do_train \
    --device "mlu"
