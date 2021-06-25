#intensive module
export SQUAD_DIR=../data
CUDA_VISIBLE_DEVICES=$1 \
python3 ./run_squad_av.py \
    --model_type albert \
    --model_name_or_path albert-base-v2 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --version_2_with_negative \
    --train_file $SQUAD_DIR/quac_train.json \
    --predict_file $SQUAD_DIR/quac_val.json \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --max_query_length=64 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=6 \
    --warmup_steps=814 \
    --output_dir squad/squad2_albert-base-v2_lr2e-5_len512_bs32_ep7_wm814_av_ce \
    --eval_all_checkpoints \
    --save_steps 3000 \
    --n_best_size=20 \
    --max_answer_length=30 \
