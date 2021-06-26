# Retrospective Reading for Machine Reading Comprehension on QuAC Dataset

The pre-trained Ansq-RoBERTa Large model (transfer model)  provided by [wqa_tanda repo](https://github.com/alexa/wqa_tanda).  The transformers directory and run_glue.py are adaptation and modification from [wqa_tanda repo](https://github.com/alexa/wqa_tanda), the rest is author's code.

## Description
This work is for advanced information retrieval course's final project. You can read all detail work's description (in bahasa) at [technical-report](https://github.com/ryanpram/AwesomeMRC-QuACQA/tree/main/technical-report)



### Requirement
* Python 3.6
* tensorboard 1.14.0
* torch 1.7.0
* sacremoses 0.0.45
* boto3 1.17.99

### Reproduce Step

* Clone This Repo
* Use Data from data. You can see preprocessing process at 
* Run Sketchy Reader
```
bash -x sh_albert_cls.sh
```
Or
```
python3 ./run_cls.py \
    --model_type albert \
    --model_name_or_path albert-base-v2 \
    --task_name squad \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ../data \
    --max_seq_length 512 \
    --per_gpu_train_batch_size=8   \
    --gradient_accumulation_steps=4 \
    --per_gpu_eval_batch_size=6   \
    --warmup_steps=814 \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --eval_all_checkpoints \
    --save_steps 2500 \
    --output_dir squad/cls_squad2_albert-base-v2_lr2e-5_len512_bs32_ep2_wm814_f$
```
* Run Intensive Reader
```
bash -x sh_albert_av.sh
```
Or
```
python3 ./run_squad_av.py \
    --model_type albert \
    --model_name_or_path albert-base-v2 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --version_2_with_negative \
    --train_file ../data/quac_train.json \
    --predict_file ../data/quac_val.json \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --max_query_length=64 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=6 \
    --gradient_accumulation_steps=4 \
    --warmup_steps=814 \
    --output_dir squad/squad2_albert-base-v2_lr2e-5_len512_bs32_ep3_wm814_av_ce \
    --eval_all_checkpoints \
    --save_steps 3000 \
    --n_best_size=20 \
    --max_answer_length=30 \
```
* Run Rear Verifier
```
python3 evaluate-v2.0.py data/quac_val.json transformer-mrc/predictions.json 
```




## Help

You can submit a GitHub issue for asking a question or help. Or you can contact me directly at ryan.pramana@ui.ac.id as well

