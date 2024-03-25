CUDA_LAUNCH_BLOCKING=1
# First argument GPU_ID, Second argument test_file
GPU_ID=1

# # Incidentaloma Infer
# eval batch_size reduced to 4
mkdir incidentaloma_models
for seed in 45; do
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_ner.py  --model_type bertspanmarker  \
    --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
    --data_dir incidentaloma  \
    --learning_rate 2e-5  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_eval  --eval_all_checkpoints  \
    --fp16  --seed $seed  --onedropout  --lminit  \
    --test_file example_input.json  \
    --output_dir incidentaloma_models/PL-Marker-incidentaloma-bertbase-$seed  --overwrite_output_dir  --output_results
done;

# # # Incidentaloma Infer 0819
# mkdir incidentaloma_models
# for seed in 45; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
#     --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
#     --data_dir incidentaloma  \
#     --learning_rate 2e-5  --per_gpu_train_batch_size  4  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_eval  --eval_all_checkpoints  \
#     --fp16  --seed $seed  --onedropout  --lminit  \
#     --test_file DB_MR_999.json  \
#     --output_dir incidentaloma_models/PL-Marker-incidentaloma-bertbase-$seed  --overwrite_output_dir  --output_results
# done;


# For ALBERT-xxlarge, change learning_rate from 2e-5 to 1e-5

# # ACE05
# mkdir ace05ner_models
# for seed in 42 43 44 45 46; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
#     --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
#     --data_dir ace05  \
#     --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2500  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed $seed  --onedropout  --lminit  \
#     --train_file train.json --dev_file dev.json --test_file test.json  \
#     --output_dir ace05ner_models/PL-Marker-ace05-bert-$seed  --overwrite_output_dir  --output_results
# done;
# Average the scores
# python3 sumup.py ace05ner PL-Marker-ace05-bert



# SciERC
# mkdir sciner_models
# for seed in 42; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
#     --model_name_or_path  bert_models/scibert-uncased  --do_lower_case  \
#     --data_dir scierc  \
#     --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed $seed  --onedropout  --lminit  \
#     --train_file train.json --dev_file dev.json --test_file test.json  \
#     --output_dir sciner_models/PL-Marker-scierc-scibert-$seed  --overwrite_output_dir  --output_results
# done;
# # Average the scores
# python3 sumup.py sciner PL-Marker-ace05-bert

# # # Incidentaloma
# mkdir incidentaloma_models
# for seed in 45; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
#     --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
#     --data_dir incidentaloma  \
#     --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  4  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed $seed  --onedropout  --lminit  \
#     --train_file data_train_PL_marker.json --dev_file data_valid_PL_marker.json --test_file data_test_PL_marker.json  \
#     --output_dir incidentaloma_models/PL-Marker-incidentaloma-bertbase-$seed  --overwrite_output_dir  --output_results
# done;
# Average the scores
# python3 sumup.py incidentaloma PL-Marker-ace05-bert

# # Incidentaloma assertion
# mkdir incidentaloma_assertion_models
# for seed in 45; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
#     --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
#     --data_dir assertion  \
#     --learning_rate 2e-5  --num_train_epochs 5  --per_gpu_train_batch_size  4  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed $seed  --onedropout  --lminit  \
#     --train_file data_train_PL_marker_assertion.json --dev_file data_valid_PL_marker_assertion.json --test_file data_test_PL_marker_assertion.json  \
#     --output_dir incidentaloma_assertion_models/PL-Marker-incidentaloma-bertbase-$seed  --overwrite_output_dir  --output_results
# done;

# anatomy_parent done
# # Incidentaloma anatomy_parent
# mkdir incidentaloma_anatomy_parent_models
# for seed in 45; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
#     --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
#     --data_dir anatomy_parent  \
#     --learning_rate 2e-5  --num_train_epochs 5  --per_gpu_train_batch_size  4  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed $seed  --onedropout  --lminit  \
#     --train_file data_train_PL_marker_anatomy_parent.json --dev_file data_valid_PL_marker_anatomy_parent.json --test_file data_test_PL_marker_anatomy_parent.json  \
#     --output_dir incidentaloma_anatomy_parent_models/PL-Marker-incidentaloma-bertbase-$seed  --overwrite_output_dir  --output_results
# done;

# # Incidentaloma anatomy_child
# mkdir incidentaloma_anatomy_child_models
# for seed in 45; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
#     --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
#     --data_dir anatomy_child  \
#     --learning_rate 2e-5  --num_train_epochs 5  --per_gpu_train_batch_size  4  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed $seed  --onedropout  --lminit  \
#     --train_file data_train_PL_marker_anatomy_child.json --dev_file data_valid_PL_marker_anatomy_child.json --test_file data_test_PL_marker_anatomy_child.json  \
#     --output_dir incidentaloma_anatomy_child_models/PL-Marker-incidentaloma-bertbase-$seed  --overwrite_output_dir  --output_results
# done;

# # Incidentaloma indication_type
# mkdir incidentaloma_indication_type_models
# for seed in 45; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
#     --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
#     --data_dir indication_type  \
#     --learning_rate 2e-5  --num_train_epochs 5  --per_gpu_train_batch_size  4  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed $seed  --onedropout  --lminit  \
#     --train_file data_train_PL_marker_indication_type.json --dev_file data_valid_PL_marker_indication_type.json --test_file data_test_PL_marker_indication_type.json  \
#     --output_dir incidentaloma_indication_type_models/PL-Marker-incidentaloma-bertbase-$seed  --overwrite_output_dir  --output_results
# done;

# # Incidentaloma size
# mkdir incidentaloma_size_models
# for seed in 45; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
#     --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
#     --data_dir size_only  \
#     --learning_rate 2e-5  --num_train_epochs 5  --per_gpu_train_batch_size  4  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed $seed  --onedropout  --lminit  \
#     --train_file data_train_PL_marker_size.json --dev_file data_valid_PL_marker_size.json --test_file data_test_PL_marker_size.json  \
#     --output_dir incidentaloma_size_models/PL-Marker-incidentaloma-bertbase-$seed  --overwrite_output_dir  --output_results
# done;

# # Incidentaloma size_trend
# mkdir incidentaloma_size_trend_models
# for seed in 43; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
#     --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
#     --data_dir size_trend  \
#     --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  4  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed $seed  --onedropout  --lminit  \
#     --train_file data_train_PL_marker_size_trend.json --dev_file data_valid_PL_marker_size_trend.json --test_file data_test_PL_marker_size_trend.json  \
#     --output_dir incidentaloma_size_trend_models/PL-Marker-incidentaloma-bertbase-$seed  --overwrite_output_dir  --output_results
# done;

# # ACE04
# mkdir ace04ner_models
# for data_spilt in 0 1 2 3 4; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
#     --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
#     --data_dir ace04  \
#     --learning_rate 2e-5  --num_train_epochs 15  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2500  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed 42  --onedropout  --lminit  \
#     --train_file train/$data_spilt.json  --dev_file dev/$data_spilt.json  --test_file test/$data_spilt.json\
#     --output_dir ace04ner_models/PL-Marker-ace04-bert-$data_spilt  --overwrite_output_dir  --output_results
# done;
# # Average the scores
# python3 sumup.py ace04ner PL-Marker-ace05-bert
#
#
# # CoNLL03
# mkdir conll03_models
# for seed in 42; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_ner.py  --model_type robertaspanmarker  \
#     --model_name_or_path  bert_models/roberta-large    \
#     --data_dir conll03  \
#     --learning_rate 1e-5  --num_train_epochs 2  --per_gpu_train_batch_size  4  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 2  \
#     --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8  \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed $seed  --onedropout  --lminit  \
#     --train_file eng.train --dev_file eng.testa --test_file eng.testb  \
#     --output_dir conll03_models/PL-Marker-conll03-roberta-$seed  --overwrite_output_dir
# done;
# # Average the scores
# python3 sumup.py conll03ner PL-Marker-conll03-roberta
#
#
# # OntoNote 5.0
# mkdir ontonotes_models
# for seed in 42 43 44 45 46; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type robertaspanmarker  \
#     --model_name_or_path  bert_models/roberta-large    \
#     --data_dir ontonotes  \
#     --learning_rate 1e-5  --num_train_epochs 4  --per_gpu_train_batch_size  4  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 2  \
#     --max_seq_length 512  --save_steps 5000  --max_pair_length 256  --max_mention_ori_length 16    \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed $seed  --onedropout  --lminit  \
#     --train_file train.json --dev_file dev.json --test_file test.json  \
#     --output_dir ontonotes_models/PL-Marker-ontonotes-roberta-$seed  --overwrite_output_dir
# done;
# # Average the scores
# python3 sumup.py ontonotesner PL-Marker-ontonotes-roberta
#
#
# # Few-NERD
# mkdir fewnerd_models
# for seed in 42 43 44 45 46; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_ner.py  --model_type robertaspanmarker  \
#     --model_name_or_path  bert_models/roberta-large    \
#     --data_dir fewnerd  \
#     --learning_rate 1e-5  --num_train_epochs 3  --per_gpu_train_batch_size  4  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 2  \
#     --max_seq_length 512  --save_steps 5000  --max_pair_length 256  --max_mention_ori_length 16    \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed $seed  --onedropout  --lminit  \
#     --train_file train.txt --dev_file dev.txt --test_file test.txt  \
#     --output_dir fewnerd_models/PL-Marker-fewnerd-roberta-$seed  --overwrite_output_dir
# done;
# # Average the scores
# python3 sumup.py fewnerdner PL-Marker-fewnerd-roberta
