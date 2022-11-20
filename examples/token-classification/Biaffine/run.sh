python run.py \
  --do_train \
  --do_predict \
  --evaluation_strategy epoch \
  --max_train_samples 500 \
  --max_eval_samples 500 \
  --max_predict_samples 500 \
  --learning_rate 5e-5 \
  --dataset_name ../Datasets/ClueNER/ClueNER.py \
  --model_name_or_path ../../model_file/bert-base-chinese \
  --output_dir ./Output/ClueNER/ \
  --cache_dir ./Cache \
  --num_train_epochs 1 \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --save_strategy epoch \
  --overwrite_output_dir

