# [分本分类任务 Text Classification](https://github.com/casuallyName/transformers_expand/tree/master/examples/text-classification)

## [单标签任务](https://github.com/casuallyName/transformers_expand/blob/master/examples/text-classification/run.py)

```bash
python text-classification/run.py \
    --do_train \
    --do_predict \
    --evaluation_strategy epoch \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --dataset_name Datasets/THUCNews/THUCNews.py \
    --loss_weights Datasets/THUCNews/loss_weights.json \
    --model_name_or_path model_file/bert-base-chinese/ \
    --output_dir ./Output/THUCNews/Multi_Class \
    --cache_dir ./Cache \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --save_strategy epoch \
    --overwrite_output_dir
```

## [多标签任务](https://github.com/casuallyName/transformers_expand/blob/master/examples/text-classification/run.py)

```bash
python text-classification/run.py \
    --do_train \
    --do_predict \
    --evaluation_strategy epoch \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --dataset_name Datasets/THUCNews/THUCNewsOneHot.py \
    --model_name_or_path model_file/bert-base-chinese/ \
    --output_dir ./Output/THUCNews/Multi_Label \
    --cache_dir ./Cache \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --save_strategy epoch \
    --overwrite_output_dir
```

# [Token分类任务（NER）Token Classification](https://github.com/casuallyName/transformers_expand/tree/master/examples/token-classification)

## [Biaffine](https://github.com/casuallyName/transformers_expand/tree/master/examples/token-classification/Biaffine/run.py)
```bash
python token-classification/Biaffine/run.py \
    --do_train \
    --do_predict \
    --evaluation_strategy epoch \
    --learning_rate 5e-5 \
    --dataset_name Datasets/ClueNER/ClueNER.py \
    --model_name_or_path model_file/bert-base-chinese \
    --output_dir ./Output/ClueNER/Biaffine \
    --cache_dir ./Cache \
    --num_train_epochs 1 \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --save_strategy epoch \
    --overwrite_output_dir
```

## [GlobalPointer](https://github.com/casuallyName/transformers_expand/tree/master/examples/token-classification/GlobalPointer/run.py)

```bash
python token-classification/GlobalPointer/run.py \
    --do_train \
    --do_predict \
    --evaluation_strategy epoch \
    --learning_rate 5e-5 \
    --dataset_name Datasets/ClueNER/ClueNER.py \
    --model_name_or_path model_file/bert-base-chinese \
    --output_dir ./Output/ClueNER/GlobalPointer \
    --cache_dir ./Cache \
    --num_train_epochs 1 \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --save_strategy epoch \
    --overwrite_output_dir
```