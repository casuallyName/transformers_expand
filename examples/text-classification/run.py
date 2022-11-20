# -*- coding: utf-8 -*-
# @Time     : 2022/11/12 00:04
# @File     : run.py
# @Author   : Zhou Hang
# @Email    : zhouhang@idataway.com
# @Software : Python 3.7
# @About    :
import os
import sys
import json
import torch
import logging
import datasets
import numpy as np
import transformers
from typing import Optional
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass, field
from datasets.utils import logging as datasets_loggings
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as transformers_logging

from transformers_expand import Trainer
from transformers_expand import TrainingArguments
from transformers_expand import MultiLabelCategoricalCrossEntropyLoss
from transformers_expand.metrics import MetricsForCommonTextClassificationTask

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    sentence1_key: Optional[str] = field(
        default='sentence',
        metadata={
            "help": "First column name of input sentence.'"
        },
    )
    sentence2_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Second column name of input sentence."
        },
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class TextClassificationTrainingArguments(TrainingArguments):
    loss_weights: str = field(
        default=None,
        metadata={
            "help": "If not None, will load loss weight in trainint loss function."
        },
    )


class DataPreProcess:
    def __init__(self,
                 tokenizer,
                 padding: Optional[bool],
                 max_seq_length: Optional[int],
                 label_type: Optional[str],
                 label_to_id: Optional[dict],
                 sentence1_key: Optional[str] = 'sentence',
                 sentence2_key: Optional[str] = None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_seq_length = max_seq_length
        self.label_type = label_type
        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key
        self.id_to_label = {v: k for k, v in label_to_id.items()}
        self.label_to_id = label_to_id

    def __call__(self, examples):
        if self.sentence2_key is not None:
            args = (examples[self.sentence1_key], examples[self.sentence2_key])
        else:
            args = (examples[self.sentence1_key],)
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)
        if self.label_type == 'ClassLabel':
            result["label"] = examples["label"]
        elif self.label_type == 'Sequence':
            if examples['label'][0] is None:
                result["label"] = [None for _ in range(len(examples["label"]))]
            else:
                one_hot_labels = [[0 for _ in range(len(self.label_to_id))] for _ in range(len(examples["label"]))]
                for idx, label in enumerate(examples["label"]):
                    for label_id in label:
                        one_hot_labels[idx][label_id] = 1
                result["label"] = one_hot_labels
        else:
            result["label"] = [(-1 if l is None else self.label_to_id[l]) for l in examples["label"]]

        return result


def detecting_last_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"输出路径 ({training_args.output_dir}) 已存在且不为空 "
                "使用 --overwrite_output_dir 覆盖路径下内容"
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"检测到检查点，正在从{last_checkpoint}恢复训练。修改 "
                "`--output_dir` 或添加 `--overwrite_output_dir` 来进行全新训练"
            )
    return last_checkpoint


def load_dataset(path, cache_dir='./Cache'):
    raw_datasets = datasets.load_dataset(path=path,
                                         cache_dir=cache_dir)
    logger.info(f"Dataset description: {raw_datasets['train'].description}")
    if hasattr(raw_datasets['train'].info.features['label'], 'names'):
        label_list = raw_datasets['train'].info.features['label'].names
        num_labels = len(label_list)
        label_to_id = {label: i for i, label in enumerate(label_list)}
        id_to_label = {v: k for k, v in label_to_id.items()}
        problem_type = "single_label_classification"
    elif hasattr(raw_datasets['train'].info.features['label'].feature, 'names'):
        label_list = raw_datasets['train'].info.features['label'].feature.names
        num_labels = len(label_list)
        label_to_id = {label: i for i, label in enumerate(label_list)}
        id_to_label = {v: k for k, v in label_to_id.items()}
        problem_type = "multi_label_classification"
    else:
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()
        num_labels = len(label_list)
        label_to_id = {label: i for i, label in enumerate(label_list)}
        id_to_label = {v: k for k, v in label_to_id.items()}
        problem_type = "single_label_classification"
    return raw_datasets, num_labels, id_to_label, label_to_id, problem_type


def load_model(model_args, num_labels):
    config = transformers.AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    return model, tokenizer, config


def load_loss_weights(path, id_to_label):
    with open(path, "r") as f:
        weights = json.loads(f.read())
    weights = [weights[id_to_label[i]] for i in range(len(id_to_label))]
    return torch.tensor(weights)


def split_and_select_dataset(raw_datasets, dataset_type: str, select=None):
    assert dataset_type in ['train', 'validation', 'test'], "'dataset_type' mast 'train' 'validation' or 'test'"
    if (
            (dataset_type in ['validation', 'test']) and
            (dataset_type not in raw_datasets) and
            (dataset_type + '_matched' not in raw_datasets)
    ) or (
            (dataset_type == 'train') and (dataset_type not in raw_datasets)
    ):
        raise ValueError("--do_{} 需要数据集中包含 {} 部分".format(
            {'train': 'train', 'validation': 'eval', 'test': 'predict'}[dataset_type],
            dataset_type))
    else:
        if select is not None:
            return raw_datasets[dataset_type].select(range(select))
        else:
            return raw_datasets[dataset_type]


def run():
    parser = transformers.HfArgumentParser((ModelArguments, DataTrainingArguments, TextClassificationTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="[%(levelname)s|%(filename)s:%(lineno)d] %(asctime)s,%(msecs)d >> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets_loggings.set_verbosity(log_level)
    transformers_logging.set_verbosity(log_level)
    transformers_logging.enable_default_handler()
    transformers_logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    transformers.set_seed(training_args.seed)

    # 加载断点
    last_checkpoint = detecting_last_checkpoint(training_args=training_args)

    # 加载数据集
    raw_datasets, num_labels, id_to_label, label_to_id, problem_type = load_dataset(path=data_args.dataset_name,
                                                                                    cache_dir=model_args.cache_dir)
    if training_args.loss_weights is not None:
        loss_weights = load_loss_weights(training_args.loss_weights, id_to_label=id_to_label)
    else:
        loss_weights = None

    # 加载模型
    model, tokenizer, config = load_model(model_args=model_args, num_labels=len(label_to_id))
    if hasattr(config, 'problem_type') and config.problem_type is not None and config.problem_type != problem_type:
        logger.warning('模型与数据集不匹配，模型可能已经使用其他数据集训练过。')
    else:
        logger.info(f"Set problem_type '{problem_type}'")
        config.problem_type = problem_type

    if model.config.label2id != transformers.PretrainedConfig(num_labels=len(label_to_id)).label2id:
        if len(model.config.label2id) == len(label_to_id):
            model.config.label2id = label_to_id
            model.config.id2label = id_to_label
        else:
            logger.warning(
                "模型标签集与数据集标签不匹配: ",
                f"模型标签: {list(sorted(model.config.label2id.keys()))}, "
                f"数据集标签: {list(sorted(label_to_id.keys()))}."
                "\n因此忽略模型标签。",
            )
    else:
        model.config.label2id = label_to_id
        model.config.id2label = id_to_label

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"设定的文本最大长度 ({data_args.max_seq_length}) 超过模型最大可以接受长度 ({tokenizer.model_max_length})。"
            f"将使用 max_seq_length={tokenizer.model_max_length} 代替。"
        )

    # 数据预处理
    preprocess_function = DataPreProcess(tokenizer=tokenizer,
                                         padding="max_length" if data_args.pad_to_max_length else False,
                                         max_seq_length=min(data_args.max_seq_length, tokenizer.model_max_length),
                                         label_type=raw_datasets['train'].features['label']._type,
                                         label_to_id=label_to_id,
                                         sentence1_key=data_args.sentence1_key,
                                         sentence2_key=data_args.sentence2_key
                                         )

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if data_args.pad_to_max_length:
        from transformers import default_data_collator
        data_collator = default_data_collator
    elif training_args.fp16:
        from transformers import DataCollatorWithPadding
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # 评估函数
    if problem_type == 'multi_label_classification':
        # 由于使用 MultiLabelCategoricalCrossEntropyLoss ，
        # 因此此处不需要做sigmoid/softmax，直接判断大于0的结果
        compute_metrics = MetricsForCommonTextClassificationTask(multi_label=True,
                                                                  mark_line=0,
                                                                  use_sigmoid=False)
    else:
        compute_metrics = MetricsForCommonTextClassificationTask()

    # 损失函数
    if config.problem_type == 'multi_label_classification':
        # 多标签默认使用MultiLabelCategoricalCrossEntropyLoss 抛弃 BCELoss
        loss_fnt = MultiLabelCategoricalCrossEntropyLoss()
    else:
        loss_fnt = CrossEntropyLoss(weight=loss_weights)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_and_select_dataset(raw_datasets=raw_datasets,
                                               dataset_type='train',
                                               select=data_args.max_train_samples) if training_args.do_train else None,
        eval_dataset=split_and_select_dataset(raw_datasets=raw_datasets,
                                              dataset_type='validation',
                                              select=data_args.max_train_samples) if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        loss_fnt=loss_fnt
    )

    if training_args.do_train:
        train_dataset = split_and_select_dataset(raw_datasets=raw_datasets,
                                                 dataset_type='train',
                                                 select=data_args.max_train_samples) if training_args.do_train else None
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_dataset = split_and_select_dataset(raw_datasets=raw_datasets,
                                                dataset_type='validation',
                                                select=data_args.max_train_samples) if training_args.do_eval else None
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_dataset = split_and_select_dataset(raw_datasets=raw_datasets,
                                                   dataset_type='test',
                                                   select=data_args.max_train_samples)
        predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        if config.problem_type == 'multi_label_classification':
            # 由于使用 MultiLabelCategoricalCrossEntropyLoss ，
            # 因此此处不需要做softmax，直接判断大于0的结果
            predictions = [';;'.join([config.id2label[item[0]] for item in np.argwhere(pred > 0)]) for pred in
                           predictions]
        else:
            predictions = [config.id2label[item] for item in np.argmax(predictions, axis=1)]

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    writer.write(f"{index}\t{item}\n")


if __name__ == '__main__':
    run()
