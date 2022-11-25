# -*- coding: utf-8 -*-
# @Time     : 2022/11/25 17:08
# @File     : test_model.py
# @Author   : Zhou Hang
# @Email    : zhouhang@idataway.com
# @Software : Python 3.7
# @About    :
import os.path
import sys

sys.path.append(os.path.abspath('../src'))


import torch
import traceback
import transformers
import transformers_expand
from transformers.utils import logging

logging.set_verbosity(logging.ERROR)
transformers.set_seed(0)

print(transformers_expand.__version__)


def get_checkpoint_name(model_name, model_obj_name, end, auto_name):
    model_name = model_name.replace('-', '_')
    if model_name == 'data2vec_text':
        name = 'data2vec'
        modeling_name = 'modeling_' + model_name
    elif model_name == '':
        name = 'microsoft/layoutlm-base-uncased'
        modeling_name = 'modeling_' + model_name
    else:
        name = model_name
        modeling_name = 'modeling_' + model_name
    if hasattr(transformers_expand, model_obj_name + end):
        try:
            model_ckp_list = getattr(getattr(getattr(getattr(transformers, 'models'), name), modeling_name),
                                     model_name.upper() + '_PRETRAINED_MODEL_ARCHIVE_LIST')
            model_1 = getattr(transformers_expand, model_obj_name + end)
            model_2 = getattr(transformers_expand, auto_name)
            return model_ckp_list, model_1, model_2
        except:
            traceback.print_exc()
            return None, None, None
    else:
        return None, None, None


def forward_func_for_biaffine(model, tokenizer):
    max_length = 10
    inputs = tokenizer('测试句子',
                       max_length=max_length,
                       truncation=True,
                       padding='max_length',
                       return_tensors='pt'
                       )
    attention_mask = inputs['attention_mask'].numpy().tolist()[0]
    seq_len = len(attention_mask)
    seq_mask = [attention_mask for i in range(sum(attention_mask))]
    zero = [0 for i in range(seq_len)]
    seq_mask.extend([zero for i in range(sum(attention_mask), seq_len)])
    inputs['sequence_mask'] = torch.tensor([seq_mask])
    span_label = [0 for i in range(max_length)]
    span_label = [span_label for i in range(max_length)]
    inputs['labels'] = torch.tensor([span_label])
    loss = model(**inputs).loss
    return loss


def forward_func_for_global_pointer(model, tokenizer):
    max_length = 10
    inputs = tokenizer('测试句子',
                       max_length=max_length,
                       truncation=True,
                       padding='max_length',
                       return_tensors='pt'
                       )
    inputs['labels'] = torch.zeros(size=(1, 2, 10, 10))
    loss = model(**inputs).loss
    return loss


def check_model(model_list, end, auto_name, forward_func):
    for model_name, model_obj_name in model_list:
        model_ckp_list, model_1, model_2 = get_checkpoint_name(model_name=model_name,
                                                               model_obj_name=model_obj_name,
                                                               end=end,
                                                               auto_name=auto_name)
        if model_ckp_list is None:
            print(f'\t{model_name:<20}: \033[33m Θ 未支持\033[30m')
        else:
            for pretrained_model_name_or_path in model_ckp_list:
                try:
                    model_1 = model_1.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
                    model_2 = model_2.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        cache_dir='./Cache',
                    )
                    loss_1 = forward_func(model_1, tokenizer)
                    loss_2 = forward_func(model_2, tokenizer)
                    print(f'\t{model_name:<20}: \033[32m ✔ 通过\033[30m')
                    break
                except:
                    print(f'\t{model_name:<20}: \033[31m ✘ 错误\033[30m')
                    traceback.print_exc()


if __name__ == '__main__':
    from transformers.models.auto.modeling_auto import MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES

    print('Biaffine Models:')
    check_model(model_list=MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES.items(),
                end='WithBiaffine',
                auto_name='AutoModelForTokenClassificationWithBiaffine',
                forward_func=forward_func_for_biaffine)

    print('GlobalPointer Models:')
    check_model(model_list=MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES.items(),
                end='WithGlobalPointer',
                auto_name='AutoModelForTokenClassificationWithGlobalPointer',
                forward_func=forward_func_for_biaffine)
