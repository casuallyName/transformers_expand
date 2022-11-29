# -*- coding: utf-8 -*-
# @Time     : 2022/11/25 17:08
# @File     : test_model.py
# @Author   : Zhou Hang
# @Email    : zhouhang@idataway.com
# @Software : Python 3.7
# @About    :

print('Start test ...')
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
    else:
        name = model_name
    modeling_name = 'modeling_' + model_name
    if hasattr(transformers_expand, model_obj_name + end):
        try:
            model_ckp_list = getattr(getattr(getattr(getattr(transformers, 'models'), name), modeling_name),
                                     model_name.upper() + '_PRETRAINED_MODEL_ARCHIVE_LIST')
            if model_name == 'layoutlm':
                model_ckp_list = ['microsoft/' + i for i in model_ckp_list]
            model_1 = getattr(transformers_expand, model_obj_name + end)
            model_2 = getattr(transformers_expand, auto_name)
            return model_ckp_list, model_1, model_2
        except:
            traceback.print_exc()
            return None, None, None
    else:
        return None, None, None


def forward_func_for_biaffine(model, tokenizer):
    max_length = 5
    try:
        inputs = tokenizer('测试句子',
                           max_length=max_length,
                           truncation=True,
                           padding='max_length',
                           return_tensors='pt'
                           )
    except ValueError:
        inputs = tokenizer(['测试', '句子'],
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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    loss = model(**inputs).loss
    return loss


def forward_func_for_global_pointer(model, tokenizer):
    max_length = 5
    try:
        inputs = tokenizer('测试句子',
                           max_length=max_length,
                           truncation=True,
                           padding='max_length',
                           return_tensors='pt'
                           )
    except ValueError:
        inputs = tokenizer(['测试', '句子'],
                           max_length=max_length,
                           truncation=True,
                           padding='max_length',
                           return_tensors='pt'
                           )
    inputs['labels'] = torch.zeros(size=(1, 2, max_length, max_length))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    loss = model(**inputs).loss
    return loss


def check_model(model_list, end, auto_name, forward_func, pass_list=None):
    res = []
    if pass_list is None:
        pass_list = []
    for model_name, model_obj_name in model_list:
        if model_name in pass_list:
            res.append(f'{model_name:<20}: \033[34m - 跳过测试\033[30m')
            continue
        print(f'Test {model_obj_name}{end} ...')
        model_ckp_list, model_1, model_2 = get_checkpoint_name(model_name=model_name,
                                                               model_obj_name=model_obj_name,
                                                               end=end,
                                                               auto_name=auto_name)
        if model_ckp_list is None:
            # res[model_name] = f'\t{model_name:<20}: \033[33m Θ 未定义模型或导入失败\033[30m'
            # print(f'\t{model_name:<20}: \033[33m Θ 未定义模型或导入失败\033[30m')
            # res.append(f'{model_name:<20}: \033[33m Θ 未定义模型或导入失败\033[30m')
            res.append(f'{model_name:<20}: \033[31m ✘ 错误 (未定义模型或导入失败) \033[30m')
            print(f'Test {model_obj_name}{end} End')
        else:
            for pretrained_model_name_or_path in model_ckp_list:
                try:
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        cache_dir='./Cache',
                    )
                    model_1 = model_1.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
                    loss_1 = forward_func(model_1, tokenizer)
                    del model_1
                    model_2 = model_2.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
                    loss_2 = forward_func(model_2, tokenizer)
                    del model_2
                    # print(f'\t{model_name:<20}: \033[32m ✔ 通过\033[30m')
                    # res[model_name] = f'\t{model_name:<20}: \033[32m ✔ 通过\033[30m'
                    res.append(f'{model_name:<20}: \033[32m ✔ 通过\033[30m')
                except NameError as e:
                    if 'attention_mask' in e:
                        res.append(f'{model_name:<20}: \033[31m ✘ 不支持此模型\033[30m')
                    else:
                        res.append(f'{model_name:<20}: \033[31m ✘ 错误\n{traceback.format_exc()}\033[30m')
                except:
                    # print(f'\t{model_name:<20}: \033[31m ✘ 错误\033[30m')
                    # res[model_name] = f'\t{model_name:<20}: \033[31m ✘ 错误\033[30m'
                    res.append(f'{model_name:<20}: \033[31m ✘ 错误\n{traceback.format_exc()}\033[30m')
                    # print(traceback.print_exc())
                else:
                    break
            print(f'Test {model_obj_name}{end} End')
    return res


if __name__ == '__main__':
    from transformers.models.auto.modeling_auto import MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES

    result = {}
    pass_list = [
        # 'gpt2',# 太大跳过测试
        'layoutlm',  # 不合适
        'layoutlmv2',  # 不合适
        'layoutlmv3',  # 不合适
        'markuplm',  # 不合适
        'megatron-bert',  # 不合适
        'fnet',  # 不支持attention_mask
        'xlm-roberta-xl',  # 太大跳过测试
    ]
    result['Biaffine'] = check_model(model_list=MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES.items(),
                                     end='WithBiaffine',
                                     auto_name='AutoModelForTokenClassificationWithBiaffine',
                                     forward_func=forward_func_for_biaffine,
                                     pass_list=pass_list
                                     )

    result['GlobalPointer'] = check_model(model_list=MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES.items(),
                                          end='WithGlobalPointer',
                                          auto_name='AutoModelForTokenClassificationWithGlobalPointer',
                                          forward_func=forward_func_for_global_pointer,
                                          pass_list=pass_list
                                          )

    print('Result:')
    for name, res in result.items():
        print(f'\t{name} Models')
        for i in res:
            print(f'\t\t{i}')
        print()
