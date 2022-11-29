# -*- coding: utf-8 -*-
# @Time     : 2022/11/20 13:25
# @File     : adversarial_utils.py.py
# @Author   : Zhou Hang
# @Email    : zhouhang@idataway.com
# @Software : Python 3.7
# @About    :

import importlib
import logging

logger = logging.getLogger(__name__)


def load_adversarial(name, trainer):
    adversarial_module = importlib.import_module("transformers_expand.nn.adversarial")
    if name is None or name == 'none':
        return None
    if hasattr(adversarial_module, name):
        return getattr(adversarial_module, name)(trainer)
    else:
        raise ImportError(f"cannot import name '{name}' from 'transformers_expand.nn.adversarial' ")

# def load_adversarial(adversarial_name,trainer):
#     return
