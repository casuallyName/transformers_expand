import torch
from torch import Tensor

from typing import Callable, Optional

__all__ = ['multi_label_categorical_cross_entropy']


def multi_label_categorical_cross_entropy(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """多标签分类的交叉熵
    说明：target和input的shape一致，target的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证input的值域是全体实数，换言之一般情况下input
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出input大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    input = (1 - 2 * target) * input
    input_neg = input - target * 1e12
    input_pos = input - (1 - target) * 1e12
    zeros = torch.zeros_like(input[..., :1])
    input_pos = torch.cat([input_pos, zeros], dim=-1)
    input_neg = torch.cat([input_neg, zeros], dim=-1)
    neg_loss = torch.logsumexp(input_neg, dim=-1)
    pos_loss = torch.logsumexp(input_pos, dim=-1)
    if reduction == 'mean':
        return (neg_loss + pos_loss).mean()
    elif reduction == 'sum':
        return (neg_loss + pos_loss).sum()
    else:
        return neg_loss + pos_loss
