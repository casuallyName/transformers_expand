# -*- coding: utf-8 -*-
# @Time     : 2022/11/15 22:14
# @File     : biaffine.py
# @Author   : Zhou Hang
# @Email    : zhouhang@idataway.com
# @Software : Python 3.7
# @About    :

import torch
import torch.nn as nn


class Biaffine(nn.Module):
    def __init__(self, in_features, out_features, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.in_features = in_features
        self.out_features = out_features

        self.U = torch.nn.Parameter(
            torch.randn(self.in_features + int(bias_x), self.out_features, self.in_features + int(bias_y))
        )

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias_x={}, bias_y={}'.format(
            self.in_features, self.out_features, self.bias_x, self.bias_y
        )
