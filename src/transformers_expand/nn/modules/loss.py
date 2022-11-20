import torch
from torch.nn.modules.loss import _Loss
from .. import functional as F


class MultiLabelCategoricalCrossEntropyLoss(_Loss):
    """
    Reference:
        [1] https://spaces.ac.cn/archives/7359

    多标签分类的交叉熵
    说明：target和input的shape一致，target的元素非0即1，1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证input的值域是全体实数，换言之一般情况下input不用加激活函数，尤其是不能加sigmoid或者softmax！
         预测阶段则输出input大于0的类。如有疑问，请仔细阅读并理解本文。
    """

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MultiLabelCategoricalCrossEntropyLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.multi_label_categorical_cross_entropy(input, target, reduction=self.reduction)


class MultiLabelCategoricalForNerCrossEntropyLoss(_Loss):
    def __init__(self, batch_size, ent_type_size, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MultiLabelCategoricalForNerCrossEntropyLoss, self).__init__(size_average, reduce, reduction)
        self.reshape_size = batch_size * ent_type_size

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.reshape(self.reshape_size, -1)
        target = target.reshape(self.reshape_size, -1)

        return F.multi_label_categorical_cross_entropy(input, target, reduction=self.reduction)


class SpanLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean') -> None:
        super(SpanLoss, self).__init__(size_average, reduce, reduction)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, span_logits, span_label, sequence_mask):
        span_label = span_label.view(size=(-1,))
        span_logits = span_logits.view(size=(-1, span_logits.shape[-1]))
        span_loss = self.loss_func(input=span_logits, target=span_label)
        span_mask = sequence_mask.view(size=(-1,))
        span_loss *= span_mask
        if self.reduction == 'mean':
            return torch.sum(span_loss) / sequence_mask.size()[0]
        else:
            return span_loss
