# `transformers_expand.MultiLabelCategoricalCrossEntropyLoss`

> Reference: <br>
> [1] https://spaces.ac.cn/archives/7359

softmax的多标签分类推广<br>

* **input和target的shape一致，target的元素非0即1，1表示对应的类为目标类，0表示对应的类为非目标类。
  保证input的值域是全体实数，换言之一般情况下input不用加激活函数，尤其是不能加sigmoid或者softmax！
  预测阶段则输出input大于0的类。**

## Parameters

* **reduction**  (str, defaults to *mean*) — `mean`, `sum`, `none`

## Example

```python
import torch
from transformers_expand import MultiLabelCategoricalCrossEntropyLoss

loss_fnt = MultiLabelCategoricalCrossEntropyLoss()

loss_fnt(torch.rand(2, 3), torch.zeros(size=(2, 3)))
# tensor(1.9542)
```

# `transformers_expand.MultiLabelCategoricalForNerCrossEntropyLoss`

softmax的多标签分类推广的NER应用<br>

* 本质与`MultiLabelCategoricalCrossEntropyLoss`相同，只是做了一步维度变化
* **input和target的shape一致，target的元素非0即1，1表示对应的类为目标类，0表示对应的类为非目标类。
  保证input的值域是全体实数，换言之一般情况下input不用加激活函数，尤其是不能加sigmoid或者softmax！
  预测阶段则输出input大于0的类。**

## Parameters

* **batch_size**  (int) — 输入的batch大小
* **ent_type_size** (int) — 实体类型数量
* **reduction**  (str, defaults to *mean*) — `mean`, `sum`, `none`

## Example

```python
import torch
from transformers_expand import MultiLabelCategoricalForNerCrossEntropyLoss

loss_fnt = MultiLabelCategoricalForNerCrossEntropyLoss(batch_size=1, ent_type_size=3)

loss_fnt(torch.rand(1, 2, 2, 3), torch.zeros(size=(1, 2, 2, 3)))
# tensor(2.1399)
```

# `transformers_expand.SpanLoss`

## Parameters

* **reduction**  (str, defaults to *mean*) — `mean`, `sum`, `none`

## Example

```python
import torch
from transformers_expand import SpanLoss

loss_fnt = SpanLoss()
loss_fnt(span_logits=torch.rand(1, 5, 5, 3),
         span_label=torch.zeros(size=(1, 5, 5), dtype=torch.int64),
         sequence_mask=torch.ones(size=(1, 5, 5), dtype=torch.int64))
# tensor(28.0590)
```