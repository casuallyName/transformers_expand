# `transformers_expand.metrics.MetricsForBiaffineTask`

* 适用于由Biaffine层构建的NER网络

## Example

```python
import numpy as np
from transformers_expand.metrics import MetricsForBiaffineTask
from transformers.trainer_utils import PredictionOutput

compute_metrics = MetricsForBiaffineTask()
output = PredictionOutput(
    predictions=np.random.rand(1, 10, 10, 2),
    label_ids=(np.random.rand(1, 10, 10) > 0.5).astype('int'),
    metrics=None
)
compute_metrics(output=output)
# {'precision': 0.56, 'recall': 0.55, 'F1_score': 0.55, 'Combined_score': 0.55}
```

# `transformers_expand.metrics.MetricsForMetricsForGlobalPointerTask`

* 适用于由GlobalPointer层构建的NER网络

## Example

```python
import numpy as np
from transformers_expand.metrics import MetricsForGlobalPointerTask
from transformers.trainer_utils import PredictionOutput

compute_metrics = MetricsForGlobalPointerTask()
output = PredictionOutput(
    predictions=np.random.rand(1, 2, 10, 10),
    label_ids=(np.random.rand(1, 2, 10, 10) > 0.5).astype('int'),
    metrics=None
)
compute_metrics(output=output)
# {'precision': 0.5, 'recall': 1.0, 'F1_score': 0.66, 'Combined_score': 0.72}
```

# `transformers_expand.metrics.MetricsForCommonTextClassificationTask`

* 适用于文本分类任务

## Parameters

* **average**  (str, defaults to *macro*) — 平均类型，`micro`, `macro`
* **inner_dim** (int, defaults to *64*) — GlobalPointer或EfficientGlobalPointer层inner_dim大小
* **multi_label** (bool, defaults to *False*) — 是否按照多标签（multi-label）计算
* **mark_line** (int, defaults to *0.5*) — 多标签模式下大于该值则被标记
* **use_sigmoid**  (bool, defaults to *True*) — 是否对预测结果使用sigmoid，该选项仅在多标签模式下生效

## Example

```python
import numpy as np
from transformers_expand.metrics import MetricsForCommonTextClassificationTask
from transformers.trainer_utils import PredictionOutput

compute_metrics = MetricsForCommonTextClassificationTask()
output = PredictionOutput(
    predictions=np.random.rand(5, 3),
    label_ids=np.array([[0], [1], [2], [0], [2]]),
    metrics=None
)
compute_metrics(output=output)
# {'Precision': 0.11, 'Recall': 0.17, 'F1_score': 0.13, 'combined_score': 0.14}
```