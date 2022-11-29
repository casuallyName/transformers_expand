# AutoModel

## `transformers_expand.AutoModelForTokenClassificationWithGlobalPointer`

### `.from_config`

#### Parameters

* **其他参数与`transformers.AutoModelForTokenClassification`参数相同**
* **inner_dim** (int, defaults to *64*) — GlobalPointer或EfficientGlobalPointer层inner_dim大小
* **use_efficient** (bool, defaults to *False*) — 是否使用 EfficientGlobalPointer 替换 GlobalPointer

#### Example

```python
from transformers import AutoConfig
from transformers_expand import AutoModelForTokenClassificationWithGlobalPointer

config = AutoConfig.from_pretrained("bert-base-chinese")
model = AutoModelForTokenClassificationWithGlobalPointer.from_config(config)
```

### `.from_pretrained`

#### Parameters

* **其他参数与`transformers.AutoModelForTokenClassification`参数相同**
* **inner_dim** (int, defaults to *64*) — GlobalPointer或EfficientGlobalPointer层inner_dim大小
* **use_efficient** (bool, defaults to *False*) — 是否使用 EfficientGlobalPointer 替换 GlobalPointer

#### Example

```python
from transformers_expand import AutoModelForTokenClassificationWithGlobalPointer

model = AutoModelForTokenClassificationWithGlobalPointer.from_pretrained("bert-base-chinese")
```

## `transformers_expand.AutoModelForTokenClassificationWithBiaffine`

### `.from_config`

#### Parameters

* **其他参数与`transformers.AutoModelForTokenClassification`参数相同**
* **biaffine_input_size** (int, defaults to *128*) — Biaffine层input_size大小
* **use_lstm** (bool, defaults to *False*) — 在Encoder层与Biaffine层之间是否添加一层LSTM

#### Example

```python
from transformers import AutoConfig
from transformers_expand import AutoModelForTokenClassificationWithBiaffine

config = AutoConfig.from_pretrained("bert-base-chinese")
model = AutoModelForTokenClassificationWithBiaffine.from_config(config)
```

### `.from_pretrained`

#### Parameters

* **其他参数与`transformers.AutoModelForTokenClassification`参数相同**
* **biaffine_input_size** (int, defaults to *128*) — Biaffine层input_size大小
* **use_lstm** (bool, defaults to *False*) — 在Encoder层与Biaffine层之间是否添加一层LSTM

#### Example

```python
from transformers_expand import AutoModelForTokenClassificationWithBiaffine

# 使用 EfficientGlobalPointer
model = AutoModelForTokenClassificationWithBiaffine.from_pretrained("bert-base-chinese")
```

# Bert

## `transformers_expand.BertForTokenClassificationWithGlobalPointer`

### `.from_config`

#### Parameters

* **其他参数与`transformers.BertForTokenClassification`参数相同**
* **inner_dim** (int, defaults to *64*) — GlobalPointer或EfficientGlobalPointer层inner_dim大小
* **use_efficient** (bool, defaults to *False*) — 是否使用 EfficientGlobalPointer 替换 GlobalPointer

#### Example

```python
from transformers import AutoConfig
from transformers_expand import BertForTokenClassificationWithGlobalPointer

config = AutoConfig.from_pretrained("bert-base-chinese")
model = BertForTokenClassificationWithGlobalPointer.from_config(config)
```

### `.from_pretrained`

#### Parameters

* **其他参数与`transformers.BertForTokenClassification`参数相同**
* **inner_dim** (int, defaults to *64*) — GlobalPointer或EfficientGlobalPointer层inner_dim大小
* **use_efficient** (bool, defaults to *False*) — 是否使用 EfficientGlobalPointer 替换 GlobalPointer

#### Example

```python
from transformers_expand import BertForTokenClassificationWithGlobalPointer

model = BertForTokenClassificationWithGlobalPointer.from_pretrained("bert-base-chinese")
```

## `transformers_expand.BertForTokenClassificationWithBiaffine`

### `.from_config`

#### Parameters

* **其他参数与`transformers.BertForTokenClassification`参数相同**
* **biaffine_input_size** (int, defaults to *128*) — Biaffine层input_size大小
* **use_lstm** (bool, defaults to *False*) — 在Encoder层与Biaffine层之间是否添加一层LSTM

#### Example

```python
from transformers import AutoConfig
from transformers_expand import BertForTokenClassificationWithBiaffine

config = AutoConfig.from_pretrained("bert-base-chinese")
model = BertForTokenClassificationWithBiaffine.from_config(config)
```

### `.from_pretrained`

#### Parameters

* **其他参数与`transformers.BertForTokenClassification`参数相同**
* **biaffine_input_size** (int, defaults to *128*) — Biaffine层input_size大小
* **use_lstm** (bool, defaults to *False*) — 在Encoder层与Biaffine层之间是否添加一层LSTM

#### Example

```python
from transformers_expand import BertForTokenClassificationWithBiaffine

# 使用 EfficientGlobalPointer
model = BertForTokenClassificationWithBiaffine.from_pretrained("bert-base-chinese")
```

# Electral

## `transformers_expand.ElectraForTokenClassificationWithGlobalPointer`

### `.from_config`

#### Parameters

* **其他参数与`transformers.ElectraForTokenClassification`参数相同**
* **inner_dim** (int, defaults to *64*) — GlobalPointer或EfficientGlobalPointer层inner_dim大小
* **use_efficient** (bool, defaults to *False*) — 是否使用 EfficientGlobalPointer 替换 GlobalPointer

#### Example

```python
from transformers import AutoConfig
from transformers_expand import ElectraForTokenClassificationWithGlobalPointer

config = AutoConfig.from_pretrained("hfl/chinese-electra-180g-base-discriminator")
model = ElectraForTokenClassificationWithGlobalPointer.from_config(config)
```

### `.from_pretrained`

#### Parameters

* **其他参数与`transformers.ElectraForTokenClassification`参数相同**
* **inner_dim** (int, defaults to *64*) — GlobalPointer或EfficientGlobalPointer层inner_dim大小
* **use_efficient** (bool, defaults to *False*) — 是否使用 EfficientGlobalPointer 替换 GlobalPointer

#### Example

```python
from transformers_expand import ElectraForTokenClassificationWithGlobalPointer

model = ElectraForTokenClassificationWithGlobalPointer.from_pretrained("hfl/chinese-electra-180g-base-discriminator")
```

## `transformers_expand.ElectraForTokenClassificationWithBiaffine`

### `.from_config`

#### Parameters

* **其他参数与`transformers.ElectraForTokenClassification`参数相同**
* **biaffine_input_size** (int, defaults to *128*) — Biaffine层input_size大小
* **use_lstm** (bool, defaults to *False*) — 在Encoder层与Biaffine层之间是否添加一层LSTM

#### Example

```python
from transformers import AutoConfig
from transformers_expand import ElectraForTokenClassificationWithBiaffine

config = AutoConfig.from_pretrained("hfl/chinese-electra-180g-base-discriminator")
model = ElectraForTokenClassificationWithBiaffine.from_config(config)
```

### `.from_pretrained`

#### Parameters

* **其他参数与`transformers.ElectraForTokenClassification`参数相同**
* **biaffine_input_size** (int, defaults to *128*) — Biaffine层input_size大小
* **use_lstm** (bool, defaults to *False*) — 在Encoder层与Biaffine层之间是否添加一层LSTM

#### Example

```python
from transformers_expand import ElectraForTokenClassificationWithBiaffine

# 使用 EfficientGlobalPointer
model = ElectraForTokenClassificationWithBiaffine.from_pretrained("hfl/chinese-electra-180g-base-discriminator")
```

# Ernie

## `transformers_expand.ErnieForTokenClassificationWithGlobalPointer`

### `.from_config`

#### Parameters

* **其他参数与`transformers.ErnieForTokenClassification`参数相同**
* **inner_dim** (int, defaults to *64*) — GlobalPointer或EfficientGlobalPointer层inner_dim大小
* **use_efficient** (bool, defaults to *False*) — 是否使用 EfficientGlobalPointer 替换 GlobalPointer

#### Example

```python
from transformers import AutoConfig
from transformers_expand import ErnieForTokenClassificationWithGlobalPointer

config = AutoConfig.from_pretrained("nghuyong/ernie-3.0-micro-zh")
model = ErnieForTokenClassificationWithGlobalPointer.from_config(config)
```

### `.from_pretrained`

#### Parameters

* **其他参数与`transformers.ErnieForTokenClassification`参数相同**
* **inner_dim** (int, defaults to *64*) — GlobalPointer或EfficientGlobalPointer层inner_dim大小
* **use_efficient** (bool, defaults to *False*) — 是否使用 EfficientGlobalPointer 替换 GlobalPointer

#### Example

```python
from transformers_expand import ErnieForTokenClassificationWithGlobalPointer

model = ErnieForTokenClassificationWithGlobalPointer.from_pretrained("nghuyong/ernie-3.0-micro-zh")
```

## `transformers_expand.ErnieForTokenClassificationWithBiaffine`

### `.from_config`

#### Parameters

* **其他参数与`transformers.ErnieForTokenClassification`参数相同**
* **biaffine_input_size** (int, defaults to *128*) — Biaffine层input_size大小
* **use_lstm** (bool, defaults to *False*) — 在Encoder层与Biaffine层之间是否添加一层LSTM

#### Example

```python
from transformers import AutoConfig
from transformers_expand import ErnieForTokenClassificationWithBiaffine

config = AutoConfig.from_pretrained("nghuyong/ernie-3.0-micro-zh")
model = ErnieForTokenClassificationWithBiaffine.from_config(config)
```

### `.from_pretrained`

#### Parameters

* **其他参数与`transformers.ErnieForTokenClassification`参数相同**
* **biaffine_input_size** (int, defaults to *128*) — Biaffine层input_size大小
* **use_lstm** (bool, defaults to *False*) — 在Encoder层与Biaffine层之间是否添加一层LSTM

#### Example

```python
from transformers_expand import ErnieForTokenClassificationWithBiaffine

# 使用 EfficientGlobalPointer
model = ErnieForTokenClassificationWithBiaffine.from_pretrained("nghuyong/ernie-3.0-micro-zh")
```