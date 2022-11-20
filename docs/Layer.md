# GlobalPointer

> Reference: <br>
[1] [GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://spaces.ac.cn/archives/8373) <br>
[2] [Efficient GlobalPointer：少点参数，多点效果](https://spaces.ac.cn/archives/8877)

## `transformers_expand.nn.GlobalPointer`

> Reference: <br>
[1] [GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://spaces.ac.cn/archives/8373)  <br>
[2] https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py

### Parameters

* **heads** (int) — 实体数量
* **head_size**(int) — inner dim
* **hidden_size** Encoder部分隐层大小
* **RoPE** (bool, defaults to *True*) 是否使用位置编码

## `transformers_expand.nn.EfficientGlobalPointer`

> Reference: <br>
[1] [Efficient GlobalPointer：少点参数，多点效果](https://spaces.ac.cn/archives/8877)<br>
[2] https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py

### Parameters

* **heads** (int) — 实体数量
* **head_size**(int) — inner dim
* **hidden_size** Encoder部分隐层大小
* **RoPE** (bool, defaults to *True*) 是否使用位置编码

## `transformers_expand.nn.Biaffine`

Biaffine双仿射机制
> Reference: <br>
[1] [Named Entity Recognition as Dependency Parsing](https://aclanthology.org/2020.acl-main.577.pdf)<br>
[2] [实体识别之Biaffine双仿射注意力机制](https://zhuanlan.zhihu.com/p/369851456) <br>
[3] https://github.com/suolyer/PyTorch_BERT_Biaffine_NER/blob/main/model/model.py

### Parameters

* **in_features** (int) — 输入大小
* **out_features** (int) — 输出大小
* **bias_x** (bool, defaults to *True*)
* **bias_y** (bool, defaults to *True*)

## `transformers_expand.nn.SinusoidalPositionEmbedding`

RoPE 位置编码
> Reference: <br>
https://github.com/bojone/bert4keras/blob/70a7eb9ace18b9f4806b6386e5183f32d024bc37/bert4keras/layers.py#L849

### Parameters

* **output_dim** (int) — 输出大小
* **merge_mode** (str, defaults to *add*) — 合并方式
* **custom_position_ids** (bool, defaults to *False*) 


