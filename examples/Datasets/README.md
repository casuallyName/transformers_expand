# THUCNews

## Multi-Class

### dataset example

```
>>> raw_datasets
DatasetDict({
    train: Dataset({
        features: ['idx', 'sentence', 'label'],
        num_rows: 5000
    })
    validation: Dataset({
        features: ['idx', 'sentence', 'label'],
        num_rows: 10000
    })
    test: Dataset({
        features: ['idx', 'sentence', 'label'],
        num_rows: 10000
    })
})

>>> raw_datasets['train'][0]
{'idx': 0, 'sentence': '传闻《最终幻想13》日版偷跑（组图）', 'label': 8}
```

### datasets features example

```python
features = datasets.Features(
    {
        "idx": datasets.Value("string"),
        "sentence": datasets.Value("string"),
        "label": datasets.ClassLabel(names=_LABELS),
    }
)
```

## Multi-Label

### dataset example

* 使用THCNNews数据集举例，由于**THCNNews**为单标签数据集，
  所以每条数据Label内都只有一条数据，实际Label内可以为多个。
* 如：`{'idx': 0, 'sentence': '传闻《最终幻想13》日版偷跑（组图）', 'label': [6, 8]}`

```
>>> raw_datasets
DatasetDict({
    train: Dataset({
        features: ['idx', 'sentence', 'label'],
        num_rows: 5000
    })
    validation: Dataset({
        features: ['idx', 'sentence', 'label'],
        num_rows: 10000
    })
    test: Dataset({
        features: ['idx', 'sentence', 'label'],
        num_rows: 10000
    })
})

>>> raw_datasets['train'][0]
{'idx': 0, 'sentence': '传闻《最终幻想13》日版偷跑（组图）', 'label': [8]}
```

### datasets features example

```python
features = datasets.Features(
    {
        "idx": datasets.Value("string"),
        "sentence": datasets.Value("string"),
        "label": datasets.Sequence(datasets.ClassLabel(names=_LABELS)),
    }
)
```

# ClueNER

## dataset example

```
>>> raw_datasets
DatasetDict({
    train: Dataset({
        features: ['tokens', 'entities'],
        num_rows: 10748
    })
    validation: Dataset({
        features: ['tokens', 'entities'],
        num_rows: 1343
    })
    test: Dataset({
        features: ['tokens', 'entities'],
        num_rows: 1345
    })
})
>>> raw_datasets['train'][0]
{
    'tokens': ['浙', '商', '银', '行', '企', '业', '信', '贷', '部', '叶', '老', '桂', '博', '士', '则', '从', '另',
               '一', '个', '角', '度', '对', '五', '道', '门', '槛', '进', '行', '了', '解', '读', '。', '叶', '老',
               '桂', '认', '为', '，', '对', '目', '前', '国', '内', '商', '业', '银', '行', '而', '言', '，'],
    'entities': {'type': [6, 2], 'start_idx': [9, 0], 'end_idx': [11, 3]}
}
```

## dataset example

```python
features = datasets.Features(
    {
        "tokens": datasets.Sequence(datasets.Value("string")),
        "entities": datasets.Sequence(datasets.Features({
            'type': datasets.features.ClassLabel(names=_Labels),
            "start_idx": datasets.Value("int64"),
            "end_idx": datasets.Value("int64"),
        })
        ),

    }
)
```
