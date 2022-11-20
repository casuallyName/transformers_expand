# `transformers_expand.Trainer`
**继承自`trasnformers.Trainer`**
## Parameters

`(model: Union[PreTrainedModel, nn.Module] = None,
args: TrainingArguments = None,
data_collator: Optional[DataCollator] = None,
train_dataset: Optional[Dataset] = None,
eval_dataset: Optional[Dataset] = None,
tokenizer: Optional[PreTrainedTokenizerBase] = None,
model_init: Callable[[], PreTrainedModel] = None,
compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
callbacks: Optional[List[TrainerCallback]] = None,
optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
loss_fnt: torch.nn.Module = None)`

* **model** (PreTrainedModel or torch.nn.Module, *optional*) — The model to train, evaluate or use for predictions. If not
  provided, a model_init must be passed.
* **arg**s (TrainingArguments, *optional*) — The arguments to tweak for training. Will default to a basic instance of
  TrainingArguments with the output_dir set to a directory named tmp_trainer in the current directory if not provided.
* **data_collator** (DataCollator, *optional*) — The function to use to form a batch from a list of elements of
  train_dataset or eval_dataset. Will default to default_data_collator() if no tokenizer is provided, an instance of
  DataCollatorWithPadding otherwise.
* **train_dataset** (torch.utils.data.Dataset or torch.utils.data.IterableDataset, *optional*) — The dataset to use for
  training. If it is a Dataset, columns not accepted by the model.forward() method are automatically removed.
* **eval_dataset** (Union[torch.utils.data.Dataset, Dict[str, torch.utils.data.Dataset]), *optional*) — The dataset to use
  for evaluation. If it is a Dataset, columns not accepted by the model.forward() method are automatically removed. If
  it is a dictionary, it will evaluate on each dataset prepending the dictionary key to the metric name.
* **tokenizer** (PreTrainedTokenizerBase, *optional*) — The tokenizer used to preprocess the data. If provided, will be
  used to automatically pad the inputs the maximum length when batching inputs, and it will be saved along the model to
  make it easier to rerun an interrupted training or reuse the fine-tuned model.
* **model_init** (Callable[[], PreTrainedModel], *optional*) — A function that instantiates the model to be used. If
  provided, each call to train() will start from a new instance of the model as given by this function.
* **compute_metrics** (Callable[[EvalPrediction], Dict], *optional*) — The function that will be used to compute metrics
  at evaluation. Must take a EvalPrediction and return a dictionary string to metric values.
* **callbacks** (List of TrainerCallback, *optional*) — A list of callbacks to customize the training loop. Will add those
  to the list of default callbacks detailed in here.
* **optimizers** (Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR], *optional*) — A tuple containing the
  optimizer and the scheduler to use. Will default to an instance of AdamW on your model and a scheduler given by
  get_linear_schedule_with_warmup() controlled by args.
* **preprocess_logits_for_metrics** (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], *optional*) — A function that
  preprocess the logits right before caching them at each evaluation step. Must take two tensors, the logits and the
  labels, and return the logits once processed as desired. The modifications made by this function will be reflected in
  the predictions received by compute_metrics.
* **loss_fnt** (torch.nn.Module, *optional*) — Loss计算方法


## Function

继承自`trasnformers.Trainer`，但重写了以下方法

* **\_\_init\_\_**  —  重新`transformers.Trainer`的`__init__`方法，用于适配以下新功能：
  1. 新增一个可选参数`loss_fnt`，用来指定Trainer内部损失部分的计算方法，默认为`None`，使用模型自带的损失计算方法
  2. 新增训练阶段的对抗学习功能，具体由[`trainsfomers_expand.TrainingArguments`](https://github.com/casuallyName/transformers_expand/blob/master/docs/TrainingArguments.md)指定。
     * 目前支持对抗训练方案：`FGM`、`PGD`

* **compute_loss** — 训练时每个batch的loss计算方式。 在初始化Trainer时新增`loss_fnt`参数，用来更换其他损失函数。**损失函数应定义为继承自`torch.nn.Module`的模型**。
