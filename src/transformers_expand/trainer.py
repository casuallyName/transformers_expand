import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import transformers

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.data.data_collator import DataCollator
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers_expand.nn import load_adversarial
from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
    logging,
)

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_apex_available():
    from apex import amp

# from transformers import BertForSequenceClassification
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


class Trainer(transformers.Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
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
            loss_fnt: nn.Module = None,
    ):
        """
        添加自定义Loss函数

        Args:
            model:
            args:
            data_collator:
            train_dataset:
            eval_dataset:
            tokenizer:
            model_init:
            compute_metrics:
            callbacks:
            optimizers:
            preprocess_logits_for_metrics:
            loss_fnt:  Loss 函数 loss_fnt(outputs.logits, labels)
        """
        super(Trainer, self).__init__(model,
                                      args,
                                      data_collator,
                                      train_dataset,
                                      eval_dataset,
                                      tokenizer,
                                      model_init,
                                      compute_metrics,
                                      callbacks,
                                      optimizers,
                                      preprocess_logits_for_metrics)
        self.loss_fnt = loss_fnt
        if self.loss_fnt is not None:
            self.loss_fnt.to(self.args.device)

        if hasattr(self.args, 'adversarial'):
            adversarial_name = getattr(self.args, 'adversarial')
            if adversarial_name is None or adversarial_name == 'none':
                self.adversarial = None
            else:

                self.adversarial = load_adversarial(name=adversarial_name, trainer=self)
                logger.info(f"Init adversarial {self.adversarial}")
        else:
            self.adversarial = None

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        使用自定义Loss函数

        Args:
            model:
            inputs:
            return_outputs:

        Returns:

        """
        if (self.loss_fnt is not None or self.label_smoother is not None) and "labels" in inputs:
            if self.adversarial:
                labels = inputs["labels"]
            else:
                labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is not None:
            if self.loss_fnt is not None:
                loss = self.loss_fnt(outputs.logits, labels)
            elif unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        if self.adversarial:
            self.adversarial(inputs)

        return loss.detach()
