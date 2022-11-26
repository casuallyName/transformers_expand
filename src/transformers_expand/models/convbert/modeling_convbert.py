# -*- coding: utf-8 -*-
# @Time     : 2022/11/26 14:29
# @File     : modeling_convbert.py
# @Author   : Zhou Hang
# @Email    : zhouhang@idataway.com
# @Software : Python 3.7
# @About    :


from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, \
    logging
from transformers.models.convbert.modeling_convbert import (
    CONVBERT_START_DOCSTRING,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    CONVBERT_INPUTS_DOCSTRING,
    ConvBertPreTrainedModel,
    ConvBertModel
)

from ...nn import (
    GlobalPointer,
    EfficientGlobalPointer,
    MultiLabelCategoricalForNerCrossEntropyLoss,
    Biaffine,
    SpanLoss,
)

logger = logging.get_logger(__name__)


@add_start_docstrings(
    """
    ConvBERT Model with a token classification head on top (a biaffine layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    CONVBERT_START_DOCSTRING,
)
class ConvBertForTokenClassificationWithBiaffine(ConvBertPreTrainedModel):
    def __init__(self, config, biaffine_input_size: int = None, use_lstm: bool = None):
        super().__init__(config)
        # 此处 +1 操作是由于实体标签类别中含有一个 "非实体" 类别，即label map中的 0
        self.num_labels = config.num_labels + 1

        if use_lstm is not None and hasattr(config, 'use_lstm') and config.use_lstm != use_lstm:
            logger.warning(
                f"Parameter conflict, user set use_lstm is {use_lstm}, but config.use_lstm is {config.use_lstm}. "
                f"Will ignore 'use_lstm={use_lstm}'.")
        elif not hasattr(config, 'use_lstm'):
            config.use_lstm = use_lstm if use_lstm is not None else False

        if biaffine_input_size is not None and hasattr(config,
                                                       'biaffine_input_size') and config.biaffine_input_size != biaffine_input_size:
            logger.warning(
                f"Parameter conflict, user set biaffine_input_size is {biaffine_input_size}, but config.biaffine_input_size is {config.biaffine_input_size}. "
                f"Will ignore 'config.biaffine_input_size={config.biaffine_input_size}' instead of 'biaffine_input_size={biaffine_input_size}'.")
        elif not hasattr(config, 'biaffine_input_size'):
            config.biaffine_input_size = biaffine_input_size if biaffine_input_size is not None else 128

        self.use_lstm = config.use_lstm
        self.biaffine_input_size = config.biaffine_input_size
        self.convbert = ConvBertModel(config)

        if self.use_lstm:
            self.lstm = torch.nn.LSTM(input_size=768,
                                      hidden_size=768,
                                      num_layers=1,
                                      batch_first=True,
                                      dropout=0.5,
                                      bidirectional=True)
            self.start_layer = torch.nn.Sequential(
                torch.nn.Linear(in_features=2 * self.config.hidden_size, out_features=self.biaffine_input_size),
                torch.nn.ReLU())
            self.end_layer = torch.nn.Sequential(
                torch.nn.Linear(in_features=2 * self.config.hidden_size, out_features=self.biaffine_input_size),
                torch.nn.ReLU())
        else:
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)
            self.start_layer = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.config.hidden_size, out_features=self.biaffine_input_size),
                torch.nn.ReLU())
            self.end_layer = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.config.hidden_size, out_features=self.biaffine_input_size),
                torch.nn.ReLU())

        self.biaffne_layer = Biaffine(self.biaffine_input_size, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="{'entity':'小明', 'type':'PER', 'start':3, 'end':4}",
    )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            sequence_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.convbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.use_lstm:
            sequence_output, _ = self.lstm(sequence_output)
        else:
            sequence_output = self.dropout(sequence_output)

        start_logits = self.start_layer(sequence_output)
        end_logits = self.end_layer(sequence_output)

        logits = self.biaffne_layer(start_logits, end_logits)
        logits = logits.contiguous()

        loss = None
        if labels is not None and sequence_mask is not None:
            loss_fct = SpanLoss()
            loss = loss_fct(span_logits=logits, span_label=labels, sequence_mask=sequence_mask)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    ConvBERT Model with a token classification head on top (a global pointer layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    CONVBERT_START_DOCSTRING,
)
class ConvBertForTokenClassificationWithGlobalPointer(ConvBertPreTrainedModel):
    def __init__(self, config, inner_dim: int = None, use_efficient: bool = None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.convbert = ConvBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        if inner_dim is not None and hasattr(config, 'inner_dim') and config.inner_dim != inner_dim:
            logger.warning(
                f"Parameter conflict, user set inner_dim is {inner_dim}, but config.inner_dim is {config.inner_dim}. "
                f"Will ignore 'inner_dim={inner_dim}'.")
        elif not hasattr(config, 'inner_dim'):
            config.inner_dim = inner_dim if inner_dim is not None else 64

        if use_efficient is not None and hasattr(config, 'use_efficient') and config.use_efficient != use_efficient:
            logger.warning(
                f"Parameter conflict, use_efficient is {use_efficient} and config.use_efficient is {config.use_efficient}. "
                f"Will ignore 'use_efficient={use_efficient}'.")
        elif not hasattr(config, 'use_efficient'):
            config.use_efficient = use_efficient if use_efficient is not None else False

        if config.use_efficient:
            self.global_pointer = EfficientGlobalPointer(
                config.num_labels,
                config.inner_dim,
                config.hidden_size
            )
        else:
            self.global_pointer = GlobalPointer(
                config.num_labels,
                config.inner_dim,
                config.hidden_size
            )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="{'entity':'小明', 'type':'PER', 'start':3, 'end':4}",
    )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.convbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        logits = self.global_pointer(sequence_output, mask=attention_mask)

        loss = None
        if labels is not None:
            batch_size, ent_type_size = labels.shape[:2]
            loss_fct = MultiLabelCategoricalForNerCrossEntropyLoss(batch_size, ent_type_size)
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
