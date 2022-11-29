# -*- coding: utf-8 -*-
# @Time     : 2022/11/27 11:20
# @File     : modeling_flaubert.py.py
# @Author   : Zhou Hang
# @Email    : zhouhang@idataway.com
# @Software : Python 3.7
# @About    :

from transformers.utils import add_start_docstrings, logging
from ..xlm.modeling_xlm import (
    # XLMForMultipleChoice,
    # XLMForQuestionAnswering,
    # XLMForQuestionAnsweringSimple,
    # XLMForSequenceClassification,
    XLMForTokenClassificationWithBiaffine,
    XLMForTokenClassificationWithGlobalPointer,
    # XLMModel,
    # XLMWithLMHeadModel,
    # get_masks,
)
from transformers.models.flaubert.modeling_flaubert import (
    FLAUBERT_START_DOCSTRING,
    FlaubertConfig,
    FlaubertModel
)

logger = logging.get_logger(__name__)


@add_start_docstrings(
    """
    Flaubert Model with a token classification head on top (a biaffine layer on top of the hidden-states output) 
    e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
class FlaubertForTokenClassificationWithBiaffine(XLMForTokenClassificationWithBiaffine):
    """
    This class overrides [`XLMForTokenClassification`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = FlaubertConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = FlaubertModel(config)
        # Initialize weights and apply final processing
        self.post_init()


@add_start_docstrings(
    """
    Flaubert Model with a token classification head on top (a global pointer layer on top of the hidden-states output) 
    e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
class FlaubertForTokenClassificationWithGlobalPointer(XLMForTokenClassificationWithGlobalPointer):
    """
    This class overrides [`XLMForTokenClassification`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = FlaubertConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = FlaubertModel(config)
        # Initialize weights and apply final processing
        self.post_init()
