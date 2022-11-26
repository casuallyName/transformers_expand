# -*- coding: utf-8 -*-
# @Time     : 2022/11/26 20:15
# @File     : modeling_xlm_roberta.py
# @Author   : Zhou Hang
# @Email    : zhouhang@idataway.com
# @Software : Python 3.7
# @About    :

from transformers.utils import add_start_docstrings, logging
from ..roberta.modeling_roberta import (
    RobertaForTokenClassificationWithBiaffine,
    RobertaForTokenClassificationWithGlobalPointer,

)
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLM_ROBERTA_START_DOCSTRING,
)

from transformers.models.xlm_roberta.configuration_xlm_roberta import (
    XLMRobertaConfig,
)

logger = logging.get_logger(__name__)


@add_start_docstrings(
    """
    XLM-RoBERTa Model with a token classification head on top (a biaffine layer on top of the hidden-states output) 
    e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForTokenClassificationWithBiaffine(RobertaForTokenClassificationWithBiaffine):
    """
    This class overrides [`RobertaForTokenClassificationWithBiaffine`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model with a token classification head on top (a global pointer layer on top of the hidden-states output) 
    e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForTokenClassificationWithGlobalPointer(RobertaForTokenClassificationWithGlobalPointer):
    """
    This class overrides [`RobertaForTokenClassificationWithGlobalPointer`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
