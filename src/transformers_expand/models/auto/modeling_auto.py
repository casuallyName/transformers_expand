import warnings
from collections import OrderedDict

from .auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

# AutoModelForTokenClassificationWithBiaffine
MODEL_FOR_TOKEN_CLASSIFICATION_WITH_BIAFFINE_MAPPING_NAMES = OrderedDict(
    [
        # # Model for Token Classification mapping
        ("albert", "AlbertForTokenClassificationWithBiaffine"),
        ("bert", "BertForTokenClassificationWithBiaffine"),
        # ("big_bird", "BigBirdForTokenClassification"),
        # ("bloom", "BloomForTokenClassification"),
        # ("camembert", "CamembertForTokenClassification"),
        # ("canine", "CanineForTokenClassification"),
        # ("convbert", "ConvBertForTokenClassification"),
        # ("data2vec-text", "Data2VecTextForTokenClassification"),
        # ("deberta", "DebertaForTokenClassification"),
        # ("deberta-v2", "DebertaV2ForTokenClassification"),
        # ("distilbert", "DistilBertForTokenClassification"),
        ("electra", "ElectraForTokenClassificationWithBiaffine"),
        ("ernie", "ErnieForTokenClassificationWithBiaffine"),
        # ("esm", "EsmForTokenClassification"),
        # ("flaubert", "FlaubertForTokenClassification"),
        # ("fnet", "FNetForTokenClassification"),
        # ("funnel", "FunnelForTokenClassification"),
        # ("gpt2", "GPT2ForTokenClassification"),
        # ("ibert", "IBertForTokenClassification"),
        # ("layoutlm", "LayoutLMForTokenClassification"),
        # ("layoutlmv2", "LayoutLMv2ForTokenClassification"),
        # ("layoutlmv3", "LayoutLMv3ForTokenClassification"),
        # ("longformer", "LongformerForTokenClassification"),
        # ("luke", "LukeForTokenClassification"),
        # ("markuplm", "MarkupLMForTokenClassification"),
        # ("megatron-bert", "MegatronBertForTokenClassification"),
        # ("mobilebert", "MobileBertForTokenClassification"),
        # ("mpnet", "MPNetForTokenClassification"),
        # ("nezha", "NezhaForTokenClassification"),
        # ("nystromformer", "NystromformerForTokenClassification"),
        # ("qdqbert", "QDQBertForTokenClassification"),
        # ("rembert", "RemBertForTokenClassification"),
        # ("roberta", "RobertaForTokenClassification"),
        # ("roformer", "RoFormerForTokenClassification"),
        # ("squeezebert", "SqueezeBertForTokenClassification"),
        # ("xlm", "XLMForTokenClassification"),
        # ("xlm-roberta", "XLMRobertaForTokenClassification"),
        # ("xlm-roberta-xl", "XLMRobertaXLForTokenClassification"),
        # ("xlnet", "XLNetForTokenClassification"),
        # ("yoso", "YosoForTokenClassification"),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_WITH_BIAFFINE_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_WITH_BIAFFINE_MAPPING_NAMES
)


class AutoModelForTokenClassificationWithBiaffine(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_WITH_BIAFFINE_MAPPING


#  AutoModelForTokenClassificationWithGlobalPointer
MODEL_FOR_TOKEN_CLASSIFICATION_WITH_GLOBAL_POINTER_MAPPING_NAMES = OrderedDict(
    [
        # # Model for Token Classification mapping
        ("albert", "AlbertForTokenClassificationWithBiaffine"),
        ("bert", "BertForTokenClassificationWithGlobalPointer"),
        # ("big_bird", "BigBirdForTokenClassification"),
        # ("bloom", "BloomForTokenClassification"),
        # ("camembert", "CamembertForTokenClassification"),
        # ("canine", "CanineForTokenClassification"),
        # ("convbert", "ConvBertForTokenClassification"),
        # ("data2vec-text", "Data2VecTextForTokenClassification"),
        # ("deberta", "DebertaForTokenClassification"),
        # ("deberta-v2", "DebertaV2ForTokenClassification"),
        # ("distilbert", "DistilBertForTokenClassification"),
        ("electra", "ElectraForTokenClassificationWithGlobalPointer"),
        ("ernie", "ErnieForTokenClassificationWithGlobalPointer"),
        # ("esm", "EsmForTokenClassification"),
        # ("flaubert", "FlaubertForTokenClassification"),
        # ("fnet", "FNetForTokenClassification"),
        # ("funnel", "FunnelForTokenClassification"),
        # ("gpt2", "GPT2ForTokenClassification"),
        # ("ibert", "IBertForTokenClassification"),
        # ("layoutlm", "LayoutLMForTokenClassification"),
        # ("layoutlmv2", "LayoutLMv2ForTokenClassification"),
        # ("layoutlmv3", "LayoutLMv3ForTokenClassification"),
        # ("longformer", "LongformerForTokenClassification"),
        # ("luke", "LukeForTokenClassification"),
        # ("markuplm", "MarkupLMForTokenClassification"),
        # ("megatron-bert", "MegatronBertForTokenClassification"),
        # ("mobilebert", "MobileBertForTokenClassification"),
        # ("mpnet", "MPNetForTokenClassification"),
        # ("nezha", "NezhaForTokenClassification"),
        # ("nystromformer", "NystromformerForTokenClassification"),
        # ("qdqbert", "QDQBertForTokenClassification"),
        # ("rembert", "RemBertForTokenClassification"),
        # ("roberta", "RobertaForTokenClassification"),
        # ("roformer", "RoFormerForTokenClassification"),
        # ("squeezebert", "SqueezeBertForTokenClassification"),
        # ("xlm", "XLMForTokenClassification"),
        # ("xlm-roberta", "XLMRobertaForTokenClassification"),
        # ("xlm-roberta-xl", "XLMRobertaXLForTokenClassification"),
        # ("xlnet", "XLNetForTokenClassification"),
        # ("yoso", "YosoForTokenClassification"),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_WITH_GLOBAL_POINTER_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_WITH_GLOBAL_POINTER_MAPPING_NAMES
)


class AutoModelForTokenClassificationWithGlobalPointer(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_WITH_GLOBAL_POINTER_MAPPING
