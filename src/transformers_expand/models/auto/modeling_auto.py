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
        ("big_bird", "BigBirdForTokenClassificationWithBiaffine"),
        ("bloom", "BloomForTokenClassificationWithBiaffine"),
        ("camembert", "CamembertForTokenClassificationWithBiaffine"),
        ("canine", "CanineForTokenClassificationWithBiaffine"),
        ("convbert", "ConvBertForTokenClassificationWithBiaffine"),
        ("data2vec-text", "Data2VecTextForTokenClassificationWithBiaffine"),
        ("deberta", "DebertaForTokenClassificationWithBiaffine"),
        ("deberta-v2", "DebertaV2ForTokenClassificationWithBiaffine"),
        ("distilbert", "DistilBertForTokenClassificationWithBiaffine"),
        ("electra", "ElectraForTokenClassificationWithBiaffine"),
        ("ernie", "ErnieForTokenClassificationWithBiaffine"),
        # ("esm", "EsmForTokenClassificationWithBiaffine"),
        # ("flaubert", "FlaubertForTokenClassificationWithBiaffine"),
        # ("fnet", "FNetForTokenClassificationWithBiaffine"),
        # ("funnel", "FunnelForTokenClassificationWithBiaffine"),
        # ("gpt2", "GPT2ForTokenClassificationWithBiaffine"),
        # ("ibert", "IBertForTokenClassificationWithBiaffine"),
        # ("layoutlm", "LayoutLMForTokenClassificationWithBiaffine"),
        # ("layoutlmv2", "LayoutLMv2ForTokenClassificationWithBiaffine"),
        # ("layoutlmv3", "LayoutLMv3ForTokenClassificationWithBiaffine"),
        # ("longformer", "LongformerForTokenClassificationWithBiaffine"),
        # ("luke", "LukeForTokenClassificationWithBiaffine"),
        # ("markuplm", "MarkupLMForTokenClassificationWithBiaffine"),
        # ("megatron-bert", "MegatronBertForTokenClassificationWithBiaffine"),
        # ("mobilebert", "MobileBertForTokenClassificationWithBiaffine"),
        # ("mpnet", "MPNetForTokenClassificationWithBiaffine"),
        # ("nezha", "NezhaForTokenClassificationWithBiaffine"),
        # ("nystromformer", "NystromformerForTokenClassificationWithBiaffine"),
        # ("qdqbert", "QDQBertForTokenClassificationWithBiaffine"),
        # ("rembert", "RemBertForTokenClassificationWithBiaffine"),
        ("roberta", "RobertaForTokenClassificationWithBiaffine"),
        ("roformer", "RoFormerForTokenClassificationWithBiaffine"),
        ("squeezebert", "SqueezeBertForTokenClassificationWithBiaffine"),
        ("xlm", "XLMForTokenClassificationWithBiaffine"),
        ("xlm-roberta", "XLMRobertaForTokenClassificationWithBiaffine"),
        ("xlm-roberta-xl", "XLMRobertaXLForTokenClassificationWithBiaffine"),
        ("xlnet", "XLNetForTokenClassificationWithBiaffine"),
        ("yoso", "YosoForTokenClassificationWithBiaffine"),
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
        ("big_bird", "BigBirdForTokenClassificationWithGlobalPointer"),
        ("bloom", "BloomForTokenClassificationWithGlobalPointer"),
        ("camembert", "CamembertForTokenClassificationWithGlobalPointer"),
        ("canine", "CanineForTokenClassificationWithGlobalPointer"),
        ("convbert", "ConvBertForTokenClassificationWithGlobalPointer"),
        ("data2vec-text", "Data2VecTextForTokenClassificationWithGlobalPointer"),
        ("deberta", "DebertaForTokenClassificationWithGlobalPointer"),
        ("deberta-v2", "DebertaV2ForTokenClassificationWithGlobalPointer"),
        ("distilbert", "DistilBertForTokenClassificationWithGlobalPointer"),
        ("electra", "ElectraForTokenClassificationWithGlobalPointer"),
        ("ernie", "ErnieForTokenClassificationWithGlobalPointer"),
        # ("esm", "EsmForTokenClassificationWithGlobalPointer"),
        # ("flaubert", "FlaubertForTokenClassificationWithGlobalPointer"),
        # ("fnet", "FNetForTokenClassificationWithGlobalPointer"),
        # ("funnel", "FunnelForTokenClassificationWithGlobalPointer"),
        # ("gpt2", "GPT2ForTokenClassificationWithGlobalPointer"),
        # ("ibert", "IBertForTokenClassificationWithGlobalPointer"),
        # ("layoutlm", "LayoutLMForTokenClassificationWithGlobalPointer"),
        # ("layoutlmv2", "LayoutLMv2ForTokenClassificationWithGlobalPointer"),
        # ("layoutlmv3", "LayoutLMv3ForTokenClassificationWithGlobalPointer"),
        # ("longformer", "LongformerForTokenClassificationWithGlobalPointer"),
        # ("luke", "LukeForTokenClassificationWithGlobalPointer"),
        # ("markuplm", "MarkupLMForTokenClassificationWithGlobalPointer"),
        # ("megatron-bert", "MegatronBertForTokenClassificationWithGlobalPointer"),
        # ("mobilebert", "MobileBertForTokenClassificationWithGlobalPointer"),
        # ("mpnet", "MPNetForTokenClassificationWithGlobalPointer"),
        # ("nezha", "NezhaForTokenClassificationWithGlobalPointer"),
        # ("nystromformer", "NystromformerForTokenClassificationWithGlobalPointer"),
        # ("qdqbert", "QDQBertForTokenClassificationWithGlobalPointer"),
        # ("rembert", "RemBertForTokenClassificationWithGlobalPointer"),
        ("roberta", "RobertaForTokenClassificationWithGlobalPointer"),
        ("roformer", "RoFormerForTokenClassificationWithGlobalPointer"),
        ("squeezebert", "SqueezeBertForTokenClassificationWithGlobalPointer"),
        ("xlm", "XLMForTokenClassificationWithGlobalPointer"),
        ("xlm-roberta", "XLMRobertaForTokenClassificationWithGlobalPointer"),
        ("xlm-roberta-xl", "XLMRobertaXLForTokenClassificationWithGlobalPointer"),
        ("xlnet", "XLNetForTokenClassificationWithGlobalPointer"),
        ("yoso", "YosoForTokenClassificationWithGlobalPointer"),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_WITH_GLOBAL_POINTER_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_WITH_GLOBAL_POINTER_MAPPING_NAMES
)


class AutoModelForTokenClassificationWithGlobalPointer(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_WITH_GLOBAL_POINTER_MAPPING
