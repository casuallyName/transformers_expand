__version__ = "0.0.1"

from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    logging,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Base objects, independent of any specific backend
_import_structure = {
    "data": ["DataCollatorForTokenClassificationWithGlobalPointer", ],
    "data.data_collator": ["DataCollatorForTokenClassificationWithGlobalPointer", ],
    "metrics": ["MetricsForCommonTextClassificationTask",
                "MetricsForBiaffineTask",
                "MetricsForGlobalPointerTask", ],
    "models": [],
    # Models
    "models.albert": [],
    "models.auto": [],
    "models.bert": [],
    "models.big_bird": [],
    "models.bloom": [],
    "models.camembert": [],
    "models.canine": [],
    "models.convbert": [],
    "models.distilbert": [],
    "models.data2vec": [],
    "models.deberta": [],
    "models.deberta_v2": [],
    "models.electra": [],
    "models.ernie": [],
    "models.esm": [],
    "models.flaubert": [],
    "models.funnel": [],
    "models.gpt2": [],
    "models.ibert": [],
    "models.longformer": [],
    "models.luke": [],
    "models.megatron_bert": [],
    "models.mobilebert": [],
    "models.mpnet": [],
    "models.nezha": [],
    "models.nystromformer": [],
    "models.qdqbert": [],
    "models.rembert": [],
    # TODO
    "models.roberta": [],
    "models.roformer": [],
    "models.squeezebert": [],
    "models.xlm": [],
    "models.xlm_roberta": [],
    "models.xlm_roberta_xl": [],
    "models.xlnet": [],
    "models.yoso": [],

    "nn": ["functional",
           "layer",
           "modules",
           "adversarial",
           "GlobalPointer",
           "EfficientGlobalPointer",
           "Biaffine",
           "MultiLabelCategoricalCrossEntropyLoss",
           "MultiLabelCategoricalForNerCrossEntropyLoss",
           "SpanLoss",
           "load_adversarial",
           "FGSM",
           "FGM",
           "PGD",
           "FreeAT",
           ],
    "nn.adversarial": ["load_adversarial",
                       "FGSM",
                       "FGM",
                       "PGD",
                       "FreeAT",
                       ],
    "nn.functional": ["multi_label_categorical_cross_entropy"
                      ],
    "nn.layer": ["GlobalPointer",
                 "EfficientGlobalPointer",
                 "Biaffine",
                 ],
    "nn.modules": ["MultiLabelCategoricalCrossEntropyLoss",
                   "MultiLabelCategoricalForNerCrossEntropyLoss",
                   ],
    "trainer": ["Trainer"],
    "training_args": ["TrainingArguments"],
    "utils.dummy_pt_objects": []
}

# PyTorch-backed objects
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects

    #
    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    _import_structure["models.albert"].extend(
        [
            "AlbertForTokenClassificationWithBiaffine",
            "AlbertForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.auto"].extend(
        [
            "AutoModelForTokenClassificationWithGlobalPointer",
            "AutoModelForTokenClassificationWithBiaffine",
        ]
    )
    _import_structure["models.bert"].extend(
        [
            "BertForTokenClassificationWithGlobalPointer",
            "BertForTokenClassificationWithBiaffine",
        ]
    )
    _import_structure["models.big_bird"].extend(
        [
            "BigBirdForTokenClassificationWithBiaffine",
            "BigBirdForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.bloom"].extend(
        [
            "BloomForTokenClassificationWithBiaffine",
            "BloomForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.camembert"].extend(
        [
            "CamembertForTokenClassificationWithBiaffine",
            "CamembertForTokenClassificationWithGlobalPointer"
        ]
    )
    _import_structure["models.canine"] = [
        "CanineForTokenClassificationWithBiaffine",
        "CanineForTokenClassificationWithGlobalPointer"
    ]

    _import_structure["models.convbert"] = [
        "ConvBertForTokenClassificationWithBiaffine",
        "ConvBertForTokenClassificationWithGlobalPointer",
    ]

    _import_structure["models.data2vec"] = [
        "Data2VecTextForTokenClassificationWithBiaffine",
        "Data2VecTextForTokenClassificationWithGlobalPointer",
    ]

    _import_structure["models.distilbert"].extend(
        [
            "DistilBertForTokenClassificationWithBiaffine",
            "DistilBertForTokenClassificationWithGlobalPointer",
        ]
    )

    _import_structure["models.deberta"] = [
        "DebertaForTokenClassificationWithBiaffine",
        "DebertaForTokenClassificationWithGlobalPointer"
    ]

    _import_structure["models.deberta_v2"] = [
        "DebertaV2ForTokenClassificationWithBiaffine",
        "DebertaV2ForTokenClassificationWithGlobalPointer"
    ]

    _import_structure["models.electra"].extend(
        [
            "ElectraForTokenClassificationWithGlobalPointer",
            "ElectraForTokenClassificationWithBiaffine",
        ]
    )
    _import_structure["models.ernie"].extend(
        [
            "ErnieForTokenClassificationWithGlobalPointer",
            "ErnieForTokenClassificationWithBiaffine",
        ]
    )
    _import_structure["models.esm"].extend(
        [
            "EsmForTokenClassificationWithBiaffine",
            "EsmForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.flaubert"].extend(
        [
            "FlaubertForTokenClassificationWithBiaffine",
            "FlaubertForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.funnel"].extend(
        [
            "FunnelForTokenClassificationWithBiaffine",
            "FunnelForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.gpt2"].extend(
        [
            "GPT2ForTokenClassificationWithBiaffine",
            "GPT2ForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.ibert"].extend(
        [
            "IBertForTokenClassificationWithBiaffine",
            "IBertForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.longformer"].extend(
        [
            "LongformerForTokenClassificationWithBiaffine",
            "LongformerForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.luke"].extend(
        [
            "LukeForTokenClassificationWithBiaffine",
            "LukeForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.mobilebert"].extend(
        [
            "MobileBertForTokenClassificationWithBiaffine",
            "MobileBertForTokenClassificationWithGlobalPointer",
        ]
    )

    _import_structure["models.mpnet"].extend(
        [
            "MPNetForTokenClassificationWithBiaffine",
            "MPNetForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.nezha"].extend(
        [
            "NezhaForTokenClassificationWithBiaffine",
            "NezhaForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.nystromformer"].extend(
        [
            "NystromformerForTokenClassificationWithBiaffine",
            "NystromformerForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.qdqbert"].extend(
        [
            "QDQBertForTokenClassificationWithGlobalPointer",
            "QDQBertForTokenClassificationWithBiaffine",
        ]
    )
    _import_structure["models.rembert"].extend(
        [
            "RemBertForTokenClassificationWithGlobalPointer",
            "RemBertForTokenClassificationWithBiaffine",
        ]
    )
    _import_structure["models.roberta"].extend(
        [
            "RobertaForTokenClassificationWithBiaffine",
            "RobertaForTokenClassificationWithGlobalPointer",
        ]
    )

    _import_structure["models.roformer"].extend(
        [
            "RoFormerForTokenClassificationWithBiaffine",
            "RoFormerForTokenClassificationWithGlobalPointer",
        ]
    )

    _import_structure["models.squeezebert"].extend(
        [
            "SqueezeBertForTokenClassificationWithBiaffine",
            "SqueezeBertForTokenClassificationWithGlobalPointer",
        ]
    )

    _import_structure["models.xlm"].extend(
        [
            "XLMForTokenClassificationWithGlobalPointer",
            "XLMForTokenClassificationWithBiaffine"
        ]
    )
    _import_structure["models.xlm_roberta"].extend(
        [
            "XLMRobertaForTokenClassificationWithGlobalPointer",
            "XLMRobertaForTokenClassificationWithBiaffine"
        ]
    )
    _import_structure["models.xlm_roberta_xl"].extend(
        [
            "XLMRobertaXLForTokenClassificationWithBiaffine",
            "XLMRobertaXLForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.xlnet"].extend(
        [
            "XLNetForTokenClassificationWithBiaffine",
            "XLNetForTokenClassificationWithGlobalPointer",
        ]
    )
    _import_structure["models.yoso"].extend(
        [
            "YosoForTokenClassificationWithBiaffine",
            "YosoForTokenClassificationWithGlobalPointer",
        ]
    )

# Direct imports for type-checking
if TYPE_CHECKING:

    # Data
    from .data import (
        DataCollatorForTokenClassificationWithGlobalPointer,
    )
    from .data.data_collator import (
        DataCollatorForTokenClassificationWithGlobalPointer,
    )

    # Modeling

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_pt_objects import *
    else:

        # PyTorch model imports
        from .models.albert import (
            AlbertForTokenClassificationWithBiaffine,
            AlbertForTokenClassificationWithGlobalPointer,
        )
        from .models.auto import (
            MODEL_FOR_TOKEN_CLASSIFICATION_WITH_BIAFFINE_POINTER_MAPPING,
            AutoModelForTokenClassificationWithBiaffine,
            MODEL_FOR_TOKEN_CLASSIFICATION_WITH_GLOBAL_POINTER_MAPPING,
            AutoModelForTokenClassificationWithGlobalPointer,
        )
        from .models.bert import (
            BertForTokenClassificationWithBiaffine,
            BertForTokenClassificationWithGlobalPointer,
        )
        from .models.big_bird import (
            BigBirdForTokenClassificationWithBiaffine,
            BigBirdForTokenClassificationWithGlobalPointer,
        )

        from .models.bloom import (
            BloomForTokenClassificationWithBiaffine,
            BloomForTokenClassificationWithGlobalPointer,
        )

        from .models.camembert import (
            CamembertForTokenClassificationWithBiaffine,
            CamembertForTokenClassificationWithGlobalPointer
        )

        from .models.canine import (
            CanineForTokenClassificationWithBiaffine,
            CanineForTokenClassificationWithGlobalPointer,
        )

        from .models.convbert import (
            ConvBertForTokenClassificationWithBiaffine,
            ConvBertForTokenClassificationWithGlobalPointer,
        )

        from .models.distilbert import (
            DistilBertForTokenClassificationWithBiaffine,
            DistilBertForTokenClassificationWithGlobalPointer,
        )

        from .models.data2vec import (
            Data2VecTextForTokenClassificationWithBiaffine,
            Data2VecTextForTokenClassificationWithGlobalPointer,
        )
        from .models.deberta import (
            DebertaForTokenClassificationWithBiaffine,
            DebertaForTokenClassificationWithGlobalPointer
        )
        from .models.deberta_v2 import (
            DebertaV2ForTokenClassificationWithBiaffine,
            DebertaV2ForTokenClassificationWithGlobalPointer,
        )

        from .models.electra import (
            ElectraForTokenClassificationWithBiaffine,
            ElectraForTokenClassificationWithGlobalPointer,
        )
        from .models.ernie import (
            ErnieForTokenClassificationWithBiaffine,
            ErnieForTokenClassificationWithGlobalPointer,
        )
        from .models.esm import (
            EsmForTokenClassificationWithBiaffine,
            EsmForTokenClassificationWithGlobalPointer,
        )
        from .models.flaubert import (
            FlaubertForTokenClassificationWithBiaffine,
            FlaubertForTokenClassificationWithGlobalPointer
        )
        from .models.funnel import (
            FunnelForTokenClassificationWithBiaffine,
            FunnelForTokenClassificationWithGlobalPointer,
        )
        from .models.gpt2 import (
            GPT2ForTokenClassificationWithBiaffine,
            GPT2ForTokenClassificationWithGlobalPointer,
        )
        from .models.ibert import (
            IBertForTokenClassificationWithGlobalPointer,
            IBertForTokenClassificationWithBiaffine,
        )
        from .models.longformer import (
            LongformerForTokenClassificationWithBiaffine,
            LongformerForTokenClassificationWithGlobalPointer,
        )
        from .models.luke import (
            LukeForTokenClassificationWithBiaffine,
            LukeForTokenClassificationWithGlobalPointer,
        )
        from .models.megatron_bert import (
            MegatronBertForTokenClassificationWithBiaffine,
            MegatronBertForTokenClassificationWithGlobalPointer,
        )
        from .models.mobilebert import (
            MobileBertForTokenClassificationWithBiaffine,
            MobileBertForTokenClassificationWithGlobalPointer,
        )
        from .models.mpnet import (
            MPNetForTokenClassificationWithBiaffine,
            MPNetForTokenClassificationWithGlobalPointer,
        )
        from .models.nezha import (
            NezhaForTokenClassificationWithBiaffine,
            NezhaForTokenClassificationWithGlobalPointer,
        )
        from .models.nystromformer import (
            NystromformerForTokenClassificationWithBiaffine,
            NystromformerForTokenClassificationWithGlobalPointer,
        )
        from .models.qdqbert import (
            QDQBertForTokenClassificationWithBiaffine,
            QDQBertForTokenClassificationWithGlobalPointer,
        )
        from .models.rembert import (
            RemBertForTokenClassificationWithBiaffine,
            RemBertForTokenClassificationWithGlobalPointer,
        )
        from .models.roberta import (
            RobertaForTokenClassificationWithBiaffine,
            RobertaForTokenClassificationWithGlobalPointer,
        )
        from .models.roformer import (
            RoFormerForTokenClassificationWithBiaffine,
            RoFormerForTokenClassificationWithGlobalPointer,
        )
        from .models.squeezebert import (
            SqueezeBertForTokenClassificationWithBiaffine,
            SqueezeBertForTokenClassificationWithGlobalPointer,
        )
        from .models.xlm import (
            XLMForTokenClassificationWithBiaffine,
            XLMForTokenClassificationWithGlobalPointer
        )
        from .models.xlm_roberta import (
            XLMRobertaForTokenClassificationWithBiaffine,
            XLMRobertaForTokenClassificationWithGlobalPointer,
        )
        from .models.xlm_roberta_xl import (
            XLMRobertaXLForTokenClassificationWithBiaffine,
            XLMRobertaXLForTokenClassificationWithGlobalPointer,
        )
        from .models.xlnet import (
            XLNetForTokenClassificationWithBiaffine,
            XLNetForTokenClassificationWithGlobalPointer,
        )
        from .models.yoso import (
            YosoForTokenClassificationWithBiaffine,
            YosoForTokenClassificationWithGlobalPointer
        )

        from .nn import (
            GlobalPointer,
            EfficientGlobalPointer,
            MultiLabelCategoricalForNerCrossEntropyLoss,
            MultiLabelCategoricalCrossEntropyLoss,
            SpanLoss,
            load_adversarial,
            FGSM,
            FGM,
            PGD,
            FreeAT,
        )

        from .nn import functional

        from .nn import layer

        from .nn import adversarial

        from .trainer import Trainer

        from .training_args import TrainingArguments

        from .metrics import (
            MetricsForCommonTextClassificationTask,
            MetricsForBiaffineTask,
            MetricsForGlobalPointerTask
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )

if not is_torch_available():
    logger.warning(
        "None of PyTorch have been found. "
        "transformers_expand require PyTorch support."
    )
