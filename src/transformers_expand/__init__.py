__version__ = "0.0.1.dev"

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
    "models.big_bird":[],
    "models.electra": [],
    "models.ernie": [],
    "models.roberta":[],
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
    _import_structure["models.roberta"] = [
        "RobertaForTokenClassificationWithBiaffine",
        "RobertaForTokenClassificationWithGlobalPointer",
    ]

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
            MODEL_FOR_TOKEN_CLASSIFICATION_WITH_GLOBAL_POINTER_MAPPING,
            AutoModelForTokenClassificationWithGlobalPointer,
            MODEL_FOR_TOKEN_CLASSIFICATION_WITH_BIAFFINE_POINTER_MAPPING,
            AutoModelForTokenClassificationWithBiaffine
        )
        from .models.bert import (
            BertForTokenClassificationWithGlobalPointer,
            BertForTokenClassificationWithBiaffine,
        )
        from .models.big_bird import (
            BigBirdForTokenClassificationWithBiaffine,
            BigBirdForTokenClassificationWithGlobalPointer,
        )
        from .models.electra import (
            ElectraForTokenClassificationWithGlobalPointer,
            ElectraForTokenClassificationWithBiaffine,
        )
        from .models.ernie import (
            ErnieForTokenClassificationWithBiaffine,
            ErnieForTokenClassificationWithGlobalPointer,
        )
        from .models.roberta import (
            RobertaForTokenClassificationWithBiaffine,
            RobertaForTokenClassificationWithGlobalPointer,
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
