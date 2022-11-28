from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

_import_structure = {}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_bert"] = [
        "BertForTokenClassificationWithBiaffine",
        "BertForTokenClassificationWithGlobalPointer",
    ]

if TYPE_CHECKING:

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_bert import (
            BertForTokenClassificationWithBiaffine,
            BertForTokenClassificationWithGlobalPointer,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
