# -*- coding: utf-8 -*-
# @Time     : 2022/11/29 12:19
# @File     : __init__.py
# @Author   : Zhou Hang
# @Email    : zhouhang@idataway.com
# @Software : Python 3.7
# @About    :
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
    _import_structure["modeling_electra"] = [
        "ElectraForTokenClassificationWithGlobalPointer",
        "ElectraForTokenClassificationWithBiaffine",
    ]

if TYPE_CHECKING:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_electra import (
            ElectraForTokenClassificationWithGlobalPointer,
            ElectraForTokenClassificationWithBiaffine,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
