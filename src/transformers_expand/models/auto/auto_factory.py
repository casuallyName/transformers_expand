# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Factory function to build auto-model classes."""
import importlib
from collections import OrderedDict

from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from transformers.models.auto.configuration_auto import AutoConfig, model_type_to_module_name

from transformers.models.auto.auto_factory import (
    logger,
    _get_model_class,
)


class _BaseAutoModelClass:
    # Base class for auto models.
    _model_mapping = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        if hasattr(config, "auto_map") and cls.__name__ in config.auto_map:
            if not trust_remote_code:
                raise ValueError(
                    "Loading this model requires you to execute the modeling file in that repo "
                    "on your local machine. Make sure you have read the code there to avoid malicious use, then set "
                    "the option `trust_remote_code=True` to remove this error."
                )
            if kwargs.get("revision", None) is None:
                logger.warning(
                    "Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure "
                    "no malicious code has been contributed in a newer revision."
                )
            class_ref = config.auto_map[cls.__name__]
            module_file, class_name = class_ref.split(".")
            model_class = get_class_from_dynamic_module(config.name_or_path, module_file + ".py", class_name, **kwargs)
            return model_class._from_config(config, **kwargs)
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class._from_config(config, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "use_auth_token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                **hub_kwargs,
                **kwargs,
            )
        if hasattr(config, "auto_map") and cls.__name__ in config.auto_map:
            if not trust_remote_code:
                raise ValueError(
                    f"Loading {pretrained_model_name_or_path} requires you to execute the modeling file in that repo "
                    "on your local machine. Make sure you have read the code there to avoid malicious use, then set "
                    "the option `trust_remote_code=True` to remove this error."
                )
            if hub_kwargs.get("revision", None) is None:
                logger.warning(
                    "Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure "
                    "no malicious code has been contributed in a newer revision."
                )
            class_ref = config.auto_map[cls.__name__]
            module_file, class_name = class_ref.split(".")
            model_class = get_class_from_dynamic_module(
                pretrained_model_name_or_path, module_file + ".py", class_name, **hub_kwargs, **kwargs
            )
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def register(cls, config_class, model_class):
        """
        Register a new model for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            model_class ([`PreTrainedModel`]):
                The model to register.
        """
        if hasattr(model_class, "config_class") and model_class.config_class != config_class:
            raise ValueError(
                "The model class you are passing has a `config_class` attribute that is not consistent with the "
                f"config class you passed (model has {model_class.config_class} and you passed {config_class}. Fix "
                "one of those so they match!"
            )
        cls._model_mapping.register(config_class, model_class)


def getattribute_from_module(module, attr):
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    transformers_module = importlib.import_module("transformers_expand")
    return getattribute_from_module(transformers_module, attr)


class _LazyAutoMapping(OrderedDict):
    """
    " A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, model_mapping):
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        # Maybe there was several model types associated with this config.
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key, default):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self):
        return bool(self.keys())

    def values(self):
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self):
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        if item in self._extra_content:
            return True
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key, value):
        """
        Register a new model in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping.keys():
                raise ValueError(f"'{key}' is already used by a Transformers model.")

        # self._extra_content[key] = values
