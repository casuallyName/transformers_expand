import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

InputDataClass = NewType("InputDataClass", Any)

DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, Any]])


@dataclass
class DataCollatorForTokenClassificationWithGlobalPointer(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    num_labels: int
    max_length: Optional[int]
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        entity_type_name = "type" if "type" in features[0][label_name].keys() else "types"
        entity_start_name = "start_idx" if "start_idx" in features[0][label_name].keys() else "start"
        entity_end_name = "end_idx" if "end_idx" in features[0][label_name].keys() else "end"

        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding='max_length',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        labels = torch.zeros(
            (len(batch['input_ids']), self.num_labels, self.max_length, self.max_length))

        for i, entities in enumerate(batch[label_name]):
            for label_idx, start, end in zip(entities[entity_type_name],
                                             entities[entity_start_name], entities[entity_end_name]):
                start, end = start + 1, end + 1
                if start < self.max_length and end < self.max_length:
                    labels[i, label_idx, start, end] = 1
        del batch["labels"]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        batch["labels"] = labels

        return batch

    def numpy_call(self, features):
        import numpy as np

        label_name = "label" if "label" in features[0].keys() else "labels"
        entity_type_name = "type" if "type" in features[0][label_name].keys() else "types"
        entity_start_name = "start_idx" if "start_idx" in features[0][label_name].keys() else "start"
        entity_end_name = "end_idx" if "end_idx" in features[0][label_name].keys() else "end"

        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding='max_length',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="np" if labels is None else None,
        )

        if labels is None:
            return batch

        labels = np.zeros(
            (len(batch['input_ids']), self.num_labels, self.max_length, self.max_length))
        for i, entities in enumerate(batch[label_name]):
            for label_idx, start, end in zip(entities[entity_type_name],
                                             entities[entity_start_name], entities[entity_end_name]):
                start, end = start + 1, end + 1
                if start < self.max_length and end < self.max_length:
                    labels[i, label_idx, start, end] = 1
        del batch["labels"]
        batch = {k: np.array(v, dtype=np.int64) for k, v in batch.items()}
        batch["labels"] = labels

        return batch


@dataclass
class DataCollatorForTokenClassificationWithBiaffine(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    num_labels: int
    max_length: Optional[int]
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def build_span_mask(self, attention_mask):
        seq_mask = [attention_mask for i in range(sum(attention_mask))]
        zero = [0 for i in range(self.max_length)]
        seq_mask.extend([zero for i in range(sum(attention_mask), self.max_length)])
        return seq_mask

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        entity_type_name = "type" if "type" in features[0][label_name].keys() else "types"
        entity_start_name = "start_idx" if "start_idx" in features[0][label_name].keys() else "start"
        entity_end_name = "end_idx" if "end_idx" in features[0][label_name].keys() else "end"

        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding='max_length',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        batch['sequence_mask'] = [self.build_span_mask(attention_mask) for attention_mask in batch['attention_mask']]

        labels = torch.zeros((len(batch['input_ids']), self.max_length, self.max_length), dtype=torch.int64)

        for i, entities in enumerate(batch[label_name]):
            for entity_type, start, end in zip(entities[entity_type_name],
                                               entities[entity_start_name], entities[entity_end_name]):
                start, end = start + 1, end + 1
                if start < self.max_length and end < self.max_length:
                    labels[i, start, end] = entity_type + 1

        del batch["labels"]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        batch["labels"] = labels

        return batch

    def numpy_call(self, features):
        import numpy as np

        label_name = "label" if "label" in features[0].keys() else "labels"
        entity_type_name = "type" if "type" in features[0][label_name].keys() else "types"
        entity_start_name = "start_idx" if "start_idx" in features[0][label_name].keys() else "start"
        entity_end_name = "end_idx" if "end_idx" in features[0][label_name].keys() else "end"

        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding='max_length',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="np" if labels is None else None,
        )

        if labels is None:
            return batch

        batch['sequence_mask'] = [self.build_span_mask(attention_mask) for attention_mask in batch['attention_mask']]

        labels = np.zeros((len(batch['input_ids']), self.max_length, self.max_length), dtype=np.int64)

        for i, entities in enumerate(batch[label_name]):
            for entity_type, start, end in zip(entities[entity_type_name],
                                               entities[entity_start_name], entities[entity_end_name]):
                start, end = start + 1, end + 1
                if start < self.max_length and end < self.max_length:
                    labels[i, start, end] = entity_type + 1

        del batch["labels"]
        batch = {k: np.array(v, dtype=np.int64) for k, v in batch.items()}
        batch["labels"] = labels

        return batch
