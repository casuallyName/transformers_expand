# -*- coding: utf-8 -*-
# @Time     : 2022/5/17 12:07
# @File     : THUCNewsOneHot.py
# @Author   : Zhou Hang
# @email    : zhouhang@idataway.com
# @Software : Python 3.7
# @About    :

import os
import datasets
import pandas as pd

_CITATION = """\

"""

_DESCRIPTION = """\
THUCNews 数据集采样
"""

_URLS = {
    "train": "train.csv",
    "dev": "dev.csv",
    "test": "test.csv",
}
_LABELS = [
    'finance',
    'realty',
    'stocks',
    'education',
    'science',
    'society',
    'politics',
    'sports',
    'game',
    'entertainment',
]


class THUCNews_SentimentDatasetConfig(datasets.BuilderConfig):

    def __init__(self, **kwargs):
        super(THUCNews_SentimentDatasetConfig, self).__init__(**kwargs)


class THUCNews_Sentiment(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        THUCNews_SentimentDatasetConfig(name="THUCNews_Sentiment",
                                        version=datasets.Version("1.0.0"),
                                        description="THUCNews"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "idx": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=_LABELS),
                }
            ),
            supervised_keys=None,
            homepage='',
            citation=_CITATION,
        )

    def _get_file_base_path(self, file_dict):
        return {k: os.path.join(self.base_path, v) for k, v in file_dict.items()}

    def _split_generators(self, dl_manager):
        urls_to_download = _URLS
        downloaded_files = self._get_file_base_path(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": downloaded_files["train"], 'filetype': 'train'}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={"filepath": downloaded_files["dev"], 'filetype': 'dev'}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": downloaded_files["test"], 'filetype': 'test'}),
        ]

    def _generate_examples(self, filepath, filetype):
        data = pd.read_csv(filepath)
        if filetype == 'test':
            for id_, row in data.iterrows():
                yield id_, {
                    "idx": id_,
                    "sentence": row.Text,
                    "label": None
                }
        else:
            for id_, row in data.iterrows():
                yield id_, {
                    "idx": id_,
                    "sentence": row.Text,
                    "label": row.Label
                }
