import datasets
import json

logger = datasets.logging.get_logger(__name__)

_CITATION = """\

"""

_DESCRIPTION = """\

"""

_URL = "./"
_TRAINING_FILE = "train.json"
_VALIDATION_FILE = 'dev.json'
_TEST_FILE = "test.json"

_Labels = [
    'address',
    'book',
    'company',
    'game',
    'government',
    'movie',
    'name',
    'organization',
    'position',
    'scene',
]


class ClueNerConfig(datasets.BuilderConfig):
    """BuilderConfig for ClueNer"""

    def __init__(self, **kwargs):
        """BuilderConfig for CLUE NER.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ClueNerConfig, self).__init__(**kwargs)


class ClueNer(datasets.GeneratorBasedBuilder):
    """Clue NER dataset."""

    BUILDER_CONFIGS = [
        ClueNerConfig(name="clue_ner", version=datasets.Version("1.0.0"), description="CLUE NER dataset"),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "tokens": datasets.Sequence(datasets.Value("string")),
                "entities": datasets.Sequence(datasets.Features({
                    'type': datasets.features.ClassLabel(names=_Labels),
                    "start_idx": datasets.Value("int64"),
                    "end_idx": datasets.Value("int64"),
                })
                ),

            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            'validation':f"{_URL}{_VALIDATION_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,gen_kwargs={"filepath": downloaded_files["validation"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                item = {
                    "tokens": [i for i in line["text"]],
                    'entities': []
                }
                for k, v in line.get('label', {}).items():
                    for spans in v.values():
                        for start, end in spans:
                            item["entities"].append({'start_idx': start, 'end_idx': end, 'type': k})
                yield idx, item
