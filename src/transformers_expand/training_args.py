import transformers
from dataclasses import asdict, dataclass, field, fields


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    adversarial: str = field(default="none", metadata={
        "help": "FGM epsilon .",
        "choices": ["none", "FGM", "PGD"]
    })

    FGM_epsilon: float = field(default=1., metadata={"help": "Adversarial model FGM 'epsilon' parma."})

    PGM_epsilon: float = field(default=1., metadata={"help": "Adversarial model PGM 'epsilon' parma."})
    PGM_K: int = field(default=3, metadata={"help": "Adversarial model PGM 'K' parma."})
    PGM_alpha: float = field(default=.3, metadata={"help": "Adversarial model PGM 'alpha' parma."})
