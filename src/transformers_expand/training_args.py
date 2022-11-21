import transformers
from dataclasses import asdict, dataclass, field, fields


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    adversarial: str = field(default="none", metadata={
        "help": "Adversarial model name .",
        "choices": ["none", "FGSM", "FGM", "PGD", "FreeAT"]
    })

    FGSM_epsilon: float = field(default=1., metadata={"help": "Adversarial model FGSM 'epsilon' parma."})

    FGM_epsilon: float = field(default=1., metadata={"help": "Adversarial model FGM 'epsilon' parma."})

    PGM_epsilon: float = field(default=1., metadata={"help": "Adversarial model PGM 'epsilon' parma."})
    PGM_steps: int = field(default=3, metadata={"help": "Adversarial model PGM 'steps' parma."})
    PGM_alpha: float = field(default=.3, metadata={"help": "Adversarial model PGM 'alpha' parma."})

    FreeAT_epsilon: float = field(default=1., metadata={"help": "Adversarial model FreeAT 'epsilon' parma."})
    FreeAT_steps: int = field(default=3, metadata={"help": "Adversarial model FreeAT 'steps' parma."})
