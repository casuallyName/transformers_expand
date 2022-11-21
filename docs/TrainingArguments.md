# `transformers_expand.TrainingArguments`

**
继承自[`trasnformers.TrainingArguments`](https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/trainer#transformers.TrainingArguments)**

## Parameters

* **adversarial** (str, optional, defaults to none) — 对抗训练方案
* **FGSM_epsilon** (float, optional, defaults to 1.0) — FGSM 扰动半径 epsilon
* **FGM_epsilon** (float, optional, defaults to 1.0) — FGM 扰动半径 epsilon
* **PGM_epsilon** (float, optional, defaults to 1.0) — PGM 扰动半径 epsilon
* **PGM_steps** (int, optional, defaults to 3) — PGM 扰动步数 steps
* **PGM_alpha** (float, optional, defaults to 0.3) — PGM alpha
* **FreeAT_epsilon** (float, optional, defaults to 1.0) — FreeAT 扰动半径 epsilon
* **FreeAT_steps** (int, optional, defaults to 3) — PGM 扰动步数 steps