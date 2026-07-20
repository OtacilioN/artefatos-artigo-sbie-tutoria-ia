# Data notice

The `data/` directory contains only derived analytical artifacts: sequences of
dialogue-act labels, pseudonymous identifiers, cluster assignments, aggregate
metrics, and longitudinal transition counts. It does not contain message text,
tutor responses, names, e-mail addresses, grades, or other direct identifiers.

The source study is:

> McNichols, H.; Ikram, F.; Lan, A. (2026). *The StudyChat Dataset: Analyzing
> Student Dialogues With ChatGPT in an Artificial Intelligence Course*.
> LAK '26. <https://doi.org/10.1145/3785022.3785029>

The MIT license in this repository applies to the code written for this
analysis. Reuse of derived data must also respect the terms and ethical
requirements of the original StudyChat release.

## Classifier inputs and outputs

The classifier code is distributed, but its raw inputs and prompt-bearing
outputs are not. In particular, Git excludes:

- the gated StudyChat messages and tutor responses;
- SwDA/Switchboard transcripts used for supervised training;
- classified CSV, JSON and JSONL files that retain StudyChat prompts;
- BERT checkpoints, optimizer state and final model weights.

StudyChat must be obtained by accepting the conditions of the official
CC BY 4.0 release at <https://huggingface.co/datasets/wmcnicho/StudyChat>.
Switchboard is catalogued as LDC97S62 and is subject to its source terms:
<https://catalog.ldc.upenn.edu/LDC97S62>. The repository records expected
checksums for the local SwDA split without redistributing its utterance text.
