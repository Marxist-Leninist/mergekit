# Bounded layer-weight editing: Pythia-70M × PIQA

**Status: experiment running; no improvement demonstrated yet.**

This experiment extends MergeKit with a strict, non-mutating raw-state-dict layer remapper, then tests small interpolations between Pythia's original weights and rearranged copies. It does not train, rent compute, change an active service, or overwrite the source checkpoint.

![Method](method.svg)

## Provenance
- Upstream toolkit: [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit), pinned `0f1423ecf0f06f46f17b045b333b6c445705390e`; LGPL-3.0-only. The fork retains upstream attribution and license.
- Model: [EleutherAI/pythia-70m](https://huggingface.co/EleutherAI/pythia-70m), pinned `a39f36b100fe8a5377810d56c3f4789b9c53ac42`; Apache-2.0.
- Original release [PIQA result](https://github.com/EleutherAI/pythia/blob/main/evals/pythia-v1/pythia-70m/zero-shot/70m_step143000.json): accuracy 59.4668%, character-normalized accuracy 59.1404%. Our unchanged baseline must be measured under our own identical settings; the historical number is context, not our control.
- [PIQA](https://yonatanbisk.com/piqa/) data from the authors' repository, with exact input hashes in protocol.json. Data is downloaded, not redistributed here.
- Formatting follows [LM Evaluation Harness v0.3.0 PIQA](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.3.0/lm_eval/tasks/piqa.py): `Question: {goal}\nAnswer:` followed by a space and each solution. Scores sum continuation token log probabilities; normalized scores divide by solution character length.

## Fixed experiment

Two rearrangements: swap middle layers 2/3, or replace each layer by the next (last stays last). Interpolate original/remapped tensors with alpha 0.01, 0.05, or 0.10 using MergeKit's linear method. Depth, parameter count, tokenizer and evaluation budget stay unchanged. This is a small controlled search, **not a claim of a new general merging algorithm**.

Calibration: first 256 training examples whose question goals do not appear in official validation. Select by normalized calibration accuracy, ties preferring identity then listed order. The best nonidentity candidate is also evaluated as an experimental challenger if identity wins. All 1,838 official validation examples are held out until selection is frozen. Publish every candidate and both final metrics, including regressions, plus paired example-level bootstrap intervals. Selection success is not guaranteed.

An initial calibration-only run was stopped after an audit found eight shared question goals between its first 256 training examples and validation. There were no identical full examples; nevertheless all heldout-goal matches are excluded in the restarted protocol. No heldout scores were observed before this correction. Original traces are retained privately for audit.

Ten remapping regression tests passed locally: identity/no aliasing, permutations, repeated donor independence, invalid indices/depth, shape/dtype/key mismatch and custom prefixes. Raw shape compatibility does not establish functional compatibility. The run uses CPU float32 and two threads. No claim about 888B scalability or general intelligence follows from this pilot.

## Reproduce

Install the fork at this branch and its dependencies in an isolated environment. Download `train.jsonl`, `train-labels.lst`, `valid.jsonl`, and `valid-labels.lst` from the PIQA authors' data directory; verify protocol hashes. Run `python pilot.py` beside these files. The model revision is pinned in the script. Use the same Torch/Transformers versions recorded with final results. Full benchmark results and checkpoint links will be added after completion.
