# Migration to the current StruJEPA method

The maintained method is now `matrix_transformer_full_lpeadd`.

## Removed method variants

- `subnet_output_warmup`
- `matrix_row`
- no-layer-position and row/column-position ablations
- output-alignment and representation-alignment StruJEPA recipes

Current completion configurations must use `width_operator_completion`,
`matrix_transformer_full`, Gaussian-CDF depth encoding, and task-only final
training.

## Python API changes

- `AlignmentTrainer` was replaced by `StruJEPATrainer`.
- `TaskCallback` and concrete `*Callback` classes were replaced by
  `TaskAdapter` and concrete `*TaskAdapter` classes.
- `callback=` constructor arguments were replaced by `task_adapter=`.
- `elastic_update` and `final_alignment` stage fields were replaced by the
  two-stage `warmup_completion` and `subnet_training` layout.
- Structural-mask alignment modules were removed. The completion module
  implementation and multi-stack wrappers remain internal modules.
- EMA/teacher options and output/representation alignment controls were
  removed from `MethodConfig`.
- Old API aliases are not provided.

## Entrypoint changes

- `WIFO/src/strujepa_main.py` became `scripts/run_wifo_strujepa.py`.
- Dated launchers, repair commands, and ablation launchers are no longer part
  of the core method directory.
- Current configurations use `configs/**/current_*`.
- External benchmark assets are resolved through `AI_RAN_BENCHMARK_ROOT`.
- Relative output paths beginning with `runs/` are resolved outside the source
  tree under `STRUJEPA_RAN_RUN_ROOT`.

## Non-Core Assets

- Git metadata, runtime outputs, result figures, portable deliverables,
  checkpoints, raw logs, cases, OAI assets, and historical experiment
  directories are kept outside `/home/users/dky/StruJEPA`.
- Use external benchmark, baseline, and StruJEPA run roots for data, baselines,
  and reproducibility evidence.
