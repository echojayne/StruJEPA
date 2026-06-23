# elastic_method

`elastic_method` contains the model-independent implementation of the current
StruJEPA method.

## Current recipe

1. Elasticize a supported Transformer stack over width and depth.
2. Warm up one shared matrix-Transformer completion module against missing
   width operators and a frozen initial reference model.
3. Train deployable elastic subnets with supervised task loss.

Current completion configurations require:

- `mode="width_operator_completion"`
- `predictor_layout="matrix_transformer_full"`
- `depth_encoding="gaussian_cdf"`
- `stage_epochs={"warmup_completion": ..., "subnet_training": ...}`
- task-only subnet training; alignment controls are not part of the API

## Public API

```python
from elastic_method import (
    StruJEPATrainer,
    CompletionStageEpochs,
    ElasticizationSpec,
    ForwardResult,
    MethodConfig,
    StructureMaskDescriptor,
    TaskAdapter,
    WidthOperatorCompletionConfig,
    elasticize_model,
)
```

Model-family wrappers and the concrete completion module are internal APIs.

## Supported block families

- `torch_encoder`
- `torch_encoder_multi`
- `wifo_vit`
- `timm_vit`
- `hf_bert`
- `hf_vit`
- `dinov3_vit`

Optional adapters are registered only when their dependencies are installed.
