# elastic_method

An isolated internal framework for elasticizing standard Transformer encoder stacks and training them with:

- supervised task loss
- optional full-view output alignment
- structural-mask representation alignment/prediction

## Supported block families

- `torch_encoder`: `torch.nn.TransformerEncoderLayer` stacks
- `timm_vit`: `timm.models.vision_transformer.Block` stacks
- `hf_bert`: HuggingFace `BertLayer` stacks
- `hf_vit`: HuggingFace `ViTLayer` stacks

## Core ideas

- Keep the original model isolated; only a user-specified `stack_path` is elasticized.
- Keep `d_model` fixed. Width only changes active attention heads and FFN hidden size.
- Depth uses uniformly selected layer indices instead of prefix early-exit.
- The wrapper returns a `ForwardResult` with:
  - `model_output`
  - `encoder_state`
  - `structure_mask`
  - `aux`

## Minimal usage

```python
from elastic_method import ElasticizationSpec, elasticize_model

elastic_model = elasticize_model(
    model,
    ElasticizationSpec(
        stack_path="encoder",
        block_family="torch_encoder",
        width_multipliers=(1.0, 0.75, 0.5),
        depth_multipliers=(1.0, 0.75, 0.5),
        width_only_epochs=1,
    ),
)

result = elastic_model(inputs, width_multiplier=0.5, depth_multiplier=0.75, return_encoder_state=True)
```

Run the toy smoke example with:

```bash
python -m elastic_method.examples.toy_torch_encoder
```

See [`examples/toy_torch_encoder.py`](/home/dky/VIT_Deconstruct/elastic_method/examples/toy_torch_encoder.py) and [`tests/`](/home/dky/VIT_Deconstruct/elastic_method/tests) for isolated coverage.
