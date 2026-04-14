# StruJEPA Core Method

This note isolates the StruJEPA method itself from any specific backbone, dataset, or downstream task.

## Scope

Included here are only the method-level pieces:

- structural subnet description
- width/depth subnet enumeration helpers
- structural-mask encoding
- representation prediction and alignment
- generic alignment-oriented training skeleton
- task callback protocol

Not included:

- any WiFo-specific code
- any downstream-task implementation
- any dataset loader
- any model-family adapter

## Method Summary

StruJEPA is organized around two views of the same sample:

- full-view: the full encoder
- subnet-view: a width/depth-constrained subnet

Each subnet is described by a discrete structural mask descriptor:

- `width_multiplier`
- `depth_multiplier`
- `selected_layer_indices`
- `active_num_heads`
- `active_ffn_dim`
- `total_layers`

The method has three loss components:

1. supervised task loss on the subnet view
2. optional full-view output alignment
3. structural-mask representation alignment / prediction

The third term is the StruJEPA-specific core:

- encode the subnet structure descriptor into a mask embedding
- combine it with the subnet representation
- predict the full-view representation target
- optimize an MSE-style representation alignment loss

## Core Files

- `core/structures.py`
  - `ElasticSubnet`
  - `ElasticizationSpec`
  - `StructureMaskDescriptor`
  - `ForwardResult`
- `core/subnet.py`
  - width/depth resolution
  - layer selection
  - subnet pool enumeration
- `method/mask.py`
  - `StructuralMaskEncoder`
  - `RepresentationAlignmentModule`
  - `StructuralMaskModule`
- `method/trainer.py`
  - `MethodConfig`
  - `AlignmentTrainer`
- `tasks/protocol.py`
  - task callback contract

## Typical Flow

1. enumerate candidate width/depth subnets
2. run the full view
3. run each sampled subnet
4. compute supervised loss on subnet outputs
5. optionally align subnet outputs to the full view
6. extract subnet/full representations
7. encode the structural mask
8. predict full-view representation from subnet representation + mask
9. optimize the joint objective

## Design Intent

The purpose of this core package is consistency:

- keep the StruJEPA objective fixed across tasks
- keep subnet semantics fixed across models
- move task-specific differences into a narrow callback interface

That way, method comparisons are not confounded by per-task reimplementation drift.
