# Width-Operator Completion with Depth-CDF Guidance

This note describes the training method, independent of any task-specific implementation. The goal is to start from a pretrained Transformer and train a single elastic model whose width-depth subnets can be deployed without auxiliary completion modules.

## Setup

Let \(F_\theta\) be the elastic model initialized from the pretrained full model, and let \(F_{\mathrm{ref}}\) be a frozen reference copy created at the beginning of the first stage. A subnet is identified by a width ratio \(w\) and a depth ratio \(d\). Width controls the active attention heads and feed-forward channels; depth controls the active layer subset.

The method introduces a lightweight width-operator completion module \(C_\phi\) only during training. For a width-pruned block, the active width slice is kept, while the missing width slice is estimated from a shared block and a learned residual predictor:

\[
\widehat{W}_{\mathrm{miss}} =
W^{\mathrm{shared}}_{\mathrm{miss}} +
C_\phi(W^{\mathrm{shared}}_{\mathrm{miss}}, S_{\mathrm{active}}, e_\ell, t).
\]

Here \(S_{\mathrm{active}}\) summarizes the active slice, \(t\) denotes the operator type, and \(e_\ell\) is the depth encoding for layer \(\ell\). The completed operator is used only to create training signals; it is not part of the final deployed subnet.

## Depth-CDF Encoding

Each layer is assigned a normalized depth position

\[
r_\ell = \frac{\ell}{L-1},
\]

where \(L\) is the number of Transformer layers. A scalar depth prior is produced by a Gaussian CDF:

\[
e_\ell = \Phi\left(\frac{r_\ell - \mu}{\sigma}\right).
\]

The parameters \(\mu\) and \(\sigma\) are fitted from the original model family's depth-performance profile, rather than chosen manually. This makes the completion module aware of where layer depth is empirically more or less sensitive.

For the current retained settings:

| Backbone | \(\mu\) | \(\sigma\) | \(R^2\) |
| --- | ---: | ---: | ---: |
| AdaFortiTran | 0.5054656177183868 | 0.1883553716886 | 0.992240 |
| A-MMSE | 0.7810391278642149 | 0.07115624244447756 | 0.999998 |
| WiFo | 0.37790416 | 0.2029252 | 0.924271 |

## Stage 1: Completion-Guided Width Warmup

The first stage trains the elastic model and the width-completion module together, while keeping the initial reference model frozen.

Only full-depth width subnets are used in this stage:

\[
(w, d=1.0), \quad w \in \mathcal{W}.
\]

For each width subnet, the missing width operators are completed using \(C_\phi\). The completed subnet is compared against the frozen reference model at two levels.

The first term aligns the completed missing operator weights:

\[
\mathcal{L}_{W}
= \left\|
\widehat{W}_{\mathrm{miss}} - W^{\mathrm{ref}}_{\mathrm{miss}}
\right\|_2^2.
\]

The second and third terms align the completed block responses to the corresponding frozen-reference responses:

\[
\mathcal{L}_{A}
= \left\|
A_{\mathrm{completed}} - A_{\mathrm{ref}}
\right\|_2^2,
\]

\[
\mathcal{L}_{F}
= \left\|
H_{\mathrm{completed}} - H_{\mathrm{ref}}
\right\|_2^2.
\]

\(A\) denotes the attention-side response and \(H\) denotes the feed-forward-side response. The first-stage objective is:

\[
\mathcal{L}_{\mathrm{stage1}}
=
\lambda_W \mathcal{L}_{W}
+ \lambda_A \mathcal{L}_{A}
+ \lambda_F \mathcal{L}_{F}.
\]

No task loss or output-alignment loss is required in this stage. The purpose is to make the shared elastic model and the width-completion module learn how missing width operators should behave relative to the frozen full model.

## Stage 2: Task Fine-Tuning for Deployable Subnets

After the completion warmup, the auxiliary completion module is removed from the training forward path. The elastic model is then trained directly using task loss over the full model and all width-depth subnets:

\[
(w, d), \quad w \in \mathcal{W}, \ d \in \mathcal{D}.
\]

The second-stage objective is:

\[
\mathcal{L}_{\mathrm{stage2}}
=
\frac{1}{|\mathcal{S}|}
\sum_{(w,d)\in \mathcal{S}}
\mathcal{L}_{\mathrm{task}}\left(F_\theta^{w,d}(x), y\right),
\]

where \(\mathcal{S}\) includes the full model and the selected subnet grid.

The deployed model is just \(F_\theta\). It contains the trained shared Transformer weights and exposes elastic width-depth subnets. The completion module and its shared completion block are training-time scaffolding and are not required at inference time.

## Practical Training Policy

For fair comparison across AdaFortiTran, A-MMSE, and WiFo, the same policy should be used:

- Stage 1 updates the shared Transformer weights and the completion module.
- Stage 1 keeps the initial reference model frozen.
- Stage 1 covers all configured width ratios at full depth.
- Stage 1 uses depth-CDF conditioning fitted from the model family's depth-performance statistics.
- Stage 2 removes completion from the forward path.
- Stage 2 covers the full width-depth subnet grid.
- Stage 2 uses task loss only. Output and representation alignment controls are
  not part of the current method.
- The final Pareto evaluation uses only the deployable elastic model, not the completion module.
