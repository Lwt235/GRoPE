# R44: learned spatial Gaussian basis inside the Transformer

2026-11-01. R43 completed and was negative. This is a different hypothesis:
localized spatial fields, not constraints on a shared global frequency bank.
Primary is spatial_full throughout. No rendering/color readout is used.

Hypothesis: learned spatial Gaussian centers and ellipsoids provide a useful
compact positional vector beyond ordinary PE in a fixed Transformer.
Alternatives: any improvement comes from adding an ordinary coordinate network,
fixed localized functions, isotropic kernels or extra positional parameters.

## Formula and fixed controls

Keep R5's166611-parameter two-block bidirectional Transformer, all Q/K/V,
residuals, FFNs, temperatures and final matching head. For each head's unit
content vector h, replace only PE with normalize(h+p(x)).
p(x)=W^T phi(x)/r0, phi_g(x)=exp(-.5 (x-m_g)^T P_g (x-m_g)).
P_g=L_g L_g^T, so covariance P_g^-1 has the GS rotation/axis-scale form.
Here centers and ellipsoids are learned SPATIAL basis functions, not physical
input primitives or uncertainty. No opacity, SH, densification, extra head,
rendering objective, RoPE hybrid or new content module.

This is a known Gaussian/RBF basis plus linear mixture, motivated by 3D Gaussian
Point Encoders (James et al., WACV2026), not a new Gaussian basis theorem or a
full reproduction of that PointNet/Mamba3D architecture. That work notes optimizer
difficulty and uses natural gradients/distillation. This fixed AdamW screen
tests a compact PE under one common recipe; a negative cannot disprove all bases.
Learnable Fourier Features (Li et al., NeurIPS2021) motivates a strong local
Fourier control; our one-projection control is not its full published MLP.
Sources: https://arxiv.org/html/2511.04797v1 and
https://proceedings.neurips.cc/paper/2021/hash/84c2d4860a0fc27bcf854c444fb8b400-Abstract.html

Use exactly36 spatial Gaussians, centers on a6x6 uniform grid in [-1,1]^2,
initial spatial std1/3. Native token-stride coordinates have chart center
(11.9375,8.9375), extent(12,9), defined by the18x24 grid, not data labels.
L=3*(diag(exp(d))+strict_lower(l)); d/l and mean offsets start0.
W is36x48, initial N(0,.02^2), drawn before role-specific random initialization
and shared identically across roles for a seed. All heads/layers/directions
share this small PE. r0 is a FIXED scalar: sqrt(mean_grid ||phi_initial||^2)
on all432 native centers. It matches initial feature energy across families;
it does not normalize each point's basis or erase spatial support amplitude.

Eight new roles and added parameter counts:
spatial_full1908, spatial_diag1872, spatial_iso1836, spatial_fixed1728,
spatial_fixed_centers1836, fourier_fixed1728, fourier_learned1782,
coordinate_mlp1836. No dummy padding. All five Gaussian roles start identically.
Fourier:18 golden-angle vectors with original token-coordinate magnitudes
logspace(-1,.6,18), both sine/cosine, same W. The learned role adds free vector
offsets in token-frequency units and phase offsets, initially0.
MLP:2->36 GELU->48, same final W; initial first matrix N(0,1/2), learned offsets
and biases start0. The coordinate chart and r0 policy are common to new roles.
Weight decay acts on zero-initialized basis changes and the common W.

Reuse the SIX unchanged R5 baseline roles'30 completed R43 runs and sealed
old_eval outputs. Backbone source, old data identities, all non-PE initial
weights, batches, optimizer/budget and selection must match. Record hashes of
all inherited artifacts; no pretending reused runs are new independent trials.
Their position functions remain unchanged, including the original point RoPE.

## Execution, verification and data

Eight new roles x5paired seeds11,22,33,44,55 =40new runs;70models with reused
controls. Exact R43/R5 old576train/64val/480old_eval population and ten input
fields; all432keys and sensor-valid queries. No new/fresh cohort access.
1000updates,batch8,AdamW(.001,.0001),clip1,cosine LR from1 to.2,NumPy batches
seed+50000. Validate0,100,...1000; earliest minimum scene-macro CE. No role-wise
LR, width, basis count, bandwidth or step search. CPU4, TF32off, deterministic.
Use idle GPUs,<=16GiB allocated,<=3GPUhours per seed shard.

Freeze sources/data/inherited baseline identities before real array access.
CPU mathematical/gradient/input-scope checks and all-role CUDA smoke first.
Save histories, selected weights and first-layer score samples. Check all new
samples with independently written NumPy Gaussian/Fourier/GELU formulas;
probability error>5e-5 is an impact-assessment warning, not automatic task failure.
Verify all new initial backbone and batch digests against inherited controls.
Seal all40 new checkpoints before old_eval export; preserve full per-query
sufficient statistics, complete frame metrics and zero-V outputs for all roles.
Replay all-query metrics in NumPy. Independent formulas/metrics do not constitute
an independently authored second complete Transformer or blinded human review.

## Prospective decisions

Separate utility from component attribution BEFORE results, unlike requiring
each tiny component to independently exceed an entire-method practical margin.
This does not change R43's frozen decision or turn its negative result positive.

Utility screen: spatial_full must exceed EACH of the strongest unchanged R5 role,
fourier_fixed, fourier_learned and coordinate_mlp by>=1pp old_eval macro PCK8,
with>=4/5positive seeds,>=4/5positive rooms and CE worsening<=2%. Report all
continuous deltas/PCK4/EPE, not only pass flags. A smaller effect is recorded,
not proclaimed established success or generality on repeatedly used rooms.

Component attribution: report full versus fixed/iso/diag/fixed-centers with
paired seed/scene differences; no invented1pp requirement on each small component.
If utility improves but shape/center learning adds no stable benefit, state that
and test any simplification under a NEW prospective plan. No posthoc relabeling
of an ablation as this round's primary, and no claim of useful learned anisotropy
without evidence. Failure of utility ends promotion of this fixed design; no
grid/width/optimizer sweeps on old_eval to rescue it.

Any positive only supports further research: actual3D Transformer coordinates,
cross-task utility, scale/noise/frame robustness, full operator/network cost and
untouched joint confirmation are outstanding. The exact Gaussian field is
covariant to joint coordinate/basis changes, not invariant to moving inputs while
holding learned centers fixed. This 2D screen is not proof of3D generality.
No claims of normalized region-area probability or physical GS R/S utility;
their historical failures remain. Rendering is an optional later transfer test.
