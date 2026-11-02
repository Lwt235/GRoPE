# R56 — native 3D center phase and conditional regional preference

Prospective protocol, 2026-11-02. Development evidence only. The complete goal
remains a compact generally useful GS-form Transformer PE, with strong controls,
appropriate geometry/robustness/cross-task evidence and untouched confirmation.

## Hypothesis and scope

R48 found no unique full-Gaussian benefit. R52–R55 found useful camera/center
structure but no stable extra full-shape benefit. Stop tuning that local-frame
route. The PRoPE discussions in branches (3)/(4) distinguish a region's shape
from a map aligning its internal features: covariance does not determine that
map uniquely. R56 assigns the Gaussian a narrower, testable role: conditional
spatial preference, alongside unit-amplitude center RoPE on content.

The primary score is T <R(x_i)q_i,R(x_j)k_j> minus
0.5 (x_j-x_i-m_i)^T P_i (x_j-x_i-m_i), with P_i=Sigma_i^-1.
Use the unchanged R47 parameter predictor and exact nine-channel quadratic
positional lift (query-only constants omitted). The inherited implementation
adds the FP64 positional product to the FP32 rotated-content score; algebraic
concatenation is tested, but no fused kernel or speedup is claimed. No renderer,
new loss, trainable mixture gate, or extra feature branch. Sigma is a learned
regional prior, not measured GS covariance or localization uncertainty. This is
not homography/texture alignment or arbitrary SO(3) invariance. Related Gaussian
attention exists; no novelty claim follows from this combination.

Competing explanations: ordinary locality is sufficient; a generic quadratic
predictor is sufficient; adding center phase helps regardless of Gaussian shape;
the network ignores the regional predictor; changes are seed/data variation.

## Fixed comparison

- New roles: joint_conditional (primary), joint_fixed, joint_free_quadratic.
- Seeds 11,22,33, three paired fits per role, nine new fits. No seed selection.
- Reuse all ten exact R48 roles for those same three seeds, with original model
  class and source/data/config/init/sampling/selection/prediction identities.
  These 30 references are not new or independent training. Do not mix their
  three-seed means with the old five-seed means. Seeds44/55 are not fresh data.
- Same four-layer 96D two-head global Transformer, XYZ content to every role,
  same optimizer/loss/dropout schedule and CONFIG:120epochs/29760updates,
  batch32,512points,Y-rotation,jitter,warmup/cosine LR,minimum validation CE
  every5epochs including epoch0; earliest epoch wins exact ties.
- Same official-train-only ModelNet split:7910 fit/965 val/965 old dev_eval.
  Copy the exact cached manifest; do not reopen raw archive or official test.
  Official ModelNet test and home_ac,hotel_barcelona,hotel_graz,mit_1_242,
  harvard_computer_lab remain untouched.
- Gaussian and free quadratic have equal1746 PE parameters and matched
  initialization/first derivative, but also differ in nonlinearities, range and
  center coupling. Their comparison does not isolate positive definiteness alone.

## Checks, selection, evaluation

Before fits, perform bounded tests of nonzero independent direct scores,
shared initialization/reference functions, gradients, sampler identity and
serialization/resumption. Check fit/val real input and all reference identities.
Freeze protocol, transitive scientific sources, manifest and reference hashes.
On CUDA smoke verify actual CUDA access before training. Run only on GPUs idle
in two observations5seconds apart; at most6workers,CPU4each. Preserve others' jobs.

Save current model, optimizer, best model/logits, histories, RNG and epoch every
5epochs. A resume reconstructs the original NumPy sampling hash by CPU replay,
checks RNG/hash identity, preserves interrupted artifacts, then restores the same
optimizer and per-update dropout seeds. No reset optimization or budget extension.
An eight-hour worker limit is operational, not model selection; do not silently
discard failed runs. Ordinary pre-run fixes retain versioned failure evidence.

All39 models are selected/sealed before any new dev_eval prediction. Independent
NumPy/SciPy direct positional arithmetic checks selected first-layer samples,
and saved validation logits independently replay selection CE. This is not a
full-network independent rewrite/second-lab replication. Small arithmetic
differences are warnings assessed by effects on outputs/conclusions, not new
universal gates. Identity errors or substantive operator errors require fixes.

Evaluate the unchanged seven R48 conditions:base,Y90,tilt Rz45@Rx30,jitter02,
256points,1024points,point-order reversal. Primary utility is base OA against
RoPE,MLP,Gaussian-alone,joint_fixed,joint_free_quadratic; also report mCA,CE,
all other references,paired seed differences and all robustness conditions.
Report parameter cost and measured wall time without claiming shared-host
throughput rankings. R48's catastrophic tilted-rotation failure remains a
limitation; no augmentation change in this comparison.

After all selection, two prespecified fixed-checkpoint diagnostics on base old
dev_eval, for joint_conditional and original conditional, all3seeds:
1. Replace Sigma by trace(Sigma)/3 * I, retaining learned center m.
2. Rotate Sigma by a fixed90degree Z rotation, retaining eigenvalues and center.
Apply consistently at all4layers/heads without retraining. This tests sensitivity
to the trained shape, not physical shape recovery; distribution shift can also
damage predictions. Interpret alongside trained locality/quadratic controls.
No tuning perturbation angles, widths, seeds, temperature or selection on dev_eval.

## Decisions

Small inconsistent differences are ties; there is no arbitrary1pp threshold.
Stable paired utility over strong controls, with compatible mCA/CE and no material
hidden robustness loss, would justify a prospectively frozen diagonal/isotropic
simplification study, then a second task and untouched confirmation. Nine fits
alone cannot establish general effectiveness. If locality/free quadratic matches
the primary, do not attribute utility to the GS structure. If the joint does not
reliably improve the strong controls, do not rescue it with repeated budget or
hyperparameter scans. Preserve all outcomes and reconsider the structural role.

Sources motivating scope:PRoPE https://arxiv.org/html/2507.10496v2#S3.SS4 ;
HRoPE https://arxiv.org/html/2610.28284v1#S3.SS2 ;
docs/round48_native_point_report.md and docs/round55_budget_diagnostic_report.md.
