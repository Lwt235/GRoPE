# R66: center distance with one log-shape distance

2026-11-04. Adaptive development design following the completed, reviewed R65
full60-class study and fixed checkpoint-policy diagnostic. Neither R65 result
established a stable advantage over both strong MLPs. All earlier geometry,
transfer and RGB failures remain in scope; the overall goal is not achieved.

## Hypothesis and competing explanations

Native Gaussian shape relationships may help Transformer attention when supplied
as a simple comparison, independently of center separation. The specific
hypothesis is that relative orientations and relative scales of ellipsoids add
useful information beyond a learned center-distance PE. Alternatives are that
scales alone suffice, common token content already supplies useful shape, or
native primitive shape differences are nuisance rather than helpful context.
The R65 representation collision and covariance-floor observations do not
establish any of these task-level explanations.

Let L_i = (1/2) log(Sigma_i / 0.04), using the original regularized covariance.
Define r_ij = ||mu_i - mu_j|| / 0.2 and
s_ij = log(1 + ||L_i - L_j||_F^2). A single two-input MLP, 2 -> 96 -> 96 with
GELU, produces the same query-relative term q_i dot MLP(r_ij, s_ij) in every
unchanged R65 attention block. No sign or monotonicity constraint forces attention
to favor identical shapes. Shape changes never attenuate the existing center
coordinates or multiply Q/K amplitudes. The second input adds 96 parameters;
the PE has 9504 parameters versus Distance MLP's 9408 and Direct MLP's 10848.

This uses the known Log-Euclidean SPD metric (with a fixed monotone compression),
not a newly invented distance or a complete invariant of two Gaussians.
[Arsigny et al., 2006, Eq.2 and similarity invariance](https://www-sop.inria.fr/asclepios/Publications/Arsigny/arsigny_mrm_2006.pdf).
It is a compact learned relative PE, not a rotary-frequency construction.
The pair inputs are invariant to a common translation and orthogonal rotation.
The shape input alone is unchanged by a common covariance scalar multiplier;
the fixed raw-covariance floor limits a physical scale-invariance claim.
The full network's center/log-covariance content is not exactly rotation invariant.
The method omits displacement-versus-shape alignment and absolute covariance size
from PE; those losses may matter. No Gaussian-overlap interpretation is claimed.

R3 used periodic, world-axis-dependent per-token shape phases. R40/R41 used a
nonpositive scalar shape-similarity bias alongside center RoPE and failed their
tasks. R44-R46 changed spatial basis functions or warped centers. R56/R57 used
quadratic/Jeffreys relations with predicted shape. R65 used eleven invariants
including inverse-covariance terms. R66 instead tests a single, joint learned
function of center separation and log-shape difference on observed native GS.
The distinction does not erase those negative results or prove novelty.

## Fixed comparison and data

Primary candidate: log_shape_distance. Equal-capacity mechanism control:
spectrum_distance, replacing L by its sorted three eigenvalues in the same
formula. This removes relative principal-axis orientation while preserving
relative scales. Both keep the full13-channel content unchanged. All common
weights and distance-MLP weights exactly match original same-seed initialization;
the new input weights start at zero without new random draws. This tests the
specified conservative extension, not all possible initialization schemes.

Train both roles from scratch on seeds11/22/33: six fits,120epochs/46800steps.
Reuse all five completed R65 CE-selected baselines, with their sealed records
and prediction bytes checked. In particular Distance MLP and higher-capacity
Direct MLP are the strong controls. No new final120 selection or per-role policy
hybrid is used. Retain the original optimizer, augmentation, batch size,512tokens,
all120 actual sampling streams and earliest-strict-minimum validation-CE selection.

Reuse the exact full native-GS cache:6227fit/765val/765dev_eval,60classes,
7757officialTRAIN objects, same normalization and covariance floor. No exclusions,
new downloads, officialTEST, held-out ModelNetTEST or reservedSUN3D access.
This used cohort is development, not fresh confirmation. Select all six models
before any new dev_eval predictions; compare all original six conditions:
base512, sparse256, dense1024, jitter02, tilt and permutation. Report all seeds,
classes and sources; source label sets differ and do not establish transfer.

## Decisions and proportional checks

Primary outcome is base mean-class accuracy. A credible development lead over
each strong MLP requires mean gain >=0.5pp, all three paired seed gains positive,
and the original paired-class bootstrap95% lower endpoint >0. The class interval
averages paired seeds before resampling; it is not seed-population uncertainty.
All secondary conditions, CE/OA and source effects are reported without selecting
a different primary outcome or hiding negative classes. The shape-versus-spectrum
contrast uses the same descriptive quantities; a full-shape-specific claim
requires the same stable-lead evidence against this equal-capacity control.

If the primary candidate fails the strong-control comparison, stop this version;
do not search frequencies, shape scales, checkpoints or extra epochs to rescue it.
If it beats both MLPs but ties spectrum, do not claim relative orientation is
necessary. If only spectrum succeeds, regard it as a new development lead to
consider prospectively, not fresh confirmation. Any apparent success still needs
review of density/rotation behavior and later applicable transfer/mechanism and
unused-data confirmation before a general-effectiveness claim.

Before launch, use independent matrix-log reference and analytic orientation
examples, initialization/output equality, permutation and finite parameter
gradient tests; verify CUDA access and live GPU occupancy. Reuse the existing
GPU training/selection/export code through explicit bindings. Check source/data
identities, training streams and selection replay. Independently recompute all
classification metrics and inspect the generated plot. Small numerical errors
are assessed by task impact, not a new universal integrator gate. These checks
are by the same agent using independent formulas, not an independent-person audit.
