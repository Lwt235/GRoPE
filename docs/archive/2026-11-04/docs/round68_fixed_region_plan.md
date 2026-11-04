# R68 preparation: fixed-region cross-view mechanism study

2026-11-04. Prepared while the single R67 training-path diagnostic runs. R67's
results will be reviewed separately; this preparation does not assert success.
The objective remains a compact, useful Gaussian-parameter PE with applicable
generalization evidence. R65/R66 classification and earlier transfer/RGB
failures remain part of the record.

Hypothesis: explicit Gaussian relations help correspondence when center-layout
signatures admit competing candidates whose regional geometry differs, beyond
having full shape in token content or giving the same geometry to an ordinary
MLP. Alternatives: center/content suffices; ordinary geometry MLP suffices;
normal plus scale suffices; shape is a nuisance; apparent density robustness
was caused by changing which queries survived.

Use only the existing R60 SUN3D development partitions:576 fit pairs in3 scenes,
64 val pairs in1 scene,480 used-eval pairs in5 scenes. Camera pose is used only
for labels, never as a model input. Two original deterministic fit sampling
realizations, one val and one evaluation realization; no new scene access.
Build64 FPS regions, each the32 nearest observations among512 points per view,
using the established pose-free normalization and covariance floor.02^2 I.
Reference region overlap and world centers come from those observed sensor
surface samples and known poses, not a Gaussian similarity formula.

The sparse diagnostic retains all64 anchors and randomly chosen other points,
256 unique observations total per view. Within each original region use only
its surviving observations to re-estimate mean, covariance and radial features.
Keep the original reference regions, overlap labels, valid queries, world
centers and strata unchanged. Report surviving counts. This is a controlled
known-region observation test: full-density region membership is supplied to
all methods. It is NOT a test of discovering regions anew from sparse clouds.
No query is removed because its sparse estimate is poor; the original reference
validity criterion and its coverage are reported, alongside all-query success.

Five roles, seeds11/22/33, all trained from scratch with identical common
initialization and actual augmentation streams:
distance_mlp; direct_mlp(relative displacement, distance, both log covariances);
normal_scale(the existing R61 unoriented normal/projector plus spectra);
gaussian_relation(the existing R58/R59 eleven-dimensional relation);
shuffled_gaussian(the same Gaussian operator but independently reassign region
covariances within each view under an identity-seeded rule, during BOTH fitting
and evaluation). Correct common content is retained for the shuffled role.
All see the same20-channel content: original11 regional features, center3 and
full symmetric log-covariance6. Prior matching content contained spectra/radial
features rather than a full covariance; do not misdescribe this as a rerun of
an already identical-information experiment. No RGB or target-pose features.

Same four-block96-channel,2-head matcher; PE uses one96-hidden-layer MLP and
enters the existing self-attention blocks. Report actual active parameters;
similar width does not mean precisely equal capacity. The direct control has
all the raw pair geometry used by the Gaussian relation. The normal control
is not a full RoITr reproduction. This is an attribution probe, not a final
novel architecture or a test that native optimized3DGS is necessary.

Use the established from-scratch60-epoch budget: batch32,1080 updates if all576
pairs are present,AdamW1e-3 to1e-5,warmup3,weight decay.01,clip1. Alternate fit
realizations and add matched independent rigid augmentations of each view.
Validate every3epochs; select earliest strict minimum bidirectional valid-query
CE. Select all15 models before exporting used-eval results. No temperature,
formula or learning-rate sweep. Only base inputs used for training.

Predefine two INPUT-only scores per query. Center ambiguity compares the sorted
eight-neighbor distance signature with all target signatures, using
1-(second-best minus best)/(fourth-best+epsilon). Shape diversity is mean
log-covariance distance among those four candidate regions, computed within
the target view's frame. Medians over all fit0/fit1 queries define low/high
thresholds before reading evaluation scores. Report all four strata and their
coverage. These are geometric proxies and association evidence, not a causal
or complete characterization of geometric ambiguity. Sparse strata reuse base.

Primary: region R@1, first average within each of five evaluation scenes, then
across scenes. Also report query-weighted scores, center PCK at.1 normalized
units, soft EPE, CE, valid coverage, unconditional success and all seeds/scenes.
Use paired seed and scene effects; many adjacent queries are not independent
scenes. A useful development lead is >=1pp mean and three positive paired seed
gains over both distance and direct MLP; show uncertainty and magnitudes rather
than multiplying universal gates. If normal/scale is similarly effective,
prefer its simpler interpretation. A mean center-PCK loss>1pp limits a claim
of joint region and localization benefit. Small secondary numerical or metric
changes are assessed by their practical effect. No general-effectiveness claim.

Decisions: if correct geometry improves actual matching beyond strong controls,
and gains occur in the predicted strata, pursue one compact mechanism for
Transformer novel-view synthesis. If only the shuffled role degrades, report
reliance without proven utility. If all strong methods tie, special Gaussian
organization is unestablished. Preserve a conditional result as conditional;
do not promote one favorable stratum into an overall success.

Checks focus on label/input separation, sparse fixed-query identity, original
raw identities, common initialization/input pairing and independent metrics.
No old quadrature prerequisites. Frozen unused-data confirmation remains later.
