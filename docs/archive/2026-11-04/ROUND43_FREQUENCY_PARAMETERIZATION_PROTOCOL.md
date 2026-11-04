# R43: grouped Gaussian frequency parameterization, development screen

2026-11-01. R42 is complete and negative; its raw physical-shape descriptor is
retired. This is a DIFFERENT hypothesis about the GS parameterization itself,
not a claim that optimized primitive R/S became useful or that R42 was rescued.

Hypothesis: constraining learned rotary frequencies by a few mean/covariance
groups improves correspondence beyond fixed PE and ordinary learned frequencies.
Alternative: it merely reparameterizes affine Fourier features, or any gain comes
from learning frequencies at all. No new Gaussian/Fourier kernel theorem claimed.

Alignment clarification, before freeze and before any R43 training: the user
resumed the goal after identifying the drift toward rendering. The research
object is the positional operator INSIDE a Transformer; correspondence is this
screen's downstream measurement. Here the GS-style mean/rotation/scale structure
parameterizes a learned frequency distribution, not an input primitive's
physical support. This interpretation is explicit and does not establish that
real-GS shape information is useful. Rendering is an optional subsequent transfer
test, not a prerequisite or the definition of success for a general PE.

## Model and controls

Keep the complete R5 two-block bidirectional cross-attention matcher (166611
non-frequency parameters), its Q/K/V, residuals, FFNs, final head and learnable
temperatures. Only replace the frequency generator; no new value/decoder/bias.
Use 24 rotary frequency pairs in 2D projected image coordinates (units8px),
shared between heads, layers and directions, as R5. This tests the 2D instance
of a general mean/SPD-covariance parameterization; 3D task utility remains open.

Four groups of six fixed standardized normal templates epsilon are generated
with NumPy seed20261102, centered and whitened once so their empirical mean is
zero and covariance is I (denominator6). They are finite templates, not an exact
continuous Gaussian integral. Initial group sigma is the RMS norm of the
corresponding six original R5 frequencies divided by sqrt(2). All seven new
roles start at EXACTLY the same finite frequency bank, without using data scores.

Primary gaussian_full: w_gm = sigma_g (mu_g + L_g epsilon_gm).
L_g is lower triangular, diagonal exp(d_g), initially I, off-diagonals initially0;
mu_g starts0. Its frequency mean is sigma_g*mu_g, covariance
sigma_g^2 L_g L_g^T, equivalently representable as R diag(s^2) R^T.
Cholesky is unique for SPD covariance and avoids eigenframe sign ambiguity.
All parameters are dimensionless changes from the common initialization.
The point rotary blocks remain unitary; physical token covariance is not input
to these new roles. This is learned FREQUENCY covariance, not uncertainty or
physical rendering support. No interpretation of one as the other is allowed.

New controls: gaussian_diag (diagonal L), gaussian_iso (scalar L),
fixed_gaussian, shared_affine (one affine map of the initial frequencies),
group_affine (one free affine map per group), free_frequency (each frequency
has a free offset). Counts of added parameters in this 2D experiment are
20/16/12/0/6/24/48 respectively. No inactive padding to claim equal capacity.
Frequency offsets and group shifts are scaled by the initial group sigma;
shared affine shift uses RMS group sigma. Weight decay acts on zero-initialized
changes, not the fixed base frequencies. Log-diagonal parameters also start0.

All six R5 controls are retrained: point,expected,normalized,isotropic,fixed,
resolvent. Total13roles x5seeds =65 runs. Primary is gaussian_full throughout.
Ordinary free/group-affine frequency controls contain the primary's finite
function family; any benefit must be empirical constraint/optimization behavior,
not greater expressiveness. Tests explicitly construct this equivalence.
RoPE-Mixed and HARoPE are prior work; these controls are local mechanism controls,
not full reproductions of their published architectures or training.

## Data and fixed execution

Use only original R5 caches: train c3/c5/c6 (576pairs), val c8 (64pairs), and the
five ALREADY EVALUATED R5 test rooms (480pairs): harvard_c11/hv_c11_2,
hotel_umd/maryland_hotel3, mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika,
mit_76_studyroom/76-1studyroom2,
mit_dorm_next_sj/dorm_next_sj_oct_30_2012_scan1_erika.
Call the latter old_eval, not fresh confirmation. R5 report and analysis already
contain their results. The separate unused home_ac/hotel_barcelona/hotel_graz/
mit_1_242/harvard_computer_lab cohort is not accessed. Metadata identities and
source/data hashes are frozen before array decoding or tests.

Preserve all432 candidates, all sensor-valid query labels and all saved pair IDs.
Same predicted depth, frozen features, known reconstruction cameras, visibility
population and bilinear correspondence labels as R5. Labels affect loss/metrics
only; forward reads the original ten-field allowlist. No point-quality filters.
Five seeds11,22,33,44,55;1000updates,batch8,AdamW initial lr.001,wd.0001,clip1.
Keep R5's cosine learning-rate multiplier from1 to.2 and RNG seed+50000 batches.
Same neural initialization and NumPy sampled batches across all roles. Training
uses visible-row cross entropy, averaged over valid rows of the whole batch.
At0,100,...1000 select earliest strict minimum scene-macro validation CE.
No per-role LR, frequency, temperature, group-count or duration search.

Train only on idle GPUs, CPU4, TF32off, deterministic algorithms. One seed shard
may run all13 roles;16GiB allocated and3GPUhours ceiling per shard. Capture real
exit states; migrate only unstarted work if another task occupies a device.
Freeze all65 selected checkpoints before old_eval export. Save all complete
per-query sufficient statistics (argmax index, soft coordinate, log partition,
logits at label indices), frame rows, selected weights and histories. Independent
metrics cover every query/pair; network-score checks use declared samples and
must be reported as samples, not a second full-network implementation.

Before training: CPU translation/norm/parameterization/gradient and input-scope
checks; CUDA toy smoke for all roles. Independently reconstruct frequencies and
first-layer rotary probabilities in NumPy for saved validation samples of every
trained model. Float32 P>5e-5 warns and requires impact assessment; near-tie
argmax changes alone are not an automatic scientific failure. No quadrature gate.

## Decisions and remaining obligations

On old_eval compare primary to EACH predeclared decisive control: strongest
retrained R5 role, fixed_gaussian, gaussian_iso, group_affine and free_frequency.
Development is promising only with >=1pp PCK8 improvement,>=4/5 positive seeds,
positive differences in>=4/5 old rooms, and CE worsening<=2% for every comparison.
Report all roles and PCK4/EPE, not just a chosen opponent. No population
significance claim from repeatedly used rooms. These criteria do not replace
the final full objective; a smaller positive effect is recorded without promotion.
Save zero-V outputs for each selected model to check whether the task uses its
attention/V path. A zero-V difference is not correspondence or generality proof.

Failure ends promotion of this fixed parameterization; do not retry group count,
bandwidth, duration or temperatures on old_eval. Success only justifies the next
Transformer task/scale/noise check, including a genuinely 3D coordinate task,
with matched controls. Full applicable geometry/robustness/cost/generalization
and untouched joint confirmation all remain unproven. The frequency mean,
orientation and scale contributions require attribution. Physical input-GS R/S
attribution remains unproven and is required if a later design claims to use that
information; it is not inferred from this different frequency hypothesis.
Prior area-integral checks
are not silently declared passed: this encoder claims relative rotary phases,
not exact area mass. Any final applicability assessment must explicitly account
for each prior requirement rather than omit inconvenient failures.
