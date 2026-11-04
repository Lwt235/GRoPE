# R17 actual-V interface: synthetic preparation only

2026-10-31. This is a numerical/software preflight, frozen before execution.
Main R17 old-VAL acceptance and scientific results are still pending and have
not been read for this preparation. No real input archive, label, checkpoint,
training dataset, GS reconstruction or fresh scene is used. This document does
not authorize real task training, select a candidate, change the main source
lock, or relax any geometry, cost, actual-V, GS or confirmation gate.

## Fixed interface

The new adapter inherits R7's exact 41,345 parameters, names, constructor RNG
order and initialization. Its modes are interfaces, not a selected scientific
role population: `trace`, `constant`, and the unchanged `singleview` control.
Trace/constant require explicit h/T. Singleview keeps h=.5/T=5 and the original
R7 quadratic self geometry, own query features/depth and float32 arithmetic.

Cross-view input fields are q, k, q_depth, k_value_depth, geometry_logits, plus
support_covariance in trace mode. Query/key features, depths, model parameters,
support C and learned V are float32. Geometry is pure finite float64 bias;
this synthetic test does not generate or validate a physical geometry cache.
Trace mode calls the frozen R17 trace implementation: full original C computes
tau, rounded float32, multiplied by current normalized learned Q/K dot products
in float32; add the product to geometry in float64 and softmax in float64.
Constant mode follows the same precision policy with constant T. Both then
perform exactly one explicit attention-to-float32 cast before multiplying by V.
No extra layer, loss weight, visibility rule, covariance predictor or parameter
is introduced. GT/visibility/boundary and unrelated fields are not model inputs.

Matched isotropic geometry would still receive original full C for tau from a
future source-checked cache; the adapter does not synthesize or guess that cache.
All real data joins, controls, training populations, checkpoint selection and
timing remain subject to a separate future locked task contract.

## Pre-execution freeze and fixtures

After source review, freeze this protocol, the new adapter, its unit tests, the
separate synthetic reference runner, and their recursive repository import
closure into a new external results directory. Record runtime, source and
fixture hashes before numerical evaluation. Use CPU with one Torch thread,
deterministic algorithms and TF32 disabled. Main paired timing has completed;
these tests must not overlap a new timing stage. An exception or failed check
is preserved; any correction requires a separate source/attempt record.

The numerical runner uses one predetermined synthetic population (seed 171710,
model initialization seed171711). The native-size
fixture has batch1, Q=K=432, feature dimension128 and V dimension64. A 24×18
coordinate grid produces finite negative squared-distance geometry; this is
an interface stress fixture, not native Gaussian cell integration evidence.
Independent normalized random q/k features and positive depth ranges [1,5)
and [1,6) avoid task labels. C is an explicit rotated 2D PSD matrix with
varying positive eigenvalues, including one exact zero covariance. A small
batch2/Q6/K7 fixture checks batched rectangular interfaces. All seeds, fixture
construction and synthetic loss are fixed in the frozen runner; no fixture
selection by outcomes. Separate fixed unit fixtures use seeds171723/171724,
batch2/Q6/K5 and h=.625/T=3 for branch/interface coverage, as specified in their
frozen test source; these are not calibration candidates. Use h=.25/T=10 for
the numerical runner's trace/constant, and test T=0 and C=0
as mathematical interface limits rather than new calibrated configurations.

## Independent numerical checks and limits

1. Reconstruct tau from actual C in NumPy float64 with the frozen scaled ratio,
   round once to float32, and compute NumPy softmax of geometry plus float32
   tau×content. Actual learned projection/content GEMM is explicitly shared
   for this same-content check. Require exact tau32 and maximum probability
   error <=1e-12; preserve every row, including any tiny probabilities.
2. Reconstruct the complete fusion forward in NumPy float64 using the same
   saved float32 parameters and inputs, explicit normalized affine Q/K,
   GELU erf, V, query/head and depth clamps. Tau retains its original float32
   rounding but the reference otherwise omits intermediate float32 rounding.
   This is a precision reference, not a second trained model. Require maximum
   probability error <=1e-5 and maximum depth relative error <=1e-5.
3. Reconstruct the full differentiable fusion calculation separately in Torch
   float64 with original float32 parameter values promoted to double. Use the
   same float32-rounded fixed tau, and the synthetic mean squared log-depth
   residual against 0.91*q_depth. Compare every named parameter gradient:
   relative L2 error <=1e-3 for nonzero reference vectors, finite values, and
   exact zero Q/K gradient at T=0. Record absolute errors and norms too; no
   measured gradient magnitude is an efficacy requirement or new gate.
4. Same-content P64 versus actual P32@V precision is checked against a double
   product using actual learned float32 V. Require max absolute error <=2e-5
   times max(1,max(abs(V))). This only validates the declared casting interface.

The complete fixture runs trace and constant at fixed h/T, trace T=0, and trace
C=0. The two positive-temperature modes also run zero-V and reversed-V; these
are diagnostics, not candidates. Unit tests additionally verify bitwise R7
initialization, unchanged singleview outputs and gradients, forbidden-field
poisoning, invalid input rejection, zero-temperature and point limits, and an
online learned Q/K change affecting attention. Zero/reversed V must be applied
after the learned V path, exactly as in the original task.

Numerical or unit-test failures stop this preparation. Passing supports only
the software/precision interface on this synthetic population. It does not
establish good gradient conditioning, meaningful content learning, improved
depth prediction, covariance training, lower cost, actual GS benefit or fresh
generalization. No real training is started by this runner or by a pass.
