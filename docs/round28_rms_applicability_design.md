# R28 preparation: fixed real-input applicability of the RMS projective law

2026-10-31. Selected after the accepted R27 fixed synthetic numerical pass.
R26's failed reference attempt remains failed. No R28 real array has yet been
decoded, no R28 production call or label access has occurred. This is an
applicability/cost screen, not an efficacy candidate or general error proof.

Reuse exactly the72 R23 prepared target-camera Gaussians: three old TRAIN
scenes c3/c5/c6, each first pair30/45, query sizes8/16/24 and the existing
eight fixed indices per size. All72 passed the old independent input audit;
the old ray implementation completed0/72 calls. Preserve that failure and
population. Do not rebuild patches, decode original R10 inputs, reselect
queries, access labels or choose a real anchor/content rule.

The old prepared archives already bind mu[3], factor[3,3], H[2,2], offset[2],
bounds[432,4], target covariance[3,3], query bounds and index. Read only those
eight fields after the new joint freeze. Build the homogeneous matrix

    A = [[H00,H01,offset0],[H10,H11,offset1],[0,0,1]],
    mu_h=A mu, factor_h=A factor.

Keep the old native bounds byte-for-byte and set bias exactly zero. The old
H=K[:2,:2]/8 and offset=K[:2,2]/8 already match old bounds
(pixel_boundary-.5)/8. Do not subtract another .5 from the principal point,
assume zero skew, clamp depth, clip projected coordinates, repair covariance,
recenter at an anchor or add camera-frame noise. The immutable R26 model adds
its declared RMS-dependent noise only after this homogeneous transformation.

First bind R27 acceptance and all R23 receipts, inventory, prepared records,
72 archive hashes and successful old input checks using metadata/file hashes.
Then jointly freeze producer/auditor source/runtime before any array decode.
Prepare and globally seal all72 transformed inputs; independently audit every
mapping before any production call. Recompute H*XY+offset*Z elementwise,
factor columns, covariance congruence, exact Z preservation, original bounds,
zero bias and canonical source-array identity. Use normalized input error
maxabs(error)/max(1,maxabs(expected))<=1e-12, matching prior input tolerance.

Run the unchanged R26 producer at h=.25, one area floor1e-30 and T0, with
4M density nodes per call and600s for all72 forward outcomes. A nonstructural
complete native432 call statically uses1,990,656 nodes. All cases remain in
the ledger; retain partial arrays, consumed work, precise reasons, and whether
a model call was actually attempted before the shared deadline. The current
producer rejects positive-density underflow: a far cell can cause an entire
call to fail even if the floor would make the final P numerically benign.
Do not fix, clip or skip such cases during this frozen screen.

Audit saved raw/floor/P arithmetic, finite nonnegative arrays, first-index
argmax/ties, direct-FOV additivity, work, budgets, source identities and
chronology. This is independent input and output arithmetic verification,
not a fresh independent probability integral on real data. Always retain
independent_probability_reference=false,
general_numerical_accuracy_established=false, scientific_gate_passed=false,
goal_complete=false, labels_read=false. Only applicability_calls_completed
can become true if all72 calls and the declared audits/budgets pass.

Even a complete screen does not resolve anchor semantics, regional/center
quality, perturbations, actual V, optimized GS or untouched confirmation.
Those require later separately frozen protocols and all prior scientific
gates. The unselected anchor review remains mathematical preparation only.
