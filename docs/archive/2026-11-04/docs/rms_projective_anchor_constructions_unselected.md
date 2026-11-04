# Unselected anchor constructions for a single projective Gaussian

2026-10-31. Static code and mathematical review only. No input archive,
labels or efficacy calculation was used to select a construction. Neither
construction is part of the R26/R27 model or selected for the next numerical
applicability screen.

R23 forms target-camera mu and factor F from the predicted patch points plus
within-pixel covariance. R13's predicted source-center depth, obtained by
bilinear sampling on the center ray and transformed by the same R,t, can
provide a target-camera anchor a3 without ground truth. The old R20 screen
anchor a2 used a target-depth lower clamp .1 and screen clipping +/-1024.
It cannot silently be identified with the strict perspective pi(a3) on
near-camera, behind-camera or clipped rows.

Two compact constructions can be defined, with different semantics:

1. In homogeneous screen/depth coordinates, let delta=pi(a3)-pi(mu) and
   T=[[I2,delta],[0,0,1]]. Transform both mu and F by T. The complete R26 raw
   screen measure is translated by delta: depth/front mass/RMS and the
   independent XY regularizer are preserved. It moves any already correct
   regional mass too, and places the mean on the anchor ray rather than
   necessarily at a3. It does not guarantee that the projected mode or a
   native-cell argmax reaches the anchor. It is undefined if either relevant
   depth is zero and can produce huge shifts near zero depth. It is a
   view-dependent shear, not a common rigid world-coordinate change.
2. Set mu'=a3 and Sigma'=F F^T+(mu-a3)(mu-a3)^T. A QR factor of the columns
   [F,mu-a3] keeps a single Gaussian without a depth division. In the full-rank
   case this is the covariance minimizing Gaussian cross entropy at a fixed
   mean; for degenerate supports the second-moment definition still makes
   sense, but an interior full-rank optimum is not claimed. It changes depth
   mean, variance, front probability and RMS regularization, so it preserves
   neither the old projective measure nor exactly the R20 2D construction.

Both constructions commute with positive rescaling of the whole geometry
when their anchors transform consistently. The 3D second-moment construction
also commutes with a shared rigid change of world coordinates. A physical
translation of an object relative to its camera changes perspective and is
not an invariance claim. For actual GS with a3=mu, both anchor modifications
reduce to the identity; the R26 projection and regularization still differ
from earlier screen-Gaussian approximations.

If later center semantics require a separately frozen experiment, the 3D
second moment has a complete no-division parameterization; the shear is a
useful translation control. Neither has empirical support here. Existing
center/region conflicts and the actual V/GS/fresh gates remain unresolved.
