# R26 preparation: RMS-depth regularized projective Gaussian

2026-10-31. Prospective mathematical design after accepted R25 failure; no R26 numerical fixture, real array, label or efficacy run. This document does not select an efficacy candidate. The proposed next stage is a separately frozen synthetic numerical preflight. No change to R22–R25 sources or their failures.

R25 fixed t4 retained center guards but worsened q24→g8 KL in all three old TRAIN scenes. This does not disprove every radial family. It does make another arbitrary fixed-tail substitution a weak next hypothesis. Instead test retaining a single full three-dimensional Gaussian, including depth/XY correlations, until perspective projection. The regularization below is a new law, not an equivalent repair of R23's screen Gaussian convolution.

Use homogeneous screen coordinates V=(Vx,Vy,Z): camera intrinsics and principal point are included in the linear transform from target-camera coordinates, so screen position is Vxy/Z. Input is one mean μ and one finite Gaussian factor F, Σ=FFᵀ; no mixture or learned module. Let s²=μz²+||Fz||²=E[Z²]. If s>0, normalize μ and F by positive s, add independent N2(0,h²I) noise only in the normalized XY coordinates, and project at strict Z>0. Equivalently before normalization, add XY covariance h²s²I. Fix h=.25 reference-8 units. Native cell mass is the unconditioned positive-depth probability, followed by one area floor1e−30 and native content softmax. Synthetic stage can test fixed T0 and prescribed bounded native bias; no learned features or real content-temperature rule is selected here.

If s=0, Z is exactly zero and all raw mass is structurally zero. If Fz=0 but μz>0, projection is a two-dimensional Gaussian with the original projected covariance plus h²I; nonpositive deterministic depth gives zero mass. No arbitrary numerical rank threshold. If ||Fz||>0, adding XY noise makes the full Gaussian positive definite, even when the original F is rank deficient. It is a declared model regularizer, not hidden Cholesky jitter.

This keeps world-scale invariance, uniform screen-unit scaling when h/floor density transform, screen translation and orthogonal axis rotation, positive-depth total mass, native-cell additivity and the positive-depth point limit. A non-isometric screen transform would require transforming h²I as a matrix; the fixed scalar-h family is not claimed invariant to arbitrary shear. It changes the meaning of smoothing: conditional added screen variance is h²E[Z²]/z². Along an exactly radial Gaussian X=Zr, the original projected support is a point; the new law can still have tails. For zero mean Z it gives a half-mass Cauchy(h), whereas R23 gives a half-mass Gaussian(h). Thus it does not preserve equivalence of identical original projected measures, or fixed screen-noise variance. No universal effect gain or novelty is claimed.

The new unconditioned density has a useful bound: conditioning on Z and the unsmoothed XY variables, the added Gaussian gives g(y)≤E[Z²1{Z>0}]/(2πh²E[Z²])≤1/(2πh²). The same conditional-mixture argument bounds any fixed spatial derivative by a Gaussian derivative constant times E[|Z/s|^(n+2)]/h^(n+2); Gaussian standardized-by-RMS moments are bounded for each finite n. These are regularity observations, not certified finite-quadrature or tiny-FOV relative-error bounds. A FOV-conditioned P can still be sharp when the original FOV mass is tiny.

The ray-density identity is established projected-normal mathematics, not a new representation claim. For a fullrank Gaussian with positive-diagonal Cholesky factor L and r=(y1,y2,1), let a=||L−1r||², b=(L−1r)·(L−1μ), c=||L−1μ||², t=b/√a. The unconditioned screen density is

    exp(−(c−t²)/2) J2(t) / (2π det(L) a^(3/2)),
    J2(t)=(1+t²)Φ(t)+tφ(t).

[Figueras, Persson & Viitasaari, §2.3](https://vmsta.org/journal/VMSTA/article/329/read) documents the general projected-normal sphere density. Our screen form follows the z² Jacobian and an elementary positive-depth integral; see the prior [general-ray draft](general_ray_reference_unselected_draft.md). The new regularizer above is a separate proposed modeling choice, not a claim from that paper.

For a more stable producer, use normalized conditional coordinates Z=m+σζ, Vxy=μxy+βζ+Bξ, where σ>0 and B is a positive-diagonal Cholesky (or sign-normalized QR-derived) factor of the XY conditional covariance including h²I. With a general nonsingular factor instead, determinant denominators below must use absolute values. Define at screen y:

    v=B−1(σy−β), u=B−1(σμxy−mβ), D=1+v·v,
    N=v·u+m, w=N/√D, t=w/σ, d=my−μxy.

The nonnegative perpendicular exponent can be formed without subtracting c−t²:

    δ=[||B−1d||² + ((σ det(y,μxy)+det(β,d))/det B)²]/D.

For t≥0 the density is

    exp(−δ/2) [(σ²+w²)Φ(t)+σwφ(t)] / (2π det B D^(3/2)).

This analytically cancels the σ² J2 product and avoids a spurious divergent factor as depth variance tends to zero. For t<0, q=−t and c=(||u||²+m²)/σ²:

    g = σ² exp(−c/2) K(q) / ((2π)^(3/2) det B D^(3/2)),
    K(q)=∫0∞ z² exp(−qz−z²/2) dz.

Moderate q permits an erfcx expression; large q needs a positive integral or a controlled alternating expansion, not cancellation of two O(q) terms. Branch precision, thresholds and rejection rules must be frozen before formal fixtures. A log form and algebraically stable scalar branches do not certify the whole screen integral.

An independent reference can condition on standard normal ζ and integrate 2D Gaussian rectangle masses in homogeneous coordinates. For positive z=m+σζ, the rectangle is zB and the conditional covariance BBᵀ is constant: division by tiny z is unnecessary. Its implementation must not call the producer's screen density. Stable narrow Gaussian-CDF differences, explicit depth/inner truncation bounds, independent refinement, raw/P gates and resource accounting remain necessary. Existing R22 depth-integral code cannot simply be called unchanged and declared fixed; it failed real applicability.

Possible future content calibration and real-input anchor construction are deliberately undecided. In particular, a three-dimensional support mean need not project to the source-center anchor; successful numerical integration would not repair this semantic gap. All old region/center, perturbation, actual V, optimized GS, strong-control and untouched-confirmation requirements remain.
