# Unselected reserve: a Gaussian ray law must specify its measure

2026-10-31. Root's theoretical/literature note while R20 implementation is
being prepared. **Not selected, frozen, implemented or experimentally tested.**
It changes neither R20 nor any continuation gate. It is not a novelty claim.

R19 concerns a designated patch center. R20 can still leave the original V
bottleneck unchanged. A possible later question is whether retaining the full
3D Gaussian, rather than only its projected 2D marginal, permits one positional
law to supply both image mass and a ray-conditioned geometric value. This is
different from adding unrelated prediction modules, but usefulness remains
unknown and plain plane/bilinear controls are mandatory.

The following is a change-of-variables derivation, not a result taken from a
rendering benchmark. Let camera-space X~N(mu,Sigma), Sigma positive definite,
normalized image coordinates (u,v), ray r=(u,v,1), and positive camera depth z.
The map X=z*r has volume Jacobian z^2. Therefore its projected *probability*
mass and conditional depth involve

    p(u,v,z) = N(z*r; mu,Sigma) z^2, z>0
    p(u,v)   = integral_0^infinity N(z*r;mu,Sigma) z^2 dz
    E[z | u,v] = integral z^3 N(z*r;mu,Sigma) dz / p(u,v).

A conditioning constant P(Z>0) cancels when normalizing a single primitive
over a fixed visible image domain, but its treatment and any additive floor
must still be defined together. Camera-pixel density also needs the intrinsic
coordinate Jacobian. Retaining all query rows is separate from restricting a
ray to positive physical depth; zero-mass/fallback behavior cannot be omitted.

With Lambda=Sigma^{-1}, A=r^T Lambda r, B=r^T Lambda mu, D=mu^T Lambda mu,
m=B/A and s^2=1/A, completing the square yields a common factor
g=(2*pi)^(-3/2)|Sigma|^(-1/2) exp[-(D-B^2/A)/2]. Let

    F_n = integral_0^infinity z^n exp[-(z-m)^2/(2s^2)] dz.
    F_0 = s sqrt(2*pi) Phi(m/s)
    F_1 = m F_0 + s^2 exp[-m^2/(2s^2)]
    F_n = m F_(n-1) + (n-1)s^2 F_(n-2), n>=2.

Then projected density is g*F_2 and conditional depth is F_3/F_2.
These real-arithmetic expressions are **not a numerically validated algorithm**:
negative-tail cancellation, exp underflow, D-B^2/A subtraction, singular/flat
support, finite cell integration and any positional smoothing remain unresolved.
An arbitrary covariance jitter would change the model, not merely stabilize it.
The ratio is query-dependent; it does not justify commuting log/GELU with V
aggregation or claiming an exact finite-dimensional Q/K feature map.

In contrast, integrating volume density with respect to camera depth dz gives
g*F_0. The physical arc-length line integral is ||r||*g*F_0 because r=(u,v,1)
is not a unit direction. Its associated mean camera depth is F_1/F_0; mean
range is ||r||*F_1/F_0. Maximum density along a ray is another
quantity. These cannot be swapped with projected probability solely because
each formula contains Gaussian ray integration. Surface/opacity rendering,
probability pushforward, finite-pixel mass and a depth readout are separate
contracts. No formula here supplies scene visibility/transmittance for free.

Primary-source checks, with their limits:

- [Celarek et al., 2025, arXiv v1](https://arxiv.org/html/2502.19318v1),
  especially sections 4–5, separates opacity splatting from extinction-based
  volume rendering and the affine projection approximation. Its experiments
  also show that more faithful rendering need not improve a model optimized
  with many primitives. This argues against assuming that exact projection
  alone will repair R9; it does not decide our attention task.
- [3DGUT, NVIDIA's project page](https://research.nvidia.com/labs/toronto-ai/3DGUT/)
  describes unscented projection and ray-based evaluation. Thus retaining
  nonlinear projection or Gaussian ray geometry is not itself a new idea.
  The page was checked; this note does not claim a full algorithm reproduction.
- [Exact-GS, CVPR 2026 official proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Exact-GS_Mathematically_Rigorous_and_Accurate_3D_Gaussian_Splatting_for_3D_CVPR_2026_paper.html)
  is explicitly for X-ray reconstruction. The official abstract and linked
  supplement show a ray-integral formulation. Browser access to the main PDF
  returned 403 in this check; its full derivation was not verified. Do not
  mislabel its line integral as the z^2 probability pushforward above, or use
  CT reconstruction scores as evidence for PE/visible-light NVS.

The separate [mathematical review](round20_ray_measure_math_review.md) checks
this measure distinction and records additional limits: native-cell depth is
the ratio of integrated first-moment density to integrated probability, and
image smoothing must act on both densities. The infinite-image distribution
can have heavy tails; finite projected moments cannot simply be assumed.

If revisited, first compare the distinct measures on synthetic slanted planes,
large depth variance, behind-camera tails and the far-field limit, with an
independent integral and a fixed numerical budget. Any actual-data diagnostic
would need a separate selection rationale and protocol, same-information
plane/bilinear controls, complete original masks, and every unchanged later
gate. This note does not authorize that experiment or offer a fallback winner
if the currently selected R20 fails.
