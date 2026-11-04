# R69: one bounded decomposition before image-task integration

2026-11-04, after the completed and reviewed R68 result. Development-only,
adaptive choice of this experiment; none of these data are fresh confirmation.
R68's Gaussian relation improves region correspondence and center localization
over distance, raw geometry MLP, and the specified normal/scale control with
identical complete content. The predicted ambiguity-specific concentration
was not established. R65/R66 classification and earlier transfer/RGB failures
remain. R67 is an independent, unfinished training-path diagnostic.

Hypothesis: direction-dependent displacement in each region's Gaussian metric
provides most of the useful relational structure; absolute spectral descriptors
and covariance compatibility may be dispensable. Alternatives: absolute scale
descriptors suffice; compatibility is necessary; all eleven inputs are useful.
This is component attribution and simplification, not a novel distance claim.

Keep the R68 caches, fixed reference queries, all20 content channels, four-block
96-wide matcher, initialization of every common parameter, exact augmented
input streams, three seeds11/22/33,60epochs/1080steps, optimizer and validation
CE selection unchanged. Reuse all five R68 methods by checkpoint/prediction
identity. Nine new fits only; no temperature, frequency, width or budget search.
Select all nine models before exporting new development predictions.

Let delta=mu_i-mu_j, r=norm(delta)/.2,
u_i=log1p(delta^T Sigma_i^-1 delta), u_j analogously,
v_ij=log1p(tr(Sigma_i^-1 Sigma_j)), v_ji analogously,
and a_i=.5 log(eigenvalues(Sigma_i)/.04). All arithmetic/scaling is exactly
the corresponding columns of the retained R68 eleven-dimensional descriptor.
Each role uses the same width96 GELU MLP with one hidden layer:

* relative5: [r,u_i,u_j,v_ij,v_ji], discarding six absolute spectral channels.
* metric3: [r,u_i,u_j], retaining displacement/shape coupling only.
* spectrum7: [r,a_i,a_j], discarding all displacement/shape and mutual-shape
  orientation relations. This is an explanatory control, not full same-pair
  information; correct complete covariance remains in common token content.

The five/three-dimensional versions retain two directions separately. They
are not the R58 fixed weighted sum followed by a sinusoid, which failed, and
not R66's independent log-shape similarity, which did not couple displacement
to shape and failed native classification. Those negative results remain.
No claim that a merely different formula is new or effective.

Primary region R@1 is the R68 scene-macro score; report center PCK, soft EPE,
CE, all seeds/scenes, both base and fixed-query sparse256, and exact coverage.
For every new role report paired differences against full11, distance and
same-information Direct MLP. Retain normal/shuffle scores in the overall table.
Show five-scene bootstrap intervals conditional on the fixed seeds. Strata
can remain in prediction metrics, but will not choose a winning subgroup.

Decision: if metric3 has positive mean and all-three-seed gains over both strong
MLPs in both conditions, mean region loss no greater than1pp versus full11 in
each condition, and mean center loss no greater than1pp versus full11, prefer
metric3 as the compact development candidate. Apply the same descriptive rule
to relative5 if metric3 does not qualify. This1pp tolerance expresses an
engineering tradeoff, not a formal noninferiority claim or universal scientific
gate. If neither qualifies, retain the already useful full11 instead of adding
modules or searching until one passes. If spectrum7 matches full11, limit any
claim that directional coupling explains the benefit. Report all outcomes.

Regardless of this decomposition, the R68 signal is sufficient to proceed to
a controlled Transformer image task. The decomposition must not become another
open-ended barrier to downstream utility. Native3DGS, generalization and fresh
confirmation remain separate requirements for the overall research goal.

Pre-run checks: exact descriptor-column equivalence, common initialization
identity, and effective GPU gradients/independent metrics. Reuse the already
reviewed data checks. Preserve other users' GPU jobs and coordinate with R67.
