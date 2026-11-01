# Round 6: Gaussian support as a positional measure

2026-10-30. The recommended next hypothesis is **a directed, mass-aware Gaussian coverage prior**, tested on area transfer between differently sized image regions. It is a single relative positional term, not a claim of a newly discovered Gaussian kernel. Two compact alternatives are retained below for different domains; they should not all be trained merely to search for a favorable result.

## Evidence that changes the research question

[Round 5](../round5_report.md) found no predeclared practical improvement from normalized Gaussian RoPE over projected point RoPE: −0.06955 percentage points PCK8, with a confidence interval spanning zero and well below the required +1pp gain. Most input supports were nearly isotropic. The point-coordinate geometric diagnostic was already strong, and the learned final head lost some finer localization precision. This motivates preserving a geometric prior and testing a task whose target actually depends on support extent.

The earlier world moment/normal RoPE, orthogonal/isometric completions, and exact density kernels are not unexplored candidates. `world_density_softmax` multiplied a signed content dot product by BC; `world_density_linear` used positive content similarity times BC followed by row normalization. Exact BC also failed to consistently beat a validation-selected projected-center RBF. A log-overlap added to ordinary content logits is a different interaction, but it must face that strong position baseline. Changing Fourier quadrature or renaming expected Gaussian rotation does not create new mathematical evidence. The old and retrained GS decompositions do not support an independent benefit from optimized primitive internal R/S covariance.

## Search and verification scope

I used [nature-academic-search](../../.codex/skills/nature-academic-search/SKILL.md)'s multi-source search and citation-verification workflow. Its specialized arXiv/CrossRef search MCPs are unavailable in this session, and the supplied direct-API preflight was blocked by the filesystem sandbox's network policy. Primary publisher/author/arXiv pages remained available through the web and GitHub tools. Records were deduplicated by DOI or title/first author. The bibliography below contains primary sources; this is a targeted design audit rather than an exhaustive novelty search.

## What is already known

Gaussian Fourier integration is established by [mip-NeRF](https://arxiv.org/html/2103.13415v3), including an alternative mean/covariance descriptor encoding in Appendix G. [WorldRoPE](https://arxiv.org/html/2610.16337v1) integrates rotary blocks over projected Gaussian mixtures. Neither the Gaussian characteristic function nor its composition with Q/K normalization is a new kernel.

[Probability Product Kernels](https://www.jmlr.org/papers/v5/jebara04a.html) includes density inner products and Bhattacharyya affinity. [Support Measure Machines, Table 1](https://papers.neurips.cc/paper_files/paper/2012/file/9bf31c7ff062936a96d3c8bd1f8f2ff3-Paper.pdf) gives the closed-form expected Gaussian-RBF kernel. [Gaussian attention bias](https://arxiv.org/html/2305.04722) also predates this work; its bias is a Gaussian-shaped table rather than the log density overlap proposed here. These are explicit precedents, not just analogous terminology.

Gaussian tokens are also established. [GPSToken](https://arxiv.org/html/2509.01109) represents position and shape with a 2D Gaussian. [GaussTR](https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_GaussTR_Foundation_Model-Aligned_Gaussian_Transformer_for_Self-Supervised_3D_Spatial_Understanding_CVPR_2025_paper.html) uses Gaussian queries, with center positional encoding. Particularly close, [OrientedFormer, §III-C/D](https://arxiv.org/html/2409.19648v1) combines Gaussian integrated positional features with an additive Wasserstein-derived relation bias. Thus “Gaussian parameters enter a Transformer positional mechanism” is already too broad to be an adequate novelty claim.

## Support extent is not location uncertainty

Let a token describe a spatial region. Its support covariance summarizes positions *within that region*. A large region can have a perfectly known center; a tiny region can have a badly estimated center. The posterior covariance of the estimated center is a different object. Gaussian support attenuation does not become an uncertainty calibration method merely because both are called covariance.

For the intended task, sample a source pixel uniformly within a query region and push it through the predicted geometry into the target image. Its distribution describes projected coverage. It does not represent an independently calibrated distribution over camera/depth errors. A global blur can regularize geometric mismatch, but its success would not validate per-token uncertainty estimates. If a future method introduces error covariance, it needs its own estimator/calibration evidence rather than relabeling the existing support covariance.

## Candidate 1: directed Gaussian coverage, primary

Operate in the **target/key image frame**. Let `(mq,Cq)` be the query region projected into that frame, and `(mk,Ck)` the key's known native footprint. Let `Ak` be key area in the same squared coordinate units. With a single pairwise blur `h²I`, define

\[
S_{qk}=C_q+C_k+h^2I,\quad
K_{qk}=A_k\,\mathcal N(m_q-m_k;0,S_{qk}),
\]
\[
\boxed{\ b_{qk}=\log A_k-\tfrac12(m_q-m_k)^TS_{qk}^{-1}(m_q-m_k)
 -\tfrac12\log|S_{qk}|-\log(2\pi)\ }.
\]

For normalized query density `pq` and key approximation `Ak pk` to an indicator, this is `∫pq(x) Ak pk(x) dx`, with heat smoothing if `h>0`. The approximation is not an exact rectangular-overlap probability: a Gaussian has noncompact tails and one covariance cannot represent a disconnected footprint. The directed factor is deliberate. Without it, the density inner product changes inversely with token area and is not region mass.

Use one ordinary score `s=content+b` in attention and, if the protocol specifies it, the final matching head. This preserves a geometric localization term independent of the signed content vector, unlike rotary multiplication. All point/fixed/full controls must use the same placement. If adding a final prior rescues performance, compare against the same final point prior before attributing it to attention or shape.

The underlying symmetric smoothed overlap is PSD: it is an inner product of heat-smoothed probability densities, equivalently an expected Gaussian RBF. The **directed** mass-weighted kernel need not be symmetric. Its **log** is not automatically a PSD kernel or an exact compact same-map Q/K inner product. Relative attention biases do not require PSD. Describe the implementation as relative positional encoding/bias, not as an unimplemented small token feature map.

### Measure consistency, and its precise limits

For positive masses, ordinary weighted attention has weights proportional to `Ak exp(content) N(...)`. Replacing one key by identical copies with identical content/value/support and masses summing to the original preserves their combined contribution. This is exact duplicate splitting, not a guarantee for merging distinct Gaussians or arbitrary image subdivisions.

The mass coefficient must be one. If a learned scalar multiplies the whole `log(Ak N)`, splitting a mass in half introduces the incorrect factor `2^(1−lambda)`. Permissible choices include fixed-strength overlap with a shared bandwidth, or `log Ak + lambda log N` with the mass kept outside the learned strength. [ToMe Eq.1](https://arxiv.org/html/2210.09461) already adds `log token_size` with coefficient one, and its [official attention code](https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py) implements this explicitly. [Continuum Attention](https://www.jmlr.org/papers/v26/24-0879.html) gives the broader integral/discretization viewpoint. The mass correction itself is not novel.

A common invertible affine coordinate change transforms `Ak` by `|det T|`, the density by its inverse, and covariance/means consistently, so `K` is unchanged if the blur transforms too. A fixed isotropic numerical blur preserves only the compatible rigid/unit-scaled version of that statement. A change of stride units must scale means, covariance, blur, and mass together. Do not claim affine invariance while keeping `h²I` fixed under a nonuniform affine transform.

### Which area to use

In the recommended target-frame construction, native key area is known from its sampling cell. It does not depend on predicted key depth, and is constant within a uniform grid. Its log therefore cancels there; a no-mass condition is an exact duplicate on that grid and informative only with unequal key areas or duplicate splitting.

There is a second cancellation in the actual primary setting: the native key covariance is common to every key in a given grid. Thus `S_qk=S_q` and `log|S_q|` are query-row constants too. After row softmax the full operator is **exactly the support-shaped Mahalanobis score** `−(mq−mk)^T S_q^-1(mq−mk)/2`. Neither log mass nor determinant normalization can explain a gain in these uniform-key experiments. Changing between 8px and 16px grids changes the known target support used in `S_q`, but does not make mass vary within a row. This is a useful compact special case, not evidence for measure correction under arbitrary token subdivision.

### Exact five-channel Q/K lift for the primary setting

Let every key have the same covariance `C*` (it need not be isotropic), `Aq=(Cq+C*+h²I)^−1`, and write all means about one common origin. Set

\[
Q_g(q)=[(A_qm_q)_x,(A_qm_q)_y,-A_{q,xx}/2,-A_{q,xy},-A_{q,yy}/2],
\qquad K_g(k)=[m_{kx},m_{ky},m_{kx}^2,m_{kx}m_{ky},m_{ky}^2].
\]

Their dot product is `mq^T Aq mk − mk^T Aq mk/2`, which differs from the full Mahalanobis score only by `mq^T Aq mq/2`, a query-row constant. Concatenating these five channels to *separately normalized* content Q/K therefore gives the exact same attention probabilities and V aggregation as the additive primary kernel. Match the content temperature and any attention implementation's `1/sqrt(d)` scale explicitly; normalizing the concatenated vector would change the operator. A varying known mass can be represented by one further query-constant/key-log-mass channel if key covariance remains common.

This is a genuine finite per-token Q/K implementation for the stated special case, not a Fourier approximation. The query PE remains conditioned on the target camera frame and common target covariance. It is not an intrinsic world-coordinate embedding independent of the target bank, and five channels do **not** implement arbitrary unequal key covariances, mixtures or general pairwise Gaussian overlap. Per-token geometry preparation is linear in token count; ordinary dense attention still has quadratic computation.

[Cordonnier, Loukas and Jaggi, ICLR 2020, Appendix E](https://arxiv.org/html/1911.03584v2) already writes anisotropic Gaussian attention using five linear/quadratic positional monomials. The polynomial Gaussian-attention algebra is established. The proposed specialization derives each query's precision from a projected 3D region support, rather than learning one covariance/offset per head. Its scientific question is whether that input-derived support improves correspondence, not whether a five-dimensional quadratic Gaussian encoding can exist.

For floating-point stability, use the same fixed image-center origin for both sides. The algebra is exact over the reals, but quadratic products and subtracting large row constants can lose float32 precision. Audit probability/gradient equality, including clipped off-image queries and small bandwidth; do not infer full-model equality merely from an unconditioned symbolic expansion. No learned projection or normalization may silently mix the added geometry channels into content.

`12 sqrt(det Ck)` recovers area for the affine image of a continuous uniform 2D rectangular footprint. Adding within-pixel covariance `I/12` to discrete pixel-center moments recovers the continuous-cell second moment on a planar constant-depth patch. Without that term, the discrete 8×8 variance is `63/12` pixels², rather than `64/12`.

For **projected** keys, determinant area is unreliable when depth/parallax creates a multimodal cloud: empty gaps increase covariance and can inflate mass. A projected pixel-cell/quad Jacobian area estimate is then a more direct diagnostic, but folds/occlusion and overlapping footprints still need care. The target-frame native-key construction avoids this particular mass-estimation problem. No Jacobian-derived predicted area should be substituted for independently projected sensor-pixel labels.

### Falsification and strong controls

Train matched `point+mass`, size-aware `fixed+mass`, same-trace `isotropic+mass`, and `full+mass`, sharing means, mass, bandwidth/strength capacity, bias placement, visual features, architecture, optimizer and selection. Fixed covariance should retain the declared native cell size for each side, not silently use a single size for every token. A BC control is a known shape-similarity diagnostic. Add no-mass only where mass varies. Keep the audited normalized rotary as a historical operator control if budget permits.

Before training, check exact mass splitting, consistent coordinate-unit rescaling, equal-support symmetry, SPD gradients, and the constant-depth planar area/moment relation. Test with independent sensor-pixel area counts, multiple query/key cell sizes, and new held-out physical places. Separate attention-only, final-only and both placements using the same strong point prior when claiming an attention mechanism. A gain against an unanchored final content head alone is insufficient.

Expected failures: wrong projected means, depth edges or nonlocal visibility where one Gaussian fills empty space; a near-singular footprint without a physical floor; spatially broad false-positive coverage; area imbalance if mass is omitted or tempered; and overly strong geometric priors suppressing valid content corrections. The method is not an occlusion model, a pose estimator, or a guarantee under arbitrary geometry error.

## Why log BC is not the same objective

For `Cbar=(Cq+Ck)/2`,

\[
\log BC=-\tfrac18\Delta^TC_{\rm bar}^{-1}\Delta
 -\tfrac12\log|C_{\rm bar}|+\tfrac14\log|C_q|+\tfrac14\log|C_k|.
\]

BC is symmetric, bounded by one, unit self-similarity, and jointly affine invariant for SPD covariances. It measures similarity of normalized region densities. A narrow region contained inside a very broad one can have low BC, although coverage of the narrow query by the broad key approaches one. In contrast, a key-area-weighted density overlap is directed and can represent that containment. Neither quantity is automatically the exact transfer label for arbitrary non-Gaussian/occluded regions.

`softmax(content + log BC)` weights the content likelihood by BC. This differs from the old `softmax(content * BC)`, and also from `softmax(log positive_content_dot + log BC)`. A new result with one interaction cannot be credited to another. BC is useful as a strong, known diagnostic rather than a new discovery or a promised coverage metric.

## Candidate 2: finite sliced-Gaussian distance PE, reserve domain

For fixed directions `nl`, define a compact independent-token descriptor

\[
e(G)=L^{-1/2}[n_l^T\mu,\sqrt{n_l^T\Sigma n_l}]_{l=1}^L.
\]

For each 1D Gaussian projection, squared Wasserstein distance is squared mean difference plus squared standard-deviation difference. Therefore `||e(Gi)−e(Gj)||²` is the finite-direction sliced Gaussian distance, and `exp(−beta ||ei−ej||²)` is PSD by its Euclidean construction. This special-case derivation follows the sliced-Wasserstein kernel approach; it is not a claim that the cited paper introduced this token PE. A log distance bias has an exact finite asymmetric Q/K lift: append `sqrt(2 beta)e` to Q/K and a query constant/key `−beta||e||²` channel; the omitted query norm is a softmax row constant.

This is distinct from the failed periodic covariance-descriptor RoPE: it uses an explicit distributional distance without phase wrapping. Still, it is a known metric-derived construction. Finite fixed directions can alias different covariances and do not give exact arbitrary rotation invariance. A covariance floor controls square-root derivatives near zero. Unrestricted multivariate Bures/Wasserstein RBFs must not be assumed PSD merely because W2 is a metric; the Euclidean finite-sliced construction avoids that assertion.

Its appropriate domain is matching **estimates of the same physical Gaussian primitive/support** across coordinate-aligned observations, where differing shape should be penalized. It is not the primary candidate for coarse/fine region coverage, where unequal extents can be the correct correspondence. Controls must include center-only Euclidean distance, matched trace/area descriptors and the old world-moment RoPE. Do not run it on the area-transfer task merely hoping that any shape distance helps.

For an explicit counterexample to an unrestricted Bures-RBF PSD claim, take zero-mean rank-one 2D covariances `Cθ=uθuθ^T` at `θ=0,π/4,π/2,3π/4`. The Bures squared distance is `2−2|cos(θ−φ)|`. The Gram matrix of `exp(−β d²)` has an eigenvalue `1+exp(−2β)−2exp[−β(2−sqrt(2))]`, approximately `−0.0675` at `β=.1`. By continuity it stays negative after adding a sufficiently small common `εI`, giving nondegenerate SPD examples. This is our direct special-case calculation, independently checked by the protocol reviewer; it is not a verbatim example from the cited papers.

## Candidate 3: additive heat-mean positional features, only if compactness is essential

Use fixed frequencies from a declared Gaussian spectral measure and per-token features

\[
\psi_G= M^{-1/2}[e^{-\omega_l^TC\omega_l/2}\cos(\omega_l^T\mu),
e^{-\omega_l^TC\omega_l/2}\sin(\omega_l^T\mu)]_l.
\]

Their inner product approximates the known expected heat/RBF kernel. Add this geometric inner product to the content score, equivalently concatenate independent geometric channels. This differs from multiplying each content channel by the expectation or tensor-lifting content by square-root-density features. It yields a real finite per-token PE, but adds channels and inherits approximation error. A finite trigonometric approximation may be negative even when the exact overlap is positive, so taking its logarithm is not automatically valid.

This is a fallback for token-independent positional features on spatial distributions, not a new Gaussian integration route. First test the **exact** additive heat-kernel score against strong position controls; if it has no benefit, do not invest in quadrature. Its bounded score, finite-frequency approximation, frequency/bandwidth selection, and additional memory can all make it inferior to the exact relative bias. Do not repeat the earlier large density-feature experiment without establishing this distinct additive interaction first.

## Generalization claim to aim for

The primary claim is limited to **posed multi-view correspondence of visible, locally single-surface regions across declared pixel/token scales, with geometry estimated from input views**. Success should reproduce on new physical scenes and within each scale condition, under common sensor validity rules. It does not establish arbitrary modalities, unposed scenes, global SO(3)-invariant attention, calibrated uncertainty, or usefulness of optimized GS internal R/S. Existing Round-5 rooms are development evidence once inspected.

Freeze success and stopping criteria before new confirmation. Report the independent area-label oracle, geometric-only point baseline, matched bias-placement controls, label-entropy/KL context when key counts differ, place/seed uncertainty and failures at depth boundaries. If a full support prior does not improve over the size-aware point/fixed controls, retain that negative result rather than adding mixture, uncertainty and learned gates simultaneously.

## Implementation audit before development interpretation

The isolated implementation is `models/gaussian_measure_pe.py`. Eleven independent tests in `tests/test_gaussian_measure_pe.py` cover the float64 linear-algebra reference, float32 agreement, exact duplicate-mass splitting even when density strength differs from one, matched covariance controls/mass information, scalar/batch/per-token fixed variances, coordinate transformations, uniform-key cancellation, positive-SPD gradients and the point limit. Four further independent tests in `tests/test_round6_quadratic_identity.py` verify five-channel probability/V and gradient identities, the unequal-key-covariance limitation, and the SPD Bures-RBF counterexample. None uses trained model outcomes as a target.

The read-only model audit found the declared placement in both cross-attention directions and the final head, shared bandwidth, common parameter counts, actual V aggregation and a target-frame Gaussian definition. `round6_region_data.make_pair_inputs` accepts only frozen RGB prediction caches and camera metadata; the exporter computes all forward arrays before loading sensor depth. The trainer passes only the ten declared forward fields. Native/target means are in the common 8px unit; changing token stride changes known native support, rather than changing the unit. Current test suites for kernel/model/data passed together (32 tests), followed by the four independent quadratic tests.

A CPU float32-versus-float64 smoke audit used **only the first validation pair** in each of four old development grid conditions, at bandwidths `.05,.25,1,4,32`. Maximum probability differences ranged from `1.97e−7` to `3.87e−7`. This is a bounded numerical check, not an empirical performance result or an audit of every future input. The runtime kernel applies no undisclosed covariance clipping; the data producer separately declares fixed mean/depth/covariance fallbacks. Positive native key covariance and the declared bandwidth keep the actual primary pair covariance away from a singularity.

Before formal confirmation, enforce actual train/validation/test place and camera sets, NPZ row counts against manifest pairs, all cache/source/checkpoint digests, validation replay and a persisted selection lock. The current development trainer hashes its runtime model/kernel/helper files and input archives but does not itself enforce all cross-split identity-set assertions. These checks belong in the confirmation audit; no running trainer or data source was edited for this review.

The later minimal candidate is `models/gaussian_quadratic_pe.py`: frozen 128D visual features concatenated with the exact five geometry channels, with **no learned network parameters**. Bandwidth and content temperature are two buffers calibrated on development data, not fitted per test scene. The matched deterministic controls receive the same calibration budget; repeated seed copies would not be independent runs. Its all-pair implementation audit is in `results/round6_quadratic_audit`: all 64 camera pairs from each of two already inspected validation places, four query/key grids, four controls at `h=.5,T=5` plus selected point at `h=1,T=10`. This is 512 scene/grid pair instances but only 128 different physical camera pairs.

The actual class and actual pair-kernel code were compared with an independent float64 solve/slogdet reference. Across all declared settings, lift64/reference64 maximum probability difference was `1.58e−12`. The selected full float32 scores versus reference64 gave maximum probability difference `3.63e−5` and visible-mass-weighted mean absolute CE difference `7.21e−7`, with no valid-row top-key disagreements. Float64 V/geometry/bandwidth gradient checks and common-unit changes also agreed numerically. The exhaustive comparison used float64 softmax/log-softmax of each implementation's produced scores; a separately recorded first/last-pair subset checks native float32 softmax/log-softmax. See the audit README for numerical scope, source snapshots and limitations.

This compact candidate preserves the mathematical prior without a new learned encoder or multiple auxiliary modules. It is still a specialization of known Gaussian/quadratic attention, conditioned on target-bank coordinates/common native support. In particular, the current dispatcher verifies target covariance after the control transformation; accepting nonuniform native key banks in a future API would also require handling original varying log mass. Formal uniform-grid inputs meet the stronger condition. The sealed Round-5 negative result remains unchanged, and development improvements in an area-coverage target do not establish improved center localization.

## Verified primary bibliography

| Work | Primary source | Relevance/status |
|---|---|---|
| Jebara, Kondor, Howard. Probability Product Kernels. JMLR 2004 | https://www.jmlr.org/papers/v5/jebara04a.html | Verified title/authors/year; density inner products and BC |
| Smola et al. A Hilbert Space Embedding for Distributions. ALT 2007 | https://is.mpg.de/ei/publications/4645 | Verified DOI `10.1007/978-3-540-75225-7_5`; distribution mean embeddings |
| Muandet et al. Learning from Distributions via Support Measure Machines. NIPS 2012 | https://papers.neurips.cc/paper_files/paper/2012/hash/9bf31c7ff062936a96d3c8bd1f8f2ff3-Abstract.html | Verified; Table1 expected Gaussian-RBF overlap |
| Barron et al. mip-NeRF. ICCV 2021 | https://arxiv.org/html/2103.13415v3 | Verified; IPE and descriptor alternative, not new here |
| Yu et al. Mip-Splatting. CVPR 2024 | https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf | Verified; scale-dependent Gaussian filtering precedent |
| Barron et al. Zip-NeRF. ICCV 2023 | https://jonbarron.info/zipnerf/ | Verified author page; multiscale anti-aliasing, not this attention operator |
| Kim et al. WorldRoPE. arXiv 2026 | https://arxiv.org/html/2610.16337v1 | Verified; probabilistic rotary integration; full reproduction not claimed |
| Bolya et al. Token Merging. ICLR 2023 | https://arxiv.org/html/2210.09461 | Verified Eq1 and official implementation; log token mass |
| Calvello et al. Continuum Attention for Neural Operators. JMLR 2025 | https://www.jmlr.org/papers/v26/24-0879.html | Verified; integral attention/discretization |
| Hermosilla et al. Monte Carlo Convolution. TOG 2018 | https://viscom.uni-ulm.de/publications/monte-carlo-convolution-for-learning-on-non-uniformly-sampled-point-clouds/ | Verified DOI `10.1145/3272127.3275110`; sampling measure matters |
| Kim et al. Understanding Gaussian Attention Bias. arXiv 2023 | https://arxiv.org/html/2305.04722 | Verified; Gaussian-shaped additive bias precedent |
| Zhang et al. GPSToken. arXiv 2025 | https://arxiv.org/html/2509.01109 | Verified; mean/shape Gaussian tokens |
| Jiang et al. GaussTR. CVPR 2025 | https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_GaussTR_Foundation_Model-Aligned_Gaussian_Transformer_for_Self-Supervised_3D_Spatial_Understanding_CVPR_2025_paper.html | Verified; Gaussian queries, center PE |
| Zhao et al. OrientedFormer. arXiv 2024 | https://arxiv.org/html/2409.19648v1 | Verified §III-C/D; Gaussian PE plus Wasserstein attention bias |
| Cordonnier, Loukas, Jaggi. On the Relationship between Self-Attention and Convolutional Layers. ICLR 2020 | https://arxiv.org/html/1911.03584v2 | Verified Appendix E; five linear/quadratic channels for anisotropic Gaussian attention |
| Feragen et al. Geodesic Exponential Kernels. CVPR 2015 | https://openaccess.thecvf.com/content_cvpr_2015/html/Feragen_Geodesic_Exponential_Kernels_2015_CVPR_paper.html | Verified by independent protocol reviewer; metric alone does not ensure every RBF is PSD |
| Kolouri et al. Sliced Wasserstein Kernels. CVPR 2016 | https://openaccess.thecvf.com/content_cvpr_2016/html/Kolouri_Sliced_Wasserstein_Kernels_CVPR_2016_paper.html | Publisher metadata verified; PDF fetch403; finite-Gaussian descriptor derived above independently |
| Meunier et al. Distribution Regression with Sliced Wasserstein Kernels. ICML 2022 | https://proceedings.mlr.press/v162/meunier22b.html | Independent primary-source check; Hilbertian sliced-distance construction |

These references establish ingredients and related designs, not empirical support for an unrun Round-6 candidate. No GPU experiment was launched for this literature audit.
