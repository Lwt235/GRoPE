# R41 fixed-checkpoint shape-component diagnosis

2026-11-01, after the complete negative R41 result. Old development only.
Decoupled failed: gate MSE .024874800 vs learned point .022275296,0/5 wins;
same-checkpoint shape-off improves to .024185743. No numeric warning.

One bounded diagnostic, no training or parameter/temperature search. For all
ten decoupled checkpoints keep Q/K, beta, geometry, features, source eligibility,
fusion, original-GS fallback and target populations unchanged. Decompose the
existing penalty exactly into trace-normalized shape distance (first6 svec
coordinates) and squared log-trace difference. Evaluate scale-only and
orientation/anisotropy-only, both with the original beta/7. Neither is a newly
validated candidate; R41's primary failure remains unchanged.

Export all primitive colors and eight target renders for both interventions.
Use the identical fixed context-selection rows to compute full,shape-off,
scale-only and shape-only RGB MSE. Aggregate over every original readable
primitive/source observation the probability mass assigned to the source's
largest10% covariance-trace keys and expected squared pixel distance from the
projected primitive center. Input-derived90th-percentile groups diagnose where
attention moves; they never replace whole-image metrics or exclude queries.
Large trace alone is not a ground-truth edge or uncertainty label.

Replay complete first-source sample probabilities through separate NumPy
formulas for both components. Full-image MSE uses all target pixels with the
old4/4 split reported separately; originals and shape-off come from sealed R41
predictions. Fixed models/predictions are sealed before this diagnostic's image
scoring. No untouched data is read. Outcome may guide a separately frozen next
design, but cannot select a winning component and relabel it confirmation.
