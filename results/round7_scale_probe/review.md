# Independent Round7 scale diagnosis

All five new conditions were evaluated with the original Round6 lock configs. This is an extension on reused data; no new independent test was used.

The executable output audit found no identity, metric arithmetic, visible-area conservation, shared supervision mask, or point-quantization ceiling mismatch across 15,200 role rows. Independent macro error was 3.33e-16; the 20,000 paired place bootstrap intervals matched exactly. The scope is arithmetic and invariant validation, not a second full prediction replay.

Full versus fixed improves area KL for every new condition in each of the five already-used test scan/place groups. Overall KL is 0.198624 versus 0.223403 (−11.092%, descriptive95% relative CI [−17.227%,−5.978%]); JS is 0.049808 versus 0.051877 (−3.988%). Full versus trace-isotropic improves KL by 1.522% overall; this contrast is exploratory.

|grid|full−fixed PCK8, pp (95% descriptive CI)|full−point_pck PCK8, pp (95% descriptive CI)|nearest-grid PCK8 ceiling|
|---|---:|---:|---:|
|q8_k24|+0.0153 [+0.0053, +0.0247]|+0.1093 [+0.0333, +0.1986]|36.181%|
|q24_k8|-0.4602 [-0.8781, -0.0706]|-0.6326 [-1.5739, +0.2723]|100.000%|
|q16_k24|+0.0099 [+0.0000, +0.0230]|+0.1222 [+0.0382, +0.2063]|36.508%|
|q24_k16|-0.0832 [-0.2586, +0.0917]|-0.3503 [-1.1421, +0.3090]|80.135%|
|q24_k24|+0.0318 [-0.0165, +0.0771]|+0.1175 [-0.0402, +0.2751]|41.661%|

The 24→8 PCK8 deficit against fixed occurs in all five old groups; its descriptive CI is negative. Against point_pck, the mean is also lower but the CI spans zero (two Brown groups are lower and the other three groups are higher). This rules out a universal point-preservation claim from these diagnostics. The source support becomes larger while the point target remains its independently sampled center; area KL and point accuracy are distinct endpoints. No condition or bandwidth/temperature was selected from this diagnosis.

For 24px key cells, PCK8 has an intrinsic grid quantization ceiling near 36–42%; it is not comparable with 8px-key PCK as an absolute matching quality measure. Target-stride PCK and its oracle ceiling are available in summary/point_intervals alongside fixed 8/4px thresholds. The source 24 pooling also changes the visual descriptor, so scale is not isolated from feature aggregation.

The first technical startup failed before any completed metric row: the streaming adapter supplied int32 point indices while the archived metric expected int64. The adapter was changed to the exact original loader long conversion, and a direct archived-metric regression was added. The original failure protocol/source snapshot is retained in results/round7_scale_probe_attempt1_dtype_failure. No model, h/T, sample, label, or decision rule changed.

There are five reused named scan/place groups, no artificial seed replication, and no claim of independent physical rooms. The two validation groups are shown separately. These observations concern RGB-predicted region covariance plus a known pixel footprint; they do not identify optimized 3DGS rotation/scale as their unique cause.

Run time 106.756s CPU1; 14 tests passed. Protocol SHA256: 0c5d3755914d7bdee84ef8f168b467c77df001c4d5ad1f693258bba18283a33c. Frozen Round5/6 files were not edited.
