# Real photographic regional support data

This dataset exports fixed 8 x 6 image regions from the figurines and temple photographic scenes. Each token contains a 6 x 6 RGB patch and the first two moments of independently visible reconstructed Gaussians. Query and key use mutually exclusive even/odd primitive subsets; `_alt` arrays swap these subsets. No primitive ID or SfM track ID is exported as a token input.

`soft_labels` count the common SfM tracks observed in each pair of regions and normalize each query row. These are sparse salient-track transfer proxies, not dense surface-area overlap and not occlusion labels. Rows with fewer than four common tracks have `label_valid=False`; evaluators must also handle `q_valid`/`k_valid`. `track_counts`, all count/fraction arrays, and label arrays are evaluator/training targets only and must never enter the matcher.

The CPU geometry exporter projects high-opacity (>=0.3) GS centers into a 192 x 144 z-buffer, retains centers within 1.5% of the first depth, and moment-fits each region, including internal primitive covariance. It does not implement alpha compositing. The Gaussian reconstruction can include all scene photos, so split cameras isolate matcher fitting, not reconstruction.

Every scene manifest records original source paths, SHA256 hashes of geometry/SfM/cameras/images, fit statistics, pair splits, and limitations. `q_pixel_mean/cov` are fixed image-region support; `q_projected_mean/cov` and `k_projected_mean/cov` project fitted world support into the query camera. Coordinates use unnormalized source-model world units. `source_snapshot.py` and `summary.json` identify exporter code.

Source outputs:
- figurines: `/home/liwt/miniColmap/outputs/ui_3dgs_20261022_171409_43d932`
- temple: `/home/liwt/miniColmap/outputs/linux_reproduction_temple312_combined_s0_545b11caae`

Reproduce with `experiments.real_support_data.export_scene`, using each manifest's `source_output`, `seed`, and `groups_per_split=18`, writing to its scene subdirectory here. The module CLI defaults export teatime/kitchen instead; they do not reproduce this external-scene export. All 108 pairs per external scene are retained.
