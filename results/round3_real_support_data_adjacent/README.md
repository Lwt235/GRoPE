# Real photographic regional support data

This dataset exports fixed 8 x 6 image regions from LERF-OVS teatime and waldo_kitchen photographs. Each token contains a 6 x 6 RGB patch and the first two moments of independently visible reconstructed Gaussians. Query and key use mutually exclusive even/odd primitive subsets; `_alt` arrays swap these subsets. No primitive ID or SfM track ID is exported as a token input.

`soft_labels` count the common SfM tracks observed in each pair of regions and normalize each query row. These are sparse salient-track transfer proxies, not dense surface-area overlap and not occlusion labels. Rows with fewer than four common tracks have `label_valid=False`; evaluators must also handle `q_valid`/`k_valid`. `track_counts`, all count/fraction arrays, and label arrays are evaluator/training targets only and must never enter the matcher.

The CPU geometry exporter projects high-opacity (>=0.3) GS centers into a 192 x 144 z-buffer, retains centers within 1.5% of the first depth, and moment-fits each region, including internal primitive covariance. It does not implement alpha compositing. The Gaussian reconstruction can include all scene photos, so split cameras isolate matcher fitting, not reconstruction.

Every scene manifest records original source paths, SHA256 hashes of geometry/SfM/cameras/images, fit statistics, pair splits, and limitations. `q_pixel_mean/cov` are fixed image-region support; `q_projected_mean/cov` and `k_projected_mean/cov` project fitted world support into the query camera. Coordinates use unnormalized source-model world units. `source_snapshot.py` and `summary.json` identify exporter code.

Source outputs:
- teatime: `/home/liwt/miniColmap/outputs/ui_3dgs_20261024_112609_a1d94a`
- waldo_kitchen: `/home/liwt/miniColmap/outputs/ui_3dgs_20261019_230921_797f2a`

Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.real_support_data`.
