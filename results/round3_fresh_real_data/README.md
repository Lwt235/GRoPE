# Previously unexported-camera data: confirmation preparation only

Pair selection was written to selection_protocol.json before new image pixels or pair labels were opened. Every prior scene-data manifest's frames/q_id/k_id union is excluded. Camera here means posed-image ID, not an intrinsic calibration ID. Remaining images are sorted by filename and grouped in threes: the first queries the second and third; a final pair is retained, a singleton is recorded unused. No shared-track threshold, prediction or matching score selects pairs. All selected pairs remain even if their labels/geometry have no valid rows.

The original geometry exporter is wrapped with a frozen make_pairs callback and restored in finally; its source file is unchanged. Source snapshots/hashes cover the actual controller, exporter and visual helper. Raw RGB108, independent even/odd-GS moment fits and the alternate swap, native query-region covariance, projected q/k support, cameras and sparse SfM-track targets retain the original schema. All pairs have split=test and subset=previously_unexported_cameras.

The sibling round3_fresh_frozen_visual_data is an explicitly authorized confirmation-preparation phase using the same official local ResNet18 weights, layer2 regional average pooling, exact saved development 128->108 projection, ImageNet normalization and full 192x144 BOX-resize. It calls the frozen extraction primitives, not a bypass of the old development-only exporter. Nonfeature arrays are verified identical. No layer, projection or preprocessing was fitted or selected here.

No matcher has been trained or evaluated on this export. The main thread must freeze candidate/settings before evaluation. Tea/kitchen have very few remaining cameras; the data imbalance is retained. New cameras are from already known scenes and can neighbor previously exported frames, so this is conditional matching on unused matcher images, not independent new-scene evidence. Source photographs may already have trained the available GS/SfM reconstruction. Sparse-track targets are not dense coverage or occlusion truth.

verification.json proves the source hashes, fresh/prior ID disjointness, output schema and exact raw/frozen nonfeature equality. It contains no model performance metrics.

Reproduce with `python -m experiments.fresh_real_pairs`.
