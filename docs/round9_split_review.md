# Round 9: source reconstruction and split audit

This began as a read-only audit of the original reconstruction metadata and
trainer source, and was followed by the new context-only exporter described
below. No historic test pair archive, test result, or target photograph was
opened. No training was launched by this audit. The final frozen Round 9
protocol is authoritative.

## Existing reconstructions cannot establish context-only geometry

The four `DEFAULT_SOURCES` in `experiments/real_support_data.py` contain explicit
appearance-training splits in `training_scene.json`:

| Scene | Training views | Appearance-validation views | Sparse points | Points with a validation observation |
|---|---:|---:|---:|---:|
| Teatime | 169 | 8 | 25,503 | 6,646 |
| Waldo kitchen | 178 | 9 | 17,475 | 3,503 |
| Figurines | 285 | 14 | 50,517 | 10,726 |
| Temple | 297 | 15 | 19,648 | 5,635 |

These figures are counts from `source_model.json` and `training_scene.json`, not
image-quality results. `minicolmap.rendering.data.load_training_scene` copies
every sparse point's XYZ and RGB into Gaussian initialization before photometric
training. The source tracks therefore explicitly include held-out appearances.
Even deleting those target observations without recomputing coordinates and
colors would retain their information in the initialized geometry.

The existing trainer samples gradients only from `scene.train_ids`, uses a fixed
iteration count, and saves the final checkpoint without a validation-best
selection. However, it loads **all** posed images, renders the first validation
image during training, and evaluates every validation image at the end. A claim
that its existing checkpoints or unmodified all-frame loader never accessed
targets would be false.

## Minimal defensible route

1. Use the supplied intrinsics and world-to-camera poses as fixed calibration.
   The dataset's `dataset_posed_all.json` contains these poses without a sparse
   cloud. These calibrations originate from full-image SfM; this must remain an
   explicit limitation. The setting is calibrated view synthesis, not full
   reconstruction from previously unseen images with newly estimated cameras.
2. Select context and target image identities before decoding RGB. Only context
   RGB enters feature extraction, matching, triangulation, initial colors,
   Gaussian fitting, visibility construction, color reading, and parameter
   selection. Do not reuse old checkpoints, old point XYZ, or old point RGB.
3. For the strongest pixel-access guarantee, extract and match features again on
   context images and triangulate with fixed context cameras. Reusing the old
   track topology, even after recomputing coordinates, retains a weaker
   all-image provenance because target matches may have connected that track.
4. Build a context-only `Reconstruction` and context-only dataset if reusing
   `minicolmap.rendering.pipeline.train`. Its internal validation must then use
   context images. An independent training loop can use every context image and
   report training-only progress instead. No final target score may select the
   geometry checkpoint or color-reading bandwidth.
5. Freeze reconstruction and source-color predictions first. Decode target RGB
   only in the evaluation stage. Hash context membership, matching and training
   inputs, trainer sources, and final checkpoints. Report any source-visibility
   fallback consistently across every color-reading ablation.

## Implemented old-development-only split

The old `make_pairs` implementation sorts by image **name**, then divides that
list into three contiguous blocks. Keeping only its first two blocks avoids
opening the sealed third image block. For Teatime this is 118 registered views,
`frame_00001.jpg` through `frame_00118.jpg`; for Waldo kitchen it is 125 views,
`frame_00001.jpg` through `frame_00125.jpg`. Numeric IDs are not chronological.

The implemented protocol selects 32 uniformly spaced indices from each
old-development block using integer `linspace`. Selected positions 3, 7, ..., 31
are targets; the other 24 views are context. The eight targets alternate between
four calibration and four gate views in name order.

| Scene | Target IDs in name order | Target filenames |
|---|---|---|
| Teatime | 19, 29, 10, 69, 60, 105, 147, 148 | frame_00012, 00027, 00042, 00057, 00072, 00087, 00102, 00118 |
| Waldo kitchen | 23, 29, 44, 73, 75, 88, 106, 131 | frame_00013, 00029, 00045, 00061, 00077, 00093, 00109, 00125 |

`experiments/round9_context_data.py` reads only the camera-only posed dataset and
the 24 selected context RGB files per scene. It rejects inherited geometry keys
in the dataset and checks context membership before hashing or opening any RGB
file. Context-only SIFT features are matched to the next 1, 2 and 3 context
neighbors (66 camera pairs), then triangulated with fixed supplied cameras.
Positive depth, two-view reprojection error at most 2 SIFT-image pixels and
ray angle at least one degree are required. The first point per normalized
1e-4-width voxel is retained, with a deterministic 12,000-point cap if needed.

The actual export produced 847 accepted / 799 unique Teatime points and 1,838
accepted / 1,728 unique Waldo points, in 8.78 seconds total on CPU. Each output
records its 24 original image SHA256s, the source calibration SHA256, source-code
SHA256, all pair filter counts, normalization, and separate target camera-only
JSON. The eight exporter tests passed, covering split membership, forbidden
target image accesses, normalization, camera resize, triangulation, deduplication
and bilinear color sampling. The point counts are initialization diagnostics,
not reconstruction quality evidence; subsequent context reconstruction quality
must be reported before interpreting a negative color-reading result.

These are reused development scenes, not an independent scene-generalization
test. Any gate passed here can justify a subsequent frozen confirmation, but
cannot by itself establish generality.

## Sources inspected

- `experiments/real_support_data.py`: `DEFAULT_SOURCES`, `make_pairs`, `load_scene`.
- `/home/liwt/miniColmap/src/minicolmap/rendering/data.py`: `split_views`,
  `load_training_scene`.
- `/home/liwt/miniColmap/src/minicolmap/rendering/pipeline.py`: `train`.
- `/home/liwt/miniColmap/src/minicolmap/io.py` and `models.py`: reconstruction
  serialization and track invariants.
- Original source metadata at the four paths in `DEFAULT_SOURCES`, and Teatime /
  Waldo input-dataset metadata. No archived evaluation image/metric was used to
  choose the suggested split.
