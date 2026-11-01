# Round 7 fresh-scan acquisition and RGB-only identity review

2026-10-30. This review constructs no sensor-depth supervision and computes no
test model scores. Metadata, raw file bytes, fixed RGB representatives and the
unchanged RGB-only frozen cache are inspected separately.

The selection is `data/round7_sun3d/selection.json`, SHA256
`be816055c073529fc450e3e210698ca26d5178fb2d1d70103c76b5a89d43598f`.
It contains 48 linearly spaced query ordinals and offsets15/60 for each of five
new top-level scan/place groups: 96 pairs each, 480 total. Every place used in
Round5 or either Round6 selection, including rejected Brown bm1, is excluded
(16 prior groups). No candidate replacement was necessary.

All 1,362 required RGB/depth files (205,258,518 bytes) match the retained
hash/size manifest and complete frame identities. All 25,931 full-list camera
poses pass the existing affine/SO(3) reader; maximum rotation orthogonality
error is about 1.08e-7. W59's maximum nearest-depth time gap is29.725ms; other
selected scans are at most16.63ms. Fixed pairs are retained without time-gap,
GT-validity, overlap, geometric-quality or method-performance filtering.
See `acquisition_audit.json`, `camera_metadata_audit.json` and the native
metadata/download manifests in the data directory.

## Fixed RGB QA

The first, middle and last selected RGB frame of every candidate were viewed.
Exact original ordinals, paths and hashes are in `preview_manifest.json`, with
`test_rgb_firstmiddlelast.png`. These were compared with the previously viewed
Round5/6 representative layouts; sensor values and model outputs were not used.

| New top-level place | Visible representative layout | Identity assessment |
|---|---|---|
| mit_46_6conf | Large meeting room, patterned gray carpet, vertical blinds, side chairs and round standing table | No definite old-room match found |
| mit_32_g725 | Red-edged pale conference tables, black chairs, wall boards and pendant fixtures | Different layout from old MIT d507's yellow-chair room |
| mit_w59_conference | Large dark conference table, blue chairs, gray walls and separate side room | No definite old-room match found |
| harvard_robotics_lab | Fixed green robot-soccer field, lab benches and whiteboards | Distinct visible use/layout from old Harvard conference rooms; yellow wall panels reflect shared university style |
| home_md | Wooden display cabinets and dining table beside patterned curtains | No definite match to old home_at/home_bksh or MIT dorm previews found; household scan spans multiple rooms |

The [official SUN3D catalog](https://sun3d.cs.princeton.edu/) and
[native directory index](https://sun3d.cs.princeton.edu/data/) list these groups
separately. Distinct names and fifteen representative images are corroborating
evidence, not complete floorplans or proof of independent buildings. The
honest evaluation unit is five named scan/place groups. Three MIT scans share
institution/collection style; Harvard test also shares institution with
training/validation. Do not claim institution-heldout generalization or exactly
five single rooms. Group and leave-one-scan results should remain visible.

## Frozen RGB-only cache

The unchanged sealed `experiments.round6_cache` CLI ran with physical GPU3
(`CUDA_VISIBLE_DEVICES=3`) on this selection and wrote
`results/round7_inputs/cache_test.json` plus681 cached frames. The producer
record matches the Round6 test producer exactly: same predictor/backbone weight
hashes, RGB preprocessing source, wrapper source and geometry version. Every
actual cache hash, sidecar, RGB byte hash, camera hash and selected original
index was checked. See `rgb_cache_audit.json` for runtime and checks.

This cache contains RGB-derived observations and shared camera metadata only.
Raw sensor depth files have been downloaded and byte-hashed, but this task has
not decoded their values or generated new test labels. Such export remains
subject to the forthcoming protocol/source/checkpoint confirmation lock.
