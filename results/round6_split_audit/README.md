# Round 6 v2 acquisition and fixed RGB place review

2026-10-30. This audit reads native metadata, file bytes and fixed RGB previews.
It does not run a label exporter, geometry predictor, matcher or test metric.
The v2 confirmation selection contains five new test place/scan groups, 96
pairs each, and Brown bm4 validation, 64 pairs. All 1,560 required RGB/depth
files (236,603,066 bytes) match the retained download hashes and exact selected
identities. See `acquisition_v2_audit.json` and the combined
`data/round6_sun3d/download_manifest_v2.json`.

V1 selection is retained unchanged: SHA256
`c7297b2cd040c271c49a90cf85e1d1e7893c4c57b2f77119a16580b72bad9631`.
V2 SHA256 is
`4925a965e311908463b9ba0ebc5f932494cb3872b160777eda434613b8d17d1d`.
The original acquisition script is unchanged; replacement metadata/frames were
acquired by the separate archived `round6_replace_scene.py`.

## Model-independent replacement

The first/middle/last selected RGB frames of old nine place groups and Brown
bm1/bm4 were reviewed, followed by twelve uniformly spaced selected RGB frames
for each Brown bm scan. Both bm1 and bm4 include offices and corridors with
similar interior/carpet style. These previews do not establish that the rooms
are identical, but cannot exclude shared physical space. Before test labels or
model outcomes, bm1 was therefore removed from confirmation and retained as
auxiliary data. It was replaced by the official native sequence
`home_bksh/home_bksh_oct_30_2012_scan2_erika`.

The exact twelve-frame identities/hashes and decision are retained in
`data/round6_sun3d/rejection_bm1.json`; the corresponding preview is
`round6_brown_bm_fixed12.png`. This is a physical-split QA decision, not a
performance-based exclusion.

## Remaining v2 representatives

The predeclared first, middle and last selected frames of every v2 scene were
reviewed. Exact original ordinals, RGB hashes and image paths are retained in
`fresh_v2_preview_manifest.json`, with `fresh_v2_first_middle_last.png`.

| Test place | Visible layout in fixed representatives | Scope of conclusion |
|---|---|---|
| brown_cogsci_1 | Conference table, dark wall display/whiteboard, black chairs, kitchenette | Visually distinct from bm office/corridor and old Harvard conference interiors; building address not established |
| brown_cs_3 | Lounge with red armchairs, coffee/sink cabinet area | Distinct layout from cogsci and bm representatives; same university/collection style remains |
| mit_32_d507 | Meeting/study tables, light yellow chairs, whiteboard/projector area | No obvious same-room match to old MIT lab/study/dorm previews; MIT institution is shared |
| home_at | Apartment dining area, Christmas tree, wooden cabinets, kitchen | Not the old MIT dorm bedroom, and visually different from home_bksh representatives; multi-room household scan |
| home_bksh | Narrow hall, bathroom with black shelf/green basket, adjoining rooms | Distinct from home_at kitchen/dining/bathroom representatives; multi-room household scan |

No definite duplicate room was identified among these fixed previews. This is
corroborating evidence, **not proof that every trajectory is physically
disjoint**. Several sequences traverse multiple rooms, so five groups must not
be described as exactly five single rooms or independent buildings.

[SUN3D's official list](https://sun3d.cs.princeton.edu/) and the
[3DMatch author data page](https://3dmatch.cs.princeton.edu/) distinguish the
named scan/place groups, but do not supply complete physical floorplans.
Top-level groups have no overlap with the previous nine matcher places. Do
not claim institution-heldout or unknown-foundation-pretraining disjointness.
Brown and home collection-group leave-out analyses should remain descriptive.

Nearest RGB/depth association is fixed by timestamps. Home_bksh has a maximum
32.057 ms gap, versus roughly 16.7 ms in the other selected scenes. All pairs
remain; no synchronization/visibility/performance threshold was used to select
a favorable subset. Sensor-derived labels will still be approximate under
pose, depth and time-association error.
