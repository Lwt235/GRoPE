# Round12 old-development depth geometry data

Freeze this protocol and producer before generating any geometry. Only the five
original R7 old-development scans and their complete original manifest pair
order are permitted: 576 train pairs (192 each Harvard c3/c5/c6), 128 val pairs
(64 Harvard c8 and 64 Brown bm4). No confirmation or test archive is opened.
No sensor image or label file is opened in any producer stage. Label SHA values
are copied from old R7 manifest metadata solely for downstream identity checks.

The shared q/k features, predicted-depth values, targets and masks remain in
original R7 files. Every original q/k feature is checked bitwise against the
corresponding frozen RGB cache in original pair order. Camera metadata, RGB
images, cache arrays/sidecars, original selection, R7 manifest and inputs are
hashed in the new data lock before computation. The earlier B/R11 selections
and source locks are verified; their completed prediction files are used only
for numerical replay. All producer sources and independent reference sources
are hashed and snapshotted before geometry output. Existing output directories
and completed stages are never overwritten.

Eight frozen roles use actual native stride8 key rectangle integration:

| Role | method | h | online content T |
|---|---|---:|---:|
| full4 | full4 | .5 | 5 |
| iso4_matched | iso4 | .5 | 5 |
| full1 | full | .25 | 0 |
| quad4 | quad4 | .5 | 5 |
| quad16 | quad16 | .5 | 5 |
| point | point | 1 | 5 |
| point_pck | point | .5 | 5 |
| fixed | fixed | .5 | 5 |

Geometry comes only from R10 dense predicted-pixel projection, including all
pixels regardless of in-front or visibility status. Physical pixels/pixel² are
converted to stride8 units by /8 and /64. Support moments are prepared in
float32 exactly as B/R11, and native cell integration uses the frozen float64
GL32 operator. Each key cell has physical edges (integer bounds−.5)/8. Mixtures
use actual area weights. The same floor density 1e−30 is integrated over native
cell area in stride8²; g8 cells have area one. No content score is stored.

Each `train|val/scene_slug/{role}.npy` is float32 `[N,432,432]` geometry log mass.
The scene manifest gives ordered pairs, absolute original R7 **scene directory**
`r7_root`, R7 manifest/input hashes, copied label hash and each role's method,
h, temperature, shape, dtype, SHA and path **relative to that scene directory**.
The top-level lock fixes all identities, and completion is written only after
all five scenes, eight roles and 704 pairs verify. Array payload is 4,204,167,168
bytes (3.916 GiB), with no duplicate compressed archive. Consumers mmap one role
at a time; learned QK content remains online and receives the manifest T.

Stages are freeze → preflight → export-scene (independent scenes) → finalize.
Before full export, the first pair [30,45] from each old val scene is replayed
against both completed B/R11 prediction archives, after the identical g8→g16
query aggregation. These are old forward predictions, not target metrics.
Every role is also checked on 12 evenly spaced fixed query rows/all432 keys
against NumPy literal rectangle support moments and SciPy CDF GL64 integrals.
Max probability error must be ≤1e−4. Float32 cached-logit absolute error must
be ≤1e−5; both preflight pairs also check probability error after float32
content addition and softmax (≤1e−4). Full export checks finite logits and cast
error for every element. No threshold is selected from depth outcomes.

For independent fresh full-pipeline timing, public `compute_geometry` performs
static g8 partition setup, support preparation and integration on each call.
Input is exactly two projected predicted-pixel tensors; GT/visibility/content
fields are rejected. `prepare_forward_observation` constructs those tensors
from frozen RGB cache plus supplied cameras. `load_first_observation` loads the
metadata-fixed first R7 pair per scene and verifies the same source identities.
The downstream timing stage can exclude shared RGB/depth inference, projection
and transfer while including fresh support preparation, geometry and actual
learned QK/attention/V/head. Cached logits must not stand in for geometry timing.
