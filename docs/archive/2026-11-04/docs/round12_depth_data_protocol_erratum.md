# Round12 depth data protocol byte-count correction

The frozen `ROUND12_DEPTH_DATA_PROTOCOL.md` prose states array payload as
4,204,167,168 bytes. The correct count is **4,204,265,472 bytes**:
`8 roles × 704 pairs × 432 × 432 × 4 float32 bytes`.

The frozen producer, lock `estimated_array_bytes`, and completion calculation
use that exact multiplication and are correct. This 98,304-byte prose typo
(0.00234%) does not affect shape, dtype, population, configuration or experiment.
The original protocol/source lock remains unchanged. NPY headers and JSON/source
metadata add a small amount beyond the array payload.
