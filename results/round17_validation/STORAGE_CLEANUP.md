# Storage cleanup 2026-11-10

Large derived NumPy arrays were partially removed under user authorization. Reports, scores, configuration, logs and checkpoints remain unchanged. Two representative large arrays per scene/group remain. Exact removed paths, sizes and SHA256 are in `results/storage_cleanup_20261110/deleted.jsonl`; the retention plan is in `plan.json` there. Historical array-level replay requires regenerating removed caches using the original producer code/configuration; this cleanup does not change scientific conclusions.
