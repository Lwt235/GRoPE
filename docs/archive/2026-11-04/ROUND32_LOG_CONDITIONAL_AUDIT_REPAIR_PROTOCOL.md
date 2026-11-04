# R32 post-seal audit serialization repair

2026-10-31. The first formal R32 producer completed all 46 reference calls.
The independent output audit stopped while writing checks.json because its
new coalescing diagnostic contains a NumPy bool scalar. The original pipeline
has an exit-1 audit stage and is not a fully successful pipeline. A premature
root acceptance attempt then stopped on missing all_stages_completed; preserve
that attempt as well. No accepted numerical conclusion exists yet.

Preserve every byte of the original producer, failed audit, execution record,
source files and snapshots. Do not repeat reference integration. Create a
separate post-seal repair directory, explicitly identified as a new audit of
already sealed results. Copy only the old audit lock/lock.sha256, frozen source
snapshot, input_verification.json and input_audit tree byte-for-byte. These
are copies of original evidence, not a new pre-forward freeze or input audit.

Freeze the new wrapper, its tests and this protocol before repaired auditing.
Its repair lock binds the complete original producer and failed-audit
inventories, the original pipeline failure and root-controller history, and
the inherited input/source evidence. Verify those immutable bindings before
and after execution. Record its actual post-seal timestamps.

Run the unchanged frozen audit function against the new directory. Its input,
source, forward, budget, arithmetic, all 46 checks and 155 comparisons remain
unchanged. Only temporarily replace the JSON write function: keep indent=2,
sort_keys=True, allow_nan=False and the trailing newline; the JSON default
handler accepts exactly numpy.bool_ and returns its Python bool value. All
other unsupported types and nonfinite JSON numbers must still fail. Restrict
every write to the new repair directory and restore the original writer in
a finally block. Do not patch numerical functions, error gates or records.

Test nested true/false NumPy booleans, rejection of unsupported scalar types
and NaN, output-path confinement and restoration after an exception using
pure toys. Preserve the source bytes before every test attempt and its log.
Before any saved-array decode, a metadata-only freeze checks the source and
inventory bindings. The repaired audit must read only the already authorized
synthetic inputs and sealed outputs; no model/reference integration or real
input/label access is allowed.

Root acceptance must separately verify the original five successful stages,
the failed sixth stage, retained failure artifacts and the new repaired audit.
It must replay all saved checks/comparisons without integrating, distinguish
successful serialization from numerical success, and report the failure and
repair history. Any failed numerical gate remains a failure. Scientific
success and goal completion remain false even if this fixed synthetic round
eventually passes.
