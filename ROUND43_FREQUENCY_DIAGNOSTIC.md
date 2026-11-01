# R43 bounded positional-path diagnosis

Declared after complete R43 old_eval metrics, before this diagnosis runs.
Primary PCK8=.919973726 versus point=.923507882; the new frequency roles are
nearly tied. No retraining, new winner, promotion or frequency/step search.

Question: did the networks use rotary positions at all, and how much do their
learned frequency changes matter at the selected weights?
Alternative: attention/V is useful but the positional phase is ignored, or
frequency changes are small/compensated by the content network.

Use ONLY the same64 old R5 validation pairs, all432 keys and all valid queries,
from the exact R43 frozen cache. Models: gaussian_full,free_frequency,
fixed_gaussian,point; all five selected seed checkpoints, unchanged weights.
Run normal and zero_phase for all20 models. Also restore the common initial
frequency bank for gaussian_full/free_frequency (ten models). Total50 forward
conditions, no training. zero_phase bypasses rotary position but preserves Q/K,
V, residuals and FFNs; initial_bank changes frequencies only.

Freeze protocol, runner, original lock and all selected checkpoint identities
before array decoding. Recompute complete per-query CE/PCK4/PCK8/EPE in NumPy
from exported final logits; save per-query sufficient statistics and frame rows.
Normal output must reproduce the saved selection rows, with numerical differences
reported rather than interpreted as task effects. These are interventions on
fixed models, not retrained ablation comparisons or new generalization evidence.

Decision: a substantial zero_phase drop establishes reliance on the positional
path, not the usefulness of Gaussian covariance constraints. A near-zero
initial_bank effect localizes the limited change to learned frequencies at these
checkpoints; it does not establish optimization causality. Either result ends
this bounded diagnosis; it does not authorize tuning or rescue of the candidate.
