# GRoPE research work

Read `docs/research_current.md` before continuing this research. The current
route follows the user's 2026-11-01 review of excessive static-kernel repair
and numerical auditing; do not automatically resume the unexecuted half-ray
prototype or treat every historical integrator gate as a universal prerequisite.

Before designing, revising, or resuming a research plan, also read
`docs/research_outline_alignment.md` and the relevant prior plans/results linked
there. Every new or revised plan must include a brief outline-alignment check:
identify the O-items it serves, distinguish core requirements from experimental
implementation choices and temporary scope limits, cite related previous
attempts and outcomes, and explain what substantive change or unresolved
question makes the experiment informative. Keep this living document's current
mapping, choices, attempt status, evidence links, and decisions synchronized
when plans change and when experiments complete, fail, or are closed. Preserve
negative results and past decisions; do not present a renamed old attempt as
new, turn a provisional choice into a user requirement, or extrapolate one
configuration's failure into a universal veto. This is a concise documentation
requirement, not an extra numerical gate or user-approval step; routine fixes
use the existing versioned failure records.

Preserve the complete scientific objective and historical negative results.
For each new experiment state the hypothesis, competing explanation, fixed
comparison and the decision different outcomes would change. Distinguish
implementation correctness, mechanism evidence, downstream utility and fresh
confirmation. A limited pass never establishes general effectiveness.

Keep source/data/configuration identities and matched controls. Audit effort
must match the claim and likely effect on conclusions. Development diagnostics
may isolate variables on old data under a prospective plan; do not tune on or
relabel used confirmation data as fresh. Ordinary pre-run fixes need versioned
failure records, not a new scientific round or a report-freezing bureaucracy.

Use available idle GPUs for learned attention and suitable batched arithmetic;
verify live occupancy and CUDA access before launch. Preserve other users' jobs.
Keep the server's old GLIBC intact; prefer the existing compatible Python/Node
runtimes and user-space wrappers. Do not modify system libc.
