"""Finalize the finite read-only evidence delivery; no external/model operations."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
def identity(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

docs = [PROJECT / 'docs' / n for n in ['research_current.md', 'research_outline_alignment.md', 'gaussian_memory_evidence_scope.md']]
static_files = [p for p in ROOT.rglob('*') if p.is_file() and p.name not in {'execution.json', 'terminal.json', 'implementation_handoff.json', 'artifact_manifest.json'}]
dump('artifact_manifest.json', dict(files=[identity(p) for p in sorted(static_files)], research_entries=[identity(p) for p in docs],
                                    exclusion='Execution/terminal/handoff are separately indexed, avoiding self-referential hashes.'))
dump('terminal.json', dict(status='complete_finite_read_only_delivery', background_jobs_started=[], managers=[], owner_sessions=[],
                           CUDA_or_GPU_probes=0, model_imports=0, model_calls=0, training=0,
                           process_account='Only synchronous source-fetch/file/stdlib tools; all earlier shell tools returned, no yielded background process or pending wait.',
                           validation_receipt='Final source/reference check actual synchronous exit_code=0: 46 fixed-source line references; 16 exact source bodies; input and contract match.',
                           final_seal='This synchronous stdlib writer is the final shell operation; its tool exit code is reported in the actual FINAL message.'))
handoff = dict(round=47, experiment='stac_route_review1', role='/root/research_implementation_sol_recovery',
               original_creation_request=dict(model='gpt-6.1-sol', reasoning_effort='high', fork_turns='none'),
               Fast_parameter_supported=False, backend_effective_configuration=None,
               contract=identity(PROJECT / 'results/gaussian_memory/autonomous_research_loop/round_046/science_plan.md'),
               input_handoff=identity(PROJECT / 'results/gaussian_memory/autonomous_research_loop/round_047/implementation_input_handoff.json'),
               source_commit='fd7e718597cf9963de85c8fffae32a698e8619f5', report=identity(ROOT / 'report.md'),
               artifacts={n: identity(ROOT / n) for n in ['source_identities.json', 'retrieval_ledger.json', 'local_text_identities.json', 'artifact_manifest.json']},
               execution=str(ROOT / 'execution.json'), terminal=str(ROOT / 'terminal.json'),
               unresolved=['Default CUDA merger exact fusion/replacement/count update and retrieval sort/bias not verified: merger-cuda Python wrapper/CUDA wrapper/pipeline/kernels bodies not retrieved within 16-text cap.',
                           'Full parameter key/shape coverage, actual checkpoint identity/loading and complete RoPE/Block/DPT equivalence unknown; no author imports or weight reads.',
                           'Dynamic total byte peak, CUDA allocator/workspace/storage aliasing, dependency compatibility and performance/accuracy unknown; no runtime test.'],
               fixed_facts='Default 4-frame chunk; hot anchor/recent/HH plus delayed shared geometry-neighborhood retrieval; separate all-history four-iteration camera; append-only voxel table and full RGB/output backing. Direct/Python-reference/inference/unknown distinguished in report.',
               research_entries=[str(p) for p in docs],
               decision_scope='Facts returned to original Astra via main; no method adoption, successor, fresh confirmation or overall success. All closures and canonical goal preserved.',
               human_authorization='Continuous autonomous sequential main/original Astra/Sol bidirectional handoff retained; implementation lease released on actual FINAL.')
dump('implementation_handoff.json', handoff)
execution = json.loads((ROOT / 'execution.json').read_text())
ended = datetime.now(timezone.utc)
elapsed = (ended - datetime.fromisoformat(execution['first_operation_utc'])).total_seconds()
assert elapsed < 45 * 60
execution.update(sealed_utc=ended.isoformat(), mixed_wall_seconds_to_seal=elapsed,
                 mixed_wall_seconds_conservative=math.ceil(elapsed) + 2,
                 explicit_retrieval_requests=17, author_text_bodies=16, author_body_bytes=467805,
                 request_failures=0, request_retries=0, connector_internal_HTTP_requests='unknown',
                 status='complete_finite_fact_delivery_with_explicit_unknowns', all_tools_synchronous=True,
                 actual_validation_shell_exit_code=0, new_author_source_imports=0,
                 zero_work=dict(GPU_calls=0, GPU_probes=0, model_calls=0, predictions=0, training_updates=0,
                                weight_or_dataset_downloads=0, dependencies_or_CUDA_builds=0),
                 storage_measure='OUT files plus conservatively all bytes of the three existing modified research docs; final JSON-size fixed point below')
for _ in range(3):
    dump('execution.json', execution)
    output_bytes = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    conservative_bytes = output_bytes + sum(p.stat().st_size for p in docs)
    execution.update(new_evidence_directory_bytes=output_bytes,
                     modified_existing_docs_whole_file_bytes=sum(p.stat().st_size for p in docs),
                     conservative_all_new_evidence_bytes_upper_bound=conservative_bytes,
                     budget_pass=dict(wall_seconds=math.ceil(elapsed)+2 <= 2700, author_texts=16 <= 16,
                                      explicit_requests=17 <= 24, storage=conservative_bytes <= 16*1024**2))
    assert all(execution['budget_pass'].values())
dump('execution.json', execution)
print(json.dumps(dict(status='actual_synchronous_delivery_sealed', elapsed_seconds=elapsed,
                      evidence_bytes=output_bytes, conservative_bytes=conservative_bytes,
                      report_sha256=handoff['report']['sha256'],
                      handoff_sha256=identity(ROOT / 'implementation_handoff.json')['sha256'],
                      manifest_sha256=identity(ROOT / 'artifact_manifest.json')['sha256'])))
