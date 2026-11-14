"""Seal existing completed evidence after actual exits; no inference or scoring."""
from experiments import native_source_resolved_data as d

def main():
    terminal=d.read(d.OUT/'actual_terminal_confirmation.json');assert not terminal['owned_science_CPU_or_GPU_module_processes_live'] and not terminal['owned_GPU_PIDs_still_present']
    runtime=d.read(d.OUT/'runtime_budget_calls.json');assert all(w['actual_wait_returned'] for w in runtime['workers'])
    sources=[*d.ROOT.glob('experiments/native_source_resolved*.py'),d.ROOT/'models/native_source_resolved_memory.py'];living=[d.ROOT/'docs'/n for n in ('research_current.md','research_outline_alignment.md','gaussian_memory_evidence_scope.md')]
    project=sum(p.stat().st_size for p in d.OUT.rglob('*') if p.is_file())+sum(p.stat().st_size for p in sources);runtime.update(project_final_bytes=project,project_peak_conservative_bytes=project+64*2**20,actual_terminal_confirmation=str(d.OUT/'actual_terminal_confirmation.json'));assert runtime['project_peak_conservative_bytes']<=2**30;d.write(d.OUT/'runtime_budget_calls.json',runtime)
    manifest=dict(complete=True,project_artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=d.sha(p)) for p in sorted(d.OUT.rglob('*')) if p.is_file() and p.name not in ('artifact_manifest.json','implementation_handoff.json','pool.lock','cost_dispatch.lock')],implementation_sources=[dict(path=str(p),bytes=p.stat().st_size,sha256=d.sha(p)) for p in sources],inherited_implementation_sources=[dict(path=str(p),sha256=d.sha(p)) for p in (d.ROOT/'models/native_track_memory.py',d.ROOT/'models/native_scene_exchange.py',d.ROOT/'experiments/native_track_memory_data.py',d.ROOT/'experiments/native_track_memory_prepare.py')],living_document_updates=[dict(path=str(p),sha256=d.sha(p)) for p in living],new_DATA_artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=d.sha(p)) for p in sorted(d.CACHE.rglob('*')) if p.is_file()],original_fields_readonly_verification=str(d.OUT/'source_field_readonly_verification.json'),old_large_weight_or_cache_hashes_not_recomputed=True)
    d.write(d.OUT/'artifact_manifest.json',manifest);h=d.read(d.OUT/'implementation_handoff.json');h.update(report=dict(path=str(d.OUT/'report.md'),sha256=d.sha(d.OUT/'report.md')),artifact_manifest_sha256=d.sha(d.OUT/'artifact_manifest.json'),actual_terminal_confirmation=str(d.OUT/'actual_terminal_confirmation.json'),actual_GPU_and_CPU_terminal_status='All GPU children actualwait; scoring, figure export and delivery CPU sessions actual exit verified; final sealing call exit checked by caller',completed_utc=d.now());d.write(d.OUT/'implementation_handoff.json',h)
    # Include the manifest/handoff themselves in final physical project bytes.
    # Only refresh this small runtime entry; new DATA hashes are computed once.
    for _ in range(6):
        actual_project=sum(p.stat().st_size for p in d.OUT.rglob('*') if p.is_file())+sum(p.stat().st_size for p in sources)
        if runtime['project_final_bytes']==actual_project:break
        runtime.update(project_final_bytes=actual_project,project_peak_conservative_bytes=actual_project+64*2**20,project_final_scope='All final OUT files including artifact manifest/handoff plus implementation source files')
        d.write(d.OUT/'runtime_budget_calls.json',runtime)
        entry=next(v for v in manifest['project_artifacts'] if v['path']==str(d.OUT/'runtime_budget_calls.json'))
        entry.update(bytes=(d.OUT/'runtime_budget_calls.json').stat().st_size,sha256=d.sha(d.OUT/'runtime_budget_calls.json'))
        d.write(d.OUT/'artifact_manifest.json',manifest)
        h['artifact_manifest_sha256']=d.sha(d.OUT/'artifact_manifest.json');d.write(d.OUT/'implementation_handoff.json',h)
    assert runtime['project_final_bytes']==sum(p.stat().st_size for p in d.OUT.rglob('*') if p.is_file())+sum(p.stat().st_size for p in sources)
    assert runtime['project_peak_conservative_bytes']<=2**30
    print(dict(report=h['report'],handoff_path=str(d.OUT/'implementation_handoff.json'),handoff_sha256=d.sha(d.OUT/'implementation_handoff.json'),GPU_worker_hours=runtime['GPU_worker_hours']),flush=True)
if __name__=='__main__':main()
