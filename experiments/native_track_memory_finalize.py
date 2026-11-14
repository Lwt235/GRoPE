"""Seal completed artifacts after actual process exits; zero model or metric calls."""
import re,json
from experiments import native_track_memory_data as d

def main():
    terminal=d.read(d.OUT/'actual_terminal_confirmation.json');assert not terminal['owned_science_CPU_or_GPU_module_processes_live'] and not terminal['owned_GPU_PIDs_still_present']
    r=d.read(d.OUT/'runtime_budget_calls.json');assert all(w.get('actual_wait_returned') for w in r['workers'])
    sources=[*d.ROOT.glob('experiments/native_track_memory*.py'),d.ROOT/'models/native_track_memory.py'];project=sum(p.stat().st_size for p in d.OUT.rglob('*') if p.is_file())+sum(p.stat().st_size for p in sources)
    r.update(project_final_bytes=project,project_peak_conservative_bytes=project+64*2**20,project_measurement_scope='Final artifacts plus implementation source at seal time, including existing manifest; final manifest reindex and receipts covered by64MiB reserve',actual_terminal_confirmation=str(d.OUT/'actual_terminal_confirmation.json'));assert r['project_peak_conservative_bytes']<2**30
    d.write(d.OUT/'runtime_budget_calls.json',r)
    p=d.OUT/'report.md';text=p.read_text();text=re.sub(r'项目含脚本最终[0-9.]+GiB、保守峰[0-9.]+GiB /1GiB',f'项目含脚本封存测得约{project/2**30:.4f}GiB、保守全产物峰{r["project_peak_conservative_bytes"]/2**30:.4f}GiB /1GiB',text);text=text.replace('所有GPU子进程实际wait返回，共35个worker','[现场退出确认](actual_terminal_confirmation.json)核实所有本轮GPU／科学CPU进程结束；所有GPU子进程实际wait返回，共35个worker');p.write_text(text)
    h=d.read(d.OUT/'implementation_handoff.json');h['report']['sha256']=d.sha(p);h['actual_terminal_confirmation']=str(d.OUT/'actual_terminal_confirmation.json');h['actual_CPU_and_GPU_terminal_confirmation']='All recorded GPU children actualwait; final fit manager, cost launcher, scorer, training/first/final figure exports and delivery process actual exit checked; host found no owned science module or GPU PID.';h['limits'].append('Earlier manager44052 stop-signal origin unknown; safe exit75/all children waited, no lost scientific updates or predictions.');h['completed_utc']=d.now()
    manifest=d.read(d.OUT/'artifact_manifest.json');manifest['project_artifacts']=[dict(path=str(p),bytes=p.stat().st_size,sha256=d.sha(p)) for p in sorted(d.OUT.rglob('*')) if p.is_file() and p.name not in ('artifact_manifest.json','implementation_handoff.json','pool.lock')];manifest['implementation_sources']=[dict(path=str(p),bytes=p.stat().st_size,sha256=d.sha(p)) for p in sources];manifest['living_document_updates']=[dict(path=str(p),sha256=d.sha(p)) for p in [d.ROOT/'docs'/n for n in ('research_current.md','research_outline_alignment.md','gaussian_memory_evidence_scope.md')]];d.write(d.OUT/'artifact_manifest.json',manifest)
    h['artifact_manifest_sha256']=d.sha(d.OUT/'artifact_manifest.json');d.write(d.OUT/'implementation_handoff.json',h)
    print(json.dumps(dict(report=h['report'],handoff=dict(path=str(d.OUT/'implementation_handoff.json'),sha256=d.sha(d.OUT/'implementation_handoff.json')),GPU_worker_hours=r['GPU_worker_hours'],project_bound_GiB=r['project_peak_conservative_bytes']/2**30)),flush=True)
if __name__=='__main__':main()
