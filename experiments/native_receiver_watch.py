"""Read actual same-user module identities and complete log lines; no signals."""
import json
import os
from pathlib import Path
import time

ROOT=Path('/home/liwt/GRoPE')
OUT=ROOT/'results/gaussian_memory/native_receiver_adaptation1'
MODULES={'experiments.native_receiver_orchestrate','experiments.native_receiver_launch',
    'experiments.native_receiver_prepare','experiments.native_receiver_train',
    'experiments.native_receiver_cost','experiments.native_receiver_finish',
    'experiments.native_receiver_storage','experiments.native_receiver_score',
    'experiments.native_receiver_checks','experiments.native_receiver_figures'}


def main():
    live=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            if path.stat().st_uid!=os.getuid():continue
            argv=[v.decode(errors='replace') for v in (path/'cmdline').read_bytes().split(b'\0') if v]
            match=MODULES.intersection(argv)
            if not match or (path/'cwd').resolve()!=ROOT:continue
            state=(path/'stat').read_text().split(') ',1)[1].split()[0]
            if state=='Z':continue
            live.append(dict(pid=int(path.name),state=state,modules=sorted(match)))
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    progress=[];completed=[]
    for folder in sorted(OUT.glob('*_2026111[012]')):
        status=folder/'status.json';log=folder/'training.jsonl'
        if status.exists() and json.loads(status.read_text()).get('complete'):completed.append(folder.name)
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-5000));lines=f.read().splitlines()
            valid=[]
            for line in lines:
                try:valid.append(json.loads(line))
                except (json.JSONDecodeError,UnicodeDecodeError):pass
            if valid:progress.append(dict(name=folder.name,step=valid[-1]['step'],loss=valid[-1]['loss']))
    queue=json.loads((OUT/'queue_status.json').read_text()) if (OUT/'queue_status.json').exists() else {}
    pre=json.loads((OUT/'preflight_progress.json').read_text()) if (OUT/'preflight_progress.json').exists() else {}
    exports=[json.loads(f.read_text()) for f in (OUT/'evaluation').glob('*.json')]
    r=dict(experiment='native_receiver_adaptation1',goal_complete=False,updated_epoch=time.time(),
        actual_host_live_processes=live,completed_fits=completed,fit_progress=progress,
        preflight_complete=(OUT/'preflight.json').exists(),preflight_records=len(pre.get('records',[])),
        expected_fits=9,expected_new_predictions=2880,new_exported_predictions=sum(x.get('predictions',0) for x in exports if x.get('complete')),
        pending=queue.get('pending',[]),failed=queue.get('failed',[]),queue_state=queue.get('state'),
        scope='Actual host /proc UID+cwd+exact argv identity; queue progress and exports are not scientific completion proof. Old sensor-assisted static data; O6 unresolved.')
    temp=OUT/'execution_status.json.tmp';temp.write_text(json.dumps(r,indent=2)+'\n');temp.replace(OUT/'execution_status.json')
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
