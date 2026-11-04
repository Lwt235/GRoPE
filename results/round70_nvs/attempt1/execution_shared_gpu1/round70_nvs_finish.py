"""Finish reporting after the already running R70 controller, without retraining."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.round65_common import read,write,sha
from experiments.round65_queue import occupancy
from experiments.round70_nvs_data import OUT,ROOT


def main():
    status=OUT/'finish_status.json'
    if status.exists(): raise ValueError('inspect existing postprocessing process first')
    meta=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        source_sha256=sha(Path(__file__)),report_source_sha256=sha(ROOT/'experiments/round70_nvs_report.py'),
        lock_sha256=sha(OUT/'lock.json'),started_unix=time.time(),goal_complete=False)
    def record(state,**kw): write(status,dict(meta,state=state,observed_unix=time.time(),**kw))
    try:
        while True:
            q=read(OUT/'queue_status.json')
            if q['state']=='complete': break
            if q['state']=='stopped': raise RuntimeError('training/export controller stopped: '+str(q.get('error')))
            proc=Path('/proc')/str(q['pid'])/'stat'
            if not proc.exists(): raise RuntimeError('training/export controller disappeared before completion')
            ticks=proc.read_text().rsplit(')',1)[1].split()[19]
            if ticks!=q['start_ticks']: raise RuntimeError('training controller process identity changed')
            record('waiting_for_training_and_export',completed_children=len(q['completed']))
            time.sleep(10)
        for action in ('baselines','summarize','audit'):
            gpu=None;previous=set()
            if action!='summarize':
                while gpu is None:
                    rows=occupancy();current={r['uuid'] for r in rows if r['free']}
                    stable=current&previous
                    if stable: gpu=next(r for r in rows if r['uuid'] in stable)
                    else:
                        previous=current;record('waiting_for_idle_gpu',action=action);time.sleep(5)
            env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                CUDA_VISIBLE_DEVICES='' if gpu is None else gpu['uuid'])
            command=[sys.executable,'-m','experiments.round70_nvs_report',action]
            with (OUT/f'post_{action}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    record('running',action=action,child_pid=child.pid,gpu=gpu);time.sleep(5)
                if child.returncode: raise RuntimeError(f'{action} failed with {child.returncode}')
            print('POSTPROCESS_COMPLETE',action,flush=True)
        record('complete',scientific_review_pending=True,visual_review_pending=True)
    except BaseException as error:
        record('stopped',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__': main()
