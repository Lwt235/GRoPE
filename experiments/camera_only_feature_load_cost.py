"""Planned cold asset materialization costs only; zero model/source/read forwards."""
import os
import subprocess
import time
import numpy as np
import torch
from experiments import camera_only_feature_data as d
from models.camera_only_feature_reader import CameraOnlyFeatureReader

def main():
    start=time.time();torch.set_num_threads(4);free,total=torch.cuda.mem_get_info();assert free>=16*2**30;torch.cuda.set_per_process_memory_fraction(10*2**30/total);assert not d.resources(admission=True)['stop_reasons']
    # Run only after both formal GPU children have exited; no third active worker.
    previous=[d.read(p) for p in (d.OUT/'runtime').glob('*.json') if d.read(p).get('GPU_session')]
    assert all(r.get('process_exited') for r in previous)
    path=d.OUT/'runtime'/f'worker0_load_cost_{time.time_ns()}.json';record=dict(GPU_session=True,worker='worker0_load_cost',gpu='0',start_epoch=start,pid=os.getpid(),foreign_shared=False,admission_free_GiB=free/2**30,cap_GiB=10,phase='cold logical materialization, no forward')
    uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    record['foreign_shared']=any(row.split(',')[0].strip()==uuid and row.split(',')[1].strip()!=str(os.getpid()) for row in apps.splitlines())
    d.write(path,record);rows=[];models=[]
    try:
        for meta in d.contract()['training_catalog'][:2]:
            rid=meta['id'];assets=[('base_GS',d.old.scene_path(rid))]+[(f'derived_{arm}',d.memory_path(arm,d.SEEDS[0],rid)) for arm in d.ARMS]
            for label,asset in assets:
                torch.cuda.synchronize();t=time.monotonic();value=d.load(asset);cpu=time.monotonic()-t;t=time.monotonic();value=d.to_device(value);torch.cuda.synchronize();gpu=time.monotonic()-t
                rows.append(dict(id=rid,asset=label,path=str(asset),bytes=asset.stat().st_size,CPU_file_load_seconds=cpu,CPU_to_GPU_synchronized_wall_seconds=gpu,logical_materialization_seconds=cpu+gpu))
                del value;torch.cuda.empty_cache()
        for arm in d.ARMS:
            torch.cuda.synchronize();t=time.monotonic();state=d.load(d.final_path(arm,d.SEEDS[0]));cpu=time.monotonic()-t;t=time.monotonic();reader=CameraOnlyFeatureReader().cuda();reader.load_state_dict(state['model']);torch.cuda.synchronize();gpu=time.monotonic()-t
            models.append(dict(arm=arm,final_recovery_path=str(d.final_path(arm,d.SEEDS[0])),bytes=d.final_path(arm,d.SEEDS[0]).stat().st_size,CPU_recovery_file_load_seconds=cpu,reader_instantiate_and_GPU_weight_load_seconds=gpu,total_seconds=cpu+gpu))
            del state,reader;torch.cuda.empty_cache()
        d.write(d.OUT/'load_cost_measured.json',dict(complete=True,assets=rows,reader_model_loads=models,scope='First explicit materialization in this supplemental load pass; OS page cache not flushed or controlled, not a physically cold-disk benchmark. Existing final recovery file includes Adam/RNG, only reader weights copied toGPU. No inference/raster/source encoding, no new timing sweep.',native_model_loads_separate='workers/*_native_load.json',forward_counts_added=0))
        record.update(process_exited=True,exit_code=0,seconds=time.time()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,finished_utc=d.now());d.write(path,record)
        print(dict(complete=True,seconds=record['seconds'],forwards=0),flush=True)
    except BaseException:
        record.update(process_exited=True,exit_code=1,seconds=time.time()-start,finished_utc=d.now());d.write(path,record);raise

if __name__=='__main__':main()
