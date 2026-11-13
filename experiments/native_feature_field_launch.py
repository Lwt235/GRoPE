"""Managed own-child runtime/admission/watchdog, no coordinator mutation."""
import argparse,os,time,fcntl,subprocess,sys
from experiments import native_feature_field_data as d

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--mode',choices=['targets','row','cost'],required=True);p.add_argument('--row',default='C');a=p.parse_args();d.BASEOUT.mkdir(exist_ok=True);folder=d.BASEOUT/'runtime';folder.mkdir(exist_ok=True)
    with (folder/'admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);gpu,free=[x.strip() for x in subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')];apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True);foreign=any(line.split(',')[0].strip()==gpu for line in apps.splitlines());active=[d.read(f) for f in folder.glob('*.json') if d.read(f).get('GPU_session') and not d.read(f).get('process_exited')];assert len(active)<2 and sum(x['foreign_shared'] for x in active)+int(foreign)<=1;assert int(free)>=16384;assert not d.resources(admission=True)['stop_reasons']
        name=f'{a.mode}_{a.row}_{time.time_ns()}';path=folder/(name+'.json');start=time.time();rec=dict(GPU_session=True,mode=a.mode,row=a.row,gpu_uuid=gpu,foreign_shared=foreign,start_epoch=start,admission_free_MiB=int(free),allocator_fraction=.20,cap_GiB=10,process_exited=False);d.write(path,rec)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PATH='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH'],TORCH_EXTENSIONS_DIR=str(d.parent.old.DATA/'torch_extensions'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6')
    module='experiments.native_feature_field_source' if a.mode=='targets' else ('experiments.native_feature_field_cost' if a.mode=='cost' else 'experiments.native_feature_field_run');cmd=[sys.executable,'-B','-u','-m',module]+(['targets'] if a.mode=='targets' else ([] if a.mode=='cost' else ['--row',a.row]));f=(d.BASEOUT/(name+'.log')).open('a');child=subprocess.Popen(cmd,cwd=d.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);rec['pid']=child.pid;d.write(path,rec);last=0;peak=0;yielding=False;previous=None
    while child.poll() is None:
        if time.monotonic()-last>=30:
            last=time.monotonic();previous=d.resources(previous);free_now=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True);own=[int(row.split(',')[2].strip()) for row in apps.splitlines() if row.split(',')[1].strip()==str(child.pid)];peak=max([peak]+own)
            if free_now<6144 or peak>10240 or previous['stop_reasons']:
                yielding=True;rec['yield_reason']=dict(resources=previous,free_MiB=free_now,own_peak_MiB=peak);child.terminate()
        time.sleep(2)
    code=child.wait();rec.update(exit_code=code,process_exited=True,seconds=time.time()-start,peak_process_MiB_sampled=peak,resource_yield=yielding,finished_utc=d.now());d.write(path,rec);print(dict(runtime=str(path),exit_code=code,seconds=rec['seconds']),flush=True);raise SystemExit(code)
if __name__=='__main__':main()
