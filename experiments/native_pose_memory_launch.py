"""Managed local launcher; no coordinator mutation or scientific plan duplication."""
import os,sys,subprocess,json
from pathlib import Path
ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'results/gaussian_memory/native_pose_memory_pe1';OUT.mkdir(parents=True,exist_ok=True)
base='/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory'
env=os.environ.copy();env.update(PATH='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH'],TORCH_EXTENSIONS_DIR=base+'/torch_extensions',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6')
args=sys.argv[1:];gpu=args.pop(0);env['CUDA_VISIBLE_DEVICES']=gpu
name='_'.join(args).replace('/','_');log=OUT/(name+'.log');f=log.open('a');proc=subprocess.Popen(['/home/liwt/miniColmap/.venv/bin/python','-u','-m','experiments.native_pose_memory_run']+args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=proc.pid,arguments=args,GPU_UUID=gpu,log=str(log))))
