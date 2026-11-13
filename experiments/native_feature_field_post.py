"""Finite CPU-only score queue and tables, without coordinator/task mutation."""
import os,time,sys,subprocess,csv
import numpy as np
import torch
from experiments import native_feature_field_data as d

def main():
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4');done=set()
    while len(done)<2:
        for row in ('C','J'):
            if row in done:continue
            workers=list((d.BASEOUT/row/'workers').glob(f'{row}.json'))
            if workers and d.read(workers[0]).get('phase')=='complete':
                f=(d.BASEOUT/f'score_{row}.log').open('a');subprocess.run([sys.executable,'-u','-m','experiments.native_feature_field_score','--row',row],cwd=d.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True);done.add(row)
        if len(done)<2:time.sleep(5)
    subprocess.run([sys.executable,'-u','-m','experiments.native_feature_field_summary'],cwd=d.ROOT,env=env,check=True)
    print('Both rows independently scored and factorial interaction complete',flush=True)
if __name__=='__main__':main()
