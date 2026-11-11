"""Execute the authorized fixed diagnostic without selecting models or data."""
import time

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_direct_data import OUT, REF_CACHE, VIEW_OUT, SEEDS, model_path, extract
from experiments.gaussian_memory_direct_train import train
from experiments.gaussian_memory_direct_evaluate import evaluate
from experiments.gaussian_memory_direct_read import run
from experiments.gaussian_memory_direct_cost import measure


def main():
    completed=[]
    stages=[('reference_extraction',extract),('training',train),('evaluation',evaluate),('cost',measure),
            ('standalone',lambda:run(VIEW_OUT / 'standalone/observed_input.npz',REF_CACHE / 'val_12.pt',
              model_path('direct',SEEDS[0]),OUT / 'standalone.npz'))]
    for name,action in stages:
        write_json(OUT / 'pipeline_status.json',{'complete':False,'running':name,'completed':completed})
        started=time.monotonic(); action(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        completed.append({'stage':name,'seconds':time.monotonic()-started})
        print({'stage_complete':name,'seconds':completed[-1]['seconds']},flush=True)
    write_json(OUT / 'pipeline_status.json',{'complete':True,'completed':completed,'analysis_review_pending':True})


if __name__=='__main__': main()
