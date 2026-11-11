"""One fixed reference-geometry experiment; no tuning loop."""
import gc
import json
import time
import traceback

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_oracle_data import OUT, prepare
from experiments.gaussian_memory_oracle_train import train
from experiments.gaussian_memory_oracle_evaluate import evaluate


def main():
    completed=[]
    for stage,fn in [('reference_geometry',prepare),('training',train),('evaluation',evaluate)]:
        write_json(OUT/'pipeline_status.json',{'complete':False,'stage':stage,'completed':completed});start=time.monotonic()
        try:fn()
        except Exception:
            write_json(OUT/f'failure_{stage}_{int(time.time())}.json',{'stage':stage,'traceback':traceback.format_exc(),'scientific_result':False,'completed':completed});raise
        gc.collect();torch.cuda.empty_cache();row={'stage':stage,'seconds':time.monotonic()-start};completed.append(row);print(json.dumps({'stage_complete':stage,**row}),flush=True)
    write_json(OUT/'pipeline_status.json',{'complete':True,'completed':completed,'analysis_review_pending':True})


if __name__=='__main__':main()
