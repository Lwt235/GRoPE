"""Fixed experiment stages; failures are retained without silently changing scope."""
import argparse
import gc
import time
import traceback

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_summary_data import OUT,extract
from experiments.gaussian_memory_summary_train import train
from experiments.gaussian_memory_summary_evaluate import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('train','extract','evaluate'));args=p.parse_args()
    start=time.monotonic()
    try:globals()[args.stage]()
    except Exception:
        write_json(OUT/f'failure_{args.stage}_{int(time.time())}.json',{'stage':args.stage,'traceback':traceback.format_exc(),'scientific_result':False});raise
    gc.collect();torch.cuda.empty_cache()
    write_json(OUT/f'{args.stage}_complete.json',{'complete':True,'seconds':time.monotonic()-start})


if __name__=='__main__':main()
