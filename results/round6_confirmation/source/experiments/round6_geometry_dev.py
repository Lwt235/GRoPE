"""Equal-budget nonlearned kernel calibration on development scenes only."""
import argparse
import json
from pathlib import Path

import torch

from experiments.round6_train import load_split, evaluate, macro, GRIDS, source_hashes
from models.round6_measure_attention import position_bias, METHODS


class Geometry(torch.nn.Module):
    def __init__(self, method, bandwidth, content_temperature=0.):
        super().__init__()
        self.method,self.width,self.temperature=method,bandwidth,content_temperature

    def forward(self,d):
        return position_bias(d,self.method,self.width)+self.temperature*(d['q']@d['k'].transpose(-1,-2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',nargs='+',type=Path,default=[Path('results/round6_region_inputs')])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--widths',nargs='+',type=float,default=[.25,.5,1.,2.,4.])
    p.add_argument('--temperatures',nargs='+',type=float,default=[0.,5.,10.,20.])
    p.add_argument('--limit',type=int)
    args=p.parse_args();torch.set_num_threads(4)
    if args.out.exists():raise ValueError('refusing overwrite')
    args.out.mkdir(parents=True)
    data,ids,sources=load_split(args.data,'val',args.device,args.limit)
    records=[];best={}
    for method in METHODS:
        for width in args.widths:
            for temperature in args.temperatures:
                rows=evaluate(Geometry(method,width,temperature),data,ids)
                record=dict(method=method,width=width,temperature=temperature,kl=macro(rows),
                    kl_by_grid={g:macro(rows,grids=[g]) for g in GRIDS},
                    pck8=macro(rows,'pck8_count','n_point',['q8_k8']))
                records.append(record)
                if method not in best or record['kl']<best[method]['kl']:
                    best[method]=record
                    (args.out/(method+'_val.json')).write_text(json.dumps(dict(rows=rows,config=record),indent=2))
        print('BEST',method,best[method],flush=True)
    (args.out/'summary.json').write_text(json.dumps(dict(records=records,best=best,input_sources=sources,
        sourcehash=source_hashes(),scope='development validation only; no trained body, no fresh test'),indent=2))


if __name__=='__main__':main()
