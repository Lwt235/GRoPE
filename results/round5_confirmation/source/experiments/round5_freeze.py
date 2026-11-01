"""Lock complete five-seed validation selections before any test evaluation."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.round5_train import sha,macro
from models.round5_cross_attention import METHODS

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError('refusing overwrite confirmation lock')
    runs={};sources=None
    for method in METHODS:
        for seed in [11,22,33,44,55]:
            d=args.runs/f'{method}_{seed}'
            if (d/'test.json').exists():raise ValueError('test evaluation already exists')
            completed=json.loads((d/'completed.json').read_text())
            history=json.loads((d/'history.json').read_text());val=json.loads((d/'val.json').read_text())
            best=min(history,key=lambda row:row['val_ce'])
            if val['selected_step']!=best['step'] or abs(macro(val['rows'])-best['val_ce'])>1e-10:
                raise ValueError('validation selection inconsistent')
            if sha(d/'checkpoints/best.pt')!=completed['checkpoint_sha256']:raise ValueError('checkpoint hash mismatch')
            if sources is None:sources=completed['sourcehash']
            if sources!=completed['sourcehash']:raise ValueError('mixed training source versions')
            runs[f'{method}_{seed}']={'selected_step':best['step'],'val_ce':best['val_ce'],
                 'val_pck8':best['val_pck8'],'checkpoint_sha256':completed['checkpoint_sha256']}
    scores={m:sum(runs[f'{m}_{s}']['val_pck8'] for s in [11,22,33,44,55])/5 for m in ['point','fixed']}
    baseline=max(scores,key=scores.get)
    result={'runs':runs,'sourcehash':sources,'validation_baseline_scores':scores,'primary_baseline':baseline,
            'primary_candidate':'normalized','primary_gate':'PCK8 gain >=.01, crossed seed/place CI lower>0, >=4/5seed gain, CE relative worsening<=.02',
            'selection_sha256':sha(ROOT/'data/round5_sun3d/selection.json'),
            'protocol_sha256':sha(ROOT/'ROUND5_PROTOCOL.md'),
            'analysis_sha256':sha(ROOT/'experiments/round5_analysis.py')}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print('FROZEN',baseline,scores,sha(args.out))

if __name__=='__main__':main()
