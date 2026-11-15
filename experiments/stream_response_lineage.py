"""CPU source/partition identity check using retained geometry, no model replay."""
import os,time,json,resource
import torch
from experiments import stream_response_data as d

def main():
 start=time.monotonic();rows=[]
 for path in sorted((d.DATA/'geometry').glob('*.pt')):
  x=d.load(path);ids=x['remote_IDs'];n=len(ids);assert len(torch.unique(ids))==n;assert not n or bool(((ids//1041>=1)&(ids//1041<=x['t']-8)&(ids.remainder(1041)>=5)).all());assert torch.isfinite(x['source_XYZ']).all();assert n==x['valid_remote'];assert n+x['invalid_remote']==x['eligible_remote']
  for key in ['coarse','fine']:
   ii=torch.cat(list(x[key].values())).long() if x[key] else torch.empty(0,dtype=torch.int64);assert torch.equal(ii.sort().values,torch.arange(n));assert len(x[key])==min(n,256 if key=='coarse' else 1022)
  rows.append(dict(id=x['id'],frame=x['t'],valid_remote=n,invalid_remote=x['invalid_remote'],coarse_groups=len(x['coarse']),fine_groups=len(x['fine']),source_frames=[int(ids.min()//1041),int(ids.max()//1041)] if n else [],exact_partition=True))
 d.write(d.OUT/'source_lineage_validation.json',dict(complete=len(rows)==36,PID=os.getpid(),CPU_only=True,model_calls=0,geometry_records=len(rows),remote_ID_constraints=True,all_nonempty_partition_coverage=True,rows=rows,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024));print(dict(geometry_records=len(rows)),flush=True)
if __name__=='__main__':main()
