"""Admit one original200-call cost worker and register it with the proc backstop."""
import subprocess,sys,time
import numpy
import torch
from experiments import native_render_parameter_resource_sidecar as m

def main():
 marker=m.OUT/'cost.json'
 if marker.exists():assert m.read(marker)['complete'];return
 assert m.read(m.OUT/'queue_status.json')['state']=='complete' and not m.active_workers()
 guard=m.read(m.FOLDER/'current.json');assert time.time()-guard['epoch']<5 and not m.reasons(guard)
 assert torch.cuda.is_available()
 states=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 candidates=[tuple(map(int,line.split(', '))) for line in states.strip().splitlines()]
 idle=[s for s in candidates if s[1]<=512 and s[2]>=20480 and s[3]<=5];assert idle
 gpu=idle[0][0];log=m.OUT/'logs/cost_complete_query.log'
 with log.open('a',buffering=1) as stream:
  p=subprocess.Popen([sys.executable,'-u','-m','experiments.native_render_parameter_launch','--gpu',str(gpu),'--cap','14','--admission','20480','experiments.native_render_parameter_cost'],cwd=m.ROOT,stdout=stream,stderr=subprocess.STDOUT)
  ident=m.identity(p.pid);assert ident and 'experiments.native_render_parameter_launch' in ident['cmdline']
  m.write(m.FOLDER/'cost_launcher_identity.json',ident)
  record=dict(epoch=time.time(),launcher=ident,gpu=gpu,exact_planned_native_calls=200,source_unchanged=True);m.write(m.OUT/'cost_dispatch.json',record)
  rc=p.wait();record.update(returncode=rc,finished_epoch=time.time(),complete=rc==0);m.write(m.OUT/'cost_dispatch.json',record)
  assert rc==0,rc
 assert m.read(marker)['complete'] and m.read(marker)['total_native_forwards']==200
 print('Exactly200-call full-query cost worker completed.')

if __name__=='__main__':main()
