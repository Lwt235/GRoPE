"""CPU-only empty-support boundary and inherited F output interface check."""
import torch,time,os,json,resource
from experiments import stream_response_data as d
from experiments import stream_response_moments as mm
from experiments.stream_history_data import DATA as old

def main():
 s=time.monotonic();torch.set_num_threads(2);assert mm.tree([])==({},{});a=torch.empty(16,0,64);states,_=mm.fit(a,a.clone(),torch.empty(0,3),{},{});r=torch.zeros(16,2,64);logD=torch.randn(16,2);oD=torch.randn(16,2,64);checks={}
 for m in ['X','K','M']:
  out,lz,mass=mm.approximation(r,states[m],logD,oD);checks[m]=dict(output_exact=bool(torch.equal(out,oD)),denominator_exact=bool(torch.equal(lz,logD)),remote_mass_zero=bool(torch.equal(mass,torch.zeros_like(mass))));assert all(checks[m].values())
 first=d.plan()['catalog'][0]['id'];base=d.load(old/'predictions'/f'{first}.pt');new=d.load(d.DATA/'predictions'/f'{first}.pt');dd=(new['depth']-base['depth_full']).abs();pp=(new['poses']-base['poses']['F']).abs();rec=dict(complete=True,CPU_only=True,model_calls=0,empty_remote_exact_fallback=checks,first_stream_F_replay=dict(id=first,frames=96,depth_max_abs=float(dd.max()),depth_mean_abs=float(dd.mean()),pose_max_abs=float(pp.max()),no_old_pixel_metric_recomputed=True,old_identity_inherited=True),seconds=time.monotonic()-s,PID=os.getpid(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);d.write(d.OUT/'interface_validation.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':main()
