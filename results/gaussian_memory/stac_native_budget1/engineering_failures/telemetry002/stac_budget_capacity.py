"""Effect-blind integer capacity calculation, including common exact recovery."""
import math,sys
from experiments import stac_budget_data as d
def lock_capacity():
 p=d.plan();rows=[d.read(d.OUT/'completed'/f"S_{m['id']}.json") for m in p['catalog']];assert all(x['complete'] and not x['GT_scored'] for x in rows)
 peaks=[x['peak_history_bytes'] for x in rows]
 for f in (d.OUT/'resources').glob('*.json'):
  x=d.read(f)
  if x.get('peak_history_bytes') is not None:peaks.append(x['peak_history_bytes'])
 B=math.ceil(max(peaks)/p['B_quantum'])*p['B_quantum'];assert B<=p['history_hard_max'],'S_history_hard_max'
 inventories=[d.read(f) for f in (d.OUT/'inventory').glob('S_*.json')];assert inventories,'Require observed structural tensor inventory; no effect values read'
 fixed_tensor=max(x['nonspatial_nonmanager']['unique_tensor_bytes'] for x in inventories)
 fixed_headers=max(x['nonspatial_nonmanager']['Python_object_bytes'] for x in inventories)
 fixed_gpu=max(x['nonspatial_nonmanager']['CUDA_tensor_bytes'] for x in inventories)
 RGB=p['compressed_readable_all12_RGB_bytes'];metadata=2**20
 # Exact native buffer dtype/shape bytes. Separate readable RGB files are charged once.
 L,H,D,t,b=24,16,64,1041,2
 def bound(arm,n):
  hot=5+n if arm=='H' else 1+n;reserved=min(96,hot+4)
  KV=2*L*H*reserved*t*D*b;indices=L*H*reserved*t*8;scores=L*H*reserved*t*8 if arm=='H' else 0
  tensors=KV+indices+scores+2*L*8+t*8
  # Gather-copy and selection tensors coexist as in original Python compaction.
  if arm=='H':workspace=2*L*H*hot*t*D*b+L*H*hot*t*16+L*H*(5*hot+2*n+reserved)*t*8
  else:workspace=2*H*hot*t*D*b+H*hot*t*8+reserved*t*24
  manager_headers=32768
  state_tensor=fixed_tensor+tensors
  normal=state_tensor+fixed_headers+manager_headers+workspace+RGB+metadata
  largest_GPU_storage=max(KV//2,96*3*392*518*4,max(x['pose_view_backing_bytes'] for x in inventories))
  # CPU migration frees device backing; serialization is CPU-only. On restore,
  # file + live mixed CPU/GPU state + one copied entire storage can coexist.
  recovery=2*state_tensor+largest_GPU_storage+fixed_headers+manager_headers+RGB+metadata
  # Permanent outputs omit the unused point head, and contain stacked depth plus
  # author's retained CPU images/depth/conf and prefix camera snapshots.
  final=state_tensor+fixed_headers+manager_headers+(96*392*518*4)+(fixed_tensor-fixed_gpu)+RGB+metadata
  return dict(capacity=n,reserved_frames=reserved,hot_frames=hot,KV_bytes=KV,manager_tensor_bytes=tensors,
              normal_bound_bytes=normal,recovery_bound_bytes=recovery,final_bound_bytes=final,
              legal_bound_bytes=max(normal,recovery,final))
 table={a:[bound(a,n) for n in range(0 if a=='H' else 4,92 if a=='H' else 96)] for a in ('H','W')}
 chosen={a:max((r for r in table[a] if r['legal_bound_bytes']<=B),key=lambda r:r['capacity']) for a in table}
 rec=dict(locked_utc=d.now(),before_GT_scoring=True,all12_S_complete=True,S_actual_peak_bytes=max(peaks),B_quantum_bytes=p['B_quantum'],B_bytes=B,hard_max_bytes=p['history_hard_max'],
          capacity={a:r['capacity'] for a,r in chosen.items()},chosen=chosen,
          next_integer={a:next((r for r in table[a] if r['capacity']==chosen[a]['capacity']+1),None) for a in table},
          nonspatial_nonmanager_fixed_tensor_bytes=fixed_tensor,nonspatial_nonmanager_Python_bytes=fixed_headers,
          readable_all12_compressed_RGB_bytes=RGB,metadata_and_file_header_allowance_bytes=metadata,
          calculation_scope='Maximal integer under explicit normal/complete-recovery/permanent-write lifecycle envelope; no GT, model predictions, capacity benchmark or scientific quota scan. Observed actual peaks and unused B reported separately. Every reserved frame is reachable within the 96-frame horizon; no horizon padding.',
          exact_normal_arithmetic='KV=2*24*16*reserved*1041*64*2; token IDs=8*24*16*reserved*1041; H scores/count=8*24*16*reserved*1041. H all-layer gather and retained selection arrays; W one-layer gather.',
          recovery='Alias-preserving in-place CPU migration and CPU-only serialization; GPU restore retains disk until first successful new chunk. Full file/live state plus largest copied storage charged. Common to S/H/W; no uncounted recovery backing.',
          no_effect_read=True,training=0,normal_full_history_H_W_identical=chosen['H']['capacity']==91 and chosen['W']['capacity']==95)
 d.write(d.OUT/'capacity_lock.json',rec)
 return rec
if __name__=='__main__':print(lock_capacity())
