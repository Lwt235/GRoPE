"""Effect-blind maximum integer capacities under the actual common lifecycle."""
import math
from experiments import stac_budget_data as d

def lock_capacity():
 p=d.plan();rows=[d.read(d.OUT/'completed'/f"S_{m['id']}.json") for m in p['catalog']];assert all(x['complete'] and not x['GT_scored'] for x in rows)
 peaks=[x['peak_history_bytes'] for x in rows]
 for f in (d.OUT/'resources').glob('*.json'):
  x=d.read(f)
  if x.get('peak_history_bytes') is not None:peaks.append(x['peak_history_bytes'])
 for f in (d.OUT/'engineering_failures').rglob('last_worker_failure.json'):
  x=d.read(f)
  if x.get('peak_history_bytes') is not None:peaks.append(x['peak_history_bytes'])
 B=math.ceil(max(peaks)/p['B_quantum'])*p['B_quantum'];assert B<=p['history_hard_max'],'S_history_hard_max'
 inventories=[d.read(f) for f in (d.OUT/'inventory').glob('S_*.json') if 'nonspatial_nonmanager' in d.read(f)];assert inventories
 fixed=max(x['nonspatial_nonmanager']['unique_tensor_bytes'] for x in inventories)
 headers=max(x['nonspatial_nonmanager']['Python_object_bytes'] for x in inventories)+2**20
 RGB=p['compressed_readable_all12_RGB_bytes'];L,H,D,t,b=24,16,64,1041,2;depth=96*392*518*4
 def manager(arm,frames):
  KV=2*L*H*frames*t*D*b;indices=L*H*frames*t*8;scores=L*H*frames*t*8 if arm=='H' else 0
  return KV+indices+scores+2*L*8+t*8
 def bound(arm,n):
  hot=5+n if arm=='H' else 1+n;reserved=min(96,hot+4);KV=2*L*H*reserved*t*D*b
  tensors=manager(arm,reserved);state=fixed+tensors
  # Source prune returns before creating gather/select arrays when live<=hot.
  # In particular hot96 never prunes, even if hot95 has a large final gather.
  if hot>=96:workspace=0
  elif arm=='H':workspace=2*L*H*hot*t*D*b+L*H*hot*t*16+L*H*(5*hot+2*n+reserved)*t*8
  else:workspace=2*H*hot*t*D*b+H*hot*t*8+reserved*t*24
  # Actual backing: minimal chunk numerical files, then stacked final plus file.
  decode_workspace=2*H*reserved*t*D*b+H*reserved*t*16+4*t*H*D*b
  common=RGB+headers;normal=state+common+max(workspace,decode_workspace)+depth
  # Physical reservation is 4 -> 8 -> reachable final capacity after prefix8.
  # Growth replaces one tensor at a time; old8 buffer + final new key is worst.
  grow=fixed+manager(arm,min(8,reserved))+KV//2+common+depth
  # Same fixed first-source prefix8 recovery for S/H/W. Conservative fixed
  # nonmanager full96 bound here only increases the prefix8 estimate.
  prefix8=fixed+manager(arm,min(8,reserved));largest=max(96*3*392*518*4,2*L*H*min(8,reserved)*t*D)
  recovery=2*prefix8+largest+common+depth
  final=state+common+3*depth # existing chunk files + stack + permanent file
  return dict(capacity=n,hot_frames=hot,reserved_frames=reserved,manager_tensor_bytes=tensors,KV_bytes=KV,prune_reachable=hot<96,prune_workspace_bytes=workspace,decode_workspace_bytes=decode_workspace,normal_bound_bytes=normal,growth_bound_bytes=grow,recovery_at_prefix8_bound_bytes=recovery,final_bound_bytes=final,legal_bound_bytes=max(normal,grow,recovery,final))
 table={a:[bound(a,n) for n in range(0 if a=='H' else 4,92 if a=='H' else 96)] for a in ('H','W')}
 chosen={a:max((r for r in table[a] if r['legal_bound_bytes']<=B),key=lambda r:r['capacity']) for a in table}
 rec=dict(version=2,locked_utc=d.now(),before_GT_scoring=True,all12_S_complete=True,S_actual_peak_bytes=max(peaks),B_quantum_bytes=p['B_quantum'],B_bytes=B,hard_max_bytes=p['history_hard_max'],capacity={a:r['capacity'] for a,r in chosen.items()},chosen=chosen,integer_lifecycle_table=table,nonspatial_nonmanager_fixed_tensor_bytes=fixed,nonspatial_nonmanager_Python_bytes=headers,readable_all12_compressed_RGB_bytes=RGB,
 calculation_scope='All finite96 integer capacities considered before GT; no effect/model benchmark. Actual common first-source prefix8 recovery; 4->8->reachable final physical reserve. No hypothetical full96 recovery applied only to controls. No gather charged when actual policy never prunes. Real subsequent resource-yield recovery remains subject to B, not assumed free.',
 recovery='Same prefix8 full exact recovery performed for each arm. Any later resource-yield state must independently obey B; failure is reported rather than silently changing capacity.',
 no_effect_read=True,training=0,normal_full_history_H_W_identical=chosen['H']['capacity']==91 and chosen['W']['capacity']==95,full_history_note='Both retain all96 without pruning if true; output attention identical source kernel, H still maintains policy scores and its extra cost is reported. Conditions may be merged before effects only after actual interface identity is established.')
 d.write(d.OUT/'capacity_lock.json',rec);return rec
if __name__=='__main__':print(lock_capacity())
