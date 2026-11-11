import os,json,hashlib,subprocess,time
from pathlib import Path
root=Path('/home/liwt/GRoPE'); os.chdir(root)
out=root/'results/storage_cleanup_20261110'
assert not (out/'deleted.jsonl').exists()
tracked=set(subprocess.check_output(['git','ls-files','-z']).decode().split('\0'))
names=['round12_stress','round12_depth_geometry','round15_feasibility','round15_regional','round17_feasibility','round17_validation','round20_anchor_fixed','round21_matched_decomposition']
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
plan=[];kept=[]
for name in names:
 base=root/'results'/name; groups={}
 search=base if name=='round12_depth_geometry' else base/'forward'
 for p in sorted(search.rglob('*')):
  if p.is_symlink() or not p.is_file() or p.suffix not in ('.npy','.npz') or p.stat().st_size<1048576:continue
  rel=p.relative_to(root)
  assert str(rel) not in tracked
  scene=p.relative_to(search).parts[0]
  groups.setdefault(scene,[]).append(p)
 for scene,files in groups.items():
  kept.extend(files[:2]); plan.extend(files[2:])
protected={str(p.relative_to(root)):digest(p) for name in names for p in (root/'results'/name).rglob('*') if p.is_file() and not p.is_symlink() and p.suffix not in ('.npz','.npy')}
(out/'plan.json').write_text(json.dumps({'delete':[str(p.relative_to(root)) for p in plan],'representatives_kept':[str(p.relative_to(root)) for p in kept],'protected_sha256':protected,'policy':'Only >1MiB legacy derived numpy arrays, first two per scene retained; all reports, scores, configurations, logs, model weights and current Gaussian memory/data retained.'},indent=2))
freed=0;started=time.time()
with (out/'deleted.jsonl').open('x',buffering=1) as log:
 for i,p in enumerate(plan):
  stat=p.stat(); record={'path':str(p.relative_to(root)),'bytes':stat.st_size,'allocated_bytes':stat.st_blocks*512,'sha256':digest(p)}
  assert p.stat().st_mtime_ns==stat.st_mtime_ns
  log.write(json.dumps(record)+'\n');log.flush();p.unlink();freed+=record['allocated_bytes']
  if (i+1)%100==0:print(json.dumps({'deleted':i+1,'freed_GiB':freed/2**30}),flush=True)
for p,h in protected.items():assert digest(root/p)==h,p
for name in names:
 (root/'results'/name/'STORAGE_CLEANUP.md').write_text('# Storage cleanup 2026-11-10\n\nLarge derived NumPy arrays were partially removed under user authorization. Reports, scores, configuration, logs and checkpoints remain unchanged. Two representative large arrays per scene/group remain. Exact removed paths, sizes and SHA256 are in `results/storage_cleanup_20261110/deleted.jsonl`; the retention plan is in `plan.json` there. Historical array-level replay requires regenerating removed caches using the original producer code/configuration; this cleanup does not change scientific conclusions.\n')
summary={'complete':True,'deleted_files':len(plan),'freed_allocated_GiB':freed/2**30,'representative_files_retained':len(kept),'protected_files_hash_verified':len(protected),'seconds':time.time()-started}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
