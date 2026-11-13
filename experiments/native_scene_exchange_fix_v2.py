"""One documented pixel-center interface correction; no scientific changes."""
import shutil
from experiments import native_scene_exchange_data as d

def main():
    assert all(d.read(p)['exit_code']==75 and d.read(p)['actual_process_wait_returned'] for p in (d.OUT/'workers').glob('formal*_v1.json'))
    failed=d.OUT/'failed/v1';failed.mkdir(parents=True,exist_ok=True)
    for name in ('fits','calls','preparation'):
        assert not (failed/name).exists();shutil.move(d.OUT/name,failed/name)
    shutil.copy2(d.OUT/'preflight.json',failed/'preflight.json');shutil.copy2(d.OUT/'preparation_complete.json',failed/'preparation_complete.json')
    assert not (d.CACHE/'failed/v1_checkpoints').exists();(d.CACHE/'failed').mkdir(exist_ok=True);shutil.move(d.CACHE/'checkpoints',d.CACHE/'failed/v1_checkpoints')
    for p in sorted((failed/'preparation').glob('*.json')):
        r=d.read(p);path=d.Path(r['path']);v=d.load(path);v['scene_pos']-=.5;d.save(path,v);r['prior_sha256']=r['sha256'];r['sha256']=d.sha(path);r['engineering_version']='v2_pixel_center_alignment';d.write(d.OUT/'preparation'/p.name,r)
    rows=[d.read(p) for p in sorted((d.OUT/'preparation').glob('*.json'))];d.write(d.OUT/'preparation_complete.json',dict(complete=True,views=rows,raster_calls=352,raster_repeated_for_fix=0,source_encoding=0,source_optimization=0,new_fields=0))
    d.identity_config();steps={p.stem:d.json.loads(p.read_text().splitlines()[-1])['step'] for p in (failed/'fits').glob('*.jsonl')};d.journal(d.OUT/'engineering_versions.jsonl',dict(utc=d.now(),version='v2',issue='Physical patch center i+.5 must map to native RoPE index i+1; scene projected position changed from pixel+1 to pixel+.5. Local grid already correct.',discarded_formal_steps=steps,discarded_formal_samples=2*sum(steps.values()),recovery_paths=str(d.CACHE/'failed/v1_checkpoints'),source_scene_or_raster_rebuilt=False,science_conditions_changed=False,paired_initialization_reset=True))
    print(dict(corrected_views=len(rows),discarded_steps=steps),flush=True)

if __name__=='__main__':main()
