"""Extend the identical sensor/joint-validity source recipe, never using queries."""
from experiments.native_coverage_data import *


@torch.no_grad()
def main():
    import experiments.gaussian_scene_anysplat as official
    from experiments.gaussian_scene_build import register_source_cameras
    from experiments.source_geometry_data import warp_sensor
    p=plan();torch.set_num_threads(4)
    # Isolate the official loader's runtime report; checkpoint identity is unchanged.
    (OUT/'setup').mkdir(exist_ok=True)
    identity=read(ROOT/'results/gaussian_memory/gaussian_scene_memory1/setup/anysplat_weights.json')
    write_json(OUT/'setup/anysplat_weights.json',identity);official.OUT=OUT
    model=official.model_load()
    for meta in p['added_training_catalog']:
        rid=meta['id'];marker=OUT/'source_predictions'/f'{rid}.json'
        if marker.exists():
            r=read(marker);assert sha256(source_pair_path(rid))==r['source_pair_sha256'];continue
        source=load(source_rgb_path(rid));images=source['rgb'][None].cuda().float()/255.
        torch.cuda.synchronize();begin=__import__('time').monotonic()
        output=model.encoder(images,global_step=0,visualization_dump=None)
        raw_depth=output.depth_dict['depth'][0].float().cpu()
        cameras=output.pred_context_pose['extrinsic'][0].float().cpu()
        scale,rotation,translation,diagnostic=register_source_cameras(cameras,source['c2w'])
        unit=scale*raw_depth.median();pred=raw_depth.permute(0,3,1,2)[:,0]*scale
        sensor=[];valid=[];files={}
        for q in p['sources']:
            f=meta['row']['frames'][q];dep=depth_source(f);h=sha256(dep)
            if 'depth_sha256' in f:assert h==f['depth_sha256']
            _,transform=transformed_rgb(f['rgb_path']);d,v=warp_sensor(dict(f,depth_path=str(dep)),transform)
            sensor.append(d);valid.append(v);files[str(dep)]=h
        sensor=torch.stack(sensor);valid=torch.stack(valid)
        common=valid&torch.isfinite(pred)&(pred>.1)&(pred<10)
        pair=dict(source,depth_pred=pred.float(),depth_sensor=sensor,common_valid=common,unit=unit)
        save_tensor(source_pair_path(rid),pair)
        write_json(marker,dict(complete=True,id=rid,source_only=True,queries_used=False,source_ids=p['sources'],
            source_payload_sha256=sha256(source_rgb_path(rid)),source_pair_sha256=sha256(source_pair_path(rid)),
            source_sensor_files=files,registration=diagnostic,unit=float(unit),common_valid_fraction=float(common.float().mean()),
            compact_saved_fields='Paired source payload contains all fields consumed by unchanged sensor construction; unused dense AnySplat Gaussians not serialized.',
            seconds=__import__('time').monotonic()-begin))
        print(dict(source_prediction=rid),flush=True)
        del output,images,pair,raw_depth,pred;torch.cuda.empty_cache()


if __name__=='__main__':main()
