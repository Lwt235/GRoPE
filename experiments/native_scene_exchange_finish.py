"""Authorized finite post-fit stages; stops for actual plot review, no successor."""
import os,subprocess,time
from experiments import native_scene_exchange_data as d

def main():
    status=d.OUT/'finish_status.json';start=time.time()
    while True:
        workers=d.formal_worker_records()
        if all(w.get('actual_process_wait_returned') for w in workers):
            assert all(w['exit_code']==0 for w in workers),workers;break
        d.write(status,dict(stage='waiting_for_actual_formal_process_exit',pid=os.getpid(),updated_utc=d.now(),fits_complete=len(list((d.OUT/'fits').glob('*_complete.json'))),predictions=len(list((d.OUT/'completed').glob('*.json'))) if (d.OUT/'completed').exists() else 0));time.sleep(15)
    def run(args,label):
        d.write(status,dict(stage=label,pid=os.getpid(),updated_utc=d.now()))
        code=subprocess.run(['/home/liwt/miniColmap/.venv/bin/python','-u','-m',*args],cwd=d.ROOT).returncode
        assert code==0,(label,code)
    run(['experiments.native_scene_exchange_check','--complete'],'completion_checks')
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    available=[x.split(',')[0].strip() for x in gpus.splitlines() if int(x.split(',')[1])<256 and int(x.split(',')[2])>30*1024];assert available,'No idle card for necessary full-query costs; preserve and report, no interference.'
    run(['experiments.native_scene_exchange_cost_launch',available[0]],'finite_actual_full_query_cost')
    run(['experiments.native_scene_exchange_score'],'independent_scores_and_fixed_bootstrap')
    run(['experiments.native_scene_exchange_figures'],'final_figures')
    d.write(status,dict(stage='all_experiment_score_cost_figures_complete_pending_actual_view_image_and_final_package',pid=os.getpid(),updated_utc=d.now(),wall_seconds=time.time()-start,new_predictions=14784,all_GPU_workers_actual_exit=True))
    print(d.read(status),flush=True)

if __name__=='__main__':main()
