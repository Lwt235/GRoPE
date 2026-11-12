"""Use an idle GPU, or resources released by Center seed3, for fixed measurements."""
import json
import time
from experiments.gaussian_memory_coarse_data import *
from experiments.gaussian_memory_coarse_finish import gpu_action
from experiments.gaussian_memory_coarse_queue import occupancy


def main():
    p=plan();verify(p)
    while True:
        state=read(OUT/'train_queue.json')
        if state['failed']:raise RuntimeError('Training failure must be resolved first')
        current=occupancy()
        if any(current[g][0]<=512 and current[g][1]>=40000 and current[g][2]<=5 for g in ('2','3','4','5','7')):break
        target=OUT/'evaluation/center_20261112.json'
        if target.exists() and read(target)['complete']:
            pid=77969
            try:alive=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z'
            except FileNotFoundError:alive=False
            if not alive:break
        time.sleep(15)
    gpu_action('references');gpu_action('timing')
    note=read(OUT/'cost_measurement_completion_note.json');note['gpu_timing_pending']=False;note['runtime_sha256']=sha256(OUT/'runtime.json');write_json(OUT/'cost_measurement_completion_note.json',note)
    write_json(OUT/'fixed_measurements_complete.json',dict(complete=True,references_sha256=sha256(OUT/'evaluation/references.json'),runtime_sha256=sha256(OUT/'runtime.json'),all_adapted_results_required_before_comparison=True,efficacy_comparison_not_started=True,plan_sha256=sha256(OUT/'plan.json')))
    print(json.dumps({'fixed_references_and_costs_complete':True,'all_fit_comparison_still_waits_for_9':True}),flush=True)


if __name__=='__main__':main()
