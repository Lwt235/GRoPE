"""Use another idle A6000 when GPU 5 is taken before any test starts."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import time

from experiments import native_speed_transition_gpu5 as transition
from experiments.round65_common import read, write, sha


def main(uuid):
    out = transition.OUT
    old = read(out/'status.json')
    assert old['active_child'] is None
    controller = old['process']
    running = transition.live(controller)
    if running:
        assert old['state'] == 'waiting_exclusive_fusion_correctness' and not old['stages']
    else:
        assert old['state'] == 'transition_failed' and len(old['stages']) == 1
        failed = old['stages'][0]
        assert failed['exit_code'] == 1
        assert 'Another process occupies this GPU' in Path(failed['log']).read_text()
        assert not (transition.ROOT/'results/efficiency_sidecar/fullbatch1/exclusive').exists()
    cards, apps = transition.resources()
    card = next(c for c in cards if c['uuid'] == uuid)
    assert card['used_mib'] < 512 and card['free_mib'] >= 24576 and card['utilization'] < 10
    assert not any(a['uuid'] == uuid for a in apps)
    assert not (out/'idle_gpu_substitution.json').exists()
    if running:
        os.kill(controller['pid'], signal.SIGSTOP)
        for _ in range(100):
            if transition.identity(controller['pid'])['state'] == 'T': break
            time.sleep(.01)
        held = read(out/'status.json')
        if held['active_child'] is not None or held['stages']:
            os.kill(controller['pid'], signal.SIGCONT)
            raise RuntimeError('Watcher launched work; leave it running, no substitution.')
        assert transition.identity(controller['pid'])['state'] == 'T'
    shutil.copy2(out/'status.json', out/'status_before_gpu_substitution.json')
    write(out/'idle_gpu_substitution.json', dict(time_unix=time.time(),
        reason='GPU 5 was occupied by another user before tests started; use an entirely idle same-model A6000.',
        requested_gpu=transition.GPU5, actual_gpu=card, resources=cards, compute_apps=apps,
        retired_waiting_controller=controller, training_workers_signalled=[],
        benchmark_protocol_unchanged=True, wrapper_sha256=sha(Path(__file__)),
        scheduling_amendment_sha256=sha(out/'gpu5_amendment.json')))
    if running:
        os.kill(controller['pid'], signal.SIGKILL)
        for _ in range(100):
            if not transition.live(controller): break
            time.sleep(.01)
    assert not transition.live(controller)
    (out/'status.json').unlink()
    transition.GPU5 = uuid
    print('EXCLUSIVE_GPU_SUBSTITUTION', card, flush=True)
    transition.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-uuid', required=True)
    main(parser.parse_args().gpu_uuid)
