"""Observe an existing native-GS experiment and download; never launch jobs.

Writes only the replaceable monitoring snapshot, outside frozen experiment
artifacts. Run on the host so /proc can identify the actual live processes.
"""
import argparse
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'results/round65_actual_gs/pilot3'
DOWNLOAD = ROOT / 'results/round65_actual_gs/preparation/acquisition_1793682705671116425'


def read(path):
    return json.loads(path.read_text())


def process(pid, expected_ticks=None):
    directory = Path('/proc') / str(pid)
    try:
        # The comm field can contain spaces; fields after its final ')' start
        # at process state (field 3), so starttime (field 22) has offset 19.
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
        return dict(pid=pid, visible=True, exists=True, state=fields[0],
                    start_ticks=fields[19], command=command,
                    identity_matches=(fields[19] == str(expected_ticks)) if expected_ticks else None)
    except FileNotFoundError:
        return dict(pid=pid, visible=True, exists=False)
    except PermissionError:
        return dict(pid=pid, visible=False, exists=None)


def snapshot(pilot=PILOT):
    queue = read(pilot / 'queue_status.json')
    live = [process(queue['pid'], queue['start_ticks']), process(99977, '1396096550')]
    live += [process(row['pid']) for row in queue.get('active', [])]
    runs = []
    for path in sorted((pilot / 'runs').glob('*/*/history.json')):
        history = read(path)
        training, validation = history['training'], history['validation']
        runs.append(dict(role=path.parent.parent.name, seed=int(path.parent.name),
            epochs=len(training), steps=training[-1]['step'] if training else 0,
            last_loss=training[-1]['loss'] if training else None,
            finite=all(math.isfinite(row[key]) for row in training for key in ('loss', 'gradient_norm')),
            validation_finite=all(math.isfinite(row[key]) for row in validation for key in ('ce', 'oa', 'mca')),
            latest_validation={key: validation[-1][key] for key in ('epoch', 'ce', 'mca')} if validation else None,
            completed=(path.parent / 'run.json').exists(), failure=(path.parent / 'failure.json').exists()))
    now = time.time()
    progress = read(DOWNLOAD / 'progress.json')
    progress_age = now-progress['unix']
    partials = []
    if progress_age > 120:
        raw = Path(read(DOWNLOAD / 'plan.json')['raw_root'])
        for path in raw.rglob('*.part'):
            try:
                stat = path.stat()
            except FileNotFoundError:  # A completed download may just have renamed it.
                continue
            partials.append(dict(file=str(path.relative_to(raw)), bytes=stat.st_size,
                                 last_write_age_seconds=max(0., now-stat.st_mtime)))
        partials.sort(key=lambda row: row['last_write_age_seconds'])
    return dict(observed_unix=now, experiment=str(pilot), live_processes=live,
        queue_state=queue['state'], queue_heartbeat_age_seconds=now-queue['observed_unix'],
        stage=queue.get('stage'), pending_jobs=queue.get('pending', []),
        queue_error=queue.get('error'), completed_children=len(queue.get('completed', [])),
        active_jobs=[dict(stage=row['stage'], job=row['job'], pid=row['pid'], gpu=row['gpu']['index'])
                     for row in queue.get('active', [])], runs=runs,
        download=progress, download_progress_age_seconds=progress_age,
        download_recent_partials=partials[:8],
        download_complete=(DOWNLOAD / 'completion.json').exists(),
        download_failed=(DOWNLOAD / 'failure.json').exists(),
        report_available=(pilot / 'result.json').exists(),
        scientific_interpretation_pending=queue.get('scientific_review_pending', True),
        observation_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=('pilot3', 'content_route1'), default='pilot3')
    args = parser.parse_args()
    pilot = PILOT.parent / args.experiment
    result = snapshot(pilot)
    target = pilot / 'monitor_status.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    print('QUEUE', result['queue_state'], 'completed children', result['completed_children'],
          'heartbeat age', round(result['queue_heartbeat_age_seconds'], 1), 'error', result['queue_error'])
    print('STAGE', result['stage'], 'pending', len(result['pending_jobs']), 'active', result['active_jobs'])
    for row in result['live_processes']:
        print('PROCESS', row['pid'], 'exists', row['exists'], 'state', row.get('state'),
              'identity', row.get('identity_matches'))
    for row in result['runs']:
        print('RUN', row['role'], row['seed'], 'epoch', row['epochs'],
              'finite', row['finite'] and row['validation_finite'], 'failure', row['failure'])
    print('DOWNLOAD', result['download'], 'complete', result['download_complete'],
          'failure', result['download_failed'])
    if result['download_recent_partials']:
        print('IN_PROGRESS_FILES', [(row['file'], row['bytes'], round(row['last_write_age_seconds'], 1))
                                   for row in result['download_recent_partials']])
    print('REPORT', result['report_available'], 'scientific review pending', result['scientific_interpretation_pending'])
