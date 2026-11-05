def queue():
    check()
    assert not (HANDOFF / 'execution_identity.json').exists()
    meta = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        source_sha256=sha(Path(__file__)), lock_sha256=sha(OUT / 'lock.json'), maximum_owned_gpu_jobs=8, controller_identity_path=str(HANDOFF / 'execution_identity.json'),
        started_unix=time.time(), resource_dependency=str(REGION), dependency_scientific_success_required=False,
        goal_complete=False)
    write(HANDOFF / 'execution_identity.json', meta)
    active, completed, pending, stage = [], [], [], 'resource_dependency'

    def record(state, **extra):
        write(OUT / 'queue_status.json', dict(meta, state=state, stage=stage,
            active=[item for _, item, _ in active], completed=completed, pending=pending,
            observed_unix=time.time(), **extra))

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def batch(action, jobs, gpu=True):
        nonlocal pending, stage
        pending, stage, previous = list(jobs), action, set()
        while pending or active:
            for process, item, stream in list(active):
                code = process.poll()
                if code is not None:
                    stream.close()
                    active.remove((process, item, stream))
                    completed.append(dict(item, exit_code=code))
                    if code == 75 and action == 'train':
                        pending.append(dict(role=item['role'], seed=item['seed']))
                    elif code:
                        raise RuntimeError(f"{item['log']}: exit {code}")
            if not pending and not active:
                break
            choices = parent.available_gpus() if gpu and pending else [None]
            available = {r['uuid'] for r in choices if r}
            stable, previous = available & previous, available
            busy = {item['gpu']['uuid'] for _, item, _ in active if item['gpu']}
            for device in choices:
                if not pending or len(active) >= (8 if gpu else 1):
                    break
                if device and (device['uuid'] not in stable or device['uuid'] in busy):
                    continue
                env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES=device['uuid'] if device else '')
                probe = None
                if device:
                    code = 'import numpy; import torch; assert torch.cuda.is_available(); x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                    probe = subprocess.check_output([sys.executable, '-c', code], env=env, text=True, timeout=60).strip()
                    if not any(r['uuid'] == device['uuid'] for r in parent.available_gpus()):
                        continue
                job = pending.pop(0)
                command = [sys.executable, '-m', MODULE, action]
                for key, value in job.items():
                    command += ['--' + key, str(value)]
                log = OUT / f'{action}_{job.get("role")}_{job.get("seed")}_{time.time_ns()}.log'
                stream = log.open('x')
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(job, action=action, pid=process.pid, gpu=device, cuda_probe=probe,
                            log=str(log), started_unix=time.time())
                active.append((process, item, stream))
                if device:
                    busy.add(device['uuid'])
                print('LAUNCHED', item, flush=True)
            record('running' if active else 'waiting_for_idle_gpu')
            time.sleep(5)

    try:
        while not region_released():
            record('waiting_for_region_resources')
            time.sleep(10)
        assert read(OUT / 'cuda_preflight.json')['passed']  # Preserve completed preflight.
        jobs = [dict(seed=seed, role=ROLES[(i+j) % 2]) for i, seed in enumerate(SEEDS) for j in range(2)]
        jobs = [job for job in jobs if not (OUT / 'runs' / job['role'] / str(job['seed']) / 'run.json').exists()]
        batch('train', jobs)
        batch('selections', [{}])
        batch('export', [dict(seed=s) for s in SEEDS])
        batch('report', [{}], False)
        stage = 'complete'
        record('complete', scientific_review_pending=True, visual_review_pending=True)
    except BaseException as error:
        for process, item, stream in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            stream.close()
        record('stopped', error=str(error))
        raise
