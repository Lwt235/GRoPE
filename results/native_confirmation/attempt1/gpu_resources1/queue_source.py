def queue():
    lock, plan = checked_seal()
    checked_acquisition(lock)
    with owned_stage('queue'):
        if (OUT / 'result.json').exists():
            raise ValueError('result exists; preserve it for actual review')
        identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                        started_unix=time.time(), evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'),
                        resource_record_sha256=sha(RESOURCE_REV / 'runtime_record.json'),
                        maximum_shared_gpu_jobs=MAX_SHARED_JOBS, allocator_cap_bytes=ALLOCATOR_CAP_BYTES)
        atomic_json(OUT / ('execution_identity_' + str(time.time_ns()) + '.json'), identity)
        pending = [dict(role=m['role'], seed=m['seed']) for m in lock['models']]
        active, completed, previous = [], [], set()

        def status(state, **extra):
            atomic_json(OUT / 'queue_status.json', dict(identity, state=state, pending=pending,
                active=[item for _, item, _ in active], completed=completed, observed_unix=time.time(),
                goal_complete=False, **extra))

        def stop(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        try:
            while pending or active:
                for process, item, stream in list(active):
                    code = process.poll()
                    if code is not None:
                        stream.close()
                        active.remove((process, item, stream))
                        completed.append(dict(item, exit_code=code))
                        if code:
                            raise RuntimeError(f"{item['log']}: exit {code}")
                devices = available_gpus() if pending else []
                available = {d['uuid'] for d in devices}
                stable, previous = available & previous, available
                busy = {item['gpu']['uuid'] for _, item, _ in active}
                for device in devices:
                    if not pending or len(active) >= lock['maximum_owned_gpu_jobs']:
                        break
                    if device['uuid'] not in stable or device['uuid'] in busy:
                        continue
                    if not shared_slot_available(device, active):
                        continue
                    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                        CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                        CUDA_VISIBLE_DEVICES=device['uuid'], PYTHONPATH=str(ROOT))
                    code = 'import numpy; import torch; assert torch.cuda.is_available(); x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                    probe = subprocess.check_output([sys.executable, '-c', code], env=env, text=True, timeout=60).strip()
                    fresh = {d['uuid']: d for d in available_gpus()}
                    if device['uuid'] not in fresh:
                        continue
                    device = fresh[device['uuid']]
                    if not shared_slot_available(device, active):
                        continue
                    job = pending.pop(0)
                    log = OUT / f"export_{job['role']}_{job['seed']}_{time.time_ns()}.log"
                    stream = log.open('x')
                    process = subprocess.Popen([sys.executable, '-m', MODULE, 'export', '--role', job['role'],
                        '--seed', str(job['seed'])], cwd=ROOT, env=env, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    item = dict(job, pid=process.pid, gpu=device, cuda_probe=probe, log=str(log))
                    active.append((process, item, stream))
                    busy.add(device['uuid'])
                    print('LAUNCHED', item, flush=True)
                status('running' if active else 'waiting_for_gpu_headroom')
                if pending or active:
                    time.sleep(5)
            status('reporting')
            final_report()
            status('complete', scientific_review_pending=True, visual_review_pending=True)
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
            status('stopped', error=str(error))
            raise
