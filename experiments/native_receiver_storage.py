"""Owned RAM-backed transient resumes and verified persistent final weights."""
import argparse
import os
import shutil
from experiments.native_receiver_data import *
from experiments.native_receiver_orchestrate import jobs,done

RAM=Path('/dev/shm')/f'liwt_native_receiver_adaptation1_{os.getuid()}'


def owned_ram():
    RAM.mkdir(mode=0o700,exist_ok=True)
    assert RAM.stat().st_uid==os.getuid() and RAM.resolve().parent==Path('/dev/shm')
    return RAM


def migrate():
    # This mode is run only after verified own CUDA workers have exited.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            args=process.joinpath('cmdline').read_bytes().split(b'\0')
            if process.stat().st_uid!=os.getuid() or process.joinpath('cwd').resolve()!=ROOT:continue
            state=process.joinpath('stat').read_text().split(') ',1)[1].split()[0]
            if state!='Z' and any(m in args for m in [b'experiments.native_receiver_train',b'experiments.native_receiver_prepare',b'experiments.native_receiver_cost']):
                raise RuntimeError('Own CUDA worker still active; refuse checkpoint-directory migration')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    owned_ram();assert shutil.disk_usage(RAM).free>80*1024**3
    records=[];parent=CACHE/'checkpoints';parent.mkdir(parents=True,exist_ok=True)
    backup=CACHE/'checkpoints_before_ram';backup.mkdir(exist_ok=True)
    for seed in SEEDS:
        for kind in KINDS:
            name=f'{kind}_{seed}';link=parent/name;ram=RAM/name;ram.mkdir(exist_ok=True)
            if link.is_symlink():
                assert link.resolve()==ram.resolve()
                records.append(dict(name=name,active_link=str(link),ram_directory=str(ram),already_migrated=True))
                continue
            old=backup/name
            if link.exists():
                assert not old.exists();link.rename(old)
            copied=[]
            for filename in ['latest.pt','step_2000.pt']:
                src=old/filename
                if not src.exists():continue
                target=ram/filename;begin=time.monotonic();temporary=target.with_suffix('.copying')
                shutil.copyfile(src,temporary)
                expected=sha256(src);assert sha256(temporary)==expected
                temporary.replace(target)
                copied.append(dict(filename=filename,source_sha256=expected,bytes=target.stat().st_size,seconds=time.monotonic()-begin))
                print(dict(migrated=name,file=filename,bytes=target.stat().st_size),flush=True)
            link.symlink_to(ram,target_is_directory=True)
            retired=[]
            for filename in ['latest.tmp','step_2000.tmp']:
                tmp=old/filename
                if tmp.exists():
                    retired.append(dict(path=str(tmp),bytes=tmp.stat().st_size,
                        reason='Own interrupted atomic-write temporary, valid preceding checkpoint preserved and RAM copy verified.'))
                    tmp.unlink()
            records.append(dict(name=name,original_directory=str(old),active_link=str(link),ram_directory=str(ram),copied=copied,retired_temporaries=retired))
    write_json(OUT/'storage_migration.json',dict(complete=True,records=records,ram_root=str(RAM),
        persistent_backup=str(backup),science_changed=False,
        scope='Same exact FP32 native/fusion/Adam tensors, path-transparent symlinks. RAM resumes are transient across host reboot; earlier persistent snapshots retained until verified final archive.'))


def persist():
    owned_ram();completed=[];definitions={j['name']:j for j in jobs() if 'kind' in j}
    while len(completed)<9:
        for seed in SEEDS:
            for kind in KINDS:
                name=f'{kind}_{seed}'
                if name in completed:continue
                marker=OUT/'persistence'/f'{name}.json'
                if marker.exists() and read(marker).get('complete'):
                    completed.append(name);continue
                status_path=OUT/name/'status.json'
                if not status_path.exists():continue
                status=read(status_path)
                if not status['complete']:continue
                ram=RAM/name;source=ram/'step_2000.pt'
                archive=CACHE/'final_archive'/name;archive.mkdir(parents=True,exist_ok=True)
                final=archive/'step_2000.pt'
                if not final.exists():
                    assert sha256(source)==status['checkpoint_sha256']
                    tmp=archive/'step_2000.copying';begin=time.monotonic();shutil.copyfile(source,tmp)
                    assert sha256(tmp)==status['checkpoint_sha256'];tmp.replace(final)
                    write_json(OUT/'persistence_progress'/f'{name}.json',dict(copied_verified=True,
                        archive=str(final),sha256=status['checkpoint_sha256'],seconds=time.monotonic()-begin))
                assert sha256(final)==status['checkpoint_sha256']
                if not done(definitions[name]):continue
                # Atomic symlink switch: old and new opens both see identical bytes.
                link=CACHE/'checkpoints'/name;assert link.is_symlink()
                assert link.resolve() in (ram.resolve(),archive.resolve())
                replacement=link.with_name(name+'.next')
                if replacement.is_symlink():replacement.unlink()
                replacement.symlink_to(archive,target_is_directory=True);replacement.replace(link)
                write_json(marker,dict(complete=True,name=name,persistent_path=str(final),
                    checkpoint_sha256=status['checkpoint_sha256'],bytes=final.stat().st_size,
                    active_path=str(link/'step_2000.pt'),active_path_now_persistent=True,
                    same_final_bytes_as_evaluation=True))
                # Only this experiment's now redundant transient/old optimizer files.
                if ram.exists():
                    for file in ram.iterdir():
                        assert file.is_file()
                        # A cost reader may still be resolving the previous
                        # symlink target. Keep that immutable final until all
                        # GPU queries finish; old/new opens then both succeed.
                        if file.name!='step_2000.pt':file.unlink()
                old=CACHE/'checkpoints_before_ram'/name
                if old.exists():
                    for file in old.iterdir():
                        assert file.name in ['latest.pt','latest.tmp','step_2000.pt','step_2000.tmp'] and file.is_file()
                        file.unlink()
                    old.rmdir()
                completed.append(name);print(dict(persisted=name,sha256=status['checkpoint_sha256']),flush=True)
        write_json(OUT/'persistence_status.json',dict(complete=len(completed)==9,completed=completed,
            pending=[n for n in definitions if n not in completed],updated_epoch=time.time()))
        if len(completed)<9:time.sleep(20)
    while not all(done(j) for j in jobs()):time.sleep(5)
    for directory in list(RAM.iterdir()):
        assert directory.is_dir() and directory.name in definitions
        for file in directory.iterdir():
            assert file.is_file();file.unlink()
        directory.rmdir()
    write_json(OUT/'persistence_complete.json',dict(complete=True,finals=9,transient_states_retired=True,
        persistent_paths=[read(OUT/'persistence'/f'{n}.json')['persistent_path'] for n in completed]))
    if RAM.exists() and not list(RAM.iterdir()):RAM.rmdir()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--migrate',action='store_true');args=parser.parse_args()
    if args.migrate:migrate()
    else:
        import fcntl
        with (OUT/'persistence.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            (OUT/'persistence.pid').write_text(str(os.getpid())+'\n')
            persist()
