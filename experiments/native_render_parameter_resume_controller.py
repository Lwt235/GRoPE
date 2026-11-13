"""Pause before original queue.main; resource admission controls the single start."""
import fcntl,os,signal,time
from experiments import native_render_parameter_resource_sidecar as m

def main():
 with (m.OUT/'queue.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  m.write(m.FOLDER/'replacement_controller_identity.json',dict(identity=m.identity(os.getpid()),epoch=time.time(),source_sha256=m.sha(__file__),calls_unchanged_original_queue=True))
  os.kill(os.getpid(),signal.SIGSTOP)
 # Release the temporary admission lock just before unchanged queue acquires it.
 from experiments.native_render_parameter_queue import main as queue_main
 queue_main()

if __name__=='__main__':main()
