"""Single CPU delivery stage; finite scoring/cost/report only, no CUDA/model work."""
import time,os,resource
from experiments import stream_response_data as d
from experiments import stream_response_lineage,stream_response_costs,stream_response_score,stream_response_report

def main():
 begin=time.monotonic();phases=[];d.write(d.OUT/'CPU_progress.json',dict(PID=os.getpid(),phase='starting',utc=d.now()))
 for name,mod in [('lineage',stream_response_lineage),('costs',stream_response_costs),('score',stream_response_score),('report_figures',stream_response_report)]:
  d.write(d.OUT/'CPU_progress.json',dict(PID=os.getpid(),phase=name,utc=d.now()));s=time.monotonic();mod.main();phases.append(dict(phase=name,seconds=time.monotonic()-s))
 d.write(d.OUT/'CPU_complete.json',dict(complete=True,PID=os.getpid(),phases=phases,seconds=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,model_GPU_calls=0,ended_utc=d.now()));print(dict(CPU_complete=True,PID=os.getpid()),flush=True)
if __name__=='__main__':main()
