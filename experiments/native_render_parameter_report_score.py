"""Reporting entry uses the locked collector/summarizer with the local JSON writer."""
import time
import torch
from experiments.native_render_parameter_data import OUT,write
from experiments import native_render_parameter_score as score


def main():
    torch.set_num_threads(4);begin=time.monotonic();rows,audit=score.collect()
    write(OUT/'scoring/metrics.json',dict(rows=rows));write(OUT/'scoring/independent_recomputation.json',audit)
    write(OUT/'scoring/summary.json',score.summarize(rows));write(OUT/'scoring/runtime.json',dict(complete=True,seconds=time.monotonic()-begin,new_predictions=6336,inherited_predictions=4224))
    print(dict(complete=True,new=6336,inherited=4224,independent_max_abs=audit['new_max_abs_metric_difference'],seconds=time.monotonic()-begin),flush=True)


if __name__=='__main__':main()
