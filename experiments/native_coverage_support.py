"""Fixed evaluation-only source support for newly added training scenes."""
from experiments.native_coverage_data import *


def main():
    import experiments.gaussian_scene_reference as reference
    p=plan();torch.set_num_threads(4)
    reference.CACHE=CACHE;reference.OUT=OUT;reference.query_path=query_path
    reference.source_path=source_rgb_path
    for meta in p['added_training_catalog']:
        reference.support(meta,dict(p,queries=list(EVAL_QUERIES)))
    write_json(OUT/'support_complete.json',dict(complete=True,added_places=20,evaluation_only=True))


if __name__=='__main__':main()
