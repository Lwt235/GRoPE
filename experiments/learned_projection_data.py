"""Paths and fixed design for the learned projection PE comparison."""
import copy

import numpy as np
import torch

from experiments.source_geometry_data import (ROOT, DATA, read, load, sha256, query_path,
    patch_path, QUERIES, CONDITIONS, SEEDS, save_tensor, write_json)

OUT = ROOT / 'results/gaussian_memory/learned_projection_pe1'
CACHE = DATA / 'learned_projection_pe_v1'
SOURCE_OUT = ROOT / 'results/gaussian_memory/source_geometry_control1/sensor'
SOURCE_CACHE = DATA / 'source_geometry_control_v1/sensor'
PARENT_OUT = ROOT / 'results/gaussian_memory/gaussian_scene_memory1'
PARENT_CACHE = DATA / 'gaussian_scene_memory_v1'
KINDS = ('P', 'S', 'G')


def plan():
    return read(OUT / 'plan.json')


def geometry_path(rid, query):
    return CACHE / rid / f'geometry_{query}.pt'


def initialize_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'plan.json').exists():
        return plan()
    p = copy.deepcopy(read(SOURCE_OUT / 'plan.json'))
    p.update(experiment='learned_projection_pe1', kinds=list(KINDS), geometry_mode='sensor',
        expected_fits=9, expected_adapted_predictions=864, expected_intervention_predictions=0,
        primary_hypothesis='Retaining learned center relations permits additional projection information to improve native outputs',
        competing_explanations=['center sufficient', 'local content or geometry remains inadequate',
            'scalar projection descriptor insufficient', 'receiver or training objective limits'],
        main_endpoint='missing/roi/centered_log_mse',
        pe=dict(network=[8,16,4], center_inputs=7, extra=dict(P='zero',
            S='log1p(N*w_sphere)', G='log1p(N*w_full)'),
            extra_column_initialization='zero; preserve original seeded common center and reader parameters',
            sphere='PE only: covariance=trace(covariance)*I/3; same means/opacity/content; no rebind',
            storage_dtype='bfloat16, matching autocast input to the PE linear layer'),
        primary_pairs=[['G','P'],['G','S']], secondary_pairs=[['S','P'],['P','parent_L'],['S','parent_L'],['G','parent_L']],
        source_sensor_plan_sha256=sha256(SOURCE_OUT/'plan.json'),
        base_parameters=1054464, pe_parameters=208, center_effective_pe_parameters=192,
        interventions=dict(kinds=[], scope='No repeated full fixed-intervention suite; inherited evidence separately identified'),
        outline=['O2','O3','O4','O5','O6'], choices_changed=['C4'],
        temporary_scope=['source sensor and joint validity privilege', 'given cameras',
            'static four-source scenes', 'frozen backbone and DPT', 'old training/development places'],
        decisions='G>P/S plus net utility supports anisotropic projection; S=G favors simpler scope; recovery without P gain closes projection mapping repair; no L utility leaves receiver/content open. No automatic successor.',
        new_confirmation=False, development_tuning=False)
    write_json(OUT/'plan.json', p)
    return p


def geometry_tensor(payload, kind):
    center = payload['center']
    t = torch.zeros_like(center[..., :1]) if kind == 'P' else payload['sphere' if kind == 'S' else 'full'][..., None]
    return torch.cat((center, t), -1).contiguous()
