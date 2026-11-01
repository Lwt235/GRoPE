"""Prospective R27 synthetic-only reference revision, frozen before formal evaluation."""
POLICY = dict(h=.25, floor_density=1e-30, quadrature_order=24,
              maximum_tile_side_in_h=2., max_cell_node_evaluations=4_000_000,
              negative_tail_threshold=10., negative_tail_series_terms=20,
              max_negative_tail_relative_remainder=1e-10,
              projection_regularization='rms_depth_xy_gaussian')
REFERENCE_POLICY = dict(orders=[32,48], outer_relative_tolerances=[1e-11,1e-12],
    outer_absolute_tolerance=0., depth_cutoff=18., inner_x_cutoff=32., standard_x_panel_width=2.,
    outer_quadrature='gk21', outer_norm='max', outer_workers=1,
    maximum_depth_evaluations=32768, maximum_depth_intervals=512,
    outer_geometry_breaks=[-8.,-1.,0.,1.,8.], inner_geometry_breaks=[-8.,-1.,0.,1.,8.],
    narrow_cdf_order=16, narrow_cdf_exponent_span=.25,
    max_call_cell_node_evaluations=250_000_000,
    max_total_cell_node_evaluations=600_000_000, max_phase_wall_seconds=1800.)
GATES = dict(probability_max_absolute=1e-4, raw_fov_relative=1e-6,
    raw_l1_over_reference_fov=1e-6, reference_probability_max_absolute=1e-8,
    reference_raw_relative=1e-8, normalization_absolute=2e-12,
    direct_fov_additivity_absolute=1e-8, integrated_identity_absolute=1e-8,
    near_constant_limit_absolute=1e-8, near_constant_limit_raw_relative=1e-8,
    truncation_raw_l1_relative=1e-10, truncation_probability_absolute=1e-8,
    production_max_phase_wall_seconds=600.)
BINDINGS = ('results/round26_rms_root_acceptance_20261031.json',
    'results/round26_rms_post_result_decision_20261031.json',
    'docs/round27_reference_revision_design.md')
SOURCE_FILES = ('ROUND27_RMS_REFERENCE_PREFLIGHT_PROTOCOL.md',
    'docs/round26_rms_quadrature_derivation.md', 'docs/round27_reference_panel_derivation.md',
    'experiments/round27_rms_contract.py', 'experiments/round27_rms_fixtures.py',
    'models/round26_rms_projective_pe.py', 'experiments/round27_rms_preflight.py',
    'experiments/round27_rms_reference.py', 'experiments/round27_rms_audit.py',
    'tests/test_round26_rms_projective_pe.py', 'tests/test_round27_rms_reference.py',
    'tests/test_round27_rms_preflight.py', 'tests/test_round27_rms_audit.py')
