"""Prospective, synthetic-only audit of an unselected Gaussian dual PE reserve."""
from __future__ import annotations

import builtins
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

START = time.perf_counter()
ROOT = Path(__file__).resolve().parent
BEFORE = json.loads((ROOT / "before_execution.json").read_text())
SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
assert SOURCE_HASH == BEFORE["source_sha256"]
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    assert os.environ.get(name) == "4", name

ORIGINAL_IMPORT = builtins.__import__
BLOCKED_IMPORT_ROOTS = {"models", "experiments", "torch", "tensorflow"}
GUARD_EVENTS = []


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] in BLOCKED_IMPORT_ROOTS:
        GUARD_EVENTS.append({"kind": "blocked_import", "name": name})
        raise RuntimeError("synthetic audit forbids model/framework imports")
    return ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


builtins.__import__ = guarded_import
import numpy as np
from threadpoolctl import threadpool_info


def forbidden_load(*args, **kwargs):
    GUARD_EVENTS.append({"kind": "blocked_numpy_load"})
    raise RuntimeError("synthetic audit forbids np.load")


np.load = forbidden_load


def write_once(name, value):
    with (ROOT / name).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def svec(matrix):
    indices = np.triu_indices(matrix.shape[0])
    return matrix[indices] * np.where(indices[0] == indices[1], 1.0, math.sqrt(2.0))


def dual(mean, covariance):
    precision = np.linalg.solve(covariance, np.eye(mean.size))
    theta = np.concatenate((precision @ mean, -0.5 * svec(precision)))
    eta = np.concatenate((mean, svec(covariance + np.outer(mean, mean))))
    return theta, eta


def kl(mean_a, cov_a, mean_b, cov_b):
    delta = mean_b - mean_a
    sign_a, logdet_a = np.linalg.slogdet(cov_a)
    sign_b, logdet_b = np.linalg.slogdet(cov_b)
    assert sign_a == sign_b == 1
    return 0.5 * (np.trace(np.linalg.solve(cov_b, cov_a))
                  + delta @ np.linalg.solve(cov_b, delta)
                  - mean_a.size + logdet_b - logdet_a)


def jeffreys_fraction(query_var, key_var, delta):
    return (key_var / query_var + query_var / key_var - 2
            + delta * delta * (1 / query_var + 1 / key_var)) / 2


def fraction_string(value):
    return f"{value.numerator}/{value.denominator}"


def main():
    # These deliberate probes must stop before any archive or model is opened.
    for probe in (lambda: np.load("forbidden_probe.npz"),
                  lambda: __import__("models.synthetic_guard_probe")):
        try:
            probe()
        except RuntimeError:
            pass
        else:
            raise AssertionError("guard probe did not fail closed")

    parent_j = jeffreys_fraction(F(1), F(1, 3), F(0))
    child_j = jeffreys_fraction(F(1), F(1, 12), F(1, 2))
    exponent_gap = (child_j - parent_j) / 2
    assert parent_j == F(2, 3)
    assert child_j == F(20, 3)
    assert exponent_gap == F(3)

    rng = np.random.default_rng(BEFORE["fixed_input_rules"]["seed"])
    rows = []
    for dimension in (2, 3):
        for case_index in range(8):
            mean_i = rng.normal(size=dimension)
            mean_j = rng.normal(size=dimension)
            factor_i = rng.normal(size=(dimension, dimension))
            factor_j = rng.normal(size=(dimension, dimension))
            cov_i = factor_i @ factor_i.T + 0.5 * np.eye(dimension)
            cov_j = factor_j @ factor_j.T + 0.5 * np.eye(dimension)
            left, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
            right, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
            affine = left @ np.diag(rng.uniform(0.5, 2.0, dimension)) @ right.T
            translation = rng.normal(size=dimension)

            theta_i, eta_i = dual(mean_i, cov_i)
            theta_j, eta_j = dual(mean_j, cov_j)
            query = np.concatenate((theta_i / math.sqrt(2.0),
                                    eta_i / math.sqrt(2.0), [1.0]))
            key = np.concatenate((eta_j / math.sqrt(2.0),
                                  theta_j / math.sqrt(2.0), [-0.5 * theta_j @ eta_j]))
            direct_j = kl(mean_i, cov_i, mean_j, cov_j) + kl(mean_j, cov_j, mean_i, cov_i)
            dual_j = (theta_i - theta_j) @ (eta_i - eta_j)
            expanded_score = query @ key - 0.5 * theta_i @ eta_i
            transformed_i = affine @ mean_i + translation
            transformed_j = affine @ mean_j + translation
            transformed_ci = affine @ cov_i @ affine.T
            transformed_cj = affine @ cov_j @ affine.T
            affine_j = (kl(transformed_i, transformed_ci, transformed_j, transformed_cj)
                        + kl(transformed_j, transformed_cj, transformed_i, transformed_ci))
            errors = {"natural_moment_identity": abs(float(direct_j - dual_j)),
                      "qk_score_identity": abs(float(-0.5 * direct_j - expanded_score)),
                      "common_affine_invariance": abs(float(direct_j - affine_j))}
            assert query.size == key.size == (11 if dimension == 2 else 19)
            assert direct_j >= -BEFORE["absolute_tolerance"]
            assert all(math.isfinite(value) and value <= BEFORE["absolute_tolerance"]
                       for value in errors.values())
            rows.append({"dimension": dimension, "case_index": case_index,
                         "query_key_channels": int(query.size), "direct_jeffreys": float(direct_j),
                         "errors": errors})

    pools = threadpool_info()
    assert pools and all(pool["num_threads"] == 4 for pool in pools)
    assert len(GUARD_EVENTS) == 2
    assert SOURCE_HASH == digest(Path(__file__))
    result = {"status": "passed_limited_synthetic_checks", "source_sha256": SOURCE_HASH,
              "exact_partition_counterexample": {
                  "parent_jeffreys": fraction_string(parent_j),
                  "child_jeffreys": fraction_string(child_j),
                  "child_minus_parent_negative_half_j_exponent_gap": fraction_string(exponent_gap),
                  "child_sum_over_parent_mass": "exp(-3)"},
              "synthetic_case_count": len(rows), "rows": rows,
              "max_errors": {name: max(row["errors"][name] for row in rows)
                             for name in rows[0]["errors"]},
              "runtime": {"python": sys.version, "numpy": np.__version__,
                          "threadpools": pools, "wall_seconds": time.perf_counter() - START},
              "guard_events": GUARD_EVENTS,
              "scope": {"real_arrays_read": False, "labels_read": False, "model_calls": 0,
                        "candidate_selected": False, "round39": False,
                        "r38_changed": False, "scientific_goal_passed": False,
                        "general_gaussian_pe_impossibility": False}}
    write_once("results.json", result)
    write_once("completion.json", {"status": "complete", "exit_code": 0,
                                   "files": {name: digest(ROOT / name)
                                             for name in ("check.py", "before_execution.json", "results.json")},
                                   "source_unchanged": True, "scope": result["scope"]})
    print(json.dumps({"status": result["status"], "cases": len(rows),
                      "max_errors": result["max_errors"], "wall_seconds": result["runtime"]["wall_seconds"]}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write_once("failure.json", {"exception": type(error).__name__, "message": str(error),
                                    "source_sha256": SOURCE_HASH, "guard_events": GUARD_EVENTS,
                                    "wall_seconds": time.perf_counter() - START})
        raise
