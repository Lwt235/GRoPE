"""Fixed GK21 driver with scalar truncation and historical vector roundoff.

The GK21 rule and scalar resasc correction are the local SciPy rule before
its scalar roundoff floor. This driver uses positive float64 vectors only.
Its error estimates are not rigorous integration or floating-point bounds.
"""
# GK21 constants/arithmetic adapted from SciPy _quad_vec.py.
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations
import math
from types import SimpleNamespace
import numpy as np

# Identical binary64 nodes and weights to SciPy's _quadrature_gk21.
X = (0.995657163025808080735527280689003,
     0.973906528517171720077964012084452,
     0.930157491355708226001207180059508,
     0.865063366688984510732096688423493,
     0.780817726586416897063717578345042,
     0.679409568299024406234327365114874,
     0.562757134668604683339000099272694,
     0.433395394129247190799265943165784,
     0.294392862701460198131126603103866,
     0.148874338981631210884826001129720, 0.,
     -0.148874338981631210884826001129720,
     -0.294392862701460198131126603103866,
     -0.433395394129247190799265943165784,
     -0.562757134668604683339000099272694,
     -0.679409568299024406234327365114874,
     -0.780817726586416897063717578345042,
     -0.865063366688984510732096688423493,
     -0.930157491355708226001207180059508,
     -0.973906528517171720077964012084452,
     -0.995657163025808080735527280689003)
W = (0.066671344308688137593568809893332,
     0.149451349150580593145776339657697,
     0.219086362515982043995534934228163,
     0.269266719309996355091226921569469,
     0.295524224714752870173892994651338,
     0.295524224714752870173892994651338,
     0.269266719309996355091226921569469,
     0.219086362515982043995534934228163,
     0.149451349150580593145776339657697,
     0.066671344308688137593568809893332)
V = (0.011694638867371874278064396062192,
     0.032558162307964727478818972459390,
     0.054755896574351996031381300244580,
     0.075039674810919952767043140916190,
     0.093125454583697605535065465083366,
     0.109387158802297641899210590325805,
     0.123491976262065851077958109831074,
     0.134709217311473325928054001771707,
     0.142775938577060080797094273138717,
     0.147739104901338491374841515972068,
     0.149445554002916905664936468389821,
     0.147739104901338491374841515972068,
     0.142775938577060080797094273138717,
     0.134709217311473325928054001771707,
     0.123491976262065851077958109831074,
     0.109387158802297641899210590325805,
     0.093125454583697605535065465083366,
     0.075039674810919952767043140916190,
     0.054755896574351996031381300244580,
     0.032558162307964727478818972459390,
     0.011694638867371874278064396062192)
EPSILON = np.finfo(np.float64).eps
TRACE_FIELDS = ('outer_interval_bounds', 'outer_interval_integrals',
                'outer_interval_truncation_error', 'outer_interval_roundoff',
                'outer_active_interval_indices')


def gk21(left, right, function):
    """One rule; return (integral vector, scalar truncation, roundoff vector)."""
    c = .5 * (left + right)
    h = .5 * (right - left)
    if not (math.isfinite(c) and math.isfinite(h) and left < c < right and h > 0):
        raise FloatingPointError('strict GK21 interval midpoint is not representable')
    values = [None] * 21
    sk = 0.0
    sk_abs = 0.0
    for i in range(21):
        value = function(c + h * X[i])
        values[i] = value
        sk += V[i] * value
        sk_abs += V[i] * abs(value)
    sg = 0.0
    for i in range(10):
        sg += W[i] * values[2 * i + 1]
    resasc = 0.0
    average = sk / 2.0
    for i in range(21):
        resasc += V[i] * abs(values[i] - average)
    difference = (sk - sg) * h
    deviation = resasc * h
    integral = h * sk
    roundoff = 50 * EPSILON * h * sk_abs
    if any(not np.isfinite(v).all() for v in (difference, deviation, integral, roundoff)):
        raise FloatingPointError('nonfinite GK21 accumulation')
    error = float(np.max(np.abs(difference)))
    dabs = float(np.max(np.abs(deviation)))
    if dabs != 0 and error != 0:
        error = dabs * min(1.0, (200 * error / dabs) ** 1.5)
    if not math.isfinite(error) or error < 0 or np.any(integral < 0) or np.any(roundoff < 0):
        raise FloatingPointError('invalid nonnegative GK21 rule result')
    return integral, error, roundoff


def quad_vec(function, a, b, *, epsabs, epsrel, norm='max', workers=1,
             quadrature='gk21', points=(), limit=512, full_output=True,
             component_count=None, available=None, metadata=None, work=None):
    """Deterministic single-split adaptive rule; preserve partial traces on error.

    Parent intervals remain active until both children finish. Every completed
    interval, including an orphan child on failure, contributes to history.
    `callback_evaluations` counts attempted callbacks, including one that raises.
    Actual model resource charging remains the caller's independent ledger.
    """
    available = {} if available is None else available
    metadata = {} if metadata is None else metadata
    rows, active = [], []
    count = component_count
    callbacks = splits = 0
    initial_count = 0
    initial_checked = False
    pending = None
    status = 'exception'
    success = False
    summary = dict(sum_active_truncation_error=None, maximum_historical_roundoff=None,
                   estimated_error=None, target=None)
    integral = None

    def checked(value):
        nonlocal callbacks, count
        callbacks += 1
        result = function(value)
        if (not isinstance(result, np.ndarray) or result.dtype != np.float64
                or result.ndim != 1 or not len(result)):
            raise ValueError('GK21 requires a nonempty float64 vector integrand')
        if count is None:
            count = len(result)
        if result.shape != (count,) or not np.isfinite(result).all() or np.any(result < 0):
            raise FloatingPointError('GK21 integrand must be finite nonnegative with fixed dimension')
        return result

    def evaluate(left, right):
        nonlocal pending
        pending = [float(left), float(right)]
        value, error, rounding = gk21(left, right, checked)
        index = len(rows)
        rows.append((float(left), float(right), value.copy(), float(error), rounding.copy()))
        pending = None
        return index

    def aggregate():
        nonlocal integral, summary
        if not rows:
            integral = np.zeros(count if type(count) is int and count > 0 else 0, np.float64)
            summary = dict(sum_active_truncation_error=0., maximum_historical_roundoff=0.,
                           estimated_error=0., target=float(epsabs)/8)
            return
        integral = np.asarray([math.fsum(float(rows[i][2][j]) for i in active)
                               for j in range(count)], np.float64)
        scalar = math.fsum(rows[i][3] for i in active)
        history = np.asarray([math.fsum(float(row[4][j]) for row in rows)
                              for j in range(count)], np.float64)
        rounding = float(np.max(history))
        error = scalar + rounding
        target = max(epsabs, epsrel * float(np.max(np.abs(integral)))) / 8
        if (not np.isfinite(integral).all() or not np.isfinite(history).all()
                or not all(math.isfinite(x) and x >= 0 for x in (scalar, rounding, error, target))):
            raise FloatingPointError('nonfinite component-roundoff global accumulation')
        summary = dict(sum_active_truncation_error=scalar, maximum_historical_roundoff=rounding,
                       estimated_error=error, target=target)

    try:
        if (norm != 'max' or workers != 1 or quadrature != 'gk21' or not full_output
                or type(limit) is not int or limit < 1
                or not math.isfinite(epsabs) or epsabs < 0
                or not math.isfinite(epsrel) or epsrel <= 0
                or not math.isfinite(a) or not math.isfinite(b) or not a < b
                or (count is not None and (type(count) is not int or count < 1))):
            raise ValueError('fixed finite positive-vector GK21 contract required')
        edges = [float(a)] + list(map(float, points)) + [float(b)]
        if any(not math.isfinite(x) for x in edges) or any(x >= y for x, y in zip(edges, edges[1:])):
            raise ValueError('initial depth edges must be finite and strictly increasing')
        initial_count = len(edges) - 1
        if initial_count > limit:
            raise RuntimeError('fixed initial depth-interval budget exceeded')
        for left, right in zip(edges, edges[1:]):
            active.append(evaluate(left, right))
        initial_checked = True
        while True:
            aggregate()
            if summary['estimated_error'] < summary['target']:
                status, success = 'converged', True
                break
            if summary['maximum_historical_roundoff'] >= summary['target']:
                status = 'roundoff_floor'
                break
            if len(active) >= limit:
                status = 'interval_limit'
                break
            parent = min(active, key=lambda i: (-max(rows[i][3], float(np.max(rows[i][4]))), rows[i][0]))
            left, right = rows[parent][:2]
            midpoint = .5 * (left + right)
            if not math.isfinite(midpoint) or not left < midpoint < right:
                raise FloatingPointError('strict refinement midpoint is not representable')
            first = evaluate(left, midpoint)
            second = evaluate(midpoint, right)
            active.remove(parent)
            active.extend((first, second))
            active.sort(key=lambda i: rows[i][0])
            splits += 1
        messages = dict(converged='Target precision reached.',
                        roundoff_floor='Historical component roundoff reaches the fixed target.',
                        interval_limit='Fixed depth-interval limit reached before convergence.')
        info = SimpleNamespace(success=success, status=status, message=messages[status], neval=callbacks,
            intervals=np.asarray([[rows[i][0], rows[i][1]] for i in active], np.float64))
        return integral, summary['estimated_error'], info
    finally:
        # On partial initial construction or an unfinished split, retain exactly
        # the completed rules and the still-active parent, never phantom rows.
        try:
            aggregate()
        finally:
            width = count if type(count) is int and count >= 0 else 0
            available.update(
                outer_interval_bounds=np.asarray([[r[0], r[1]] for r in rows], np.float64).reshape(-1, 2),
                outer_interval_integrals=np.asarray([r[2] for r in rows], np.float64).reshape(len(rows), width),
                outer_interval_truncation_error=np.asarray([r[3] for r in rows], np.float64),
                outer_interval_roundoff=np.asarray([r[4] for r in rows], np.float64).reshape(len(rows), width),
                outer_active_interval_indices=np.asarray(active, np.int64))
            metadata['outer_component_roundoff'] = dict(
                complete=bool(initial_checked and pending is None), success=success, status=status,
                **summary, initial_interval_count=initial_count,
                completed_interval_evaluations=len(rows), split_count=splits,
                callback_evaluations=callbacks, initial_checked_before_split=initial_checked,
                roundoff_epsilon=float(EPSILON), roundoff_factor=50.,
                pending_interval_bounds=pending,
                error_estimate_is_not_true_error_certificate=True)
            if work is not None:
                work['outer_intervals'] = len(active)
