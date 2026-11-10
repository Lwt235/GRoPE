# AOT ID: ['1_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/liwt/GRoPE/results/efficiency_sidecar/deployment1/cache/combination/distance_mlp/inductor/2z/c2z4r23kogspc2ukz2dkbb4mrvlswysh6ki5cnthvam6ylvurwfd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_25, [0], True), kwargs = {})
triton_red_fused_sum_0 = async_compile.triton('triton_red_fused_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4D7C73DD31AFBA87844AD567D170FF169B484D1EC1087599863CF60FB70FE14D', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 96)
    x1 = xindex // 96
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 96*r2 + 12288*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/liwt/GRoPE/results/efficiency_sidecar/deployment1/cache/combination/distance_mlp/inductor/4o/c4o26zalhe4i5qpwnstaxw5wseh5nks4tbohy7hff36pstinyjpe.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_25, [0], True), kwargs = {})
triton_per_fused_sum_1 = async_compile.triton('triton_per_fused_sum_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 64},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4D7C73DD31AFBA87844AD567D170FF169B484D1EC1087599863CF60FB70FE14D', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sum_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*r1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /home/liwt/GRoPE/results/efficiency_sidecar/deployment1/cache/combination/distance_mlp/inductor/4i/c4ic6ogoyh4h36nbrlkiylqhnlxk35wazuadf6r2erslykahs2lj.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu, aten.gelu_backward]
# Source node to ATen node mapping:
#   input_2 => add_6, erf, mul_5
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_5,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.5), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %view_22), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, -0.5), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_10,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, 0.3989422804014327), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %mul_11), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_12), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_27, %add_9), kwargs = {})
triton_poi_fused_gelu_gelu_backward_2 = async_compile.triton('triton_poi_fused_gelu_gelu_backward_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4D7C73DD31AFBA87844AD567D170FF169B484D1EC1087599863CF60FB70FE14D', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_gelu_backward_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl_math.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''', device_str='cuda')


# kernel path: /home/liwt/GRoPE/results/efficiency_sidecar/deployment1/cache/combination/distance_mlp/inductor/gz/cgzg2ntvfqlycrq4eie3k6etq62rfkl5nfyeowl67a3hnwq7roas.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_28, [0], True), kwargs = {})
triton_red_fused_sum_3 = async_compile.triton('triton_red_fused_sum_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4D7C73DD31AFBA87844AD567D170FF169B484D1EC1087599863CF60FB70FE14D', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_3(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 192)
    x1 = xindex // 192
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 192*r2 + 24576*x1), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/liwt/GRoPE/results/efficiency_sidecar/deployment1/cache/combination/distance_mlp/inductor/fb/cfbtogvihsqt4od5f73yzxmw4ows4ivgqno7toyfyclub46am2ri.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_28, [0], True), kwargs = {})
triton_per_fused_sum_4 = async_compile.triton('triton_per_fused_sum_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 64},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4D7C73DD31AFBA87844AD567D170FF169B484D1EC1087599863CF60FB70FE14D', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sum_4(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 192*r1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /home/liwt/GRoPE/results/efficiency_sidecar/deployment1/cache/combination/distance_mlp/inductor/f3/cf3wcl45mhpiq2a7dnxupzdjk7l3bo47efzhltzxadct77etton3.py
# Topologically Sorted Source Nodes: [x, layer_norm_1], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => mul_2, sub_2
#   x => add_3
# Graph fragment:
#   %mul_15 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_30, %primals_10), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, 96), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_15, [2], True), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %view_20), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_6), kwargs = {})
#   %mul_2 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %mul_2), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_17, [2], True), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %sum_5), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_16, %sum_4), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_4, %mul_18), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 96), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sub_5), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %mul_19), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': '4D7C73DD31AFBA87844AD567D170FF169B484D1EC1087599863CF60FB70FE14D', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 96
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 96*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + 96*x0), rmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1 + 96*x0), rmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (r1 + 96*x0), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tmp2 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 0.010416666666666666
    tmp21 = tmp12 * tmp20
    tmp22 = 96.0
    tmp23 = tmp2 * tmp22
    tmp24 = tmp23 - tmp6
    tmp25 = tmp13 * tmp18
    tmp26 = tmp24 - tmp25
    tmp27 = tmp21 * tmp26
    tmp28 = tmp19 + tmp27
    tl.store(out_ptr2 + (r1 + 96*x0), tmp28, rmask)
''', device_str='cuda')


# kernel path: /home/liwt/GRoPE/results/efficiency_sidecar/deployment1/cache/combination/distance_mlp/inductor/qg/cqgtappzcfrpfvcgkjjmm4x42du2smjgitr4hypvfykpok4nblmt.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_29,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4D7C73DD31AFBA87844AD567D170FF169B484D1EC1087599863CF60FB70FE14D', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 48)
    x1 = ((xindex // 48) % 512)
    x2 = ((xindex // 24576) % 2)
    x3 = xindex // 49152
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 48*x2 + 96*x1 + 49152*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /home/liwt/GRoPE/results/efficiency_sidecar/deployment1/cache/combination/distance_mlp/inductor/eb/cebrvf47olxbbe6us473ytx3c77zmkvbtqnbufynvu37zvpnqriv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_21 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_37, %div_1), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_21, [-1], True), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div_1,), kwargs = {})
#   %fma : [num_users=1] = call_function[target=torch.ops.prims.fma.default](args = (%neg, %sum_9, %mul_21), kwargs = {})
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%fma, 6.928203230275509), kwargs = {})
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_32,), kwargs = {memory_format: torch.contiguous_format})
triton_per_fused__softmax_backward_data_clone_div_7 = async_compile.triton('triton_per_fused__softmax_backward_data_clone_div_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_clone_div_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '4D7C73DD31AFBA87844AD567D170FF169B484D1EC1087599863CF60FB70FE14D', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_backward_data_clone_div_7(in_ptr0, in_ptr1, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 16384
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = (xindex % 512)
    x3 = ((xindex // 512) % 2)
    x4 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (r1 + 512*x0), None)
    tmp1 = tl.load(in_ptr1 + (r1 + 512*x0), None)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = -tmp1
    tmp7 = libdevice.fma(tmp6, tmp5, tmp2)
    tmp8 = 0.14433756729740646
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr1 + (r1 + 512*x3 + 1024*x2 + 524288*x4), tmp9, None)
    tl.store(out_ptr2 + (r1 + 512*x0), tmp9, None)
''', device_str='cuda')


# kernel path: /home/liwt/GRoPE/results/efficiency_sidecar/deployment1/cache/combination/distance_mlp/inductor/nw/cnwhgzzs4qkgujk3l6wt6evzxrswedag2rsuuy4cjuf4r2444zcw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_49,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_8 = async_compile.triton('triton_poi_fused_clone_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '4D7C73DD31AFBA87844AD567D170FF169B484D1EC1087599863CF60FB70FE14D', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 96) % 3)
    x4 = xindex // 147456
    x0 = (xindex % 48)
    x1 = ((xindex // 48) % 2)
    x3 = ((xindex // 288) % 512)
    x5 = (xindex % 96)
    x6 = xindex
    tmp0 = x4 + 16*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 48*x3 + 24576*(x4 + 16*x2) + 393216*x1), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 48*x3 + 24576*x1 + 49152*(x4 + 16*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 32, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + (x3 + 512*x5 + 49152*((-16) + x4 + 16*x2)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp0 >= tmp11
    tmp16 = tl.full([1], 48, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr3 + (x0 + 48*x3 + 24576*x1 + 49152*((-32) + x4 + 16*x2)), tmp15, other=0.0)
    tmp19 = tl.where(tmp13, tmp14, tmp18)
    tmp20 = tl.where(tmp4, tmp9, tmp19)
    tl.store(out_ptr0 + (x6), tmp20, None)
''', device_str='cuda')


# kernel path: /home/liwt/GRoPE/results/efficiency_sidecar/deployment1/cache/combination/distance_mlp/inductor/fx/cfxjnl6i5edipqxez53r4buxkmvs7jbftrxhm2iqx2htummsc2xh.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_52, [0], True), kwargs = {})
triton_red_fused_sum_9 = async_compile.triton('triton_red_fused_sum_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4D7C73DD31AFBA87844AD567D170FF169B484D1EC1087599863CF60FB70FE14D', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_9(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 288)
    x1 = xindex // 288
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 288*r2 + 36864*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/liwt/GRoPE/results/efficiency_sidecar/deployment1/cache/combination/distance_mlp/inductor/6i/c6iysegwr5knyx7ke2teg7y476vlgimoiov7csn3hvkpgwsj4uqy.py
# Topologically Sorted Source Nodes: [x, layer_norm_1, layer_norm], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm => mul, sub
#   layer_norm_1 => mul_2, sub_2
#   x => add_3
# Graph fragment:
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %view_20), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_6), kwargs = {})
#   %mul_2 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_30, %mul_2), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_20, [0, 1]), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_30, [0, 1]), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %getitem_1), kwargs = {})
#   %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_54, %mul), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_28, [0, 1]), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_54, [0, 1]), kwargs = {})
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_10 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '4D7C73DD31AFBA87844AD567D170FF169B484D1EC1087599863CF60FB70FE14D', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 96)
    x1 = xindex // 96
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 96*r2 + 12288*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 96*r2 + 12288*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 96*r2 + 12288*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r2 + 128*x1), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr4 + (r2 + 128*x1), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr5 + (x0 + 96*r2 + 12288*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr6 + (r2 + 128*x1), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr7 + (r2 + 128*x1), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp3 - tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp12 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp17 = tmp1 - tmp16
        tmp19 = tmp17 * tmp18
        tmp20 = tmp15 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
        tmp24 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp13, xmask)
    tl.store(out_ptr2 + (x3), tmp22, xmask)
    tl.store(out_ptr3 + (x3), tmp25, xmask)
''', device_str='cuda')


# kernel path: /home/liwt/GRoPE/results/efficiency_sidecar/deployment1/cache/combination/distance_mlp/inductor/zr/czrodulcdfa4gdyslqklvxxejt75glytfox5v6oyxleerz4zeoh6.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm, aten.add]
# Source node to ATen node mapping:
#   layer_norm => mul, sub
# Graph fragment:
#   %mul_23 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_54, %primals_2), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, 96), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_23, [2], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %getitem_1), kwargs = {})
#   %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %mul), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_25, [2], True), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %sum_12), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_24, %sum_11), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_7, %mul_26), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 96), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sub_8), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %mul_27), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_11 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '4D7C73DD31AFBA87844AD567D170FF169B484D1EC1087599863CF60FB70FE14D', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 96
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 96*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + 96*x0), rmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r1 + 96*x0), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp9 = tmp7 - tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.010416666666666666
    tmp19 = tmp10 * tmp18
    tmp20 = 96.0
    tmp21 = tmp2 * tmp20
    tmp22 = tmp21 - tmp6
    tmp23 = tmp11 * tmp16
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp26 = tmp17 + tmp25
    tl.store(in_out_ptr0 + (r1 + 96*x0), tmp26, rmask)
''', device_str='cuda')


# kernel path: /home/liwt/GRoPE/results/efficiency_sidecar/deployment1/cache/combination/distance_mlp/inductor/vi/cvi74rblfzm54nmgemfzgp4ayhrxmr5zcbrdtnjdwepxagxswm7l.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_52, [0], True), kwargs = {})
triton_per_fused_sum_12 = async_compile.triton('triton_per_fused_sum_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 64},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4D7C73DD31AFBA87844AD567D170FF169B484D1EC1087599863CF60FB70FE14D', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sum_12(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 288*r1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_10, getitem_1, rsqrt, view, div_1, view_19, addmm_1, getitem_6, rsqrt_1, view_21, addmm_2, view_23, permute_17, permute_21, permute_25, permute_31, permute_33, permute_34, permute_40, permute_41, permute_46, permute_47, permute_50, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, 512, 96), (49152, 96, 1))
    assert_size_stride(primals_2, (96, ), (1, ))
    assert_size_stride(primals_10, (96, ), (1, ))
    assert_size_stride(getitem_1, (16, 512, 1), (512, 1, 1))
    assert_size_stride(rsqrt, (16, 512, 1), (512, 1, 1))
    assert_size_stride(view, (8192, 96), (96, 1))
    assert_size_stride(div_1, (16, 2, 512, 512), (524288, 262144, 512, 1))
    assert_size_stride(view_19, (8192, 96), (96, 1))
    assert_size_stride(addmm_1, (8192, 96), (96, 1))
    assert_size_stride(getitem_6, (16, 512, 1), (512, 1, 1))
    assert_size_stride(rsqrt_1, (16, 512, 1), (512, 1, 1))
    assert_size_stride(view_21, (8192, 96), (96, 1))
    assert_size_stride(addmm_2, (8192, 192), (192, 1))
    assert_size_stride(view_23, (8192, 192), (192, 1))
    assert_size_stride(permute_17, (96, 192), (192, 1))
    assert_size_stride(permute_21, (192, 96), (96, 1))
    assert_size_stride(permute_25, (96, 96), (96, 1))
    assert_size_stride(permute_31, (32, 48, 512), (24576, 1, 48))
    assert_size_stride(permute_33, (8192, 96, 2), (96, 1, 786432))
    assert_size_stride(permute_34, (8192, 512, 96), (49152, 96, 1))
    assert_size_stride(permute_40, (2, 48, 8192), (48, 1, 288))
    assert_size_stride(permute_41, (2, 96, 48), (4608, 1, 96))
    assert_size_stride(permute_46, (32, 48, 512), (24576, 1, 48))
    assert_size_stride(permute_47, (32, 512, 48), (24576, 1, 512))
    assert_size_stride(permute_50, (288, 96), (96, 1))
    assert_size_stride(tangents_1, (16, 512, 96), (49152, 96, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((1, 96, 64), (6144, 1, 96), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_0.run(tangents_1, buf2, 6144, 128, grid=grid(6144), stream=stream0)
        buf3 = empty_strided_cuda((1, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_1.run(buf2, buf3, 96, 64, grid=grid(96), stream=stream0)
        buf1 = empty_strided_cuda((96, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (96, 8192), (1, 96), 0), view_23, out=buf1)
        del view_23
        buf0 = empty_strided_cuda((8192, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (8192, 96), (96, 1), 0), permute_17, out=buf0)
        del permute_17
        buf4 = reinterpret_tensor(buf0, (16, 512, 192), (98304, 192, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu, aten.gelu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_2.run(buf4, addmm_2, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_2
        buf6 = empty_strided_cuda((192, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (192, 8192), (1, 192), 0), view_21, out=buf6)
        del view_21
        buf7 = empty_strided_cuda((1, 192, 64), (12288, 1, 192), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_3.run(buf4, buf7, 12288, 128, grid=grid(12288), stream=stream0)
        buf5 = empty_strided_cuda((8192, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (8192, 192), (192, 1), 0), permute_21, out=buf5)
        del permute_21
        buf8 = empty_strided_cuda((1, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_4.run(buf7, buf8, 192, 64, grid=grid(192), stream=stream0)
        del buf7
        buf15 = empty_strided_cuda((16, 512, 96), (49152, 96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, layer_norm_1], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf5, primals_10, primals_1, addmm_1, getitem_6, rsqrt_1, tangents_1, buf15, 8192, 96, grid=grid(8192), stream=stream0)
        del primals_10
        del tangents_1
        buf17 = empty_strided_cuda((96, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (96, 8192), (1, 96), 0), view_19, out=buf17)
        del view_19
        buf18 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_0.run(buf15, buf18, 6144, 128, grid=grid(6144), stream=stream0)
        buf19 = empty_strided_cuda((1, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_1.run(buf18, buf19, 96, 64, grid=grid(96), stream=stream0)
        buf16 = empty_strided_cuda((8192, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (8192, 96), (96, 1), 0), permute_25, out=buf16)
        del permute_25
        buf20 = empty_strided_cuda((16, 2, 512, 48), (49152, 24576, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf16, buf20, 786432, grid=grid(786432), stream=stream0)
        buf21 = reinterpret_tensor(buf16, (32, 512, 48), (24576, 48, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(div_1, (32, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf20, (32, 512, 48), (24576, 48, 1), 0), out=buf21)
        buf22 = empty_strided_cuda((32, 512, 512), (262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (32, 512, 48), (24576, 48, 1), 0), permute_31, out=buf22)
        del permute_31
        buf24 = empty_strided_cuda((16, 512, 2, 1, 512), (524288, 1024, 512, 512, 1), torch.float32)
        buf29 = empty_strided_cuda((16, 2, 512, 512), (524288, 262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_backward_data_clone_div_7.run(buf22, div_1, buf24, buf29, 16384, 512, grid=grid(16384), stream=stream0)
        del buf22
        del div_1
        buf26 = reinterpret_tensor(buf4, (8192, 2, 96), (192, 96, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (8192, 2, 512), (1024, 512, 1), 0), permute_34, out=buf26)
        del permute_34
        buf27 = empty_strided_cuda((2, 48, 96), (4608, 96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_40, reinterpret_tensor(buf26, (2, 8192, 96), (96, 192, 1), 0), out=buf27)
        del permute_40
        buf28 = reinterpret_tensor(buf20, (2, 8192, 48), (393216, 48, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf26, (2, 8192, 96), (96, 192, 1), 0), permute_41, out=buf28)
        del buf26
        del permute_41
        buf30 = empty_strided_cuda((32, 48, 512), (24576, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_46, reinterpret_tensor(buf29, (32, 512, 512), (262144, 512, 1), 0), out=buf30)
        del permute_46
        buf31 = empty_strided_cuda((32, 512, 48), (24576, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (32, 512, 512), (262144, 512, 1), 0), permute_47, out=buf31)
        del buf29
        del permute_47
        buf32 = empty_strided_cuda((16, 512, 3, 2, 48), (147456, 288, 96, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf28, buf31, buf30, buf21, buf32, 2359296, grid=grid(2359296), stream=stream0)
        del buf21
        del buf28
        del buf30
        buf34 = empty_strided_cuda((288, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (288, 8192), (1, 288), 0), view, out=buf34)
        del view
        buf35 = empty_strided_cuda((1, 288, 64), (18432, 1, 288), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_9.run(buf32, buf35, 18432, 128, grid=grid(18432), stream=stream0)
        buf33 = reinterpret_tensor(buf31, (8192, 96), (96, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (8192, 288), (288, 1), 0), permute_50, out=buf33)
        del buf32
        del permute_50
        buf11 = reinterpret_tensor(buf18, (96, 64), (1, 96), 0); del buf18  # reuse
        buf13 = empty_strided_cuda((96, 64), (1, 96), torch.float32)
        buf39 = empty_strided_cuda((96, 64), (1, 96), torch.float32)
        buf41 = empty_strided_cuda((96, 64), (1, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x, layer_norm_1, layer_norm], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf5, primals_1, addmm_1, getitem_6, rsqrt_1, buf33, getitem_1, rsqrt, buf11, buf13, buf39, buf41, 6144, 128, grid=grid(6144), stream=stream0)
        del addmm_1
        del buf5
        del getitem_6
        del rsqrt_1
        buf43 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_11.run(buf43, buf33, primals_2, primals_1, getitem_1, rsqrt, 8192, 96, grid=grid(8192), stream=stream0)
        del buf33
        del getitem_1
        del primals_1
        del primals_2
        del rsqrt
        buf36 = empty_strided_cuda((1, 288), (288, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_12.run(buf35, buf36, 288, 64, grid=grid(288), stream=stream0)
        del buf35
        buf12 = empty_strided_cuda((96, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_1.run(buf11, buf12, 96, 64, grid=grid(96), stream=stream0)
        del buf11
        buf14 = empty_strided_cuda((96, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_1.run(buf13, buf14, 96, 64, grid=grid(96), stream=stream0)
        del buf13
        buf40 = empty_strided_cuda((96, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_1.run(buf39, buf40, 96, 64, grid=grid(96), stream=stream0)
        del buf39
        buf42 = empty_strided_cuda((96, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_1.run(buf41, buf42, 96, 64, grid=grid(96), stream=stream0)
        del buf41
        buf25 = empty_strided_cuda((8192, 96, 512), (49152, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_33, reinterpret_tensor(buf24, (8192, 2, 512), (1024, 512, 1), 0), out=buf25)
        del buf24
        del permute_33
    return (buf43, buf40, buf42, buf34, reinterpret_tensor(buf36, (288, ), (1, ), 0), reinterpret_tensor(buf27, (96, 96), (96, 1), 0), reinterpret_tensor(buf25, (16, 512, 512, 96), (25165824, 49152, 1, 512), 0), buf17, reinterpret_tensor(buf19, (96, ), (1, ), 0), buf12, buf14, buf6, reinterpret_tensor(buf8, (192, ), (1, ), 0), buf1, reinterpret_tensor(buf3, (96, ), (1, ), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 512, 96), (49152, 96, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((8192, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((16, 2, 512, 512), (524288, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((8192, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((8192, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((8192, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((8192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((8192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_17 = rand_strided((96, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_21 = rand_strided((192, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_25 = rand_strided((96, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    permute_31 = rand_strided((32, 48, 512), (24576, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_33 = rand_strided((8192, 96, 2), (96, 1, 786432), device='cuda:0', dtype=torch.float32)
    permute_34 = rand_strided((8192, 512, 96), (49152, 96, 1), device='cuda:0', dtype=torch.float32)
    permute_40 = rand_strided((2, 48, 8192), (48, 1, 288), device='cuda:0', dtype=torch.float32)
    permute_41 = rand_strided((2, 96, 48), (4608, 1, 96), device='cuda:0', dtype=torch.float32)
    permute_46 = rand_strided((32, 48, 512), (24576, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_47 = rand_strided((32, 512, 48), (24576, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_50 = rand_strided((288, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((16, 512, 96), (49152, 96, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_10, getitem_1, rsqrt, view, div_1, view_19, addmm_1, getitem_6, rsqrt_1, view_21, addmm_2, view_23, permute_17, permute_21, permute_25, permute_31, permute_33, permute_34, permute_40, permute_41, permute_46, permute_47, permute_50, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
