# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU-specific Pallas primitives."""

from __future__ import annotations

from jax._src import core as jax_core
from jax._src import effects
from jax._src import state
from jax._src.state import discharge
from jax._src.lib.mlir.dialects import nvvm as nvvm_dialect
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering
from jax.experimental.mosaic.gpu import dsl as mgpu
from jax._src.interpreters import mlir
import jax.numpy as jnp

async_copy_p = jax_core.Primitive("async_copy")
async_copy_p.multiple_results = True


@async_copy_p.def_effectful_abstract_eval
def _async_copy_abstract_eval(*avals):
  del avals  # Unused.
  return (), {state.ReadEffect(0), state.WriteEffect(1)}


@lowering.register_lowering_rule(async_copy_p)
def _async_copy_lowering_rule(
    ctx: lowering.LoweringRuleContext, src, dst, barrier=None
):
  ctx.launch_ctx.async_copy(src_ref=src, dst_ref=dst, barrier=barrier)
  return ()


def async_copy_smem_to_gmem(
    src: pallas_core.AbstractMemoryRef, dst: pallas_core.AbstractMemoryRef
) -> None:
  if src.memory_space is not gpu_core.SMEM:
    raise TypeError(f"src must be a SMEM reference, got {src.memory_space}")
  if dst.memory_space is not gpu_core.GMEM:
    raise ValueError(f"dst must be a GMEM reference, got {dst.memory_space}")
  async_copy_p.bind(src, dst)
  return None


def async_copy_gmem_to_smem(
    src: pallas_core.AbstractMemoryRef,
    dst: pallas_core.AbstractMemoryRef,
    *,
    barrier: pallas_core.AbstractMemoryRef,
) -> None:
  if src.memory_space is not gpu_core.GMEM:
    raise TypeError(f"src must be a GMEM reference, got {src.memory_space}")
  if dst.memory_space is not gpu_core.SMEM:
    raise ValueError(f"dst must be a SMEM reference, got {dst.memory_space}")
  async_copy_p.bind(src, dst, barrier)
  return None


class WaitEffect(jax_core.Effect):
  ...


wait_effect = WaitEffect()


wait_p = jax_core.Primitive("wait")
wait_p.multiple_results = True


@wait_p.def_effectful_abstract_eval
def _wait_abstract_eval(*avals, **params):
  del avals, params  # Unused.
  return (), {wait_effect}


@lowering.register_lowering_rule(wait_p)
def _wait_lowering_rule(
    ctx: lowering.LoweringRuleContext, barrier=None, allow_groups=None,
):
  if barrier is not None:
    barrier.wait()
  else:
    assert allow_groups is not None
    ctx.launch_ctx.await_async_copy(allow_groups=allow_groups)
  return ()


def wait_smem_to_gmem(allow_groups: int) -> None:
  """Waits until there are no more than the given number of SMEM->GMEM copies in flight."""
  wait_p.bind(allow_groups=allow_groups)


def wait_barrier(barrier: pallas_core.AbstractMemoryRef) -> None:
  """Waits on the given barrier."""
  wait_p.bind(barrier)


class _WGMMAPipelineEffect(effects.Effect):
  pass


_wgmma_pipeline_effect = _WGMMAPipelineEffect()
effects.control_flow_allowed_effects.add_type(_WGMMAPipelineEffect)

wgmma_discharged_p = jax_core.Primitive("wgmma_discharged")

def wgmma(acc, a, b, *, rhs_transpose: bool | None = None, swizzle: int = 128):
  """Asynchronous warp group matmul.

  The sm90 wgmma instruction, essentially acc[...] += a @ b. Requires
  that accumulator is an accumualtion register reference.

  Args:
    acc: The accumulator register.
    a: The left hand side operand.
    b: The right hand side operand.
    transpose: Whether to transpose b.
    n_tile: The number of tiles to use.
    swizzle: The swizzle pattern.
  """
  rhs_transpose = (
      (jnp.dtype(b.dtype).itemsize == 2)
      if rhs_transpose is None
      else rhs_transpose
  )

  ma, ka, tma, tka = a.shape
  kb, nb, tkb, tnb = b.shape
  mc, nc = acc.shape

  if rhs_transpose:
    kb, nb, tkb, tnb = nb, kb, tnb, tkb

  if tma * ma != mc or nb * tnb != nc or ka != kb or tka != tkb:
    raise ValueError(f"Incompatible shapes: {a.shape=}, {b.shape=}, {acc.shape=}, {rhs_transpose=}")

  return wgmma_p.bind(acc, a, b, swizzle=swizzle, rhs_transpose=rhs_transpose)

@wgmma_discharged_p.def_effectful_abstract_eval
def _wgmma_discharged_effectful_abstract_eval(acc, *args, **kwargs):
  del args, kwargs
  return acc, {
      _wgmma_pipeline_effect,
      state.ReadEffect(1),
      state.ReadEffect(2),
  }

@lowering.register_lowering_rule(wgmma_discharged_p)
def _wgmma_discharged_lowering_rule(
    ctx: lowering.LoweringRuleContext,
    acc,
    a,
    b,
    swizzle,
    rhs_transpose,
):
  del ctx
  new_acc = mgpu.wgmma(
      acc,
      a,
      b,
      swizzle=swizzle,
      b_order=mgpu.WGMMALayout.COL_MAJOR
      if rhs_transpose
      else mgpu.WGMMALayout.ROW_MAJOR,
  )
  nvvm_dialect.wgmma_commit_group_sync_aligned()
  return new_acc

wgmma_p = jax_core.Primitive("wgmma")
wgmma_p.multiple_results = True

@wgmma_p.def_effectful_abstract_eval
def _wgmma_effectful_abstract_eval(acc, *args, **kwargs):
  del acc, args, kwargs
  return [], {
      _wgmma_pipeline_effect,
      state.WriteEffect(0),
      state.ReadEffect(0),
      state.ReadEffect(1),
      state.ReadEffect(2),
  }

@discharge.register_discharge_rule(wgmma_p)
def _wgmma_discharge_rule(
    in_avals, out_avals,
    acc,
    a,
    b,
    swizzle,
    rhs_transpose,
):
  del in_avals, out_avals
  return (
      wgmma_discharged_p.bind(
          acc, a, b, swizzle=swizzle, rhs_transpose=rhs_transpose
      ),
      None,
      None,
  ), []

wgmma_wait_p = jax_core.Primitive("wgmma_wait")
wgmma_wait_p.multiple_results = True

def wgmma_wait(i: int):
  """Wait until all but the last `i` WGMMA operations are done."""
  return wgmma_wait_p.bind(i)


@wgmma_wait_p.def_effectful_abstract_eval
def wgmma_wait_effectful_abstract_eval(_):
  return [], {_wgmma_pipeline_effect}

@lowering.register_lowering_rule(wgmma_wait_p)
def _wgmma_wait_lowering_rule(ctx: lowering.LoweringRuleContext, allow_groups):
  del ctx
  nvvm_dialect.wgmma_wait_group_sync_aligned(allow_groups)
  return ()

zero_accumulator_p =  jax_core.Primitive("zero_accumulator")
def zero_accumulator(shape, dtype):
  return zero_accumulator_p.bind(shape=shape, dtype=dtype)

@zero_accumulator_p.def_abstract_eval
def _zero_accumulator_abstract_eval(shape, dtype):
  return jax_core.ShapedArray(shape=shape, dtype=dtype)


@lowering.register_lowering_rule(zero_accumulator_p)
def _zero_accumulator_lowering_rule(
    ctx: lowering.LoweringRuleContext, shape, dtype
):
  del ctx
  return mgpu.WGMMAAccumulator.zero(*shape, dtype=mlir.dtype_to_ir_type(jnp.dtype(dtype)))
