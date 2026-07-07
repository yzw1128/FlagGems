import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@libentry()
@triton.jit
def count_nonzero_kernel_1(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int32)
    nonzero_count = tl.sum(is_nonzero, axis=0)
    tl.atomic_add(out_ptr, nonzero_count)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("count_nonzero"), key=["numel"])
@triton.jit
def count_nonzero_kernel(x_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr):
    n_workers = ext.num_programs(0)
    pid = ext.program_id(0)

    n_tasks = tl.cdiv(numel, N)
    tasks_per_worker = tl.cdiv(n_tasks, n_workers)

    for task_index in range(tasks_per_worker):
        task_id = pid + task_index * n_workers
        nonzero_count = tl.full((), value=0, dtype=out_ptr.dtype.element_ty)
        for start_n in range(0, N, BLOCK_SIZE):
            cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
            offset = task_id * N + cols_offsets
            mask = offset < numel and cols_offsets < N
            x = tl.load(x_ptr + offset, mask=mask, other=0)
            is_nonzero = (x != 0).to(tl.int64)
            nonzero_count += tl.sum(is_nonzero)

        tl.store(out_ptr + task_id, nonzero_count)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("count_nonzero"), key=["numel"])
@triton.jit
def count_nonzero_combin_kernel_1(x_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr):
    pid_x = ext.program_id(0)
    nonzero_count = tl.full((), value=0, dtype=out_ptr.dtype.element_ty)
    for start_n in range(0, N, BLOCK_SIZE):
        cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        offset = pid_x * N + cols_offsets
        mask = offset < numel and cols_offsets < N
        x = tl.load(x_ptr + offset, mask=mask, other=0)
        nonzero_count += tl.sum(x)
    tl.store(out_ptr + pid_x, nonzero_count)


@libentry()
@triton.jit
def count_nonzero_combin_kernel(
    x_ptr, combin_ptr, N, combin_N, numel, BLOCK_SIZE: tl.constexpr
):
    pid_x = ext.program_id(0)
    pid_y = ext.program_id(1)
    cols_offsets = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset = pid_x * N + cols_offsets
    mask = offset < numel and cols_offsets < N
    x = tl.load(x_ptr + offset, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int64)
    nonzero_count = tl.sum(is_nonzero)
    tl.store(combin_ptr + pid_x * combin_N + pid_y, nonzero_count)


def count_nonzero(x, dim=None):
    logger.debug("GEMS_ASCEND COUNT NONZERO")
    if dim is not None:
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        if dim < 0:
            dim = dim + x.ndim
        # Use simple tensor ops to avoid triton compiler issues on ascend
        return (x != 0).to(torch.int64).sum(dim=dim)
    else:
        return (x != 0).to(torch.int64).sum()
