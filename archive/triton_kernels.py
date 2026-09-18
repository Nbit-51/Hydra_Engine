import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"num_warps": 4}, num_stages=1),
        triton.Config({"num_warps": 4}, num_stages=2),
        triton.Config({"num_warps": 4}, num_stages=3),
        triton.Config({"num_warps": 8}, num_stages=1),
        triton.Config({"num_warps": 8}, num_stages=2),
        triton.Config({"num_warps": 8}, num_stages=3),
        triton.Config({"num_warps": 8}, num_stages=4),
        triton.Config({"num_warps": 16}, num_stages=2),
        triton.Config({"num_warps": 16}, num_stages=3),
        triton.Config({"num_warps": 16}, num_stages=4),
        triton.Config({"num_warps": 32}, num_stages=2),
        triton.Config({"num_warps": 32}, num_stages=4),
    ],
    key=["n_cols"],
)
@triton.jit
def fused_rms_norm_kernel(
    X,
    Y,
    W,
    R,
    stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    IS_POWER_OF_2: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_offset = row_idx * stride
    offsets = tl.arange(0, BLOCK_SIZE)
    row_start_ptr = X + row_offset
    residual_start_ptr = R + row_offset
    output_start_ptr = Y + row_offset

    if IS_POWER_OF_2:
        x = tl.load(row_start_ptr + offsets).to(tl.float32)
        residual = tl.load(residual_start_ptr + offsets).to(tl.float32)
        weight = tl.load(W + offsets).to(tl.float32)
        x = x + residual
        tl.store(residual_start_ptr + offsets, x.to(R.dtype.element_ty))
    else:
        mask = offsets < n_cols
        x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(
            residual_start_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(W + offsets, mask=mask, other=1.0).to(tl.float32)
        x = x + residual
        tl.store(
            residual_start_ptr + offsets,
            x.to(R.dtype.element_ty),
            mask=mask,
        )

    variance = tl.sum(x * x, axis=0) * (1.0 / n_cols)
    normalized = x * tl.math.rsqrt(variance + eps) * weight

    if IS_POWER_OF_2:
        tl.store(output_start_ptr + offsets, normalized.to(Y.dtype.element_ty))
    else:
        tl.store(
            output_start_ptr + offsets,
            normalized.to(Y.dtype.element_ty),
            mask=mask,
        )


def fast_fused_norm(x, residual, weight, eps=1e-6):
    original_shape = x.shape
    x_flat = x.view(-1, original_shape[-1])
    residual_flat = residual.view(-1, original_shape[-1])
    n_rows, n_cols = x_flat.shape
    output = torch.empty_like(x_flat)
    block_size = triton.next_power_of_2(n_cols)

    fused_rms_norm_kernel[(n_rows,)](
        x_flat,
        output,
        weight,
        residual_flat,
        x_flat.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        IS_POWER_OF_2=(n_cols & (n_cols - 1)) == 0,
    )
    return output.view(*original_shape)
@triton.autotune(
    configs=[
        triton.Config({"num_warps": 4}, num_stages=2),
        triton.Config({"num_warps": 8}, num_stages=2),
        triton.Config({"num_warps": 8}, num_stages=3),
        triton.Config({"num_warps": 16}, num_stages=2),
        triton.Config({"num_warps": 16}, num_stages=4),
        triton.Config({"num_warps": 32}, num_stages=2),
    ],
    key=["n_cols"],
)
@triton.jit
def rms_norm_kernel(
    X,
    Y,
    W,
    stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    IS_POWER_OF_2: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_offset = row_idx * stride
    offsets = tl.arange(0, BLOCK_SIZE)

    if IS_POWER_OF_2:
        x = tl.load(X + row_offset + offsets).to(tl.float32)
        weight = tl.load(W + offsets).to(tl.float32)
    else:
        mask = offsets < n_cols
        x = tl.load(X + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(W + offsets, mask=mask, other=1.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) * (1.0 / n_cols)
    normalized = x * tl.math.rsqrt(variance + eps) * weight

    if IS_POWER_OF_2:
        tl.store(Y + row_offset + offsets, normalized.to(Y.dtype.element_ty))
    else:
        tl.store(
            Y + row_offset + offsets,
            normalized.to(Y.dtype.element_ty),
            mask=mask,
        )


def fast_rms_norm(x, weight, eps=1e-6):
    original_shape = x.shape
    x_flat = x.view(-1, original_shape[-1])
    n_rows, n_cols = x_flat.shape
    output = torch.empty_like(x_flat)
    block_size = triton.next_power_of_2(n_cols)

    rms_norm_kernel[(n_rows,)](
        x_flat,
        output,
        weight,
        x_flat.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        IS_POWER_OF_2=(n_cols & (n_cols - 1)) == 0,
    )
    return output.view(*original_shape)
