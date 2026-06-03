import torch
import triton
import triton.language as tl

@triton.jit
def fused_rms_norm_kernel(X, Y, W, R, stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = X + row_idx * stride
    res_start_ptr = R + row_idx * stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = x + r 
    
    w = tl.load(W + offsets, mask=mask, other=1.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / n_cols
    rsqrt = tl.math.rsqrt(mean_sq + eps)
    y = x * rsqrt * w
    tl.store(Y + row_idx * stride + offsets, y.to(tl.float16), mask=mask)

def fast_fused_norm(x, residual, weight, eps=1e-6):
    orig_shape = x.shape
    x_flat = x.view(-1, orig_shape[-1])
    res_flat = residual.view(-1, orig_shape[-1])
    n_rows, n_cols = x_flat.shape
    y_flat = torch.empty_like(x_flat)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    fused_rms_norm_kernel[(n_rows,)](x_flat, y_flat, weight, res_flat, x_flat.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y_flat.view(*orig_shape)
