import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4}, num_stages=2),
        triton.Config({'num_warps': 8}, num_stages=2),
        triton.Config({'num_warps': 16}, num_stages=2),
        triton.Config({'num_warps': 4}, num_stages=4),
        triton.Config({'num_warps': 8}, num_stages=4),
        triton.Config({'num_warps': 16}, num_stages=4),
    ],
    key=['n_cols'],
)
@triton.jit
def fused_rms_norm_kernel(
    X, Y, W, R, stride, n_cols, eps, 
    BLOCK_SIZE: tl.constexpr, 
    IS_POWER_OF_2: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = X + row_idx * stride
    res_start_ptr = R + row_idx * stride
    offsets = tl.arange(0, BLOCK_SIZE)

    # 1. Mask-free specialization if n_cols is power of 2
    if IS_POWER_OF_2:
        x = tl.load(row_start_ptr + offsets).to(tl.float32)
        r = tl.load(res_start_ptr + offsets).to(tl.float32)
        
        # In-place residual addition
        x_new = x + r
        
        # Store sum back to residual in-place
        tl.store(res_start_ptr + offsets, x_new.to(R.dtype.element_ty))
        
        # Load weight
        w = tl.load(W + offsets).to(tl.float32)
    else:
        mask = offsets < n_cols
        x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # In-place residual addition
        x_new = x + r
        
        # Store sum back to residual in-place
        tl.store(res_start_ptr + offsets, x_new.to(R.dtype.element_ty), mask=mask)
        
        # Load weight
        w = tl.load(W + offsets, mask=mask, other=1.0).to(tl.float32)
    
    # 2. RMSNorm computation (using fast reciprocal multiplication instead of division)
    inv_cols = 1.0 / n_cols
    mean_sq = tl.sum(x_new * x_new, axis=0) * inv_cols
    rsqrt = tl.math.rsqrt(mean_sq + eps)
    y = x_new * rsqrt * w
    
    # 3. Store normalized output
    if IS_POWER_OF_2:
        tl.store(Y + row_idx * stride + offsets, y.to(Y.dtype.element_ty))
    else:
        tl.store(Y + row_idx * stride + offsets, y.to(Y.dtype.element_ty), mask=mask)

def fast_fused_norm(x, residual, weight, eps=1e-6):
    orig_shape = x.shape
    x_flat = x.view(-1, orig_shape[-1])
    res_flat = residual.view(-1, orig_shape[-1])
    n_rows, n_cols = x_flat.shape
    
    y_flat = torch.empty_like(x_flat)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Check if n_cols is a power of 2 (enables mask-free path)
    is_power_of_2 = (n_cols & (n_cols - 1)) == 0
    
    # Launch autotuned JIT kernel
    fused_rms_norm_kernel[(n_rows,)](
        x_flat, y_flat, weight, res_flat, 
        x_flat.stride(0), n_cols, eps, 
        BLOCK_SIZE=BLOCK_SIZE, 
        IS_POWER_OF_2=is_power_of_2
    )
    return y_flat.view(*orig_shape)
