"""Explicit, optional Triton spline kernels; imported only for the CUDA backend.

These kernels are original implementations of the two-active-basis gather,
multiply and scatter formula. They keep edge messages in registers rather than
allocating E-by-C tensors. CUDA atomics change summation order, not the operator.
The reference remains available as spline_backend='torch'.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "start", "edge_count"], debug=True)
def _forward_kernel(
    projected, source, destination, indices, basis, output,
    start, edge_count,
    N, K: tl.constexpr, C: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
    S0: tl.constexpr, D0: tl.constexpr, I0: tl.constexpr, I1: tl.constexpr,
    B0: tl.constexpr, B1: tl.constexpr, O0: tl.constexpr, O1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    edge = start + offset // C
    channel = offset % C
    mask = offset < edge_count * C
    src = tl.load(source + edge * S0, mask, other=0).to(tl.int64)
    dst = tl.load(destination + edge * D0, mask, other=0).to(tl.int64)
    left = tl.load(indices + edge * I0, mask, other=0).to(tl.int64)
    right = tl.load(indices + edge * I0 + I1, mask, other=0).to(tl.int64)
    valid = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
    valid = valid & (left >= 0) & (left < K) & (right >= 0) & (right < K)
    tl.device_assert((~mask) | valid, "Spline topology index out of bounds")
    mask = mask & valid
    v0 = tl.load(projected + src * P0 + left * P1 + channel * P2, mask, other=0)
    v1 = tl.load(projected + src * P0 + right * P1 + channel * P2, mask, other=0)
    b0 = tl.load(basis + edge * B0, mask, other=0).to(projected.dtype.element_ty)
    b1 = tl.load(basis + edge * B0 + B1, mask, other=0).to(projected.dtype.element_ty)
    # Explicit conversions preserve multiplication rounding under FP16/BF16 AMP.
    m0 = (v0 * b0).to(projected.dtype.element_ty).to(output.dtype.element_ty)
    m1 = (v1 * b1).to(projected.dtype.element_ty).to(output.dtype.element_ty)
    combined = (m0 + m1).to(output.dtype.element_ty)
    tl.atomic_add(output + dst * O0 + channel * O1, combined, mask, sem="relaxed")


@triton.jit(do_not_specialize=["N", "start", "edge_count"], debug=True)
def _projected_backward_kernel(
    grad_output, source, destination, indices, basis, grad_projected,
    start, edge_count,
    N, K: tl.constexpr, C: tl.constexpr,
    G0: tl.constexpr, G1: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
    S0: tl.constexpr, D0: tl.constexpr, I0: tl.constexpr, I1: tl.constexpr,
    B0: tl.constexpr, B1: tl.constexpr, BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    edge = start + offset // C
    channel = offset % C
    mask = offset < edge_count * C
    src = tl.load(source + edge * S0, mask, other=0).to(tl.int64)
    dst = tl.load(destination + edge * D0, mask, other=0).to(tl.int64)
    left = tl.load(indices + edge * I0, mask, other=0).to(tl.int64)
    right = tl.load(indices + edge * I0 + I1, mask, other=0).to(tl.int64)
    valid = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
    valid = valid & (left >= 0) & (left < K) & (right >= 0) & (right < K)
    tl.device_assert((~mask) | valid, "Spline topology index out of bounds")
    mask = mask & valid
    gradient = tl.load(grad_output + dst * G0 + channel * G1, mask, other=0)
    gradient = gradient.to(grad_projected.dtype.element_ty)
    b0 = tl.load(basis + edge * B0, mask, other=0).to(grad_projected.dtype.element_ty)
    b1 = tl.load(basis + edge * B0 + B1, mask, other=0).to(grad_projected.dtype.element_ty)
    g0 = (gradient * b0).to(grad_projected.dtype.element_ty)
    g1 = (gradient * b1).to(grad_projected.dtype.element_ty)
    tl.atomic_add(grad_projected + src * P0 + left * P1 + channel * P2, g0, mask, sem="relaxed")
    tl.atomic_add(grad_projected + src * P0 + right * P1 + channel * P2, g1, mask, sem="relaxed")


@triton.jit(do_not_specialize=["N", "start"], debug=True)
def _basis_backward_kernel(
    grad_output, projected, source, destination, indices, grad_basis,
    start,
    N, K: tl.constexpr, C: tl.constexpr,
    G0: tl.constexpr, G1: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
    S0: tl.constexpr, D0: tl.constexpr, I0: tl.constexpr, I1: tl.constexpr,
    B0: tl.constexpr, B1: tl.constexpr, BLOCK_C: tl.constexpr,
):
    edge = start + tl.program_id(0)
    channel = tl.arange(0, BLOCK_C)
    src = tl.load(source + edge * S0).to(tl.int64)
    dst = tl.load(destination + edge * D0).to(tl.int64)
    left = tl.load(indices + edge * I0).to(tl.int64)
    right = tl.load(indices + edge * I0 + I1).to(tl.int64)
    valid = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
    valid = valid & (left >= 0) & (left < K) & (right >= 0) & (right < K)
    tl.device_assert(valid, "Spline topology index out of bounds")
    mask = (channel < C) & valid
    gradient = tl.load(grad_output + dst * G0 + channel * G1, mask, other=0)
    gradient = gradient.to(projected.dtype.element_ty)
    v0 = tl.load(projected + src * P0 + left * P1 + channel * P2, mask, other=0)
    v1 = tl.load(projected + src * P0 + right * P1 + channel * P2, mask, other=0)
    products0 = (gradient * v0).to(projected.dtype.element_ty)
    products1 = (gradient * v1).to(projected.dtype.element_ty)
    # Torch sums half/bfloat16 products in FP32, then rounds the result back to
    # their input dtype before the explicit conversion to the basis dtype.
    if projected.dtype.element_ty == tl.float64:
        total0 = tl.sum(products0.to(tl.float64), axis=0)
        total1 = tl.sum(products1.to(tl.float64), axis=0)
    else:
        total0 = tl.sum(products0.to(tl.float32), axis=0)
        total1 = tl.sum(products1.to(tl.float32), axis=0)
    total0 = total0.to(projected.dtype.element_ty).to(grad_basis.dtype.element_ty)
    total1 = total1.to(projected.dtype.element_ty).to(grad_basis.dtype.element_ty)
    tl.store(grad_basis + edge * B0, total0, valid)
    tl.store(grad_basis + edge * B0 + B1, total1, valid)


def spline_forward(projected, source, destination, indices, basis, output, chunk_size):
    if source.numel() == 0 or projected.shape[2] == 0:
        return
    with torch.cuda.device(projected.device):
        for start in range(0, source.numel(), chunk_size):
            count = min(chunk_size, source.numel() - start)
            _forward_kernel[(triton.cdiv(count * projected.shape[2], 256),)](
                projected, source, destination, indices, basis, output, start, count,
                *projected.shape, *projected.stride(), source.stride(0), destination.stride(0),
                *indices.stride(), *basis.stride(), *output.stride(),
                BLOCK=256, num_warps=4, enable_fp_fusion=False, debug=True,
            )


def spline_backward(
    grad_output, source, destination, indices, basis, grad_projected,
    projected, grad_basis, projected_shape, projected_dtype, chunk_size,
):
    del projected_dtype  # Dtype is carried by grad_projected or saved projected tensors.
    if source.numel() == 0 or projected_shape[2] == 0:
        return
    with torch.cuda.device(grad_output.device):
        for start in range(0, source.numel(), chunk_size):
            count = min(chunk_size, source.numel() - start)
            if grad_projected is not None:
                _projected_backward_kernel[(triton.cdiv(count * projected_shape[2], 256),)](
                    grad_output, source, destination, indices, basis, grad_projected, start, count,
                    *projected_shape, *grad_output.stride(), *grad_projected.stride(),
                    source.stride(0), destination.stride(0), *indices.stride(), *basis.stride(),
                    BLOCK=256, num_warps=4, enable_fp_fusion=False, debug=True,
                )
            if grad_basis is not None:
                _basis_backward_kernel[(count,)](
                    grad_output, projected, source, destination, indices, grad_basis, start,
                    *projected_shape, *grad_output.stride(), *projected.stride(),
                    source.stride(0), destination.stride(0), *indices.stride(), *grad_basis.stride(),
                    BLOCK_C=triton.next_power_of_2(projected_shape[2]),
                    num_warps=4, enable_fp_fusion=False, debug=True,
                )
