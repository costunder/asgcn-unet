"""Memory-bounded tensor operations used by the graph encoder."""

from __future__ import annotations

import torch

SPLINE_BACKENDS = frozenset({"torch", "torch_fused", "triton"})


class _SplineSum(torch.autograd.Function):
    """Gather, weight and scatter without retaining an E-by-C tensor per basis.

    Projection remains outside this Function so PyTorch supplies its matrix-product
    derivatives. Ordinary backward only needs topology and scalar basis weights;
    projected node features are saved only when basis derivatives are requested.
    Backward uses differentiable tensor operations to also support double backward.
    """

    @staticmethod
    def forward(
        ctx,
        projected: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        indices: torch.Tensor,
        basis: torch.Tensor,
        chunk_size: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        ctx.projected_shape = projected.shape
        ctx.projected_dtype = projected.dtype
        ctx.chunk_size = chunk_size
        # Do not retain N*K*C projections when only their derivative is needed.
        saved = (source, destination, indices, basis)
        if ctx.needs_input_grad[4]:
            saved += (projected,)
        ctx.save_for_backward(*saved)
        output = torch.zeros(
            (projected.shape[0], projected.shape[2]),
            dtype=output_dtype,
            device=projected.device,
        )
        edge_count = source.numel()
        # Preserve the reference's basis-major accumulation and AMP cast order.
        for active_basis in range(2):
            for start in range(0, edge_count, chunk_size):
                stop = min(start + chunk_size, edge_count)
                values = projected[source[start:stop], indices[start:stop, active_basis]]
                messages = values * basis[start:stop, active_basis, None].to(values.dtype)
                output.index_add_(0, destination[start:stop], messages.to(output_dtype))
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        source, destination, indices, basis, *saved_projection = ctx.saved_tensors
        node_count, kernel_size, channels = ctx.projected_shape
        projected_dtype = ctx.projected_dtype
        grad_projected = (
            torch.zeros(
                (node_count, kernel_size, channels),
                device=grad_output.device,
                dtype=projected_dtype,
            )
            if ctx.needs_input_grad[0]
            else None
        )
        grad_basis = torch.zeros_like(basis) if ctx.needs_input_grad[4] else None
        edge_count = source.numel()
        for start in range(0, edge_count, ctx.chunk_size):
            stop = min(start + ctx.chunk_size, edge_count)
            local_source = source[start:stop]
            # Both basis terms use the same destination derivative. Gather it
            # once per chunk, rather than materializing it twice per edge.
            grad_messages = grad_output[destination[start:stop]].to(projected_dtype)
            for active_basis in range(2):
                local_basis = indices[start:stop, active_basis]
                # Forward casts each message to output_dtype before index_add_.
                # Its derivative must cast back before multiplying/reducing.
                if grad_projected is not None:
                    weighted = grad_messages * basis[start:stop, active_basis, None].to(
                        projected_dtype
                    )
                    # This is the derivative of the original two-index gather.
                    # Accumulate directly into N*K*C rather than flattening row
                    # indices and constructing a full-sized gradient per chunk.
                    grad_projected.index_put_(
                        (local_source, local_basis), weighted, accumulate=True
                    )
                if grad_basis is not None:
                    values = saved_projection[0][local_source, local_basis]
                    grad_basis[start:stop, active_basis] = (grad_messages * values).sum(
                        dim=-1
                    ).to(basis.dtype)
        return grad_projected, None, None, None, grad_basis, None, None


def _save_spline_context(ctx, projected, source, destination, indices, basis, chunk_size):
    ctx.projected_shape = projected.shape
    ctx.projected_dtype = projected.dtype
    ctx.chunk_size = chunk_size
    saved = (source, destination, indices, basis)
    if ctx.needs_input_grad[4]:
        saved += (projected,)
    ctx.save_for_backward(*saved)


class _SplineSumFused(torch.autograd.Function):
    """Combine the two basis terms before one destination scatter per chunk.

    This portable candidate halves basis-loop gather/scatter dispatches. Working
    tensors contain at most two basis terms per configured edge chunk, and no
    E-by-C messages are retained for backward. Unlike the reference, accumulation
    is edge-major; floating-point comparisons require dtype-appropriate tolerances.
    """

    @staticmethod
    def forward(ctx, projected, source, destination, indices, basis, chunk_size, output_dtype):
        _save_spline_context(ctx, projected, source, destination, indices, basis, chunk_size)
        output = torch.zeros(
            (projected.shape[0], projected.shape[2]),
            dtype=output_dtype, device=projected.device,
        )
        for start in range(0, source.numel(), chunk_size):
            stop = min(start + chunk_size, source.numel())
            values = projected[source[start:stop, None], indices[start:stop]]
            messages = values * basis[start:stop, :, None].to(projected.dtype)
            # Round each product in projected dtype before converting to the
            # accumulation dtype, matching the reference's AMP cast boundaries.
            combined = messages.to(output_dtype).sum(dim=1)
            output.index_add_(0, destination[start:stop], combined)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        source, destination, indices, basis, *saved_projection = ctx.saved_tensors
        node_count, kernel_size, channels = ctx.projected_shape
        grad_projected = (
            torch.zeros(ctx.projected_shape, device=grad_output.device, dtype=ctx.projected_dtype)
            if ctx.needs_input_grad[0] else None
        )
        grad_basis = torch.zeros_like(basis) if ctx.needs_input_grad[4] else None
        for start in range(0, source.numel(), ctx.chunk_size):
            stop = min(start + ctx.chunk_size, source.numel())
            local_source, local_indices = source[start:stop], indices[start:stop]
            messages = grad_output[destination[start:stop]].to(ctx.projected_dtype)
            if grad_projected is not None:
                weighted = messages[:, None, :] * basis[start:stop, :, None].to(ctx.projected_dtype)
                rows = local_source[:, None] * kernel_size + local_indices
                grad_projected.reshape(node_count * kernel_size, channels).index_add_(
                    0, rows.reshape(-1), weighted.reshape(-1, channels)
                )
            if grad_basis is not None:
                values = saved_projection[0][local_source[:, None], local_indices]
                grad_basis[start:stop] = (messages[:, None, :] * values).sum(dim=-1).to(basis.dtype)
        return grad_projected, None, None, None, grad_basis, None, None


def _triton_ops():
    """Import the optional CUDA implementation only after explicit selection."""
    try:
        from . import ops_cuda
    except ModuleNotFoundError as error:
        if error.name == "triton" or (error.name or "").startswith("triton."):
            raise RuntimeError(
                "spline_backend='triton' requires the Triton package in the CUDA Python "
                "environment. Install the server's supported PyTorch/Triton combination "
                "or explicitly select 'torch'/'torch_fused'; no backend fallback was applied."
            ) from error
        raise
    return ops_cuda


def require_spline_backend(backend: str, device: torch.device) -> None:
    """Fail explicitly before execution, including graphs without any edges."""
    if backend not in SPLINE_BACKENDS:
        raise ValueError(f"Unknown spline_backend: {backend!r}")
    if backend == "triton":
        if device.type != "cuda":
            raise ValueError("spline_backend='triton' requires CUDA tensors; no CPU fallback is used")
        if torch.version.hip is not None or torch.cuda.get_device_capability(device)[0] < 8:
            raise RuntimeError("The Triton spline candidate requires an NVIDIA SM80-or-newer GPU")
        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError(
                "The Triton spline candidate uses atomic reductions and cannot satisfy "
                "torch deterministic-algorithms mode; explicitly select the reference backend"
            )
        _triton_ops()


class _SplineSumTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, source, destination, indices, basis, chunk_size, output_dtype):
        _save_spline_context(ctx, projected, source, destination, indices, basis, chunk_size)
        output = torch.zeros(
            (projected.shape[0], projected.shape[2]),
            dtype=output_dtype, device=projected.device,
        )
        _triton_ops().spline_forward(projected, source, destination, indices, basis, output, chunk_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            # Native kernels do not construct an autograd graph. Explicitly use
            # the differentiable portable formula for requested higher derivatives.
            return _SplineSumFused.backward(ctx, grad_output)
        source, destination, indices, basis, *saved_projection = ctx.saved_tensors
        grad_projected = (
            torch.zeros(ctx.projected_shape, device=grad_output.device, dtype=ctx.projected_dtype)
            if ctx.needs_input_grad[0] else None
        )
        grad_basis = torch.zeros_like(basis) if ctx.needs_input_grad[4] else None
        _triton_ops().spline_backward(
            grad_output, source, destination, indices, basis, grad_projected,
            saved_projection[0] if saved_projection else None, grad_basis,
            ctx.projected_shape, ctx.projected_dtype, ctx.chunk_size,
        )
        return grad_projected, None, None, None, grad_basis, None, None


def weighted_spline_sum(
    projected: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    indices: torch.Tensor,
    basis: torch.Tensor,
    chunk_size: int,
    output_dtype: torch.dtype,
    *,
    backend: str = "torch",
) -> torch.Tensor:
    """Sum the two active degree-1 spline terms into destination node features.

    The caller validates topology/basis and performs degree normalization. Native
    pointer-based execution additionally masks and asserts invalid index bounds
    on-device without a per-layer host synchronization. As with CUDA tensor
    indexing, an invalid input may leave the CUDA context unusable. The
    configured chunk size bounds edge-message working storage, not the node
    projections or the complete graph. torch_fused uses two terms per chunk;
    triton keeps edge messages in kernel registers. Both are explicit candidate
    backends, not default speedup claims. Higher-order backward can retain its own
    graph and therefore does not have the ordinary-backward memory bound. Triton
    uses differentiable torch_fused backward only when create_graph=True.
    """
    if backend not in SPLINE_BACKENDS:
        raise ValueError(f"Unknown spline_backend: {backend!r}")
    operation = _SplineSum
    if backend == "torch_fused":
        operation = _SplineSumFused
    elif backend == "triton":
        require_spline_backend(backend, projected.device)
        if any(value.layout != torch.strided for value in (projected, source, destination, indices, basis)):
            raise TypeError("The Triton spline backend requires strided tensor layouts")
        floating = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
        if projected.dtype not in floating or basis.dtype not in floating or output_dtype not in floating:
            raise TypeError("The Triton spline backend requires real floating-point tensors")
        if any(value.device != projected.device for value in (source, destination, indices, basis)):
            raise ValueError("The Triton spline backend requires all tensors on one CUDA device")
        if any(value.dtype not in {torch.int32, torch.int64} for value in (source, destination, indices)):
            raise TypeError("The Triton spline backend requires integer topology indices")
        if (
            projected.ndim != 3 or source.ndim != 1 or destination.shape != source.shape
            or indices.shape != (source.numel(), 2) or basis.shape != indices.shape
            or isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1
        ):
            raise ValueError("Invalid Triton spline tensor shapes or chunk size")
        operation = _SplineSumTriton
    return operation.apply(
        projected, source, destination, indices, basis, chunk_size, output_dtype
    )
