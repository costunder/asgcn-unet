"""Memory-bounded tensor operations used by the graph encoder."""

from __future__ import annotations

import torch


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


def weighted_spline_sum(
    projected: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    indices: torch.Tensor,
    basis: torch.Tensor,
    chunk_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Sum the two active degree-1 spline terms into destination node features.

    The caller validates topology/basis and performs degree normalization. The
    configured chunk size bounds edge-message working storage, not the node
    projections or the complete graph. Higher-order backward can retain its own
    graph and therefore does not have the ordinary-backward memory bound.
    """
    return _SplineSum.apply(
        projected, source, destination, indices, basis, chunk_size, output_dtype
    )
