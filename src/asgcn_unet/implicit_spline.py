"""Spline aggregation with O(N) saved topology and recomputed edge chunks.

Projection, root/bias, full-degree normalization, and BN remain with the caller.
Ordinary first-order backward retains no E-sized endpoints or basis tensors.
Higher derivatives can retain their own autograd graph and do not have that bound.
Coordinates are fixed non-gradient input geometry. Different accumulation order
requires dtype-appropriate tolerances, not a bitwise-equivalence claim.
"""

from __future__ import annotations

import torch

from .implicit_radius import ImplicitRadiusGraph, _positive_integer
from .ops import _triton_ops, require_spline_backend


def _message_chunks(index, selected_dest, active_sources, kernel_size, edge_chunk_size):
    local_map = None
    if selected_dest is not None:
        local_map = index.node_batch.new_full((index.positions.shape[0],), -1)
        local_map[selected_dest] = torch.arange(selected_dest.numel(), device=selected_dest.device)
    for source, destination, pseudo in index.iter_directed_neighbors(
            selected_dest, active_sources=active_sources):
        for start in range(0, source.numel(), edge_chunk_size):
            stop = min(start + edge_chunk_size, source.numel())
            local_destination = destination[start:stop]
            if local_map is not None:
                local_destination = local_map[local_destination]
            # Identical degree-1 formula to linear_open_bspline_basis. The index
            # already guarantees finite pseudo in [0,1], so do not repeat its
            # host-synchronizing validation reductions for every edge chunk.
            scaled = pseudo[start:stop, 0] * float(kernel_size - 1)
            cell = torch.floor(scaled)
            left = cell.to(torch.long).remainder(kernel_size)
            indices = torch.stack((left, (left + 1).remainder(kernel_size)), dim=-1)
            right_weight = scaled - cell
            basis = torch.stack((1.0 - right_weight, right_weight), dim=-1)
            yield source[start:stop], local_destination, indices, basis


class _ImplicitSplineSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, index, selected_dest, active_sources, edge_chunk_size, output_dtype, backend):
        ctx.projected_shape, ctx.projected_dtype = projected.shape, projected.dtype
        ctx.index, ctx.edge_chunk_size, ctx.backend = index, edge_chunk_size, backend
        ctx.has_selection, ctx.has_active_sources = selected_dest is not None, active_sources is not None
        saved = (index.positions, index.node_batch)
        if selected_dest is not None:
            saved += (selected_dest,)
        if active_sources is not None:
            saved += (active_sources,)
        ctx.save_for_backward(*saved)
        count = projected.shape[0] if selected_dest is None else selected_dest.numel()
        output = torch.zeros((count, projected.shape[2]), dtype=output_dtype, device=projected.device)
        native = _triton_ops() if backend == "triton" else None
        for source, destination, indices, basis in _message_chunks(
                index, selected_dest, active_sources, projected.shape[1], edge_chunk_size):
            if native is not None:
                native.spline_forward(projected, source, destination, indices, basis, output, edge_chunk_size)
            elif backend == "torch_fused":
                values = projected[source[:, None], indices]
                messages = values * basis[:, :, None].to(values.dtype)
                output.index_add_(0, destination, messages.to(output_dtype).sum(dim=1))
            else:
                for active_basis in range(2):
                    values = projected[source, indices[:, active_basis]]
                    messages = values * basis[:, active_basis, None].to(values.dtype)
                    output.index_add_(0, destination, messages.to(output_dtype))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors  # Enforce PyTorch version checks on geometry/masks.
        selected_dest = saved[2] if ctx.has_selection else None
        active_sources = saved[2 + int(ctx.has_selection)] if ctx.has_active_sources else None
        grad_projected = torch.zeros(ctx.projected_shape, dtype=ctx.projected_dtype, device=grad_output.device)
        # As with the existing explicit Triton operator, higher-order derivatives
        # use differentiable tensor arithmetic, explicitly documented above.
        native = _triton_ops() if ctx.backend == "triton" and not torch.is_grad_enabled() else None
        for source, destination, indices, basis in _message_chunks(
                ctx.index, selected_dest, active_sources, ctx.projected_shape[1], ctx.edge_chunk_size):
            if native is not None:
                native.spline_backward(
                    grad_output, source, destination, indices, basis, grad_projected,
                    None, None, ctx.projected_shape, ctx.projected_dtype, ctx.edge_chunk_size,
                )
            else:
                messages = grad_output[destination].to(ctx.projected_dtype)
                for active_basis in range(2):
                    weighted = messages * basis[:, active_basis, None].to(ctx.projected_dtype)
                    grad_projected.index_put_((source, indices[:, active_basis]), weighted, accumulate=True)
        return grad_projected, None, None, None, None, None, None


def implicit_weighted_spline_sum(projected, graph, selected_dest=None, *, edge_chunk_size=65_536,
                                 output_dtype=None, backend="torch", active_sources=None):
    """Unnormalized incoming message sum; optionally omit inactive source messages.

    ``active_sources`` does not change graph topology or the normalization degree.
    Omitted sources have zero message derivative. Selected output rows retain the
    supplied unique destination order. Each configured backend executes itself;
    unsupported CUDA/Triton settings fail explicitly, without a torch fallback.
    CUDA performance and peak memory still require representative measurement.
    """
    if not isinstance(graph, ImplicitRadiusGraph):
        raise TypeError("implicit_weighted_spline_sum requires ImplicitRadiusGraph")
    graph.validate_integrity()
    if (not isinstance(projected, torch.Tensor) or projected.layout != torch.strided
            or projected.ndim != 3 or projected.shape[0] != graph.positions.shape[0]
            or projected.shape[1] < 2 or projected.shape[2] < 1
            or not projected.is_floating_point() or projected.device != graph.positions.device):
        raise ValueError("projected must be a floating [N,K>=2,C>=1] tensor on the graph device")
    _positive_integer(edge_chunk_size, "edge_chunk_size")
    output_dtype = projected.dtype if output_dtype is None else output_dtype
    if output_dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError("output_dtype must be a real floating-point dtype")
    require_spline_backend(backend, projected.device)
    index = graph.index
    if selected_dest is not None:
        index.validate_destinations(selected_dest)
    if active_sources is not None and (
            not isinstance(active_sources, torch.Tensor) or active_sources.layout != torch.strided
            or active_sources.shape != (projected.shape[0],) or active_sources.dtype != torch.bool
            or active_sources.device != projected.device):
        raise ValueError("active_sources must be a bool [N] tensor on the graph device")
    return _ImplicitSplineSum.apply(
        projected, index, selected_dest, active_sources, edge_chunk_size, output_dtype, backend,
    )
