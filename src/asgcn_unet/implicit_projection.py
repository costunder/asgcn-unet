"""One-pass selected-neighbor projection/aggregation for no-grad inference.

Each needed source is projected once, on its first encountered message chunk.
No support-discovery radius pass or E-sized topology cache is retained. The
O(N*K*C) projection cache is bounded by nodes, as in the original implementation;
only bounded message chunks and an O(N) seen-source mask are added. This is not
a training shortcut: gradient-enabled callers use the ordinary autograd path.
"""

from __future__ import annotations

import torch

from .implicit_spline import _message_chunks
from .ops import _triton_ops, require_spline_backend


def selected_spline_sum(layer, x, graph, destinations, *, active_sources=None):
    """Return the unnormalized sum and actual projected-source/message counts.

    The configured spline backend, message cast boundaries and full incoming
    degree remain unchanged. Lazy matrix products can have a different row
    count; comparisons require dtype-appropriate tolerance, not bitwise equality.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("Selected lazy projection is inference-only; use the autograd spline path")
    require_spline_backend(layer.spline_backend, x.device)
    graph.validate_integrity()
    index = graph.index
    index.validate_destinations(destinations)
    seen = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    output = x.new_zeros((destinations.numel(), layer.out_channels))
    projected = None
    projected_sources = message_edges = 0
    chunk_size = layer.edge_chunk_size or graph.candidate_pair_budget
    native = _triton_ops() if layer.spline_backend == "triton" else None
    for source, destination, indices, basis in _message_chunks(
            index, destinations, active_sources, layer.kernel_size, chunk_size):
        if not source.numel():
            continue
        unseen = torch.unique(source[~seen[source]], sorted=True)
        if unseen.numel():
            values = torch.einsum("ni,kio->nko", x[unseen], layer.weight)
            if projected is None:
                # Allocate in actual projection dtype, including explicit AMP.
                projected = values.new_zeros((len(x), layer.kernel_size, layer.out_channels))
            projected[unseen] = values
            seen[unseen] = True
            projected_sources += unseen.numel()
        if projected is None:
            raise RuntimeError("Nonempty message chunk has no initialized source projection")
        message_edges += source.numel()
        if native is not None:
            native.spline_forward(projected, source, destination, indices, basis, output, chunk_size)
        elif layer.spline_backend == "torch_fused":
            values = projected[source[:, None], indices]
            messages = values * basis[:, :, None].to(values.dtype)
            output.index_add_(0, destination, messages.to(x.dtype).sum(dim=1))
        else:
            for active_basis in range(2):
                values = projected[source, indices[:, active_basis]]
                messages = values * basis[:, active_basis, None].to(values.dtype)
                output.index_add_(0, destination, messages.to(x.dtype))
    return output, projected_sources, message_edges
