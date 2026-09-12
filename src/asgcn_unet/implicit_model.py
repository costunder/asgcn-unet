"""Exact spline layers over streamed radius neighborhoods, without full edge storage.

This changes storage/execution, not sampling, pooling or the IF clock contract.
It does not claim original-paper reproduction or measured accelerator speed.
"""

from __future__ import annotations

import torch

from .graph import _safe_batch_norm
from .implicit_spline import implicit_weighted_spline_sum
from .ops import require_spline_backend


def affine(layer, x, graph, destinations=None, *, omit_zero_sources=False):
    require_spline_backend(layer.spline_backend, x.device)
    # Projection is per node/control point, never per edge. Zero-valued SNN
    # sources are removed BEFORE projection; they still contribute to degree.
    active_sources = x.ne(0).any(dim=1) if omit_zero_sources else None
    if destinations is not None and not torch.is_grad_enabled():
        from .implicit_projection import selected_spline_sum

        output, projected_sources, edges = selected_spline_sum(
            layer, x, graph, destinations, active_sources=active_sources,
        )
        degree = graph.in_degree[destinations]
        output = output / degree.to(output.dtype).clamp_min(1).unsqueeze(1)
        if layer.root is not None:
            output = output + x[destinations] @ layer.root
        if layer.bias is not None:
            output = output + layer.bias
        return output, projected_sources, edges
    edges = graph.edge_count
    if destinations is not None:
        used = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        edges = 0
        for source, _destination, _distance in graph.iter_directed_neighbors(
                destinations, active_sources=active_sources):
            used[source] = True
            edges += len(source)
        support = torch.nonzero(used, as_tuple=True)[0]
    elif active_sources is not None:
        support = torch.nonzero(active_sources, as_tuple=True)[0]
    else:
        support = None
    if support is not None:
        values = torch.einsum("ni,kio->nko", x[support], layer.weight)
        projected = values.new_zeros((len(x), layer.kernel_size, layer.out_channels))
        projected = projected.index_copy(0, support, values)
    else:
        support = None
        projected = torch.einsum("ni,kio->nko", x, layer.weight)
    output = implicit_weighted_spline_sum(
        projected, graph, selected_dest=destinations,
        edge_chunk_size=layer.edge_chunk_size or graph.candidate_pair_budget,
        output_dtype=x.dtype, backend=layer.spline_backend,
        active_sources=active_sources,
    )
    degree = graph.in_degree if destinations is None else graph.in_degree[destinations]
    output = output / degree.to(output.dtype).clamp_min(1).unsqueeze(1)
    own = x if destinations is None else x[destinations]
    if layer.root is not None:
        output = output + own @ layer.root
    if layer.bias is not None:
        output = output + layer.bias
    return output, len(x) if support is None else len(support), edges


def forward_ann(encoder, graph, *, return_activations=False):
    hidden, activations = graph.node_features, []
    for layer in encoder.layers:
        values, _, _ = affine(layer, hidden, graph)
        values = values if layer._bn_is_folded else _safe_batch_norm(layer.norm, values)
        hidden = torch.relu(values)
        if return_activations:
            activations.append(hidden)
    return hidden, activations
