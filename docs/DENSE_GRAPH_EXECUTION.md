# Dense-graph execution audit and exact optimizations

This change preserves the selected node set, strict radius edges, window,
sampling factor, model layers/channels, physical batch and training schedule.
It is not evidence of real EventHDR/EventAid GPU speed or quality.

## Why the recorded graph can be large

The implemented geometry is `x/(W-1), y/(H-1), (t-origin)/time_scale`.
Radius 0.08 is **not 0.08 pixels**. At EventAid's 1180x720 resolution its spatial
semiaxes are 94.32 and 57.52 pixels. At a 0.05-second time scale the temporal
semiaxis is 0.004 seconds. These are separate-axis bounds of an ellipsoid,
not a rectangular neighborhood. A 50-ms lifetime is a separate window setting.

No duplicate-edge or double-time-normalization defect was established in this
audit. The unordered-pair ownership and reverse-edge construction are tested
against independent dense oracles. The historical 424,096,036 directed-edge
readout can therefore not be dismissed as merely a storage bug. Whether that
support is the right reconstruction design still needs actual-data evidence.

The [ASGCN paper](https://ojs.aaai.org/index.php/AAAI/article/download/32154/34309)
describes uniform event sampling, radius graphs and affected K-hop computation.
Its classification results do not select the physical radius, lifetime or
sampling factor for this reconstruction task. Those values are not changed
automatically. In particular, graph pooling after layer four does not eliminate
the first four layers' fine-graph work.

## Exact changes in the executed paths

1. **Reject impossible cell candidates before pair expansion.** Actual float64
   member-coordinate bounds give a lower distance bound for an occupied cell.
   Only provably outside cells are rejected, with a conservative rounding margin.
   Boundary candidates still use the original strict float64 distance predicate.
   This runs in materialized graph creation, implicit queries and topology scans.
2. **Count certified complete blocks without enumerating their edges.** A
   query-to-cell farthest-corner bound can prove that every member is inside the
   radius. Integer prefix/cell counts then account for selected/unselected nodes,
   arrival/expiry and readout/union masks with the original unordered ownership.
   Mixed or uncertain cells still receive exact bounded pairwise processing.
   This is a count-only optimization, not a substitute for learned messages.
3. **Exclude inactive SNN sources before candidate expansion.** The complete
   graph degree remains unchanged, including zero-spike neighbors. No spike,
   root transform or bias update is discarded.
4. **One neighbor pass for selected-destination inference.** Needed source
   projections are computed on first use and reused within the call, replacing
   the support-discovery pass followed by a second identical radius search.
   Projection cache remains O(N*K*C), never O(E). Training keeps its differentiable
   operator; all configured torch/torch_fused/Triton dispatch paths remain explicit.

The scan now records `readout_mean_in_degree`, `readout_edge_density`, and
`radius_geometry` per frame. `candidate_pairs_visited` counts actually expanded
lookup candidates, including candidates rejected later. `bulk_query_blocks`
counts certified complete blocks; `bulk_pairwise_evaluations_avoided` counts
unordered strict-radius pairs whose individual distance evaluation was avoided.
These are different units and must not be combined as a single edge count.

## Bounded CPU diagnostics, not research results

On one local two-thread float64 synthetic check (seed 762, 2,048 nodes in 16
independent lanes, radius .08), the query visited 22,148 candidates before the
cell-bound filter and 3,608 afterwards; both produced exactly 504 directed edges.
Single-run query times were about 76 ms and 24 ms; this is not a steady-state
benchmark or a GPU speedup claim.

For 1,024 coincident synthetic nodes, all 1,047,552 directed edges were counted.
Expanded candidates fell from 1,048,576 to zero; 523,776 unordered pairs were
counted through certified blocks. Coincident nodes are a deliberately favorable
complete-block test, not a representative EventHDR window. Spatially/time-spread
cells may fail certification and keep their pairwise cost.

Independent tests cover strict nextafter boundaries, translated coordinates,
extreme finite scales, mixed full/boundary cells, multiple streams, source masks,
append/expiry, full-degree normalization, output/gradient equivalence, IF clocks,
and raw-state checkpoint continuation. CPU tests cannot validate CUDA kernels or
establish that physical batch 16 fits a 9.5-GiB MIG slice.

Final local verification on 2026-09-12: full CPU regression **3,243 passed,
84 skipped**, including synthetic train/calibrate/evaluate/benchmark smoke and
stateful resume tests. Ruff and Git whitespace checks passed. One existing
PyTorch quantized-tensor deprecation warning remained. The Windows Git ownership
exception was supplied only to the test process, not written to global config.
No real dataset training/evaluation, CUDA/Triton execution or server run occurred.

## Remaining work and safe use

Dense learned message aggregation still costs O(E); backward regenerates exact
neighbors, and early GNN layers still see the fine graph. Cache/index state and
the U-Net still consume real memory. No 25-hour-to-X-minute promise follows from
these diagnostics. Actual full-model forward/loss/backward/optimizer memory and
timing, dense causal contexts and both datasets' ANN/SNN inference need allocated
device measurements before a final study. An early probe is not full coverage.

Existing reports/checkpoints and source-bound exact-resume contracts are not
rewritten or relabelled. A checkpoint from different executable source is not
silently accepted or replayed from zero. No SSH session, server, GPU selection,
training job or full-dataset evaluation is started by these code changes.
