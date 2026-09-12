# Hierarchical ASGCN reconstruction (architecture v4)

## What is now connected

The complete reconstruction path is:

`physical events → persistent sequence-uniform sampler → sliding radius graph →
4 spline graph convolutions → mean cluster pooling + quotient edge remapping →
2 spline graph convolutions → raster → existing recurrent U-Net → image/loss`.

All six 64-channel layers, their trainable spline/root/bias/BN parameters, the
base-48 U-Net, and its recurrent state remain connected. `EncoderStage` is a
non-owning view, not a copied or separately optimized encoder. The production
preparation retains physical batch 16, 40 epochs, complete datasets/resolution,
the chosen physical window/time scale, and radius. It never edits old studies.

V4 is a new architecture requiring independent ANN training and chronological
calibration. V2/v3 checkpoints and their measured results are not v4 results.
Storage-only implicit/materialized parity does not justify reusing an unpooled
checkpoint for a pooled model, even though parameter tensor shapes match.

## Paper evidence versus reconstruction choices

The [ASGCN paper, equations 18/19, p.1624](https://ojs.aaai.org/index.php/AAAI/article/download/32154/34309)
specifies feature averaging within clusters, remapping the original edges, and
convolution on the resulting graph. Its sampling precedes graph construction;
event-local graph/spike updates avoid computing unaffected destinations. It is a
classification paper, not a U-Net HDR reconstruction specification. The checked
primary channels did not provide verified author code or exact pool placements,
cell sizes, coordinate units, or reconstruction hyperparameters.

The following are explicitly **this reconstruction's design**, not claimed
author settings:

- One pool after layer four, leaving two learned graph convolutions afterwards.
  This places coarsening after high-level features without removing any layer.
- Fixed XYZ cells, including the independent sequence/batch namespace. The
  preparation's spatial cell matches the existing raster stride (4 pixels); its
  temporal cell equals `graph_radius * time_scale_seconds` (4 ms for radius .08
  and scale .05 s). Neither setting silently changes the raw radius or window.
- Coordinates are member means. The fixed sequence origin, not the moving
  readout/window boundary, anchors temporal cell identity.
- Coarse edges are the unique endpoint-remapped original fine edges, excluding
  intra-cluster self-loops. They are NOT a newly constructed centroid-radius graph.
- Each coarse edge's scalar spline pseudo-coordinate is the arithmetic mean of
  its contributing fine-edge `distance/radius` values. This stays in the spline
  domain without clipping a possibly larger centroid distance. It is an explicit
  pooling edge-feature convention; the paper does not resolve it.

The endpoint remapping, self-loop removal, coalescing and position mean follow
the [PyG pooling primitive](https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/nn/pool/pool.py)
used by the [AEGNN author implementation](https://github.com/uzh-rpg/aegnn/blob/d96e13b2f80f3c7515a65baf966544e0914d068d/aegnn/models/layer/max_pool.py).
AEGNN's feature **max** is not copied: ASGCN's feature **mean** is implemented.

## Sampling without frame-boundary drift

`event_sampling_factor=R` selects raw sequence ordinals `0,R,2R,...`. The ordinal
counter advances for every raw event, including discarded events and frames with
zero selections. It survives training checkpoints, clone/detach/device moves,
empty windows and readouts. Equal-timestamp arrival counts are recomputed after
sampling; different timestamps are never merged. Raw IDs and chronology are
validated before selection, so sampling cannot hide reordered or duplicate input.

The default remains **R=1**, retaining all events. No new event cap, hidden subset,
or automatic OOM sampling fallback is present. The paper's N-Cars R sweep is not
evidence that a particular R is correct for EventHDR/EventAid. Any R>1 study must
be explicitly selected and reported, not retroactively applied to old results.

## Exact incremental pooling and SNN clocks

Pool state keeps member counts, feature/position/time sums, fine-to-coarse node
assignments, and coarse-edge contributor counts/pseudo sums. Arrival adds only
new incident edge contributions; expiry subtracts expired contributions. A coarse
edge disappears only when its last contributor disappears. Existing member
feature changes update their cluster sums. Empty clusters disappear, with an
explicit old-to-new remapping of downstream neuron caches.

ANN inference has a frozen-BN full-snapshot oracle. Training/calibration recompute
the entire current snapshot and differentiable feature means; stale learned
caches are never carried across optimizer updates. Calibration records all six
layers on their respective graphs. Channelwise conversion scaling commutes with
member means, preserving the single contiguous six-layer normalization chain.

SNN inference interleaves `prefix tick → current-spike mean pool → suffix tick`
for each local sweep. Pooling cumulative spike rates as if they were instantaneous
spikes would change the model and is not used. The final suffix's cumulative
local rate is scaled and rasterized only for a frame readout.

Structural graph changes and transient input pulse changes have separate masks.
Only structural seeds persist across an update's T sweeps and enter every layer;
pulse-on/off changes initially affect the first suffix layer, then propagate only
through actual emitted/ending spikes. Repeated identical nonzero pulses still
advance their dependants. An idle independent sequence does not advance because
another sequence received an event. Pending pulses, previous local spikes,
membranes and cumulative readout statistics are distinct state.

## Execution costs and measurement boundaries

The [dense-graph execution audit](DENSE_GRAPH_EXECUTION.md) documents the physical
meaning of the radius, exact cell-bound/bulk-count optimizations and one-pass
selected-neighbor inference. These reduce redundant work, not the true graph.

The recommended explicit raw storage option is `graph_storage=implicit_radius`.
It retains every selected radius edge mathematically, but regenerates bounded
neighbor chunks rather than allocating a full raw edge/basis array. Materialized
storage remains a compatibility/reference option; selecting its incident edges
and validating retained edges still scans the raw edge array. Pool work counters
make this cost explicit.

Coarse-edge witnesses use a tensor open-addressed hash accumulator. Full pair
equality resolves collisions, growth rehashes geometrically, and only incoming
chunks are coalesced; the complete Q-key result is sorted at export rather than
after every raw edge chunk. This removes the repeated accumulated-Q sorting
cost. Hash-table scratch grows with Q plus the bounded chunk, not raw E. Collision
probe rounds and capacity/rehash counters are recorded; GPU dispatch and dynamic
shape synchronization costs still require measurement on the allocation.
Pulse-only local ticks reuse the existing quotient edges/cells, without another
hash build or accumulated-Q sort. Frame `stream_execution.pooling` records
query/contributor work, accumulator rounds/capacity and quotient-reuse counts.

Pooling reduces the downstream node/edge representation as part of the requested
architecture; it does not eliminate the raw graph or the first four layers' work.
With R=1, the previously reported 424-million-edge raw window is still a dense
raw graph. Exact O(E) computation, node indexing/state copies and coarse-edge
storage remain real costs. This implementation does not promise that graph fits
a 9.5-GiB MIG slice, improves FPS, or achieves a given PSNR/SSIM.

Before any final study, the allocated device must pass the actual full-model,
physical-batch preflight and memory reserve. The complete chronological scan and
stateful probes remain required; a successful early probe is not full coverage.
No GPU is selected, no SSH/server is started, and no existing result is overwritten
by configuration preparation. Prepare v4 with the explicit `--hierarchical`
option in `scripts/prepare_streaming_experiment.py`; timestamp scales and the new
output directory remain mandatory, and sampling defaults to R=1.

## Verification status

Local tests cover differentiable means/quotient edges, append/expiry/refcounts,
feature deltas and repeated pulses, exact ANN snapshot equivalence, independent
batched ANN/SNN execution, state clone/replay and training resume counters, all
six layers' gradients/optimizer updates and calibration, and an independent
scalar IF/mean-pool oracle. Synthetic fixtures are explicitly not final training.

CUDA/Triton execution, full EventHDR/EventAid training/evaluation, real accelerator
memory/throughput and reconstruction quality are not established by CPU tests.
The classifier/head, datasets and undisclosed author choices are not reproduced;
this is an equation-derived ASGCN **reconstruction adaptation**, now including
its sampling and intermediate pooling path, not a claim of official reproduction.
