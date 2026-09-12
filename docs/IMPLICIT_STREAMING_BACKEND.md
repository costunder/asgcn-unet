# Exact implicit-radius storage for streaming reconstruction

This document explains the storage backend independently of architecture. It is
also used by [hierarchical ASGCN v4](HIERARCHICAL_ASGCN.md); v4 adds actual pooling
and pooled-edge remapping as a separate explicit architecture, not as an implicit
storage fallback. References below to missing pooling describe the preserved v3.

## What this change does, and does not, claim

`model.graph_storage="implicit_radius"` changes how the existing architecture-v3
streaming radius graph is stored and traversed. It does not remove edges, replace
the graph with a zero-edge placeholder, or introduce graph pooling. The alternative
`"materialized"` storage remains the default; existing configurations and results
are not silently converted.

For the agreed 50 ms study, the window, fixed physical time scale, all retained
events (`R=1`, `max_events=null`), radius 0.08, six 64-channel spline layers,
base-48 recurrent U-Net, full sensor resolution/data, physical training batch 16,
and 40 training epochs remain unchanged. The preparation API still requires
explicit physical time scales; selecting implicit storage does not invent or
override those values. Changing a clock scale changes the graph and is not a
storage optimization.

This is **not a claim to reproduce the original ASGCN architecture, accuracy,
speed, or energy results**. It remains this project's event-to-frame reconstruction
adaptation. The original graph clustering/pooling and pooled-edge remapping
pipeline are not implemented here. There is no pooled hierarchy or classification
head before the raster: the unpooled graph features feed the analog reconstruction
decoder. Adding that hierarchy requires a separate, explicit architecture design;
its undisclosed choices must not be invented as a way to make this implementation
fit memory. See [the v3 model contract](STREAMING_ASGCN.md) for the input/readout
adaptations and distinction from the paper.

## Preserved graph and numerical operator

The implicit index uses the same collision-free occupied-cell addressing and
strict float64 distance predicate as the materialized streaming updater:
`norm((position_i - position_j) / radius) < 1`. Self edges are excluded; every
accepted pair has both directed edges. The batch identifier is part of the cell
key, so independent streams cannot acquire cross-stream edges. Nodes at the
expiry cutoff are retained. No candidate chunk truncates the node or edge set.

An incoming-neighbor query for selected destinations includes their complete
neighborhood, not just selected sources. Cached degrees include zero-spike sources.
Degree-one open spline weights, projection, incoming-degree mean, root transform,
bias, BatchNorm/conversion parameters, and decoder are preserved. Coordinates are
fixed physical input geometry, not trainable inputs to this operator; requesting
position gradients fails explicitly.

The forward and ordinary backward regenerate bounded edge/basis chunks. They do
not retain full `edge_index`, `edge_attr`, or per-edge basis tensors for backward.
Feature and spline-weight gradients pass through the actual node projection;
root/bias/normalization/decoder remain in the connected training computation.
Accumulation order can differ from the materialized implementation. The guarantee
is exact topology and the same mathematical operator, with dtype-appropriate
floating-point tolerances, **not bitwise equality of every training trajectory**.

## Costs that remain

Let N be the number of live nodes in the packed graph and E its directed edges.

| Component | Stored or temporary work | Remaining limitation |
|---|---|---|
| Raw graph | O(N) features, float64 positions/times, lane IDs, degrees/counts | Node storage and state copies still grow with the full window |
| Occupied-cell index | O(N) rows, sorted node IDs and cell boundaries | Index construction still uses sorting/unique over live nodes |
| Spline projection | O(N × kernel size × channels) | No E-sized projection, but the node projection can still be large |
| Candidate and message chunks | Explicit bounded candidate pairs and message chunks | Bounds constrain scratch, not total pairs examined or edges processed |
| Persistent encoder/decoder state | Full retained per-node layer/IF caches and recurrent image state | Remapping, cloning, optimizer/activation buffers and decoder memory remain |

The current candidate-pair budget is 1,048,576 pairs per chunk; it is a scratch
limit, not an edge cap. Graph query chunks and spline message chunks are separately
bounded. These values do not establish a complete RAM/VRAM peak bound. A preflight
field that counts only necessary raw-node storage explicitly excludes the index,
projection, scratch, activations, optimizer and decoder; it is not proof of fit.

Append/expire updates reuse exact old degrees and query incident neighbors of
arrivals or removed nodes instead of recounting unchanged old-old edges. A new
live-node index can nevertheless require O(N) storage and whole-node indexing.
Layer inference queries complete incoming neighborhoods of affected destinations;
zero-valued SNN sources are omitted before learned projection, without changing
degree normalization. Dense K-hop dependency closure may touch the entire graph.

**O(E) message computation has not disappeared.** Exact dense neighborhoods still
have to be processed, and backward recomputes them. Candidate comparisons can
outnumber accepted edges. The portable query implementation also has a scalar
candidate-count synchronization per query chunk and dependent layer/local-tick
loops. Independent streams are tensor-batched, but causal arrival waves cannot be
merged to manufacture throughput. A reported `topology_indexed_edges=0` means no
full-E incidence/CSR array was built; it does not mean zero topology-query work.

This implementation may trade lower persistent edge memory for repeated radius
queries and kernel dispatches. It does not yet demonstrate a speedup, a particular
FPS, or successful training on a 1g.10gb MIG allocation.

## Training, calibration, and IF clocks

ANN training computes the full current causal-window snapshot, with pooled-node
BatchNorm and ordinary loss/backward/optimizer updates. Learned activation caches
cannot be reused after changing weights or training statistics. Training context
retains raw graph data and detached recurrent decoder state, not a stale learned
encoder cache. Calibration observes chronological, folded-BN ANN activations
before the existing parameter-normalization conversion is finalized.

ANN inference uses frozen-BN incremental affected-node updates. V3 SNN inference
retains local membranes, previous local spikes, cumulative spike sums/tick counts,
and pending one-sweep pulse endings. `literal_eq15` includes the previous local
spike; `standard_if` does not. T4 means four local sweeps per actual stream update,
not four sensor timestamps and not four whole-graph v2 time steps. Idle lanes do
not tick just because another lane receives events. The analog recurrent decoder
advances once per frame readout, not once per event.

The old v2 static-window checkpoints/results and their freshly initialized
whole-graph T-clock SNN remain different experiments. Implicit storage neither
makes their scores v3 results nor makes the new local clock equivalent to v2.

## Backend selection, provenance, and recovery

`scripts/prepare_streaming_experiment.py` exposes explicit
`--graph-storage implicit_radius`. It requires a new, nonoverlapping `--output-root`
inside the checkout and the same explicit raw-event/frame-interval clock scales.
Only a new configuration tree is prepared; preparation does not start training,
evaluation, SSH, a server, or a GPU operation. Existing output roots are rejected.
This document deliberately provides no server execution command or instruction
to rerun the lengthy scan.

`graph_storage` and `spline_backend` are separate settings. The checked-in full
baseline preparation inherits its explicit `spline_backend="triton"`; choosing
implicit storage does not silently replace it with torch. `torch`, `torch_fused`,
and `triton` route to their named message backend. Triton directly receives each
regenerated bounded chunk, requires a supported CUDA environment, and fails when
unavailable. As in the existing Triton operator, requested higher-order derivatives
use documented differentiable tensor arithmetic; that path does not have the
ordinary-backward saved-memory bound. Actual CUDA/Triton execution is unverified
by the CPU evidence below.

The storage kind is part of the v3 stream/model provenance. Existing materialized
state cannot be fed into an implicit sequence, or vice versa. The implicit raw
training-state payload uses its own representation metadata and stores nodes,
degrees, counts and clocks, not fabricated edge arrays. Untrusted restore verifies
the exact geometric degrees using bounded neighbor queries; a merely plausible
degree sum is not sufficient. Old checkpoints/configs are not silently relabelled.

The old `ad1b0af` partial progress JSON did not contain the live raw stream states.
It therefore cannot resume that scan exactly at its displayed cursor. Counts and
completed-row metadata do not reconstruct live nodes, positions, clocks and lane
state. The new raw-state scan checkpoint mechanism requires matching source,
configuration, dataset, batch schedule and report identity; it does not recover
information absent from an older JSON or authorize a cross-version state migration.
Preserve that partial report and all experiments. Do not automatically restart
another roughly 25-hour scan or describe a fresh scan as a resumed one.

## Verification recorded for this change

The final frozen-source local regression run passed **2,939 tests**, with **84
environment-dependent skips** and one existing PyTorch quantized-tensor
deprecation warning. `ruff check src scripts tests` and `git diff --check` passed.
This was a one-thread CPU test run, not server/GPU or real-data certification.
No commit or push was performed as part of this verification.

Both count-only scanning and chronological training probes now use the configured
CPU DataLoader workers and prefetch settings, retaining the full physical batch
schedule. The loader checks a host-memory plan before spawning, uses an isolated
explicit RNG seed, records decode/collate versus consumer-wait/transfer timings,
and closes only its own workers. It never changes a worker count, batch, or data
scope to make a failed plan pass. Raw-state scan checkpoints bind the completed
batch cursor, source, configuration, data and schedule; prefetched-but-unconsumed
batches do not advance that cursor. These tests do not establish server throughput.

The backend's 76 small synthetic CPU unit tests and 19 small-input CPU integration
tests passed. The integration models kept the full six layers, width 64, base-48
decoder and 4,409,617 parameters; only synthetic test inputs were small. They
checked exact directed topology/pseudo-coordinates, chunk bounds, first/higher
derivatives, casts and selected/masked sums; actual model outputs/loss/all-parameter
gradients and optimizer steps; calibration; ANN and both SNN T4 local clocks/spike
counts; expiry, empty/cold states, later arrivals, lane independence, cloned-state
replay, incompatible storage-state rejection, and explicit refusal to run an
implicit graph through the static global-T SNN API. These fixtures are not a reduced
production model or dataset and do not replace full-scale testing.

To repeat only these local CPU checks from the repository's Windows checkout:

```powershell
.venv/Scripts/python.exe -B -c "import tempfile, pytest, torch; torch.set_num_threads(1); raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','--basetemp',tempfile.mkdtemp(prefix='asgcn-implicit-cpu-'),'tests/test_implicit_radius.py','tests/test_implicit_spline.py','tests/test_implicit_stream_graph.py','tests/test_implicit_stream_model.py']))"
```

No real-data full training or evaluation, CUDA/Triton run, allocated-device memory
measurement, PSNR/SSIM validation, FPS improvement, or energy result is established
by these tests. Source/config/data provenance, an explicitly assigned accelerator,
safe full-scale resource evidence, and new matched quality/latency measurements
are still required before making those claims.
