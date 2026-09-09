# Stateful event-driven ASGCN reconstruction (architecture v3)

## Scope and evidence

This is an event-to-frame reconstruction adaptation of the public
[ASGCN paper](https://ojs.aaai.org/index.php/AAAI/article/download/32154/34309),
not its authors' classification implementation, trained weights, pooling/classifier
pipeline, or published accuracy/energy reproduction. Public equations do not
specify this project's physical clock scales, sliding duration, reconstruction
readout, or event-local clock scheduling. Those choices are explicit below.

The old v2 paths (`fast`, `graph_unet`/C/D, and their existing checkpoints/results)
remain static-window baselines. No saved result is relabelled as asynchronous.
V3 requires new training; exact model/config checks reject v2 checkpoint reuse.
`paper_core_version=2` still identifies the shared spline/BN/conversion equations;
`model_config.architecture_version=3` identifies the new input/execution contract.

Implemented code paths are tested with **synthetic CPU unit/integration fixtures**.
Those tests are not real-data training, CUDA kernel validation, PSNR/SSIM evidence,
an inference-speed improvement, or measured energy consumption. Actual device
preflight and full quality/latency evaluation remain required on the allocation.

## Input and spatial/temporal geometry

- `event_time_contract=physical_seconds_v1` keeps every post-ROI event. Production
  preparation retains full resolution, so it does not introduce an ROI/subset.
- `max_events=null` and model sampling factor `R=1` remove the future-dependent
  8,192-event linspace selection. No node/edge cap silently trims the graph.
- Raw event times are converted with `timestamp_scale_to_seconds`; frame boundaries
  separately use `interval_timestamp_scale_to_seconds`. Both must be explicitly
  supplied. A shared clock does **not** mean a shared storage unit. Timestamp
  offsets, mismatches, nonfinite values, and lost time precision are not guessed away.
- Events retain float64 physical times and stable source row IDs. Equal timestamps
  are simultaneous; distinct timestamps are never merged into a faster logical step.
- Spatial coordinates are x/(W-1), y/(H-1). Topology time is (event_seconds - fixed
  sequence_origin_seconds)/time_scale_seconds, never a window's first/last-event
  normalization. Existing nodes keep the same coordinates across arrivals/expiry.
- The four node features are normalized x/y, **physical offset from the previous
  frame boundary divided by time_scale_seconds**, and +/- polarity. That birth-frame
  offset remains fixed for a node's lifetime; it does not depend on future events or
  the current sliding-window endpoint. It is a declared reconstruction-specific feature.
- Nodes older than current watermark minus `window_seconds` expire. Equality at
  the cutoff is retained. Watermarks advance on arrivals and frame readouts, not
  on another sequence's arrivals. Radius and all dimensions remain explicit.
- EventHDR's official stored/recovered predecessor-index boundary is retained and
  checked against adjacent raw rows. A proven late predecessor is identified in
  metadata; arbitrary clock offsets are rejected. EventAid event/frame clocks are
  strictly checked. File/part identity prevents unrelated sequences sharing state.

The existing radius 0.08 is numerically preserved, but its temporal meaning now
depends on the declared physical scale. Old and new inputs are **not** an identical
experimental condition. All older quality tables remain valid only for their
recorded contracts, not as matched controls for this new study.

## Computation and persistent state

The shared network remains six 64-channel spline layers, scalar open degree-one
kernel size five, root transform, bias, and incoming-degree mean aggregation.
The raster and base-48 recurrent U-Net/ConvGRU/head remain analog. The decoder runs
once at a frame readout; an event update does not silently advance ConvGRU.

`stream_graph.evolve_stream_graph` reuses surviving edges and attributes. Only new
nodes issue radius neighbor queries. Removed edges change both affected endpoints;
zero-valued messages still count in the complete incoming degree. Independent
streams use one disjoint namespace with no inter-stream edge.

`stream_encoder.update_encoder` remaps retained per-node caches and computes
learned projections/messages only for affected destinations and their incident
sources. Later layers propagate changes through the actual graph dependencies.
SNN messages omit zero-valued sources before projection, not after a full dense
projection. Root and folded bias remain part of the affine computation.

ANN **training** deliberately recomputes the complete current causal-window graph
with pooled-node BatchNorm, loss, backward and optimizer updates. Learned activation
caches cannot survive changed weights or training BN statistics. The next training
state contains only raw graph data plus detached decoder context. ANN **inference**
uses frozen-BN incremental updates, tested against full snapshot recomputation.
Full chronological calibration observes the same causal-window ANN features and
retains no profile/sample maxima in its final commitment.

### Explicit event-local IF clock

V3 SNN is not equivalent to v2's freshly initialized whole-graph T-clock output.
Each actual arrival/expiry/readout update runs T local sweeps, layer 0 through layer
5 in feedforward order. Only affected neurons advance their local clock. Thus bias
is added at a local update, not continuously in wall-clock time. No unsupported
claim of equivalence to a global-clock bias-driven scheduler is made.

New neurons start at half threshold; retained neurons retain membrane, previous
local spike, emitted-spike sum, and local tick count. `literal_eq15` includes the
previous local spike; `standard_if` omits it. Both use threshold-valued spikes and
soft reset. Previous local spikes and transmitted one-sweep pulses are separate:
a pending pulse-off propagates on the next active update, never by holding a
spike indefinitely. Idle graph lanes retain pending pulses and do not tick when
another graph receives more arrivals.

Readout is each live node's cumulative threshold-valued spike sum divided by its
own local tick count, then the existing output normalization scale and cell-mean
raster. Reported firing counts/denominators instead count **actual operations in
the current frame invocation**, avoiding repeated aggregation of old history.
T4 here is four local sweeps per update, not four sensor timestamps and not the
old static T4 experiment. It is not established as optimal.

### Costs and limitations

The implementation still scans/remaps some full node/edge metadata, builds spatial
cell/CSR indices, and allocates remapped caches. These costs are exposed; an
O(K-hop)-only wall-time or memory claim is false. Dense graphs can make the affected
closure nearly the whole graph. Sequential causal arrival waves cannot be merged
without changing this IF model. Independent streams in a wave are batched on the
device; layers/local ticks have real causal dependencies.

Consequently GPU FPS may be lower or higher depending on graph density, event rate,
indexing overhead and sparsity. Lower operation counts are not proof of lower GPU
latency or measured power. Profile before claiming either speed or energy benefits.

## Safe preparation and execution

`scripts/prepare_streaming_experiment.py` requires a new `--output-root` inside the
checkout and these **six explicitly chosen** positive time quantities:

| Flag | Meaning |
|---|---|
| `--window-seconds` | Physical sliding graph lifetime |
| `--time-scale-seconds` | Fixed physical normalization for graph distance/features |
| `--hdr-timestamp-scale-to-seconds` | EventHDR event storage unit to seconds |
| `--hdr-interval-timestamp-scale-to-seconds` | EventHDR image-clock unit to seconds |
| `--aid-timestamp-scale-to-seconds` | EventAid TXT event storage unit to seconds |
| `--aid-interval-timestamp-scale-to-seconds` | EventAid frame-clock unit to seconds |

No production time constant is silently selected. Use the original format's clock
specification and measured data intervals to choose them. The preparation report
lists every difference from the existing configs. It preserves depth/width/radius,
full data/resolution, 40 epochs and physical batch 16, and creates only new files.
It does not connect SSH, change GPU masks, start training or alter any old run.

After preparation, use the ordinary CLI with the generated config paths:

1. `profile --config <new-root>/configs/train.json --output <new-root>/stream-profile.json`
2. `train --config <new-root>/configs/train.json --preflight-report <new-root>/stream-profile.json`
3. `calibrate` with that same training config and the new ANN checkpoint.
4. `evaluate` and `benchmark` with `<new-root>/configs/hdr.json` and `aid.json`,
   the new checkpoints, and explicit ANN/SNN mode/T/dynamics as appropriate.

The lines above list CLI arguments, not shell-ready commands with guessed time
values or GPU IDs. `python -B -m asgcn_unet.cli --help` documents the command syntax.
Preserve the scheduler/container's actual allocation. No script chooses GPU 0/4.

Streaming preflight uses a separate schema and rejects old static profiles, scan
reuse, and the unverified-preflight bypass. It checks the whole training stream,
records actual readout topology and separately labelled conservative prefix-union
bounds, then performs real stateful full-physical-batch CUDA training probes.
A conservative bound is not an observed intra-frame maximum. Guard refusals never
drop events or automatically raise memory limits. Neither sampled CUDA probes nor
snapshot resource checks are an absolute whole-run memory guarantee.

Calibration/evaluation batch and worker candidates are measured. Stateful
inference profile trials bootstrap the causal prefix, with that preparation cost
explicitly included in their timing; they are not steady-state throughput claims.
Benchmarking replays full sequence prefixes even if the decoder's recurrence is
disabled: the graph/SNN is still stateful. Prefix replay may be expensive and is
outside the scored frame set. Full quality evaluation, sample compute-only speed,
evaluation-loop throughput and total job duration remain distinct measurements.

Legacy single-frame graph previews, saved-result graph generators, single-sample
probes and `scan-eval-topology` explicitly refuse v3/physical inputs: they do not
have predecessor graph/IF state and must not silently normalize a physical frame
as v2. Existing v2 visualization is preserved; newly evaluated prediction/GT PNGs
are still saved and viewable. A stateful graph visualization exporter is not
implemented by this change.

Training checkpoints store a validated, CPU-owned raw-graph/decoder state under
context schema v2; legacy tensor-only context v1 remains supported. A failed AMP
attempt receives the same immutable incoming state on retry. No failed attempt
commits a graph update, kills a session, overwrites another experiment, or switches
to a smaller model/CPU fallback.
