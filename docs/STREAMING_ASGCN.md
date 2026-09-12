# Stateful event-driven ASGCN reconstruction (architecture v3)

The preserved unpooled v3 contract is documented here. The new
[hierarchical v4 ASGCN path](HIERARCHICAL_ASGCN.md) adds actual intermediate mean
pooling, quotient edge remapping and sequence-global sampling state; it requires
a separately prepared study and new training. Existing v3 experiments are unchanged.

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

An explicitly selected [implicit-radius storage backend](IMPLICIT_STREAMING_BACKEND.md)
preserves this v3 reconstruction graph/operator while avoiding persistent full-edge
arrays. It does not add the original paper's graph clustering/pooling or pooled-edge
remapping, and it is not evidence of full ASGCN reproduction or a measured speedup.

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

With default `graph_storage="materialized"`, `stream_graph.evolve_stream_graph`
reuses surviving edge arrays and attributes; only new nodes issue radius queries.
The optional `implicit_radius` backend instead retains exact node degrees/counts
and queries incident neighbors of arrivals or expired nodes without storing E-sized
edge arrays. In both cases removed edges change the surviving affected endpoints;
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

Both backends scan/remap full node metadata, build spatial cell indices and
allocate remapped caches. The materialized backend additionally scans full edge
metadata and builds E-sized incidence/CSR arrays; implicit storage regenerates
bounded neighbor/message chunks instead. Neither eliminates dense O(E) message
work or guarantees O(K-hop)-only wall time/memory. Dense graphs can make the affected
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

Streaming preflight uses a separate schema and rejects old static profiles,
JSON-only partial scan reuse, and the unverified-preflight bypass. New raw-state
scan checkpoints can resume only under the exact matching contract described
below; an old progress JSON alone cannot supply that state. It checks the whole training stream,
records actual readout topology and separately labelled conservative prefix-union
bounds, then performs real stateful full-physical-batch CUDA training probes.
A conservative bound is not an observed intra-frame maximum. Guard refusals never
drop events or automatically raise memory limits. Neither sampled CUDA probes nor
snapshot resource checks are an absolute whole-run memory guarantee.

### Recovering a streaming preflight edge-guard failure

The topology phase is count-only: it keeps raw positions and clocks, not complete
edge-index/attribute tensors. Candidate-pair scratch is bounded independently of
edge count, including a dense single occupied cell. It uses the updater's fixed
float64 coordinates, collision-free occupied cells, and strict `norm(delta/r)<1`
predicate. Counts above the configured model guard are still recorded.
Unchanged prior edges reuse their exact previous counts; only pairs incident to
arrivals or expiring nodes are queried again. This saves repeated old-old radius
queries without caching a full edge index or approximating the graph. Partial
progress, failing batch indices and failure stage survive an ordinary failure or
interrupt; a partial scan never authorizes training.

For an already prepared experiment, do **not** repeat preparation, overwrite the
failed profile, or copy a v2 static dense-frame guard. To start a new scan when
there is no prior interrupted recovery to inspect, run:

```bash
python -B scripts/recover_streaming_preflight.py --experiment-root runs/streaming-v3-50ms --use-measured-edge-guard --reserve-vram-mib 1024 --cpu-threads 4
```

The path above refers to the separately chosen 50 ms experiment, not a new default
window. The tool preserves the current CUDA allocation; it does not select a GPU,
connect SSH, terminate sessions, start training, or start calibration. It creates
a unique `preflight-recovery-*` subdirectory on every invocation, so the original
configs, failed report, checkpoints and other experiments remain unchanged.

For an interrupted recovery, use `--resume-from` with its actual directory:

```bash
python -B scripts/recover_streaming_preflight.py --experiment-root runs/streaming-v3-50ms --use-measured-edge-guard --reserve-vram-mib 1024 --cpu-threads 4 --resume-from runs/streaming-v3-50ms/preflight-recovery-REPLACE_WITH_EXISTING_DIRECTORY
```

This first checks the saved recovery/profile commitments, prepared configuration
identities, recorded source/data contract and internally consistent per-frame
counts, without starting a CUDA probe or rescanning the dataset. A historical
materialized graph's necessary storage can already exceed its recorded device
capacity after the requested reserve; that diagnosis is **not** a new measurement
of the current data, executable source or device. The actual readout edge count,
not the conservative prefix-union bound, supplies this one-graph storage proof.

Older partial JSON reports did not serialize live streams and incremental count
state. They cannot provide exact resume, even when their configuration matches.
The tool preserves their diagnosis and refuses before a new GPU probe or full
scan; it never silently substitutes a frame-zero replay. Changed executable
source also prevents raw-state migration. A matching Git commit alone is not
sufficient; executable source bytes must match, while Git metadata is recorded
separately. Do not rerun a proven-infeasible materialized experiment unchanged.

New scans save raw scanner checkpoints under `scan-checkpoint/`, with an atomic
`latest.json` manifest, ownership and content hashes. Recovery metadata is
committed before the scan starts, and checkpoint progress profiles are committed
as they are saved. A valid checkpoint is still only a candidate: current source,
configuration, raw-data content, schedule and runtime contract must match before
the saved causal states are loaded. Resume continues after the last committed
batch; work after that checkpoint may be repeated. A partial or resumed scan
never upgrades reporting eligibility without completing the full scan and CUDA
gate. Every retry writes a separate recovery directory; prior checkpoints and
metadata remain untouched.

`--use-measured-edge-guard` explicitly authorizes a **new config** with the larger
of its existing guard and the measured complete-training prefix-union bound (at
least one). It does not reduce nodes, edges, radius, window, data, resolution,
epochs, model, or physical batch size. Ordinary `profile` without this opt-in
still stops before its model probe when the completed count exceeds its guard.

Before allocating a model, recovery compares a necessary graph/spline-basis
storage floor for the actual packed readout batches and resident preceding streams
with current device memory minus the requested reserve. This is only a lower
bound: activation/autograd buffers, optimizer, decoder context and temporary
copies require additional memory. Passing it is not proof that training fits.
The subsequent full physical-batch CUDA probe measures input loading through
forward/loss/backward/optimizer **and state commit/release**, plus allocator peaks
over causal replay. Live memory reserve checks surround every replay/probe batch;
they are snapshots, not hard GPU memory isolation. Failure never shrinks the model
or batch, suppresses an OOM, or grants a training certificate.

Only CUDA-eligible success prints the exact new config/profile training command
and the subsequent full-data calibration command. The script does not execute
either command. A nonempty existing training run instead requires explicit resume
review; it is never overwritten. Matching HDR/Aid configs are generated with the
same model guard and new evaluation output paths. The inherited static Aid guard
is explicitly removed and recorded; a training scan does **not** certify either
evaluation dataset or event-driven SNN inference memory.

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
