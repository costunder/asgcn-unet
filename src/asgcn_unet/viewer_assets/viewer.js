/* Local, read-only viewer. No model inference or external dependencies. */
"use strict";
(() => {
  const byId = (id) => document.getElementById(id);
  const text = (value) => value == null ? "" : typeof value === "string" ? value : JSON.stringify(value);
  const number = (value, digits = 0) => typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("ko-KR", { maximumFractionDigits: digits }) : "확인 불가";
  const base = new URL(".", window.location.href);
  const state = {
    catalog: null, dataset: null, frames: [], frame: null, revision: 0,
    catalogRevision: 0, imageRevision: 0, catalogRequest: null, frameRequest: null,
    graphRequest: null, neighborRequest: null, pair: null, graph: null,
    selectedNode: null, neighbors: [], neighborRevision: 0
  };

  function showError(error, context) {
    byId("error-message").textContent = context + ": " + (error instanceof Error ? error.message : text(error));
    byId("error-panel").hidden = false;
  }
  function clearError() { byId("error-panel").hidden = true; }
  function abort(name) {
    if (state[name]) state[name].abort();
    state[name] = null;
  }
  function relativeURL(value) {
    if (typeof value !== "string" || !value || value.startsWith("/") ||
        value.includes("\\") || /^[a-z][a-z0-9+.-]*:/i.test(value)) {
      throw new Error("서버 이미지 주소가 로컬 상대경로가 아닙니다.");
    }
    const resolved = new URL(value, base);
    if (resolved.origin !== base.origin || !resolved.pathname.startsWith(base.pathname)) {
      throw new Error("서버 이미지 주소가 현재 뷰어 범위를 벗어납니다.");
    }
    return value;
  }
  async function api(route, params, signal) {
    const query = new URLSearchParams(params);
    const response = await fetch("api/" + route + (query.size ? "?" + query.toString() : ""), {
      signal, cache: "no-store", credentials: "same-origin", headers: { Accept: "application/json" }
    });
    let result;
    try { result = await response.json(); }
    catch (_) { throw new Error("JSON 응답을 읽을 수 없습니다. HTTP " + response.status); }
    if (!response.ok) throw new Error(text(result.error || result.message || "HTTP " + response.status));
    return result;
  }
  function option(value, label) {
    const element = document.createElement("option");
    element.value = String(value);
    element.textContent = label;
    return element;
  }
  function cells(element, entries) {
    element.replaceChildren();
    for (const [label, value] of entries) {
      const item = document.createElement("div");
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      term.textContent = label;
      detail.textContent = value;
      item.append(term, detail);
      element.append(item);
    }
  }
  function modeLabel(mode) {
    const found = state.dataset && state.dataset.modes.find((item) => String(item.id) === String(mode));
    return found ? text(found.label || found.id) : text(mode);
  }
  function clearImage(id) {
    const image = byId(id);
    image.onload = image.onerror = null;
    image.removeAttribute("src");
  }
  function showImage(id, url, revision = state.revision, imageRevision = state.imageRevision) {
    const image = byId(id);
    image.onerror = () => {
      if (revision === state.revision && (imageRevision == null || imageRevision === state.imageRevision)) {
        showError(new Error("저장 이미지 파일을 읽을 수 없습니다."), image.alt);
      }
    };
    image.src = relativeURL(url);
  }
  function resetGraph() {
    abort("graphRequest");
    abort("neighborRequest");
    state.neighborRevision++;
    state.graph = null;
    state.selectedNode = null;
    state.neighbors = [];
    byId("graph-content").hidden = true;
    byId("raw-target-card").hidden = true;
    clearImage("raw-target-image");
    byId("node-index").value = "";
    byId("neighbor-status").textContent = "노드를 선택하면 표시용 엣지 부분집합과 별도로, 해당 노드의 정확한 전체 이웃을 강조합니다.";
    byId("load-graph").textContent = "그래프 불러오기";
    graphView.reset();
  }
  function updateCoverage() {
    if (!state.dataset) return;
    const total = number(state.dataset.total_frames);
    const saved = number(state.frames.length);
    const modes = state.dataset.modes.map((mode) =>
      text(mode.label || mode.id) + " " + number(mode.saved_frames) + "장").join(" · ");
    byId("coverage").textContent = "저장 이미지가 있는 프레임 " + saved +
      "개 / 전체 평가 프레임 " + total + "개. PNG 저장 수는 평가 샘플 수와 다릅니다." +
      (modes ? "  모드별 PNG: " + modes : "");
  }
  async function loadCatalog() {
    const revision = ++state.catalogRevision;
    abort("catalogRequest");
    abort("frameRequest");
    state.revision++;
    state.imageRevision++;
    resetGraph();
    state.catalogRequest = new AbortController();
    byId("loading-status").textContent = "결과 목록을 불러오는 중…";
    byId("reload-catalog").disabled = true;
    byId("dataset-select").disabled = true;
    byId("frame-select").disabled = true;
    byId("previous-frame").disabled = byId("next-frame").disabled = true;
    byId("comparison").hidden = byId("graph-panel").hidden = true;
    byId("empty-state").hidden = true;
    clearError();
    try {
      const catalog = await api("catalog", {}, state.catalogRequest.signal);
      if (revision !== state.catalogRevision) return;
      if (catalog.readonly !== true || !Array.isArray(catalog.datasets)) {
        throw new Error("읽기 전용 결과 목록의 형식이 올바르지 않습니다.");
      }
      state.catalog = catalog;
      byId("warnings").replaceChildren();
      for (const warning of catalog.warnings || []) {
        const item = document.createElement("li");
        item.textContent = text(warning);
        byId("warnings").append(item);
      }
      byId("warning-panel").hidden = !byId("warnings").children.length;
      const select = byId("dataset-select");
      select.replaceChildren();
      for (const dataset of catalog.datasets) select.append(option(dataset.id, text(dataset.label || dataset.id)));
      select.disabled = !catalog.datasets.length;
      if (!catalog.datasets.length) {
        state.dataset = null;
        select.append(option("", "결과 없음"));
        byId("coverage").textContent = "연결된 평가 결과가 없습니다.";
        byId("loading-status").textContent = "";
        byId("empty-state").hidden = false;
        return;
      }
      chooseDataset(select.value);
    } catch (error) {
      if (error.name !== "AbortError" && revision === state.catalogRevision) {
        byId("loading-status").textContent = "결과 목록을 불러오지 못했습니다. 목록 새로고침으로 다시 시도할 수 있습니다.";
        showError(error, "결과 목록");
      }
    } finally {
      if (revision === state.catalogRevision) byId("reload-catalog").disabled = false;
    }
  }
  function chooseDataset(id) {
    state.dataset = state.catalog.datasets.find((dataset) => String(dataset.id) === id);
    if (!state.dataset || !Array.isArray(state.dataset.frames) || !Array.isArray(state.dataset.modes)) {
      showError(new Error("데이터셋 목록의 형식이 올바르지 않습니다."), "데이터셋 선택");
      return;
    }
    state.frames = [...state.dataset.frames].sort((a, b) => a.index - b.index);
    const select = byId("frame-select");
    select.replaceChildren();
    for (const frame of state.frames) {
      select.append(option(frame.index, "#" + frame.index + " · " + text(frame.sample_id) +
        (frame.group ? " · " + text(frame.group) : "")));
    }
    select.disabled = !state.frames.length;
    updateCoverage();
    if (!state.frames.length) {
      state.revision++;
      state.imageRevision++;
      abort("frameRequest");
      resetGraph();
      state.frame = null;
      select.append(option("", "저장된 프레임 없음"));
      byId("comparison").hidden = byId("graph-panel").hidden = true;
      byId("empty-state").hidden = false;
      byId("previous-frame").disabled = byId("next-frame").disabled = true;
      byId("loading-status").textContent = "";
      return;
    }
    loadFrame(Number(select.value));
  }
  async function loadFrame(index) {
    const revision = ++state.revision;
    state.imageRevision++;
    abort("frameRequest");
    resetGraph();
    state.frame = null;
    state.pair = null;
    const id = String(state.dataset.id);
    const position = state.frames.findIndex((frame) => frame.index === index);
    byId("previous-frame").disabled = position <= 0;
    byId("next-frame").disabled = position < 0 || position >= state.frames.length - 1;
    byId("comparison").hidden = byId("graph-panel").hidden = true;
    byId("empty-state").hidden = true;
    byId("loading-status").textContent = "프레임 #" + index + "의 저장 결과를 불러오는 중…";
    state.frameRequest = new AbortController();
    clearError();
    try {
      const frame = await api("frame", { dataset: id, index }, state.frameRequest.signal);
      if (revision !== state.revision) return;
      if (String(frame.dataset) !== id || frame.index !== index || !Array.isArray(frame.images)) {
        throw new Error("요청한 데이터셋·프레임과 응답이 일치하지 않습니다.");
      }
      state.frame = frame;
      byId("loading-status").textContent = "";
      byId("sample-label").textContent = "#" + index + " · " + text(frame.sample_id);
      byId("frame-note").textContent = text(frame.note);
      byId("evaluation-domain").textContent = "평가용 정답: 그레이스케일 · 로그 톤매핑" +
        (frame.evaluation_domain ? " / " + text(frame.evaluation_domain) : "");
      const primary = byId("mode-select");
      const previousMode = primary.value;
      primary.replaceChildren();
      const second = byId("second-mode-select");
      second.replaceChildren(option("", "선택 안 함"));
      for (const image of frame.images) {
        primary.append(option(image.mode, modeLabel(image.mode)));
        second.append(option(image.mode, modeLabel(image.mode)));
      }
      if (frame.images.some((image) => String(image.mode) === previousMode)) primary.value = previousMode;
      byId("comparison").hidden = !frame.images.length;
      byId("empty-state").hidden = !!frame.images.length;
      byId("graph-panel").hidden = false;
      byId("load-graph").disabled = frame.graph_available !== true;
      byId("graph-status").textContent = frame.graph_available === true
        ? "버튼을 누르면 이 프레임의 원본 데이터를 CPU로 읽습니다. 모델 추론이나 GPU 작업은 시작하지 않습니다."
        : "이 프레임은 그래프를 불러올 수 없습니다. 저장된 이미지는 그대로 비교할 수 있습니다.";
      if (frame.images.length) renderImages();
    } catch (error) {
      if (error.name !== "AbortError" && revision === state.revision) {
        byId("loading-status").textContent = "프레임 결과를 불러오지 못했습니다.";
        showError(error, "프레임 #" + index);
      }
    }
  }
  function renderMetrics(id, metrics) {
    cells(byId(id), [
      ["PSNR · dB", number(metrics && metrics.psnr, 5)],
      ["SSIM", number(metrics && metrics.ssim, 5)],
      ["RMSE", number(metrics && metrics.rmse, 5)]
    ]);
  }
  function eligibility(id, value) {
    const element = byId(id);
    element.className = "badge" + (value === true ? " valid" : value === false ? " invalid" : "");
    element.textContent = value === true ? "보고 적격" : value === false ? "비보고·진단" : "적격성 미확인";
  }
  function renderImages() {
    if (!state.frame) return;
    const images = state.frame.images;
    const selected = images.find((image) => String(image.mode) === byId("mode-select").value);
    const second = images.find((image) => String(image.mode) === byId("second-mode-select").value);
    state.imageRevision++;
    state.pair = null;
    byId("difference-canvas").width = byId("difference-canvas").height = 0;
    byId("difference-status").textContent = "";
    try {
      if (!selected) throw new Error("선택한 모드의 저장 결과가 없습니다.");
      showImage("gt-image", selected.gt_url);
      showImage("prediction-image", selected.url);
      byId("prediction-label").textContent = modeLabel(selected.mode);
      eligibility("eligibility", selected.report_eligible);
      renderMetrics("prediction-metrics", selected.metrics);
      byId("second-card").hidden = !second;
      byId("image-grid").classList.toggle("with-second", !!second);
      if (second) {
        showImage("second-image", second.url);
        byId("second-label").textContent = modeLabel(second.mode);
        eligibility("second-eligibility", second.report_eligible);
        renderMetrics("second-metrics", second.metrics);
      } else clearImage("second-image");
      state.pair = selected;
      if (byId("difference-details").open) drawDifference();
    } catch (error) { showError(error, "이미지 비교"); }
  }
  function decodedImage(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("차이 영상을 위한 PNG를 읽을 수 없습니다."));
      image.src = relativeURL(url);
    });
  }
  async function drawDifference() {
    if (!state.pair || !byId("difference-details").open) return;
    const revision = state.revision;
    const imageRevision = state.imageRevision;
    const pair = state.pair;
    byId("difference-status").textContent = "저장된 8-bit PNG 픽셀을 읽는 중…";
    try {
      const [prediction, target] = await Promise.all([decodedImage(pair.url), decodedImage(pair.gt_url)]);
      if (revision !== state.revision || imageRevision !== state.imageRevision || !byId("difference-details").open) return;
      if (prediction.naturalWidth !== target.naturalWidth || prediction.naturalHeight !== target.naturalHeight) {
        throw new Error("예측과 정답 PNG 크기가 다릅니다. 크기를 임의로 바꿔 비교하지 않습니다.");
      }
      const canvas = byId("difference-canvas");
      canvas.width = prediction.naturalWidth;
      canvas.height = prediction.naturalHeight;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) throw new Error("브라우저에서 차이 영상 Canvas를 사용할 수 없습니다.");
      context.drawImage(prediction, 0, 0);
      const predicted = context.getImageData(0, 0, canvas.width, canvas.height);
      context.drawImage(target, 0, 0);
      const expected = context.getImageData(0, 0, canvas.width, canvas.height);
      const gain = Number(byId("difference-gain").value);
      for (let i = 0; i < predicted.data.length; i += 4) {
        for (let channel = 0; channel < 3; channel++) {
          predicted.data[i + channel] = Math.min(255, Math.abs(predicted.data[i + channel] - expected.data[i + channel]) * gain);
        }
        predicted.data[i + 3] = 255;
      }
      context.putImageData(predicted, 0, 0);
      byId("difference-status").textContent = canvas.width + " × " + canvas.height + " px · 절댓값 차이 " + gain + "× · metric 재계산 아님";
    } catch (error) {
      if (revision === state.revision && imageRevision === state.imageRevision) {
        byId("difference-status").textContent = "차이 영상을 만들지 못했습니다.";
        showError(error, "PNG 차이 영상");
      }
    }
  }
  function validateGraph(graph) {
    if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges) || !graph.statistics) {
      throw new Error("그래프 응답의 형식이 올바르지 않습니다.");
    }
    for (const node of graph.nodes) {
      if (!Array.isArray(node) || node.length !== 4 || !node.every(Number.isFinite)) {
        throw new Error("그래프에 유효하지 않은 이벤트 좌표가 있습니다.");
      }
    }
    for (const edge of graph.edges) {
      if (!Array.isArray(edge) || edge.length !== 2 ||
          !edge.every((id) => Number.isSafeInteger(id) && id >= 0 && id < graph.nodes.length) || edge[0] === edge[1]) {
        throw new Error("그래프에 유효하지 않은 엣지 또는 self-edge가 있습니다.");
      }
    }
    if (graph.statistics.nodes !== graph.nodes.length || graph.statistics.displayed_edges !== graph.edges.length) {
      throw new Error("그래프의 표시 데이터와 노드·엣지 개수 보고가 일치하지 않습니다.");
    }
  }
  async function loadGraph() {
    if (!state.frame || state.frame.graph_available !== true) return;
    const revision = state.revision;
    resetGraph();
    state.graphRequest = new AbortController();
    const request = state.graphRequest;
    byId("load-graph").disabled = true;
    byId("load-graph").textContent = "CPU로 읽는 중…";
    byId("graph-status").textContent = "원본 이벤트와 그래프를 CPU로 읽고 있습니다. 데이터 크기에 따라 시간이 걸릴 수 있습니다.";
    clearError();
    try {
      const graph = await api("graph", { dataset: state.frame.dataset, index: state.frame.index }, request.signal);
      if (revision !== state.revision || request !== state.graphRequest) return;
      validateGraph(graph);
      state.graph = graph;
      state.selectedNode = null;
      state.neighbors = [];
      const stats = graph.statistics;
      cells(byId("graph-statistics"), [
        ["전체 노드 · 모두 표시", number(stats.nodes)],
        ["실제 방향별 엣지", number(stats.actual_directed_edges)],
        ["표시용 엣지 부분집합", number(stats.displayed_edges)],
        ["고립 노드", number(stats.isolated_nodes)],
        ["최대 degree", number(stats.max_degree)]
      ]);
      byId("graph-rule").textContent = "연결 규칙: 정규화 " +
        (graph.position_dims === 2 ? "(x, y)" : "(x, y, t)") + " 거리 < " + number(graph.radius, 6) +
        " · self-edge 제외 · 양방향 연결. 방향별 엣지는 같은 화면 선으로 겹쳐 보일 수 있습니다.";
      const metadata = graph.metadata || {};
      cells(byId("graph-metadata"), [
        ["원본 이벤트", number(metadata.raw_events)],
        ["입력에 남긴 이벤트", number(metadata.retained_events)],
        ["센서 H × W", Array.isArray(metadata.sensor_size) ? metadata.sensor_size.map((n) => number(n)).join(" × ") : "확인 불가"],
        ["시간 시작 · μs", number(metadata.t0_us, 3)],
        ["시간 종료 · μs", number(metadata.t1_us, 3)]
      ]);
      byId("graph-provenance").textContent = text(graph.provenance_note);
      byId("node-index").max = Math.max(0, graph.nodes.length - 1);
      byId("graph-content").hidden = false;
      byId("graph-status").textContent = "노드 " + number(graph.nodes.length) + "개를 모두 표시합니다. 전체 그래프의 엣지를 줄인 것이 아니라, 화면에 그릴 엣지만 부분집합으로 표시합니다.";
      byId("raw-target-card").hidden = !graph.raw_target_url;
      if (graph.raw_target_url) showImage("raw-target-image", graph.raw_target_url, revision, null);
      graphView.reset();
      graphView.schedule();
    } catch (error) {
      if (error.name !== "AbortError" && revision === state.revision && request === state.graphRequest) {
        byId("graph-status").textContent = "그래프를 불러오지 못했습니다. 저장 이미지 비교는 계속 사용할 수 있습니다.";
        showError(error, "그래프");
      }
    } finally {
      if (revision === state.revision && request === state.graphRequest) {
        byId("load-graph").disabled = false;
        byId("load-graph").textContent = state.graph ? "그래프 다시 불러오기" : "그래프 불러오기";
      }
    }
  }
  async function selectNode(id) {
    if (!state.graph || !Number.isSafeInteger(id) || id < 0 || id >= state.graph.nodes.length) {
      showError(new Error("현재 그래프에 있는 노드 번호를 입력하세요."), "노드 선택");
      return;
    }
    abort("neighborRequest");
    state.neighborRequest = new AbortController();
    const request = state.neighborRequest;
    const revision = state.revision;
    const neighborRevision = ++state.neighborRevision;
    state.selectedNode = id;
    state.neighbors = [];
    byId("node-index").value = String(id);
    byId("neighbor-status").textContent = "노드 #" + id + "의 정확한 전체 이웃을 CPU로 조회하는 중…";
    graphView.schedule();
    try {
      const result = await api("neighbors", { dataset: state.frame.dataset, index: state.frame.index, node: id }, request.signal);
      if (revision !== state.revision || neighborRevision !== state.neighborRevision || request !== state.neighborRequest) return;
      if (result.node !== id || !Array.isArray(result.neighbors) ||
          !result.neighbors.every((neighbor) => Number.isSafeInteger(neighbor) && neighbor >= 0 &&
            neighbor < state.graph.nodes.length && neighbor !== id) ||
          new Set(result.neighbors).size !== result.neighbors.length || result.degree !== result.neighbors.length) {
        throw new Error("노드의 전체 이웃 응답과 degree가 일치하지 않습니다.");
      }
      state.neighbors = result.neighbors;
      const node = state.graph.nodes[id];
      byId("neighbor-status").textContent = "노드 #" + id + " · 정확한 전체 이웃 " + number(result.degree) +
        "개 · (x, y, t) = (" + node.slice(0, 3).map((value) => number(value, 4)).join(", ") +
        ") · 극성 " + (node[3] > 0 ? "+" : "−") + ". 주황색 연결은 표시용 부분집합이 아닌 이 노드의 전체 이웃입니다.";
      graphView.schedule();
    } catch (error) {
      if (error.name !== "AbortError" && revision === state.revision && neighborRevision === state.neighborRevision) {
        byId("neighbor-status").textContent = "전체 이웃을 조회하지 못했습니다. 현재 선택의 이웃 연결은 표시하지 않습니다.";
        showError(error, "노드 #" + id);
      }
    }
  }

  const graphView = (() => {
    const canvas = byId("graph-canvas");
    const context = canvas.getContext("2d");
    let view = "3d", yaw = -0.48, pitch = 0.3, zoom = 1, panX = 0, panY = 0;
    let points = [], width = 0, height = 0, scheduled = false, drag = null;
    function project(node) {
      let x = node[0] - 0.5, y = 0.5 - node[1], z = node[2] - 0.5;
      if (view === "xt") y = z;
      else if (view === "3d") {
        const turnedX = x * Math.cos(yaw) + z * Math.sin(yaw);
        const turnedZ = -x * Math.sin(yaw) + z * Math.cos(yaw);
        const turnedY = y * Math.cos(pitch) - turnedZ * Math.sin(pitch);
        z = y * Math.sin(pitch) + turnedZ * Math.cos(pitch);
        x = turnedX;
        y = turnedY;
      }
      const scale = Math.min(width, height) * (view === "3d" ? 0.65 : 0.78) * zoom;
      return [width / 2 + x * scale + panX, height / 2 - y * scale + panY, z];
    }
    function line(first, second) {
      context.moveTo(first[0], first[1]);
      context.lineTo(second[0], second[1]);
    }
    function axes() {
      const origin = project([0, 1, 0]);
      const ends = view === "xy" ? [[[1, 1, 0], "x"], [[0, 0, 0], "y"]]
        : view === "xt" ? [[[1, 1, 0], "x"], [[0, 1, 1], "t"]]
          : [[[1, 1, 0], "x"], [[0, 0, 0], "y"], [[0, 1, 1], "t"]];
      context.strokeStyle = "#63788e";
      context.fillStyle = "#b8cadc";
      context.lineWidth = 1;
      context.font = "12px system-ui, sans-serif";
      for (const [point, label] of ends) {
        const end = project(point);
        context.beginPath();
        line(origin, end);
        context.stroke();
        context.fillText(label, end[0] + 7, end[1] - 6);
      }
    }
    function draw() {
      scheduled = false;
      if (!context || !state.graph || byId("graph-content").hidden) return;
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      if (!width || !height) return;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);
      points = state.graph.nodes.map(project);
      axes();
      if (byId("show-edges").checked) {
        context.strokeStyle = "rgba(116,158,189,0.07)";
        context.lineWidth = 0.6;
        context.beginPath();
        for (const edge of state.graph.edges) line(points[edge[0]], points[edge[1]]);
        context.stroke();
      }
      for (const positive of [false, true]) {
        context.fillStyle = positive ? "#fc8f7c" : "#77c9ff";
        context.beginPath();
        for (let i = 0; i < points.length; i++) {
          if ((state.graph.nodes[i][3] > 0) === positive) {
            context.moveTo(points[i][0] + 1.65, points[i][1]);
            context.arc(points[i][0], points[i][1], 1.65, 0, Math.PI * 2);
          }
        }
        context.fill();
      }
      if (state.selectedNode != null && points[state.selectedNode]) {
        const selected = points[state.selectedNode];
        context.strokeStyle = "rgba(255,200,101,0.42)";
        context.lineWidth = 1;
        context.beginPath();
        for (const neighbor of state.neighbors) line(selected, points[neighbor]);
        context.stroke();
        context.fillStyle = "#ffc865";
        context.beginPath();
        for (const neighbor of state.neighbors) {
          context.moveTo(points[neighbor][0] + 2.7, points[neighbor][1]);
          context.arc(points[neighbor][0], points[neighbor][1], 2.7, 0, Math.PI * 2);
        }
        context.fill();
        context.fillStyle = "#ffffff";
        context.beginPath();
        context.arc(selected[0], selected[1], 5, 0, Math.PI * 2);
        context.fill();
        context.strokeStyle = "#ffca6f";
        context.lineWidth = 2;
        context.beginPath();
        context.arc(selected[0], selected[1], 8, 0, Math.PI * 2);
        context.stroke();
      }
    }
    function schedule() {
      if (!scheduled) { scheduled = true; window.requestAnimationFrame(draw); }
    }
    function reset() {
      yaw = -0.48; pitch = 0.3; zoom = 1; panX = panY = 0; drag = null; points = [];
      schedule();
    }
    function setView(value) {
      view = value;
      for (const button of document.querySelectorAll("[data-view]")) button.setAttribute("aria-pressed", String(button.dataset.view === view));
      byId("graph-view-label").textContent = view === "xy" ? "x · y / XY" : view === "xt" ? "x · t / XT" : "x · y · t / 3D";
      reset();
    }
    canvas.addEventListener("pointerdown", (event) => {
      if (!state.graph || event.button !== 0) return;
      drag = { x: event.clientX, y: event.clientY, yaw, pitch, panX, panY, moved: false };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      if (Math.hypot(dx, dy) > 4) drag.moved = true;
      if (view === "3d") {
        yaw = drag.yaw + dx * 0.007;
        pitch = Math.max(-1.5, Math.min(1.5, drag.pitch + dy * 0.007));
      } else { panX = drag.panX + dx; panY = drag.panY + dy; }
      schedule();
    });
    canvas.addEventListener("pointerup", (event) => {
      if (!drag) return;
      const clicked = !drag.moved;
      drag = null;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      if (!clicked || !state.graph) return;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left, y = event.clientY - rect.top;
      let nearest = null, distance = 144;
      for (let i = 0; i < points.length; i++) {
        const squared = (points[i][0] - x) ** 2 + (points[i][1] - y) ** 2;
        if (squared < distance) { distance = squared; nearest = i; }
      }
      if (nearest != null) selectNode(nearest);
    });
    canvas.addEventListener("pointercancel", () => { drag = null; });
    canvas.addEventListener("wheel", (event) => {
      if (!state.graph) return;
      event.preventDefault();
      zoom = Math.max(0.25, Math.min(6, zoom * Math.exp(-event.deltaY * 0.001)));
      schedule();
    }, { passive: false });
    canvas.addEventListener("keydown", (event) => {
      if (!state.graph) return;
      if (event.key === "0") reset();
      else if (event.key === "+" || event.key === "=") zoom = Math.min(6, zoom * 1.15);
      else if (event.key === "-") zoom = Math.max(0.25, zoom / 1.15);
      else if (event.key === "ArrowLeft") { if (view === "3d") yaw -= 0.1; else panX -= 15; }
      else if (event.key === "ArrowRight") { if (view === "3d") yaw += 0.1; else panX += 15; }
      else if (event.key === "ArrowUp") { if (view === "3d") pitch = Math.max(-1.5, pitch - 0.1); else panY -= 15; }
      else if (event.key === "ArrowDown") { if (view === "3d") pitch = Math.min(1.5, pitch + 0.1); else panY += 15; }
      else return;
      event.preventDefault();
      schedule();
    });
    if (typeof ResizeObserver === "function") new ResizeObserver(schedule).observe(canvas);
    else window.addEventListener("resize", schedule);
    return { schedule, reset, setView, supported: !!context };
  })();

  byId("dismiss-error").addEventListener("click", clearError);
  byId("reload-catalog").addEventListener("click", loadCatalog);
  byId("dataset-select").addEventListener("change", (event) => chooseDataset(event.target.value));
  byId("frame-select").addEventListener("change", (event) => loadFrame(Number(event.target.value)));
  function stepFrame(offset) {
    const position = state.frames.findIndex((frame) => frame.index === Number(byId("frame-select").value));
    const next = state.frames[position + offset];
    if (next) { byId("frame-select").value = String(next.index); loadFrame(next.index); }
  }
  byId("previous-frame").addEventListener("click", () => stepFrame(-1));
  byId("next-frame").addEventListener("click", () => stepFrame(1));
  byId("mode-select").addEventListener("change", renderImages);
  byId("second-mode-select").addEventListener("change", renderImages);
  byId("difference-details").addEventListener("toggle", drawDifference);
  byId("difference-gain").addEventListener("input", () => {
    byId("difference-gain-label").textContent = byId("difference-gain").value + "×";
    drawDifference();
  });
  byId("load-graph").addEventListener("click", () => {
    if (!graphView.supported) showError(new Error("브라우저에서 2D Canvas를 사용할 수 없습니다."), "그래프");
    else loadGraph();
  });
  for (const button of document.querySelectorAll("[data-view]")) button.addEventListener("click", () => graphView.setView(button.dataset.view));
  byId("show-edges").addEventListener("change", graphView.schedule);
  byId("reset-view").addEventListener("click", graphView.reset);
  byId("node-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const raw = byId("node-index").value.trim();
    selectNode(raw === "" ? NaN : Number(raw));
  });
  window.addEventListener("pagehide", () => {
    for (const name of ["catalogRequest", "frameRequest", "graphRequest", "neighborRequest"]) abort(name);
  });
  loadCatalog();
})();
