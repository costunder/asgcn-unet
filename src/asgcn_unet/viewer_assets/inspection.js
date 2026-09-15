/* Offline inspection only. No request, model, neighbor inference, or node cap. */
(() => {
  "use strict";
  const root = document.getElementById("graph-inspection");
  const el = id => document.getElementById(id);
  const integer = value => Number.isSafeInteger(value) && value >= 0;
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const number = value => finite(value) ? value.toLocaleString("en-US", {maximumFractionDigits: 9}) : "미측정";
  const exact = value => String(value);
  const put = (id, value) => { el(id).textContent = value; };
  const fail = message => {
    put("inspection-error", `그래프 표시를 중단했습니다: ${message}`);
    el("inspection-error").hidden = false;
    el("inspection-content").hidden = true;
  };
  let data, nodes, nodeMap, queries;
  try {
    data = JSON.parse(el("inspection-data").textContent);
    if (data.schema !== "asgcn_graph_inspection_view_v1" || !Array.isArray(data.nodes)
        || !Array.isArray(data.queries) || !data.window || !data.geometry || !data.source
        || !["all_window_nodes", "query_neighborhoods_only"].includes(data.coverage)
        || typeof data.synthetic_fixture !== "boolean") throw Error("지원하지 않는 검사 자료 형식입니다.");
    const size = data.window.sensor_size;
    if (!Array.isArray(size) || size.length !== 2 || !size.every(value => integer(value) && value > 0)
        || !integer(data.window.nodes) || ![data.window.readout_seconds, data.window.cutoff_seconds,
          data.window.window_seconds, data.geometry.radius, data.geometry.time_scale_seconds,
          data.geometry.sequence_origin_seconds].every(finite)
        || data.window.window_seconds <= 0 || data.geometry.radius <= 0 || data.geometry.time_scale_seconds <= 0)
      throw Error("센서 크기 또는 시간·반경 기록이 올바르지 않습니다.");
    nodes = data.nodes;
    nodeMap = new Map();
    const rowIds = new Set();
    for (const node of nodes) {
      if (!Array.isArray(node) || node.length !== 9 || !integer(node[0]) || !integer(node[1])
          || !node.slice(2).every(finite) || node[0] >= data.window.nodes
          || nodeMap.has(node[0]) || rowIds.has(node[1]) || ![-1, 0, 1].includes(node[5])
          || node[2] < 0 || node[2] >= size[1] || node[3] < 0 || node[3] >= size[0])
        throw Error("노드 좌표·원시 행 번호가 잘못되었거나 중복되었습니다.");
      nodeMap.set(node[0], node);
      rowIds.add(node[1]);
    }
    if (nodes.length > data.window.nodes
        || (data.coverage === "all_window_nodes" && nodes.length !== data.window.nodes))
      throw Error("표시 노드 수와 선언한 검사 범위가 일치하지 않습니다.");
    if (data.total_directed_edges !== null && !integer(data.total_directed_edges))
      throw Error("전체 엣지 수 기록이 올바르지 않습니다.");
    queries = data.queries;
    const queryIds = new Set();
    for (const query of queries) {
      if (!nodeMap.has(query.node_index) || queryIds.has(query.node_index)
          || query.raw_row_id !== nodeMap.get(query.node_index)[1]
          || query.oracle_match !== true || !integer(query.in_degree)
          || !Array.isArray(query.neighbors) || query.in_degree !== query.neighbors.length
          || new Set(query.neighbors).size !== query.neighbors.length
          || query.neighbors.some(id => !integer(id) || id === query.node_index || !nodeMap.has(id)))
        throw Error("검증된 기준 노드와 이웃 목록이 일치하지 않습니다. 미검증 연결은 표시하지 않습니다.");
      queryIds.add(query.node_index);
    }
  } catch (error) { fail(error.message); return; }

  const [sensorH, sensorW] = data.window.sensor_size;
  let activeQuery = queries[0] || null;
  let inspected = activeQuery ? activeQuery.node_index : (nodes[0]?.[0] ?? null);
  let yaw = 35, pitch = 25, scheduled = false, focused = false;
  const canvases = [el("xy-canvas"), el("space-canvas")];
  const projected = new Map();
  const hitBounds = new Map();
  const nodeOffsets = new Map(nodes.map((node, index) => [node[0], index]));
  const colors = () => {
    const style = getComputedStyle(root);
    return Object.fromEntries(["surface", "ink", "muted", "line", "grid", "positive", "negative", "neighbor", "focus"]
      .map(name => [name, style.getPropertyValue(`--${name}`).trim()]));
  };
  const bounds = [[Infinity, -Infinity], [Infinity, -Infinity], [Infinity, -Infinity]];
  for (const node of nodes) for (let axis = 0; axis < 3; axis++) {
    bounds[axis][0] = Math.min(bounds[axis][0], node[6 + axis]);
    bounds[axis][1] = Math.max(bounds[axis][1], node[6 + axis]);
  }
  if (!nodes.length) bounds.forEach(range => { range[0] = range[1] = 0; });
  function viewNodes() { return focused && activeQuery ? [activeQuery.node_index, ...activeQuery.neighbors].map(id => nodeMap.get(id)) : nodes; }
  function viewBounds(offset) {
    const result = Array.from({length: 3}, () => [Infinity, -Infinity]);
    for (const node of viewNodes()) for (let axis = 0; axis < 3; axis++) {
      result[axis][0] = Math.min(result[axis][0], node[offset + axis]);
      result[axis][1] = Math.max(result[axis][1], node[offset + axis]);
    }
    if (!viewNodes().length) result.forEach(range => { range[0] = range[1] = 0; });
    return result;
  }

  put("source-label", `${data.source.file_name || "원본 파일"} · 파일 내 프레임 ${data.source.frame_index ?? "미기록"} · readout ${exact(data.window.readout_seconds)} s`);
  el("fixture-notice").hidden = !data.synthetic_fixture;
  put("coverage-notice", data.coverage === "all_window_nodes"
    ? "이 파일에는 윈도우의 모든 노드가 포함되어 있습니다. 전체 엣지 그림이 아니라, 선택한 기준 노드의 검증된 이웃 연결만 표시합니다."
    : "부분 표시: 검증한 기준 노드들과 그 이웃만 포함되어 있습니다. 전체 윈도우의 이벤트 분포나 전체 그래프 모습으로 해석하면 안 됩니다.");
  put("node-count", `${number(nodes.length)} / ${number(data.window.nodes)}`);
  put("edge-count", data.total_directed_edges === null ? "미측정 · 알 수 없음" : number(data.total_directed_edges));
  put("metric-ranges", bounds.map((range, index) => `${["X", "Y", "T"][index]}: ${number(range[0])} … ${number(range[1])}`).join("  |  "));
  const select = el("query-select");
  for (const query of queries) {
    const option = document.createElement("option");
    option.value = String(query.node_index);
    option.textContent = `노드 ${query.node_index} · 원시 행 ${query.raw_row_id} · 이웃 ${number(query.in_degree)}`;
    select.append(option);
  }
  if (!queries.length) {
    const option = document.createElement("option");
    option.textContent = "검증된 기준 노드 없음";
    select.append(option);
    select.disabled = true;
    el("focus-neighbors").disabled = true;
  }
  const addDetails = (target, rows) => {
    target.replaceChildren();
    for (const [label, value] of rows) {
      const item = document.createElement("div"), term = document.createElement("dt"), detail = document.createElement("dd");
      term.textContent = label; detail.textContent = value; item.append(term, detail); target.append(item);
    }
  };
  addDetails(el("contract-detail"), [
    ["윈도우", `${exact(data.window.cutoff_seconds)} ≤ t ≤ ${exact(data.window.readout_seconds)} s (이미 전달된 이벤트)`],
    ["시간 길이", `${exact(data.window.window_seconds)} s`], ["고정 시간 원점", `${exact(data.geometry.sequence_origin_seconds)} s`],
    ["시간 척도", `${exact(data.geometry.time_scale_seconds)} s`], ["연결 반경", `${exact(data.geometry.radius)} (그래프 좌표 단위)`],
    ["거리 조건", data.geometry.predicate || "저장된 반경 조건 미기록"], ["센서 H × W", `${sensorH} × ${sensorW} px`],
    ["표시 연결 방향", "각 이웃(source) → 선택한 기준 노드(target)"],
    ["전체 엣지 수", data.total_directed_edges === null ? "미측정: 이웃 수를 전체 E로 대체하지 않음" : `${number(data.total_directed_edges)} 방향 엣지`],
  ]);
  function updateNode() {
    const node = nodeMap.get(inspected);
    if (!node) { put("node-status", "표시할 노드가 없습니다."); el("node-detail").replaceChildren(); return; }
    el("node-index").value = String(node[0]);
    put("node-status", `노드 ${node[0]} · 원시 행 ${node[1]}${activeQuery?.node_index === node[0] ? " · 현재 검증 기준 노드" : " · 상세 선택은 표시 연결의 기준을 바꾸지 않습니다."}`);
    addDetails(el("node-detail"), [["윈도우 내 노드 번호", exact(node[0])], ["원본 이벤트 행", exact(node[1])],
      ["센서 X · 전처리 후", `${exact(node[2])} px`], ["센서 Y · 전처리 후", `${exact(node[3])} px`],
      ["물리 timestamp", `${exact(node[4])} s`], ["표시용 극성 부호", exact(node[5])],
      ["그래프 좌표 X / Y", `${exact(node[6])} / ${exact(node[7])}`], ["그래프 시간 좌표 T", exact(node[8])]]);
  }
  function updateQuery() {
    put("degree-count", activeQuery ? `${number(activeQuery.in_degree)}개` : "기준 노드 없음");
    put("query-status", activeQuery ? `저장된 독립 검사 결과: 일치 · ${number(activeQuery.in_degree)}개 이웃 전체 → 노드 ${activeQuery.node_index}` : "검증된 이웃 연결 기록이 없습니다.");
    updateNode(); updateViewScope(); schedule();
  }
  function updateViewScope() {
    root.dataset.viewScope = focused ? "query_neighborhood" : "available_nodes";
    el("focus-neighbors").setAttribute("aria-pressed", String(focused));
    el("show-all").setAttribute("aria-pressed", String(!focused));
    put("view-scope", focused
      ? "선택 노드와 모든 검증 이웃이 보이도록 화면만 확대합니다. 화면 밖 노드는 잘릴 수 있으며, 원본 노드·연결은 변경하지 않습니다."
      : "XY는 전체 센서, 3D는 파일에 포함된 전체 노드 범위입니다. 확대하면 선택 이웃의 연결 모양을 자세히 볼 수 있습니다.");
  }
  function setup(canvas) {
    const width = canvas.clientWidth, height = canvas.clientHeight;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(width * ratio)); canvas.height = Math.max(1, Math.round(height * ratio));
    const ctx = canvas.getContext("2d");
    if (!ctx) throw Error("이 브라우저에서 2D 캔버스를 사용할 수 없습니다.");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.font = "12px system-ui, sans-serif"; ctx.lineWidth = 1;
    return {ctx, width, height};
  }
  function cloud(ctx, points, palette) {
    // Every exported node is drawn. No downsampling, nearest-neighbor inference,
    // hidden edge/node limit, or CPU graph reconstruction occurs in this view.
    for (const positive of [false, true]) {
      ctx.fillStyle = positive ? palette.positive : palette.negative; ctx.globalAlpha = 0.58; ctx.beginPath();
      for (let i = 0; i < nodes.length; i++) if ((nodes[i][5] > 0) === positive) {
        const [x, y] = points[i];
        if (positive) { ctx.moveTo(x + 1.7, y); ctx.arc(x, y, 1.7, 0, Math.PI * 2); }
        else ctx.rect(x - 1.5, y - 1.5, 3, 3);
      }
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    const pointFor = id => points[nodeOffsets.get(id)];
    if (activeQuery) {
      const target = pointFor(activeQuery.node_index);
      ctx.strokeStyle = palette.neighbor; ctx.lineWidth = 1; ctx.globalAlpha = 0.45; ctx.beginPath();
      for (const id of activeQuery.neighbors) {
        const point = pointFor(id), dx = target[0] - point[0], dy = target[1] - point[1], length = Math.hypot(dx, dy);
        ctx.moveTo(point[0], point[1]); ctx.lineTo(target[0], target[1]);
        if (length > 12) {
          const ux = dx / length, uy = dy / length, ax = target[0] - ux * 5, ay = target[1] - uy * 5;
          ctx.moveTo(ax - ux * 5 - uy * 2.5, ay - uy * 5 + ux * 2.5); ctx.lineTo(ax, ay);
          ctx.lineTo(ax - ux * 5 + uy * 2.5, ay - uy * 5 - ux * 2.5);
        }
      }
      ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = palette.neighbor; ctx.beginPath();
      for (const id of activeQuery.neighbors) { const p = pointFor(id); ctx.moveTo(p[0] + 2, p[1]); ctx.arc(p[0], p[1], 2, 0, Math.PI * 2); }
      ctx.fill();
      ctx.fillStyle = palette.surface; ctx.strokeStyle = palette.ink; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(target[0], target[1], 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
    if (nodeOffsets.has(inspected) && inspected !== activeQuery?.node_index) {
      const point = pointFor(inspected); ctx.strokeStyle = palette.focus; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(point[0], point[1], 6, 0, Math.PI * 2); ctx.stroke();
    }
  }
  function drawXY(palette) {
    const canvas = canvases[0], {ctx, width, height} = setup(canvas);
    const ranges = focused ? viewBounds(2).slice(0, 2).map(range => {
      const pad = Math.max(0.75, (range[1] - range[0]) * 0.12);
      return [range[0] - pad, range[1] + pad];
    }) : [[0, sensorW - 1], [0, sensorH - 1]];
    const [xRange, yRange] = ranges;
    const spanX = Math.max(xRange[1] - xRange[0], 1), spanY = Math.max(yRange[1] - yRange[0], 1);
    const availableW = Math.max(1, width - 84), availableH = Math.max(1, height - 76);
    const scale = Math.min(availableW / spanX, availableH / spanY);
    const plotW = spanX * scale, plotH = spanY * scale;
    const left = 54 + (availableW - plotW) / 2, top = 18 + (availableH - plotH) / 2;
    ctx.fillStyle = palette.muted; ctx.strokeStyle = palette.grid;
    for (let i = 0; i <= 3; i++) {
      const x = left + plotW * i / 3, y = top + plotH * i / 3;
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotH); ctx.moveTo(left, y); ctx.lineTo(left + plotW, y); ctx.stroke();
      ctx.textAlign = "center"; ctx.fillText(Number((xRange[0] + (xRange[1] - xRange[0]) * i / 3).toFixed(2)).toString(), x, top + plotH + 20);
      ctx.textAlign = "right"; ctx.fillText(Number((yRange[0] + (yRange[1] - yRange[0]) * i / 3).toFixed(2)).toString(), left - 9, y + 4);
    }
    ctx.strokeStyle = palette.line; ctx.strokeRect(left, top, plotW, plotH);
    ctx.fillStyle = palette.ink; ctx.textAlign = "center"; ctx.fillText("x [px] →", left + plotW / 2, top + plotH + 43);
    ctx.save(); ctx.translate(15, top + plotH / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("y [px] ↓", 0, 0); ctx.restore();
    const points = nodes.map(node => [left + (node[2] - xRange[0]) * scale, top + (node[3] - yRange[0]) * scale]);
    projected.set(canvas, points); hitBounds.set(canvas, [left, top, left + plotW, top + plotH]);
    ctx.save(); ctx.beginPath(); ctx.rect(left - 6, top - 6, plotW + 12, plotH + 12); ctx.clip(); cloud(ctx, points, palette); ctx.restore();
    if (!nodes.length) { ctx.textAlign = "center"; ctx.fillStyle = palette.muted; ctx.fillText("이 윈도우에 유지된 노드가 없습니다", width / 2, height / 2); }
  }
  function drawSpace(palette) {
    const canvas = canvases[1], {ctx, width, height} = setup(canvas);
    const visibleBounds = focused ? viewBounds(6) : bounds;
    const center = visibleBounds.map(range => range[0] + (range[1] - range[0]) / 2);
    const diameter = Math.hypot(...visibleBounds.map(range => range[1] - range[0]));
    const ry = yaw * Math.PI / 180, rp = pitch * Math.PI / 180;
    const scale = Math.min(width - 92, height - 80) * 0.8 / (diameter || 1);
    function project(point) {
      const [x, y, t] = point.map((value, axis) => value - center[axis]);
      const xx = Math.cos(ry) * x - Math.sin(ry) * t, depth = Math.sin(ry) * x + Math.cos(ry) * t;
      const yy = Math.cos(rp) * y - Math.sin(rp) * depth;
      return [width / 2 + xx * scale, height / 2 + yy * scale];
    }
    const base = visibleBounds.map(range => range[0]);
    const axisEnds = visibleBounds.map((range, axis) => base.map((value, index) => index === axis ? range[1] : value));
    const origin = project(base);
    ctx.strokeStyle = palette.line; ctx.fillStyle = palette.muted; ctx.textAlign = "center";
    for (let axis = 0; axis < 3; axis++) {
      const end = project(axisEnds[axis]); ctx.beginPath(); ctx.moveTo(...origin); ctx.lineTo(...end); ctx.stroke();
      ctx.fillText(["X", "Y", "T"][axis], end[0], end[1] - 10 - (diameter === 0 ? axis * 16 : 0));
    }
    const points = nodes.map(node => project(node.slice(6, 9)));
    projected.set(canvas, points); hitBounds.set(canvas, [0, 0, width, height]); cloud(ctx, points, palette);
    ctx.fillStyle = palette.muted; ctx.textAlign = "left"; ctx.fillText("동일 배율 · 직교 투영 · 드래그로 회전", 16, height - 16);
    if (!nodes.length) { ctx.textAlign = "center"; ctx.fillText("이 윈도우에 유지된 노드가 없습니다", width / 2, height / 2); }
  }
  function schedule() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => { scheduled = false; try { const palette = colors(); drawXY(palette); drawSpace(palette); } catch (error) { fail(error.message); } });
  }
  function inspectAt(canvas, event) {
    const rect = canvas.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
    let best = 12 * 12, selected = null;
    (projected.get(canvas) || []).forEach((point, index) => {
      const [left, top, right, bottom] = hitBounds.get(canvas);
      if (point[0] < left || point[0] > right || point[1] < top || point[1] > bottom) return;
      const squared = (point[0] - x) ** 2 + (point[1] - y) ** 2;
      if (squared <= best) { best = squared; selected = nodes[index][0]; }
    });
    if (selected !== null) { inspected = selected; updateNode(); schedule(); }
  }
  select.addEventListener("change", () => { activeQuery = queries.find(query => query.node_index === Number(select.value)); inspected = activeQuery.node_index; updateQuery(); });
  el("focus-neighbors").addEventListener("click", () => { focused = true; updateViewScope(); schedule(); });
  el("show-all").addEventListener("click", () => { focused = false; updateViewScope(); schedule(); });
  el("node-form").addEventListener("submit", event => {
    event.preventDefault(); const id = Number(el("node-index").value);
    if (!integer(id) || !nodeMap.has(id)) { put("node-status", "해당 번호는 이 파일에 포함된 노드가 아닙니다. 누락된 노드나 이웃을 생성하지 않습니다."); return; }
    inspected = id; updateNode(); schedule();
  });
  function rotate() { yaw = Number(el("yaw").value); pitch = Number(el("pitch").value); put("yaw-value", `${yaw}°`); put("pitch-value", `${pitch}°`); schedule(); }
  el("yaw").addEventListener("input", rotate); el("pitch").addEventListener("input", rotate);
  el("reset-view").addEventListener("click", () => { el("yaw").value = "35"; el("pitch").value = "25"; rotate(); });
  canvases[0].addEventListener("click", event => inspectAt(canvases[0], event));
  let drag = null;
  canvases[1].addEventListener("pointerdown", event => { if (event.button !== 0) return; drag = {id: event.pointerId, x: event.clientX, y: event.clientY, yaw, pitch, moved: false}; canvases[1].setPointerCapture(event.pointerId); });
  canvases[1].addEventListener("pointermove", event => {
    if (!drag || drag.id !== event.pointerId) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
    el("yaw").value = String(Math.round(((drag.yaw + dx * 0.5 + 540) % 360) - 180));
    el("pitch").value = String(Math.max(-85, Math.min(85, Math.round(drag.pitch - dy * 0.5)))); rotate();
  });
  canvases[1].addEventListener("pointerup", event => { if (!drag || drag.id !== event.pointerId) return; const clicked = !drag.moved; drag = null; if (canvases[1].hasPointerCapture(event.pointerId)) canvases[1].releasePointerCapture(event.pointerId); if (clicked) inspectAt(canvases[1], event); });
  canvases[1].addEventListener("pointercancel", () => { drag = null; });
  const observer = new ResizeObserver(schedule); canvases.forEach(canvas => observer.observe(canvas));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
  el("inspection-content").hidden = false;
  updateQuery();
})();
