/* Standalone file viewer: no requests, imports, inference, or graph construction. */
"use strict";
(() => {
  const el = id => document.getElementById(id);
  const fmt = (v, digits=0) => typeof v === "number" && Number.isFinite(v)
    ? v.toLocaleString("ko-KR", {minimumFractionDigits:digits, maximumFractionDigits:digits}) : "—";
  const plain = v => v == null ? "" : typeof v === "string" ? v : JSON.stringify(v);
  const integer = v => Number.isSafeInteger(v) && v >= 0;
  const state = {data:null, dataset:null, frame:null, graph:null, selected:null, projected:[],
    imageRevision:0, fileRevision:0, pngRevision:{gt:0,pred:0}, graphFileRevision:0,
    manual:false, files:{}, urls:{}, imported:[],
    yaw:-0.55, pitch:0.35, scale:1, drag:null, pendingDraw:false};
  function fail(error) {el("error").textContent=error instanceof Error ? error.message : plain(error);el("error").hidden=false;}
  function clearError() {el("error").hidden=true;}
  function on(id, event, handler) {
    el(id).addEventListener(event, e => {
      try {const result=handler(e);if(result && typeof result.catch==="function") result.catch(fail);}
      catch(error){fail(error);}
    });
  }
  function option(value, text){const o=document.createElement("option");o.value=String(value);o.textContent=text;return o;}
  function lines(id, values) {
    el(id).replaceChildren(...values.map(v => {const li=document.createElement("li");li.textContent=plain(v);return li;}));
  }
  function terms(id, values, grouped=false) {
    el(id).replaceChildren();
    for(const [key,value] of values) {
      const dt=document.createElement("dt"),dd=document.createElement("dd");
      dt.textContent=key;dd.textContent=plain(value);
      if(grouped){const d=document.createElement("div");d.append(dt,dd);el(id).append(d);}
      else el(id).append(dt,dd);
    }
  }
  function tab(panel) {
    for(const b of document.querySelectorAll("[data-panel]")) b.setAttribute("aria-pressed",String(b.dataset.panel===panel));
    for(const name of ["compare","graph","summary"]) el(name+"-panel").hidden=name!==panel;
    if(panel==="graph") scheduleDraw();
  }
  function summary() {
    const dataset=state.dataset, modes=dataset?.modes || [], average=el("average-select").value;
    el("summary-title").textContent=(dataset?.label || "")+" · 평가 결과";
    el("summary-empty").hidden=modes.length>0;
    el("summary-body").replaceChildren();
    for(const mode of modes) {
      const tr=document.createElement("tr"), q=mode.quality || {}, b=mode.benchmark || {};
      if(mode.id==="ann") tr.className="baseline";
      const eligibility=v=>v===true?"yes":v===false?"no":"—";
      for(const value of [mode.label||mode.id,fmt(q.frames??dataset.total_frames),fmt(q[average]?.psnr,5),
        fmt(q[average]?.ssim,5),fmt(b.mean_ms,3),fmt(b.fps,3),fmt(b.peak_gpu_memory_mb,2),
        eligibility(mode.report_eligible)+" / "+eligibility(b.report_eligible)]) {
        const td=document.createElement("td");td.textContent=plain(value);tr.append(td);
      }
      el("summary-body").append(tr);
    }
  }
  function hideImages() {
    state.imageRevision++;
    el("image-content").hidden=true;el("image-empty").hidden=false;el("frame-metrics").replaceChildren();
    for(const id of ["gt-image","pred-image","wipe-gt","wipe-pred"]) el(id).removeAttribute("src");
    el("gt-size").textContent="";el("pred-size").textContent="";
    el("diagnostic-previews").hidden=true;
    el("events-preview").removeAttribute("src");el("graph-preview-png").removeAttribute("src");
  }
  function imageURL(key) {
    const item=state.data.images?.[key];
    if(!item || item.mime!=="image/png" || typeof item.data_url!=="string" ||
      !/^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/.test(item.data_url)) throw new Error("포함된 PNG 데이터가 없거나 형식이 잘못되었습니다.");
    return item.data_url;
  }
  async function inspectImage(url) {
    const image=new Image();image.decoding="async";
    await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error("PNG를 표시할 수 없습니다."));image.src=url;});
    return {width:image.naturalWidth,height:image.naturalHeight};
  }
  async function pair(gt,pred,label,metrics=null) {
    const revision=++state.imageRevision;
    el("image-content").hidden=true;el("image-empty").hidden=false;
    const [a,b]=await Promise.all([inspectImage(gt),inspectImage(pred)]);
    if(revision!==state.imageRevision)return;
    if(a.width!==b.width || a.height!==b.height)throw new Error("GT와 복원 PNG의 해상도가 다릅니다. 대응하는 두 파일을 선택하세요.");
    for(const [id,url] of [["gt-image",gt],["pred-image",pred],["wipe-gt",gt],["wipe-pred",pred]])el(id).src=url;
    el("gt-size").textContent=a.width+" × "+a.height;el("pred-size").textContent=b.width+" × "+b.height;
    el("prediction-label").textContent=label;
    terms("frame-metrics",metrics ? [["PSNR · 저장된 float",fmt(metrics.psnr,5)],["SSIM · 저장된 float",fmt(metrics.ssim,5)],
      ["RMSE",fmt(metrics.rmse,5)],["Temporal L1",fmt(metrics.temporal_l1,5)]] : [["직접 선택한 PNG","저장된 평가 지표와 연결하지 않음"]],true);
    el("image-empty").hidden=true;el("image-content").hidden=false;layout();
    const previews=state.manual?null:state.frame?.diagnostic_images;
    if(previews?.events && previews?.graph){
      el("events-preview").src=imageURL(previews.events);
      el("graph-preview-png").src=imageURL(previews.graph);
      el("diagnostic-previews").hidden=false;
    }
  }
  function layout() {
    const wipe=el("comparison-layout").value==="wipe";
    el("side-view").hidden=wipe;el("wipe-view").hidden=!wipe;
    const zoom=Number(el("zoom").value);el("zoom-value").value=zoom+"%";
    el("gt-image").style.width=zoom+"%";el("pred-image").style.width=zoom+"%";el("wipe-stage").style.width=zoom+"%";
    const percent=Number(el("wipe").value);el("wipe-value").value=percent+"%";
    el("wipe-gt").style.clipPath="inset(0 "+(100-percent)+"% 0 0)";el("wipe-line").style.left=percent+"%";
  }
  async function selectMode() {
    if(state.manual)return;
    const item=state.frame?.images?.find(x=>x.mode===el("mode-select").value);
    if(!item){hideImages();return;}
    clearError();
    await pair(imageURL(item.target),imageURL(item.prediction),item.mode+" · 복원",item.metrics);
  }
  function clearManual() {
    state.fileRevision++;state.manual=false;state.files={};
    for(const value of Object.values(state.urls))URL.revokeObjectURL(value);
    state.urls={};el("local-gt").value="";el("local-pred").value="";
    el("local-status").textContent="";
  }
  async function selectFrame() {
    clearError();hideImages();clearManual();
    state.graphFileRevision++;state.imported=[];
    el("imported-graph-label").hidden=true;el("imported-graph-select").replaceChildren();
    el("graph-file-status").textContent="";el("local-graph").value="";
    state.frame=state.dataset?.frames?.find(x=>String(x.index)===el("frame-select").value)||null;
    const images=state.frame?.images||[],old=el("mode-select").value;
    el("mode-select").replaceChildren(...images.map(x=>option(x.mode,x.mode)));
    el("mode-select").disabled=!images.length;
    if(images.some(x=>x.mode===old))el("mode-select").value=old;
    const saved=state.dataset?.frames?.length||0;
    el("frame-status").textContent=state.frame ? state.frame.sample_id+" · "+(state.frame.group||"")+" · 저장 PNG "+fmt(saved)+" / 평가 "+fmt(state.dataset.total_frames)+" 프레임" :
      "저장 PNG 0장 · 평가 수치가 있어도 저장되지 않은 복원 이미지를 새로 만들지 않습니다.";
    setGraph(state.frame?.graph||null,false);
    await selectMode();
  }
  async function selectDataset() {
    state.dataset=state.data.datasets.find(x=>x.id===el("dataset-select").value)||null;
    const frames=state.dataset?.frames||[];
    el("frame-select").replaceChildren(...frames.map(f=>option(f.index,String(f.index)+" · "+f.sample_id)));
    if(!frames.length)el("frame-select").append(option("","저장 이미지 없음"));
    el("frame-select").disabled=!frames.length;
    summary();await selectFrame();
  }
  async function readPNG(file) {
    if(file.size>32*1024*1024)throw new Error("PNG 파일의 32 MiB 읽기 안전 한도를 초과했습니다. 축소하거나 일부만 읽지 않았습니다.");
    const bytes=new Uint8Array(await file.slice(0,33).arrayBuffer());
    if(bytes.length<33 || ![137,80,78,71,13,10,26,10].every((b,i)=>bytes[i]===b) ||
      String.fromCharCode(...bytes.slice(12,16))!=="IHDR")throw new Error("실제 PNG 파일을 선택하세요.");
    const view=new DataView(bytes.buffer),w=view.getUint32(16),h=view.getUint32(20);
    if(!w||!h||w*h*4>128*1024*1024)throw new Error("PNG의 예상 RGBA 메모리가 128 MiB 안전 한도를 초과하거나 해상도가 잘못되었습니다.");
    return URL.createObjectURL(file);
  }
  async function localPNG(kind) {
    const file=el("local-"+kind).files[0];if(!file)return;
    const generation=state.fileRevision,revision=++state.pngRevision[kind];
    clearError();hideImages();state.manual=true;
    if(state.urls[kind])URL.revokeObjectURL(state.urls[kind]);
    delete state.urls[kind];delete state.files[kind];
    el("local-status").textContent="PNG 확인 중: "+file.name;
    const url=await readPNG(file);
    if(generation!==state.fileRevision||revision!==state.pngRevision[kind]){URL.revokeObjectURL(url);return;}
    state.urls[kind]=url;state.files[kind]=file.name;
    el("local-status").textContent="PC 파일: GT "+(state.files.gt||"미선택")+" / 복원 "+(state.files.pred||"미선택");
    el("frame-status").textContent="수동 PNG 비교 · 선택된 데이터셋/프레임/평가 수치와 연결 미검증";
    if(state.urls.gt&&state.urls.pred)await pair(state.urls.gt,state.urls.pred,"직접 선택 · "+state.files.pred);
  }
  function validateGraph(g) {
    if(!g || !Array.isArray(g.nodes)||!Array.isArray(g.edges)||!g.statistics)throw new Error("nodes / edges / statistics가 있는 저장 그래프 JSON이 필요합니다.");
    if(!g.nodes.every(n=>Array.isArray(n)&&n.length===4&&n.every(x=>typeof x==="number"&&Number.isFinite(x))))throw new Error("노드는 유한한 [x,y,t,polarity] 배열이어야 합니다.");
    if(g.statistics.nodes!==g.nodes.length || !integer(g.statistics.actual_directed_edges) ||
      g.statistics.actual_directed_edges<g.edges.length || g.statistics.displayed_edges!==g.edges.length)
      throw new Error("그래프 노드·엣지 통계와 포함된 배열의 크기가 맞지 않습니다.");
    if(!g.edges.every(e=>Array.isArray(e)&&e.length===2&&e.every(x=>integer(x)&&x<g.nodes.length)))throw new Error("엣지의 노드 index가 잘못되었습니다.");
    return g;
  }
  function setGraph(graph,manual) {
    state.graph=null;state.selected=null;state.projected=[];
    el("graph-content").hidden=true;el("graph-empty").hidden=false;
    el("node-index").value="";el("node-detail").textContent="노드를 선택하세요.";
    if(!graph)return;
    state.graph=validateGraph(graph);
    el("graph-content").hidden=false;el("graph-empty").hidden=true;
    el("graph-note").textContent=(manual?"직접 읽은 그래프 파일 · 현재 영상/프레임과 연결 미검증. ":"포함된 저장 그래프 · 과거 GPU tensor와의 동일성을 재검증하지 않았습니다. ")+
      "모든 포함 노드를 표시합니다. 선의 표시 개수만 조절하며 그래프를 다시 생성하지 않습니다. 이웃은 파일에 포함된 엣지만 대상으로 합니다. "+plain(graph.provenance_note||"");
    scheduleDraw();
  }
  function project(node,width,height) {
    let x=node[0]-.5,y=.5-node[1],z=node[2]-.5;
    const mode=el("projection").value;
    if(mode==="xt")y=z;
    if(mode==="3d"){
      const a=x*Math.cos(state.yaw)+z*Math.sin(state.yaw),b=-x*Math.sin(state.yaw)+z*Math.cos(state.yaw);
      x=a;y=y*Math.cos(state.pitch)-b*Math.sin(state.pitch);
    }
    const s=Math.min(width,height)*.61*state.scale;
    return [width/2+x*s,height/2-y*s];
  }
  function scheduleDraw(){
    if(state.pendingDraw)return;
    state.pendingDraw=true;requestAnimationFrame(()=>{state.pendingDraw=false;try{drawGraph();}catch(error){fail(error);}});
  }
  function drawGraph() {
    const g=state.graph;if(!g || el("graph-panel").hidden)return;
    const canvas=el("graph-canvas"),width=canvas.clientWidth,height=canvas.clientHeight;if(!width||!height)return;
    const ratio=window.devicePixelRatio||1;canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
    const ctx=canvas.getContext("2d");ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
    state.projected=g.nodes.map(n=>project(n,width,height));
    const colors=getComputedStyle(document.documentElement),muted=colors.getPropertyValue("--muted").trim();
    const pos=colors.getPropertyValue("--accent").trim(),neg=colors.getPropertyValue("--negative").trim();
    const budget=Number(el("edge-count").value);if(!integer(budget))throw new Error("표시할 선 개수는 0 이상의 정수여야 합니다.");
    const shown=Math.min(budget,g.edges.length);
    ctx.strokeStyle=muted;ctx.globalAlpha=.13;ctx.lineWidth=.65;ctx.beginPath();
    for(let i=0;i<shown;i++){
      const at=shown===1?0:Math.floor(i*(g.edges.length-1)/Math.max(1,shown-1)),edge=g.edges[at];
      const a=state.projected[edge[0]],b=state.projected[edge[1]];ctx.moveTo(...a);ctx.lineTo(...b);
    }
    ctx.stroke();ctx.globalAlpha=1;
    const selected=state.selected,neighbors=new Set();
    if(selected!==null){
      ctx.strokeStyle=pos;ctx.lineWidth=1.3;ctx.globalAlpha=.65;ctx.beginPath();
      for(const [a,b] of g.edges)if(a===selected){neighbors.add(b);ctx.moveTo(...state.projected[a]);ctx.lineTo(...state.projected[b]);}
      ctx.stroke();ctx.globalAlpha=1;
    }
    for(let i=0;i<g.nodes.length;i++){
      const [x,y]=state.projected[i];ctx.fillStyle=i===selected?"#fff":neighbors.has(i)?pos:g.nodes[i][3]>0?pos:neg;
      ctx.globalAlpha=(selected===null||i===selected||neighbors.has(i)) ? .9 : .24;
      ctx.beginPath();ctx.arc(x,y,i===selected?5:2.1,0,Math.PI*2);ctx.fill();
    }
    ctx.globalAlpha=1;ctx.font="12px Segoe UI";ctx.fillStyle=muted;
    for(const [p,label] of [[[1,.5,.5],"x"],[[.5,0,.5],"y"],[[.5,.5,1],"t"]]){
      if(el("projection").value==="xy"&&label==="t"||el("projection").value==="xt"&&label==="y")continue;
      const a=project([.5,.5,.5],width,height),b=project(p,width,height);
      ctx.strokeStyle=muted;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.fillText(label,b[0]+6,b[1]-6);
    }
    terms("graph-stats",[["노드 · 포함 전체",fmt(g.nodes.length)],["실제 방향 엣지 · 기록값",fmt(g.statistics.actual_directed_edges)],
      ["파일에 포함된 엣지",fmt(g.edges.length)],["현재 그린 선",fmt(shown)],["반경 · 기록값",fmt(g.radius,4)],
      ["위치 차원",fmt(g.position_dims)],["고립 노드 · 기록값",fmt(g.statistics.isolated_nodes)]]);
    if(selected!==null){
      const n=g.nodes[selected];
      el("node-detail").textContent="node "+selected+"\nx / y / t: "+n.slice(0,3).map(x=>fmt(x,4)).join(" / ")+
        "\npolarity: "+fmt(n[3])+"\n포함된 outgoing 엣지의 이웃: "+neighbors.size+"개\n전체 그래프 degree로 간주하지 않습니다.";
    }
  }
  async function loadGraphFile(){
    const file=el("local-graph").files[0];if(!file)return;clearError();
    const revision=++state.graphFileRevision;
    state.imported=[];setGraph(null,false);
    el("imported-graph-label").hidden=true;el("imported-graph-select").replaceChildren();
    el("graph-file-status").textContent="JSON 확인 중: "+file.name;
    if(file.size>32*1024*1024)throw new Error("그래프 JSON의 32 MiB 읽기 안전 한도를 초과했습니다. 일부만 읽지 않았습니다.");
    const text=await file.text();if(revision!==state.graphFileRevision)return;
    const value=JSON.parse(text);
    const entries=value.schema==="asgcn_offline_graphs_v1"?value.graphs:[{graph:value}];
    if(!Array.isArray(entries)||!entries.length)throw new Error("파일에 그래프가 없습니다.");
    const checked=entries.map((v,i)=>({label:v.sample_id?plain(v.dataset)+" / "+plain(v.sample_id):"그래프 "+(i+1),graph:validateGraph(v.graph)}));
    state.imported=checked;
    el("imported-graph-select").replaceChildren(...checked.map((v,i)=>option(i,v.label)));
    el("imported-graph-label").hidden=checked.length<2;
    el("graph-file-status").textContent=file.name+" · 별도 파일 · 현재 영상/프레임과 연결 미검증";
    setGraph(checked[0].graph,true);
  }
  async function main(){
    state.data=JSON.parse(el("offline-data").textContent);
    if(state.data.schema!=="asgcn_offline_results_v1" || !Array.isArray(state.data.datasets))throw new Error("지원하지 않는 오프라인 결과 형식입니다.");
    el("page-title").textContent=state.data.title||"복원 결과 살펴보기";
    document.title=(state.data.title||"ASGCN 결과")+" · 오프라인";
    lines("notes",state.data.notes||[]);lines("warnings",state.data.warnings||[]);
    el("source-note").textContent=(state.data.notes||[]).slice(0,2).map(plain).join(" ");
    el("dataset-select").replaceChildren(...state.data.datasets.map(d=>option(d.id,d.label||d.id)));
    el("dataset-select").disabled=!state.data.datasets.length;
    for(const b of document.querySelectorAll("[data-panel]"))b.addEventListener("click",()=>tab(b.dataset.panel));
    on("dataset-select","change",selectDataset);on("frame-select","change",selectFrame);on("mode-select","change",selectMode);
    on("comparison-layout","change",layout);on("zoom","input",layout);on("wipe","input",layout);
    on("average-select","change",summary);on("local-gt","change",()=>localPNG("gt"));on("local-pred","change",()=>localPNG("pred"));
    on("use-saved","click",selectFrame);on("local-graph","change",loadGraphFile);
    on("imported-graph-select","change",()=>setGraph(state.imported[Number(el("imported-graph-select").value)].graph,true));
    on("projection","change",scheduleDraw);on("edge-count","change",scheduleDraw);
    on("select-node","click",()=>{
      const raw=el("node-index").value,i=Number(raw);
      if(!state.graph)throw new Error("먼저 저장된 그래프를 열어 주세요.");
      if(!raw||!integer(i)||i>=state.graph.nodes.length)throw new Error("그래프 안에 있는 node index를 입력하세요.");
      clearError();state.selected=i;scheduleDraw();
    });
    on("reset-graph","click",()=>{state.yaw=-.55;state.pitch=.35;state.scale=1;state.selected=null;el("node-detail").textContent="노드를 선택하세요.";scheduleDraw();});
    on("graph-canvas","pointerdown",e=>{state.drag={x:e.clientX,y:e.clientY,moved:false};el("graph-canvas").setPointerCapture(e.pointerId);});
    on("graph-canvas","pointermove",e=>{
      const drag=state.drag;if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
      if(Math.abs(dx)+Math.abs(dy)>2)drag.moved=true;
      state.yaw+=dx*.008;state.pitch+=dy*.008;drag.x=e.clientX;drag.y=e.clientY;scheduleDraw();
    });
    on("graph-canvas","pointerup",e=>{
      const drag=state.drag;state.drag=null;if(!drag||drag.moved||!state.graph)return;
      const rect=el("graph-canvas").getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;
      let best=12,index=null;
      for(let i=0;i<state.projected.length;i++){const p=state.projected[i],d=Math.hypot(x-p[0],y-p[1]);if(d<best){best=d;index=i;}}
      if(index!==null){state.selected=index;el("node-index").value=index;scheduleDraw();}
    });
    on("graph-canvas","pointercancel",()=>{state.drag=null;});
    el("graph-canvas").addEventListener("wheel",e=>{e.preventDefault();state.scale=Math.max(.25,Math.min(5,state.scale*Math.exp(-e.deltaY*.001)));scheduleDraw();},{passive:false});
    new ResizeObserver(scheduleDraw).observe(el("graph-canvas"));
    const scrolls=document.querySelectorAll("#side-view .image-scroll");
    for(const source of scrolls)source.addEventListener("scroll",()=>{
      for(const target of scrolls)if(target!==source){target.scrollLeft=source.scrollLeft;target.scrollTop=source.scrollTop;}
    });
    await selectDataset();
    if(!state.data.datasets.some(d=>d.frames?.length)&&state.data.datasets.some(d=>d.modes?.length))tab("summary");
  }
  main().catch(fail);
})();
