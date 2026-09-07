/* Local file:// browser smoke. Only the explicitly synthetic fixture is used. */
const {chromium}=require(process.env.ASGCN_PLAYWRIGHT_MODULE || "playwright");
const {pathToFileURL}=require("node:url");
const path=require("node:path");
const assert=require("node:assert/strict");
const fs=require("node:fs");
(async()=>{
 const root=path.resolve(process.argv[2]),summary=path.resolve(process.argv[3]);
 const browser=await chromium.launch({headless:true,channel:process.env.ASGCN_BROWSER_CHANNEL,
   args:["--disable-gpu"]});
 try{
  const context=await browser.newContext({viewport:{width:1400,height:1050},offline:true});
  const page=await context.newPage(),errors=[],network=[];
  page.on("pageerror",e=>errors.push(e.message));
  page.on("request",r=>{if(/^https?:|^wss?:/.test(r.url()))network.push(r.url());});
  await page.goto(pathToFileURL(path.join(root,"smoke.html")).href);
  await page.waitForFunction(()=>!document.getElementById("image-content").hidden);
  assert.equal(await page.locator("#gt-image").evaluate(i=>i.naturalWidth),64);
  assert.equal(await page.evaluate(()=>window.INJECTED),undefined);
  assert.match(await page.locator("#frame-status").textContent(),/__OFFLINE_JS__/);
  const original=await page.locator("#pred-image").getAttribute("src");
  await page.selectOption("#mode-select","snn_standard_if_T4");
  await page.waitForFunction(src=>document.getElementById("pred-image").src!==src,original);
  assert.match(await page.locator("#frame-metrics").textContent(),/11\.12500/);
  await page.selectOption("#comparison-layout","wipe");
  assert.equal(await page.locator("#wipe-view").isVisible(),true);
  await page.locator("#wipe").fill("25");
  await page.locator("#wipe").dispatchEvent("input");
  assert.match(await page.locator("#wipe-gt").getAttribute("style"),/75%/);
  await page.locator("#zoom").fill("200");await page.locator("#zoom").dispatchEvent("input");
  assert.equal(await page.locator("#wipe-stage").evaluate(e=>e.style.width),"200%");
  await page.selectOption("#comparison-layout","side");
  await page.locator("#zoom").fill("100");await page.locator("#zoom").dispatchEvent("input");
  await page.screenshot({path:path.join(root,"compare-desktop.png"),fullPage:true});
  await page.click('[data-panel="graph"]');
  await page.waitForFunction(()=>document.getElementById("graph-stats").textContent.includes("파일에 포함된 엣지"));
  await page.fill("#node-index","0");await page.click("#select-node");
  await page.waitForFunction(()=>document.getElementById("node-detail").textContent.includes("2개"));
  await page.screenshot({path:path.join(root,"graph-desktop.png"),fullPage:true});
  await page.click('[data-panel="compare"]');
  await page.selectOption("#frame-select","1");
  await page.click('[data-panel="graph"]');
  assert.equal(await page.locator("#graph-empty").isVisible(),true);
  await page.locator("#graph-panel details summary").click();
  await page.setInputFiles("#local-graph",path.join(root,"graph.json"));
  await page.waitForFunction(()=>!document.getElementById("graph-content").hidden);
  assert.match(await page.locator("#graph-note").textContent(),/연결 미검증/);
  await page.click('[data-panel="compare"]');
  await page.locator("#compare-panel details summary").click();
  await Promise.all([
   page.setInputFiles("#local-gt",path.join(root,"gt.png")),
   page.setInputFiles("#local-pred",path.join(root,"ann.png"))
  ]);
  await page.waitForFunction(()=>!document.getElementById("image-content").hidden&&document.getElementById("frame-status").textContent.includes("수동"));
  assert.match(await page.locator("#frame-metrics").textContent(),/연결하지 않음/);
  await page.setInputFiles("#local-gt",path.join(root,"invalid.png"));
  await page.waitForFunction(()=>!document.getElementById("error").hidden);
  assert.equal(await page.locator("#image-content").isVisible(),false);
  await page.setInputFiles("#local-pred",path.join(root,"snn.png"));
  assert.equal(await page.locator("#image-content").isVisible(),false);
  await page.setInputFiles("#local-gt",path.join(root,"gt.png"));
  await page.waitForFunction(()=>!document.getElementById("image-content").hidden);
  await page.click("#use-saved");
  await page.waitForFunction(()=>!document.getElementById("image-content").hidden);
  await page.click('[data-panel="summary"]');
  await page.selectOption("#average-select","macro");
  assert.match(await page.locator("#summary-body").textContent(),/11\.50000/);
  await page.setViewportSize({width:360,height:820});
  for(const panel of ["summary","compare","graph"]){
   await page.click('[data-panel="'+panel+'"]');
   await page.screenshot({path:path.join(root,panel+"-mobile.png"),fullPage:true});
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,panel+" page overflow");
  }
  await page.goto(pathToFileURL(summary).href);
  await page.waitForFunction(()=>document.querySelectorAll("#summary-body tr").length===9);
  assert.equal(await page.locator("#summary-panel").isVisible(),true);
  assert.match(await page.locator("#summary-body").textContent(),/10\.59523/);
  await page.selectOption("#dataset-select","hdr");
  assert.match(await page.locator("#summary-body").textContent(),/14\.78788/);
  await page.setViewportSize({width:1400,height:1050});
  await page.screenshot({path:path.join(root,"actual-pasted-summary.png"),fullPage:true});
  await page.click('[data-panel="compare"]');
  assert.equal(await page.locator("#image-empty").isVisible(),true);
  await page.click('[data-panel="graph"]');
  assert.equal(await page.locator("#graph-empty").isVisible(),true);
  assert.deepEqual(errors,[],"page JS errors");assert.deepEqual(network,[],"network requests");
  console.log(JSON.stringify({status:"passed",fixture:"SYNTHETIC UI ONLY",network_requests:network.length,
   page_errors:errors.length,source_summary:"user pasted metrics; real images not present",
   screenshots:fs.readdirSync(root).filter(x=>x.endsWith(".png")&&x.includes("-"))},null,2));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
