/* Generated SYNTHETIC integration artifacts only; no server, inference, or GPU. */
"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const {pathToFileURL} = require("node:url");
const {chromium} = require(process.env.ASGCN_PLAYWRIGHT_MODULE || "playwright");

const hash = value => crypto.createHash("sha256").update(value).digest("hex");
const formatted = value => Number(value).toLocaleString("ko-KR");

function fixtureAt(directory) {
  assert.ok(directory, "Pass the generated SYNTHETIC integration output directory");
  const root = path.resolve(directory);
  const index = path.join(root, "index.html");
  assert.ok(fs.statSync(index).size <= 32 * 1024 * 1024,
    "Browser smoke accepts only small synthetic HTML fixtures (32 MiB maximum)");
  const report = JSON.parse(fs.readFileSync(path.join(root, "generation.json"), "utf8"));
  assert.equal(report.schema, "asgcn_result_visualizations_v1");
  assert.equal(report.complete, true, "The generator must have completed successfully");
  assert.equal(report.graph_reconstruction, true);
  assert.equal(report.model_inference, false);
  assert.equal(report.report_eligible, false);
  const html = fs.readFileSync(index, "utf8");
  const match = html.match(/<script id="offline-data" type="application\/json">([\s\S]*?)<\/script>/);
  assert.ok(match, "Generated HTML has an embedded results payload");
  const data = JSON.parse(match[1]);
  assert.equal(data.schema, "asgcn_offline_results_v1");
  assert.ok(data.datasets.length > 0);
  assert.ok(data.datasets.every(dataset => ["aid", "hdr"].includes(dataset.id)));
  const frames = data.datasets.flatMap(dataset => dataset.frames);
  assert.ok(frames.length >= 2 && frames.length <= 16,
    "Use a small synthetic integration fixture with at least two real generated frames");
  assert.ok(frames.every(frame => /synthetic/i.test(frame.group)),
    "Refusing non-synthetic inputs: every fixture frame must be labeled SYNTHETIC");
  assert.ok(frames.every(frame => frame.images.every(item => item.report_eligible === false)));
  assert.equal(report.generated_frames, frames.length);
  return {root, index, data, report};
}

function pngDescriptor(fixture, imageId, file) {
  const info = fixture.data.images[imageId];
  assert.ok(info && info.mime === "image/png");
  assert.match(info.data_url, /^data:image\/png;base64,/);
  const bytes = fs.readFileSync(file);
  assert.equal(hash(bytes), info.sha256, "The displayed PNG must match its generated file");
  assert.equal(hash(Buffer.from(info.data_url.split(",")[1], "base64")), info.sha256);
  assert.equal(bytes.readUInt32BE(16), info.width);
  assert.equal(bytes.readUInt32BE(20), info.height);
  return info;
}

async function imagePair(page, gt, prediction) {
  await page.waitForFunction(({a, b}) => {
    const left = document.getElementById("gt-image"), right = document.getElementById("pred-image");
    return !document.getElementById("image-content").hidden &&
      left.getAttribute("src") === a && right.getAttribute("src") === b &&
      left.complete && right.complete && left.naturalWidth > 0 && right.naturalWidth > 0;
  }, {a: gt.data_url, b: prediction.data_url});
  assert.deepEqual(await page.locator("#gt-image").evaluate(image =>
    [image.naturalWidth, image.naturalHeight]), [gt.width, gt.height]);
  assert.deepEqual(await page.locator("#pred-image").evaluate(image =>
    [image.naturalWidth, image.naturalHeight]), [prediction.width, prediction.height]);
  assert.equal(await page.locator("#error").isVisible(), false);
}

async function painted(page) {
  await page.evaluate(() => new Promise(resolve =>
    requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function graphChecks(page, graph) {
  await page.click('[data-panel="graph"]');
  await page.waitForFunction(() => !document.getElementById("graph-content").hidden &&
    document.querySelectorAll("#graph-stats dd").length === 7);
  await painted(page);
  const stats = await page.locator("#graph-stats dd").allTextContents();
  assert.equal(stats[0], formatted(graph.nodes.length));
  assert.equal(stats[1], formatted(graph.statistics.actual_directed_edges));
  assert.equal(stats[2], formatted(graph.edges.length));
  assert.equal(await page.locator("#graph-empty").isVisible(), false);
  assert.match(await page.locator("#graph-note").textContent(), /CPU/);
  assert.equal(await page.locator("#graph-canvas").evaluate(canvas => {
    if (!canvas.width || !canvas.height) return false;
    const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    return pixels.some((value, index) => index % 4 === 3 && value > 0);
  }), true, "Graph canvas contains rendered pixels");
  if (graph.nodes.length) {
    const selected = graph.edges.length ? graph.edges[0][0] : 0;
    const neighbors = new Set(graph.edges.filter(edge => edge[0] === selected).map(edge => edge[1]));
    await page.fill("#node-index", String(selected));
    await page.click("#select-node");
    await page.waitForFunction(node => document.getElementById("node-detail").textContent
      .startsWith("node " + node + "\n"), selected);
    assert.match(await page.locator("#node-detail").textContent(),
      new RegExp("포함된 outgoing 엣지의 이웃: " + neighbors.size + "개"));
    assert.match(await page.locator("#node-detail").textContent(), /전체 그래프 degree로 간주하지 않습니다/);
  }
  await page.fill("#edge-count", "0");
  await page.locator("#edge-count").dispatchEvent("change");
  await painted(page);
  assert.equal(await page.locator("#graph-stats dd").nth(3).textContent(), "0");
  assert.equal(await page.locator("#graph-stats dd").nth(1).textContent(),
    formatted(graph.statistics.actual_directed_edges), "Display control must not change actual topology");
  await page.fill("#edge-count", String(graph.edges.length));
  await page.locator("#edge-count").dispatchEvent("change");
  for (const projection of ["xy", "xt", "3d"]) {
    await page.selectOption("#projection", projection);
    await painted(page);
    assert.equal(await page.locator("#error").isVisible(), false);
  }
  const canvas = page.locator("#graph-canvas");
  await canvas.scrollIntoViewIfNeeded();
  const before = await canvas.evaluate(element => element.toDataURL());
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 46, box.y + box.height / 2 + 24, {steps: 4});
  await page.mouse.up();
  await painted(page);
  assert.notEqual(await canvas.evaluate(element => element.toDataURL()), before,
    "Dragging rotates the actual graph canvas");
}

(async () => {
  const fixture = fixtureAt(process.argv[2]);
  const screenshots = fs.mkdtempSync(path.join(fixture.root, "synthetic-browser-smoke-"));
  const errors = [], consoleErrors = [], network = [], sockets = [];
  const browser = await chromium.launch({headless: true,
    channel: process.env.ASGCN_BROWSER_CHANNEL || "chrome", args: ["--disable-gpu"]});
  try {
    const context = await browser.newContext({viewport: {width: 1400, height: 1050},
      offline: true, colorScheme: "light"});
    context.on("request", request => {
      if (/^(?:https?|wss?):/i.test(request.url())) network.push(request.url());
    });
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(error.message));
    page.on("console", message => { if (message.type() === "error") consoleErrors.push(message.text()); });
    page.on("websocket", socket => sockets.push(socket.url()));
    await page.goto(pathToFileURL(fixture.index).href);
    await page.waitForFunction(() => !document.getElementById("image-content").hidden);
    let checkedModes = 0, checkedFrames = 0;
    for (const dataset of fixture.data.datasets) {
      await page.selectOption("#dataset-select", dataset.id);
      assert.ok(dataset.frames.length >= 2, "Fixture exercises frame changes per dataset");
      for (const frame of dataset.frames) {
        await page.click('[data-panel="compare"]');
        await page.selectOption("#frame-select", String(frame.index));
        await page.waitForFunction(sample => document.getElementById("frame-status").textContent
          .includes(sample), frame.sample_id);
        const directory = path.join(fixture.root, dataset.id, String(frame.index).padStart(8, "0"));
        assert.ok(frame.images.some(item => item.mode === "ann"), "Fixture includes ANN");
        assert.ok(frame.images.some(item => item.mode.startsWith("snn_")), "Fixture includes SNN");
        for (const item of frame.images) {
          const gt = pngDescriptor(fixture, item.target, path.join(directory, "gt.png"));
          const prediction = pngDescriptor(fixture, item.prediction,
            path.join(directory, item.mode + ".png"));
          await page.selectOption("#mode-select", item.mode);
          await imagePair(page, gt, prediction);
          assert.match(await page.locator("#prediction-label").textContent(), new RegExp(item.mode));
          if (Number.isFinite(item.metrics.psnr)) {
            const psnr = item.metrics.psnr.toLocaleString("ko-KR", {
              minimumFractionDigits: 5, maximumFractionDigits: 5});
            assert.ok((await page.locator("#frame-metrics").textContent()).includes(psnr));
          }
          checkedModes++;
        }
        assert.equal(await page.locator("#diagnostic-previews").isVisible(), true);
        if (!await page.locator("#diagnostic-previews").evaluate(element => element.open)) {
          await page.locator("#diagnostic-previews > summary").click();
        }
        for (const [key, selector, filename] of [
          ["events", "#events-preview", "events-xy.png"],
          ["graph", "#graph-preview-png", "graph-xyt.png"],
        ]) {
          const info = pngDescriptor(fixture, frame.diagnostic_images[key], path.join(directory, filename));
          await page.waitForFunction(({id, src}) => {
            const image = document.querySelector(id);
            return image.getAttribute("src") === src && image.complete && image.naturalWidth > 0;
          }, {id: selector, src: info.data_url});
          assert.deepEqual(await page.locator(selector).evaluate(image =>
            [image.naturalWidth, image.naturalHeight]), [info.width, info.height]);
        }
        assert.ok(frame.graph && frame.graph.nodes.length, "Every synthetic frame has an actual generated graph");
        assert.deepEqual(JSON.parse(fs.readFileSync(path.join(directory, "graph.json"), "utf8")), frame.graph);
        if (!checkedFrames) {
          await page.screenshot({path: path.join(screenshots, "compare-desktop.png"), fullPage: true});
        }
        await graphChecks(page, frame.graph);
        if (!checkedFrames) {
          await page.screenshot({path: path.join(screenshots, "graph-desktop.png"), fullPage: true});
        }
        checkedFrames++;
      }
    }
    await page.setViewportSize({width: 360, height: 820});
    for (const panel of ["compare", "graph", "summary"]) {
      await page.click('[data-panel="' + panel + '"]');
      await painted(page);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1),
        false, panel + " mobile page has no horizontal overflow");
      await page.screenshot({path: path.join(screenshots, panel + "-mobile.png"), fullPage: true});
    }
    await page.setViewportSize({width: 1400, height: 1050});
    await page.emulateMedia({colorScheme: "dark"});
    await page.click('[data-panel="graph"]');
    await painted(page);
    await page.screenshot({path: path.join(screenshots, "graph-desktop-dark.png"), fullPage: true});
    assert.equal(await page.locator("#error").isVisible(), false);
    assert.deepEqual(errors, [], "Uncaught page JavaScript errors");
    assert.deepEqual(consoleErrors, [], "Browser console errors");
    assert.deepEqual(network, [], "HTTP/HTTPS/WS requests");
    assert.deepEqual(sockets, [], "WebSockets");
    console.log(JSON.stringify({status: "passed", fixture: "SYNTHETIC CPU INTEGRATION ONLY",
      generated_frames: checkedFrames, mode_frame_checks: checkedModes,
      natural_resolution_preserved: true, actual_generated_png_graph_binding: true,
      http_requests: network.length, websockets: sockets.length,
      page_errors: errors.length, console_errors: consoleErrors.length,
      browser_gpu_disabled: true, context_offline: true, screenshots}, null, 2));
    await context.close();
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
