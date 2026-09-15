/* Synthetic-only file:// smoke. No server, real data, network, or model. */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {pathToFileURL} = require("node:url");

async function main() {
  const input = process.argv[2];
  assert(input, "Usage: node graph_inspection_browser_smoke.cjs <synthetic graph.html>");
  const file = path.resolve(input);
  const html = fs.readFileSync(file, "utf8");
  const match = html.match(/<script\s+id="inspection-data"\s+type="application\/json">([\s\S]*?)<\/script>/);
  assert(match, "A self-contained inspection payload is required");
  const fixture = JSON.parse(match[1]);
  assert.equal(fixture.schema, "asgcn_graph_inspection_view_v1");
  assert.equal(fixture.synthetic_fixture, true, "Refusing to open anything except an explicitly synthetic fixture");
  assert(fixture.nodes.length > 0 && fixture.nodes.length <= 32768,
    "Test-only workload bound: this synthetic browser smoke accepts at most 32768 fixture nodes; production display has no node cap");
  assert(fixture.queries.length > 0, "Fixture must include an audited query");
  const modulePath = process.env.ASGCN_PLAYWRIGHT_MODULE;
  assert(modulePath, "Set ASGCN_PLAYWRIGHT_MODULE to the installed Playwright package; no package download is performed");
  const {chromium} = require(modulePath);
  const browser = await chromium.launch({headless: true, args: ["--disable-gpu"],
    ...(process.env.ASGCN_BROWSER_EXECUTABLE ? {executablePath: process.env.ASGCN_BROWSER_EXECUTABLE} : {})});
  const pageErrors = [], forbiddenRequests = [];
  try {
    const context = await browser.newContext({viewport: {width: 1360, height: 1040}, colorScheme: "light"});
    const page = await context.newPage();
    page.on("pageerror", error => pageErrors.push(error.message));
    await context.route("**/*", route => {
      if (route.request().url() === pathToFileURL(file).href) return route.continue();
      forbiddenRequests.push(route.request().url());
      return route.abort();
    });
    await page.goto(pathToFileURL(file).href, {waitUntil: "load"});
    assert.equal(await page.locator("#inspection-error").isVisible(), false,
      await page.locator("#inspection-error").textContent());
    await page.locator("#inspection-content").waitFor({state: "visible"});
    assert.equal(await page.locator("#fixture-notice").isVisible(), true);
    await page.waitForFunction(() => ["xy-canvas", "space-canvas"].every(id => {
      const canvas = document.getElementById(id);
      const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      let painted = 0;
      for (let index = 3; index < pixels.length; index += 4) if (pixels[index]) painted++;
      return painted > 40;
    }));
    assert.equal(await page.locator("#query-select option").count(), fixture.queries.length);
    for (const query of fixture.queries) {
      await page.locator("#query-select").selectOption(String(query.node_index));
      assert((await page.locator("#query-status").textContent()).includes(`노드 ${query.node_index}`));
      assert((await page.locator("#degree-count").textContent()).includes(query.in_degree.toLocaleString("en-US")));
    }
    const chosen = fixture.queries[fixture.queries.length - 1];
    const selectedValue = await page.locator("#query-select").inputValue();
    await page.locator("#focus-neighbors").click();
    assert.equal(await page.locator("#graph-inspection").getAttribute("data-view-scope"), "query_neighborhood");
    assert.equal(await page.locator("#focus-neighbors").getAttribute("aria-pressed"), "true");
    assert.equal(await page.locator("#query-select").inputValue(), selectedValue);
    const before = await page.locator("#space-canvas").evaluate(canvas => canvas.toDataURL());
    await page.locator("#yaw").focus();
    await page.keyboard.press("ArrowRight");
    assert.equal(await page.locator("#yaw-value").textContent(), "36°");
    await page.waitForFunction(old => document.getElementById("space-canvas").toDataURL() !== old, before);
    await page.locator("#reset-view").click();
    assert.equal(await page.locator("#yaw").inputValue(), "35");
    await page.locator("#show-all").click();
    assert.equal(await page.locator("#graph-inspection").getAttribute("data-view-scope"), "available_nodes");
    const inspected = fixture.nodes.find(node => node[0] !== chosen.node_index) || fixture.nodes[0];
    await page.locator("#node-index").fill(String(inspected[0]));
    await page.locator("#node-form button").click();
    assert((await page.locator("#node-status").textContent()).includes(`원시 행 ${inspected[1]}`));
    assert.equal(await page.locator("#query-select").inputValue(), selectedValue, "Node inspection must not invent a new query");
    await page.locator("#node-index").fill(String(fixture.window.nodes + 10));
    await page.locator("#node-form button").click();
    assert((await page.locator("#node-status").textContent()).includes("포함된 노드가 아닙니다"));
    assert((await page.locator("#edge-count").textContent()).includes(
      fixture.total_directed_edges === null ? "미측정" : fixture.total_directed_edges.toLocaleString("en-US")));
    for (const width of [1360, 390, 320]) {
      await page.setViewportSize({width, height: 1040});
      await page.waitForFunction(() => document.documentElement.scrollWidth <= window.innerWidth + 1);
    }
    await page.emulateMedia({colorScheme: "dark"});
    await page.locator("#focus-neighbors").click();
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert.equal(await page.locator("#inspection-error").isVisible(), false);
    // The screenshot is a new test artifact only, never an original run/result.
    await page.emulateMedia({colorScheme: "light"});
    await page.setViewportSize({width: 1360, height: 1040});
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const screenshot = path.join(path.dirname(file), "inspection-browser-synthetic.png");
    assert(!fs.existsSync(screenshot), "Refusing to overwrite an existing screenshot");
    await page.screenshot({path: screenshot, fullPage: true});
    for (const variant of ["partial", "empty"]) {
      const sibling = path.join(path.dirname(file), `${variant}.html`);
      const siblingHtml = fs.readFileSync(sibling, "utf8");
      const siblingMatch = siblingHtml.match(/<script\s+id="inspection-data"\s+type="application\/json">([\s\S]*?)<\/script>/);
      assert(siblingMatch, `${variant} fixture must contain inline data`);
      const siblingData = JSON.parse(siblingMatch[1]);
      assert.equal(siblingData.synthetic_fixture, true, "Variant must also be explicitly synthetic");
      assert(siblingData.nodes.length <= 32768, "Variant exceeds the synthetic-only test workload");
      await context.unroute("**/*");
      await context.route("**/*", route => {
        if (route.request().url() === pathToFileURL(sibling).href) return route.continue();
        forbiddenRequests.push(route.request().url()); return route.abort();
      });
      await page.goto(pathToFileURL(sibling).href, {waitUntil: "load"});
      await page.locator("#inspection-content").waitFor({state: "visible"});
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      assert.equal(await page.locator("#inspection-error").isVisible(), false);
      assert.equal(await page.locator("#fixture-notice").isVisible(), true);
      if (variant === "partial") {
        assert.equal(siblingData.coverage, "query_neighborhoods_only");
        assert((await page.locator("#coverage-notice").textContent()).includes("부분 표시"));
      } else {
        assert.equal(siblingData.nodes.length, 0);
        assert.equal(await page.locator("#query-select").isDisabled(), true);
        assert.equal(await page.locator("#focus-neighbors").isDisabled(), true);
        assert((await page.locator("#node-count").textContent()).includes("0 / 0"));
      }
    }
    assert.deepEqual(pageErrors, []);
    assert.deepEqual(forbiddenRequests, [], "No extra file/network requests are permitted");
    console.log(`Synthetic-only offline graph browser smoke passed: ${fixture.nodes.length} nodes, 2 canvases, query/focus/rotation/node controls, partial/empty views, 320px layout, zero network requests. Screenshot: ${screenshot}`);
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
