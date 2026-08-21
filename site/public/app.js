const COLORS = ["#58e2c2", "#ff8a45", "#e9c77b", "#a995ff", "#6ea8ff", "#ff6b5f"];
const state = { summary: null, runs: [], performance: [], capability: [] };

const $ = (selector) => document.querySelector(selector);
const svg = (name, attributes = {}) => {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
};

async function load() {
  try {
    const [summary, runs, performance, capability] = await Promise.all([
      get("/api/v1/summary"), get("/api/v1/runs?limit=40"),
      get("/api/v1/performance?limit=5000"), get("/api/v1/capability?limit=1000"),
    ]);
    Object.assign(state, { summary, runs: runs.runs, performance: performance.results, capability: capability.snapshots });
    renderSummary(); renderProgramOptions(); renderPerformance(); renderSpeedups(); renderCapability(); renderRuns();
    $("#live-status").className = "live-status ready"; $("#live-status").lastChild.textContent = " live database";
  } catch (error) {
    console.error(error);
    $("#live-status").className = "live-status failed"; $("#live-status").lastChild.textContent = " database error";
    $("#runs-body").innerHTML = `<tr><td colspan="7" class="loading">The results API is temporarily unavailable.</td></tr>`;
  }
}

async function get(url) { const response = await fetch(url); if (!response.ok) throw new Error(`${url}: ${response.status}`); return response.json(); }

function renderSummary() {
  const { counts, latest_run: latest, latest_capability: capability } = state.summary;
  $("#count-runs").textContent = formatInteger(counts.runs);
  $("#count-programs").textContent = formatInteger(capability?.corpus_programs ?? counts.programs);
  $("#count-glyphs").textContent = capability?.glyphs_total ?? "—";
  $("#count-correct").textContent = formatInteger(counts.correct_results);
  $("#correct-denominator").textContent = `of ${formatInteger(counts.results)} recorded results`;
  if (latest) {
    $("#latest-commit").textContent = shortCommit(latest.project_commit);
    $("#latest-commit").href = `https://github.com/spencerhhubert/bqn-gpu/commit/${latest.project_commit}`;
    $("#latest-date").textContent = formatDate(latest.started_at);
  }
}

function renderProgramOptions() {
  const select = $("#program-filter");
  const ids = [...new Set(state.performance.map((item) => item.program_id))].sort();
  for (const id of ids) { const option = document.createElement("option"); option.value = id; option.textContent = id; select.append(option); }
  select.value = ids.includes("program.long_pipeline_reduce.05")
    ? "program.long_pipeline_reduce.05"
    : (ids[0] ?? "");
}

function renderPerformance() {
  const chart = $("#performance-chart"); chart.replaceChildren();
  const requestedProgram = $("#program-filter").value;
  const points = state.performance.filter((item) => !requestedProgram || item.program_id === requestedProgram);
  const empty = $("#performance-empty");
  if (!points.length) { chart.hidden = true; empty.hidden = false; return; }
  chart.hidden = false; empty.hidden = true;

  const seriesMap = new Map();
  for (const point of points) {
    const key = `${point.backend} · ${point.device}`;
    if (!seriesMap.has(key)) seriesMap.set(key, []);
    seriesMap.get(key).push(point);
  }
  const allDates = [...new Set(points.map((item) => item.started_at))].sort();
  const allValues = points.map((item) => item.median_ns / 1e6).filter((value) => value > 0);
  const log = $("#scale-filter").value === "log";
  const transformed = allValues.map((value) => log ? Math.log10(value) : value);
  const rawMin = Math.min(...transformed), rawMax = Math.max(...transformed);
  const spread = Math.max(rawMax - rawMin, log ? .5 : rawMax * .2 || 1);
  const min = rawMin - spread * .12, max = rawMax + spread * .12;
  const left = 72, right = 975, top = 24, bottom = 342;
  const x = (date) => allDates.length === 1 ? (left + right) / 2 : left + allDates.indexOf(date) / (allDates.length - 1) * (right - left);
  const y = (value) => bottom - ((log ? Math.log10(value) : value) - min) / (max - min) * (bottom - top);

  for (let index = 0; index <= 5; index++) {
    const py = top + index / 5 * (bottom - top);
    const transformedValue = max - index / 5 * (max - min);
    chart.append(svg("line", { x1: left, x2: right, y1: py, y2: py, class: "grid-line" }));
    const label = svg("text", { x: left - 12, y: py + 4, "text-anchor": "end", class: "axis-label" });
    label.textContent = formatMs(log ? 10 ** transformedValue : transformedValue); chart.append(label);
  }
  const shownDates = allDates.filter((_, index) => allDates.length <= 6 || index % Math.ceil(allDates.length / 6) === 0 || index === allDates.length - 1);
  for (const date of shownDates) {
    const label = svg("text", { x: x(date), y: 374, "text-anchor": "middle", class: "axis-label" }); label.textContent = shortDate(date); chart.append(label);
  }
  const legend = $("#performance-legend"); legend.replaceChildren();
  [...seriesMap.entries()].forEach(([name, values], index) => {
    const color = COLORS[index % COLORS.length]; values.sort((a, b) => a.started_at.localeCompare(b.started_at));
    const coordinates = values.map((point) => `${x(point.started_at)},${y(point.median_ns / 1e6)}`).join(" ");
    chart.append(svg("polyline", { points: coordinates, class: "chart-path", stroke: color }));
    for (const point of values) { const dot = svg("circle", { cx: x(point.started_at), cy: y(point.median_ns / 1e6), r: 4.5, fill: color, class: "chart-dot" }); dot.append(svg("title")); dot.firstChild.textContent = `${name}\n${point.program_id}\n${formatMs(point.median_ns / 1e6)}\n${shortCommit(point.project_commit)}`; chart.append(dot); }
    const item = document.createElement("span"); item.innerHTML = `<i style="background:${color}"></i>${escapeHtml(name)}`; legend.append(item);
  });
}

function renderSpeedups() {
  const groups = new Map();
  for (const item of state.performance) {
    const key = `${item.run_id}\u0000${item.program_id}\u0000${item.timing_scope}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  }
  const comparisons = [];
  for (const values of groups.values()) {
    const cpu = values.find((item) => item.backend === "cbqn");
    for (const gpu of values.filter((item) => item.backend !== "cbqn" && /cuda|gpu/i.test(item.device))) {
      if (cpu?.median_ns && gpu.median_ns) comparisons.push({ ...gpu, speedup: cpu.median_ns / gpu.median_ns });
    }
  }
  comparisons.sort((a, b) => b.started_at.localeCompare(a.started_at) || b.speedup - a.speedup);
  const latestRun = comparisons[0]?.run_id;
  const latest = comparisons.filter((item) => item.run_id === latestRun).sort((a, b) => b.speedup - a.speedup).slice(0, 7);
  const container = $("#speedup-bars"); container.replaceChildren();
  $("#speedup-empty").hidden = latest.length > 0;
  if (!latest.length) return;
  const max = Math.max(...latest.map((item) => item.speedup), 1);
  for (const item of latest) {
    const row = document.createElement("div"); row.className = "speed-row";
    row.innerHTML = `<span class="speed-label" title="${escapeHtml(item.program_id)}">${escapeHtml(item.program_id)}</span><span class="speed-track"><i class="speed-fill" style="width:${Math.max(1, item.speedup / max * 100)}%"></i></span><strong class="speed-value">${item.speedup.toFixed(item.speedup >= 10 ? 1 : 2)}×</strong>`;
    container.append(row);
  }
}

function renderCapability() {
  const chart = $("#capability-chart"); chart.replaceChildren();
  const snapshots = state.capability.filter((item) => item.backend === "tinygrad");
  $("#capability-empty").hidden = snapshots.length > 0; chart.hidden = snapshots.length === 0; if (!snapshots.length) return;
  const fields = [["monadic_supported", "#58e2c2"], ["dyadic_supported", "#ff8a45"], ["folds_supported", "#e9c77b"]];
  const maxValue = Math.max(1, ...snapshots.flatMap((item) => fields.map(([field]) => item[field])));
  const left = 42, right = 600, top = 20, bottom = 260;
  const x = (index) => snapshots.length === 1 ? (left + right) / 2 : left + index / (snapshots.length - 1) * (right - left);
  const y = (value) => bottom - value / maxValue * (bottom - top);
  for (let index = 0; index <= 4; index++) {
    const py = top + index / 4 * (bottom - top); chart.append(svg("line", { x1: left, x2: right, y1: py, y2: py, class: "grid-line" }));
    const label = svg("text", { x: left - 10, y: py + 4, "text-anchor": "end", class: "axis-label" }); label.textContent = String(Math.round(maxValue * (1 - index / 4))); chart.append(label);
  }
  for (const [field, color] of fields) {
    const coordinates = snapshots.map((item, index) => `${x(index)},${y(item[field])}`).join(" "); chart.append(svg("polyline", { points: coordinates, class: "chart-path", stroke: color }));
    snapshots.forEach((item, index) => { const dot = svg("circle", { cx: x(index), cy: y(item[field]), r: 4, fill: color, class: "chart-dot" }); dot.append(svg("title")); dot.firstChild.textContent = `${field.replace("_supported", "")} ${item[field]}\n${shortCommit(item.project_commit)}`; chart.append(dot); });
  }
  snapshots.forEach((item, index) => { const label = svg("text", { x: x(index), y: 291, "text-anchor": "middle", class: "axis-label" }); label.textContent = shortCommit(item.project_commit); chart.append(label); });
}

function renderRuns() {
  const body = $("#runs-body"); body.replaceChildren();
  if (!state.runs.length) { body.innerHTML = `<tr><td colspan="7" class="loading">No benchmark runs have been published yet.</td></tr>`; return; }
  for (const run of state.runs) {
    const row = document.createElement("tr");
    const hardware = run.accelerator_models || run.cpu_model || run.environment_label || "unlabeled";
    row.innerHTML = `<td><a class="run-link" href="/api/v1/runs/${encodeURIComponent(run.id)}">${escapeHtml(run.id)}</a></td><td><a class="commit" href="https://github.com/spencerhhubert/bqn-gpu/commit/${run.project_commit}">${shortCommit(run.project_commit)}</a></td><td>${escapeHtml(run.suite)}</td><td class="hardware" title="${escapeHtml(hardware)}">${escapeHtml(truncate(hardware, 30))}</td><td>${escapeHtml(run.timing_scope)}</td><td>${formatInteger(run.result_count)}</td><td class="status-${run.status}">${escapeHtml(run.status)}</td>`;
    body.append(row);
  }
}

function shortCommit(value) { return value ? value.slice(0, 8) : "unknown"; }
function shortDate(value) { return new Intl.DateTimeFormat("en", { month: "short", day: "numeric" }).format(new Date(value)); }
function formatDate(value) { return new Intl.DateTimeFormat("en", { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(value)); }
function formatInteger(value) { return new Intl.NumberFormat("en-US").format(value ?? 0); }
function formatMs(value) { if (value < .001) return `${Math.round(value * 1e6)} ns`; if (value < 1) return `${(value * 1e3).toFixed(value < .01 ? 2 : 1)} µs`; if (value < 1000) return `${value.toFixed(value < 10 ? 2 : 1)} ms`; return `${(value / 1000).toFixed(1)} s`; }
function truncate(value, length) { return value.length <= length ? value : `${value.slice(0, length - 1)}…`; }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]); }

$("#program-filter").addEventListener("change", renderPerformance);
$("#scale-filter").addEventListener("change", renderPerformance);
load();
