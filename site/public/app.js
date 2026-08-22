const BQN_GPU_BACKEND = "bqn-gpu-tinygrad";
const REFERENCE_BACKENDS = [
  { backend: "cbqn", label: "cBQN", color: "#475569" },
  { backend: "native-tinygrad", label: "native tinygrad", color: "#15803d" },
  { backend: "native-torch", label: "native Torch", color: "#c2410c" },
];
const PRIMITIVE_GLYPHS = new Set(Array.from("+-×÷|⌊⌈⋆√¬∧∨<≤=≥>≠≡≢⊣⊢⥊∾≍⋈↑↓↕»«⌽⍉/⍋⍒⊏⊑⊐⊒∊⍷⊔!˙˜∘○⊸⟜⊘◶⌾⎊˘¨⌜⎉⛇⁼⍟´˝`"));
const state = { summary: null, runs: [], performance: [], capability: [], programs: [], selectedProgram: null };

const $ = (selector) => document.querySelector(selector);
const svg = (name, attributes = {}) => {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
};

async function load() {
  try {
    const [summary, runs, performance, capability, programs] = await Promise.all([
      get("/api/v1/summary"),
      get("/api/v1/runs?limit=100"),
      get("/api/v1/performance?limit=5000"),
      get("/api/v1/capability?limit=1000"),
      get("/api/v1/programs?limit=1000"),
    ]);
    Object.assign(state, {
      summary,
      runs: runs.runs,
      performance: performance.results,
      capability: capability.snapshots,
      programs: programs.programs,
    });
    state.selectedProgram = state.programs.find((program) => program.id === "program.long_pipeline_reduce.05")?.id
      ?? state.programs[0]?.id ?? null;
    initializeFilters();
    renderAll();
    renderRuns();
    renderPrograms();
  } catch (error) {
    console.error(error);
    $("#page-context").textContent = "The results API could not be loaded.";
    $("#distribution-chart").hidden = true;
    $("#distribution-empty").hidden = false;
    $("#runs-body").innerHTML = '<tr><td colspan="7" class="px-4 py-10 text-center text-red-700">The results API could not be loaded.</td></tr>';
  }
}

async function get(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

function initializeFilters() {
  const latestCommit = state.summary.latest_run?.project_commit;
  const commits = unique(state.performance.map((row) => row.project_commit)).sort().reverse();
  setOptions($("#commit-filter"), commits.map((value) => ({ value, label: shortCommit(value) })), latestCommit ?? commits[0]);
  setOptions($("#size-filter"), [
    { value: "all", label: "All sizes" },
    ...unique(state.performance.map((row) => row.input_size).filter(Number.isFinite)).sort((a, b) => a - b)
      .map((value) => ({ value: String(value), label: `${formatInteger(value)} elements` })),
  ], "all");
  const categories = unique(state.programs.map((program) => program.category).filter(Boolean)).sort();
  setOptions($("#category-filter"), [{ value: "all", label: "All workloads" }, ...categories.map((value) => ({ value, label: capitalize(value) }))], "all");
  const hardware = unique(state.performance.map(hardwareName).filter(Boolean)).sort();
  setOptions($("#hardware-filter"), [{ value: "all", label: "All machines" }, ...hardware.map((value) => ({ value, label: truncate(value, 42) }))], hardware.length === 1 ? hardware[0] : "all");
  const timings = unique(state.performance.map((row) => row.timing_scope).filter(Boolean)).sort();
  const preferredTiming = timings.includes("resident-compute") ? "resident-compute" : timings[0];
  setOptions($("#timing-filter"), timings.map((value) => ({ value, label: value })), preferredTiming);
}

function setOptions(select, options, selected) {
  select.replaceChildren();
  for (const option of options) {
    const element = document.createElement("option");
    element.value = option.value;
    element.textContent = option.label;
    select.append(element);
  }
  if (selected !== undefined) select.value = String(selected);
}

function renderAll() {
  const rows = filteredRows();
  renderContext(rows);
  renderMetrics(rows);
  renderDistribution(rows);
  renderConditionTables(rows);
  renderCapability();
  renderMeasurementConditions(rows);
  if (state.selectedProgram) renderProgramDetail(state.selectedProgram);
}

function filteredRows() {
  const commit = $("#commit-filter").value;
  const size = $("#size-filter").value;
  const category = $("#category-filter").value;
  const hardware = $("#hardware-filter").value;
  const timing = $("#timing-filter").value;
  return state.performance.filter((row) => {
    const program = programById(row.program_id);
    return (!commit || row.project_commit === commit)
      && (size === "all" || row.input_size === Number(size))
      && (category === "all" || program?.category === category)
      && (hardware === "all" || hardwareName(row) === hardware)
      && (!timing || row.timing_scope === timing);
  });
}

function renderContext(rows) {
  const latest = [...rows].sort((a, b) => b.started_at.localeCompare(a.started_at))[0];
  if (!latest) {
    $("#page-context").textContent = "No measurements match these filters.";
    return;
  }
  const sizeCount = unique(rows.map((row) => row.input_size)).length;
  const hardware = hardwareName(latest);
  const programCount = unique(rows.map((row) => row.program_id)).length;
  $("#page-context").textContent = `${shortCommit(latest.project_commit)} · ${formatDate(latest.started_at)} · ${hardware} · ${programCount} programs · ${sizeCount} input size${sizeCount === 1 ? "" : "s"}`;
}

function renderMetrics(rows) {
  const vsCbqn = pairedRatios(rows, BQN_GPU_BACKEND, "cbqn");
  const vsNativeTinygrad = pairedRatios(rows, BQN_GPU_BACKEND, "native-tinygrad");
  const vsNativeTorch = pairedRatios(rows, BQN_GPU_BACKEND, "native-torch");
  const correct = rows.filter((row) => Boolean(row.correct)).length;
  renderComparisonMetric($("#metric-cbqn"), median(vsCbqn.map((item) => item.ratio)), "cBQN CPU");
  renderComparisonMetric($("#metric-native-tinygrad"), median(vsNativeTinygrad.map((item) => item.ratio)), "native tinygrad");
  renderComparisonMetric($("#metric-native-torch"), median(vsNativeTorch.map((item) => item.ratio)), "native Torch");
  $("#metric-correct").textContent = rows.length ? `${formatPercent(correct / rows.length)}` : "—";
  $("#metric-correct-detail").textContent = `${formatInteger(correct)} of ${formatInteger(rows.length)} results`;
}

function renderDistribution(rows) {
  const chart = $("#distribution-chart");
  const table = $("#distribution-table");
  chart.replaceChildren();
  table.replaceChildren();
  const definitions = distributionConditions(rows);
  const datasets = [];
  for (const condition of definitions) {
    const conditionRows = rows.filter((row) => condition.matches(programById(row.program_id), row));
    for (const reference of REFERENCE_BACKENDS) {
      const comparisons = pairedRatios(conditionRows, BQN_GPU_BACKEND, reference.backend);
      const values = comparisons.map((item) => item.ratio).filter((value) => value > 0);
      if (!values.length) continue;
      const devices = unique(comparisons.map((item) => resultDevice(item.reference)));
      datasets.push({ condition: condition.label, reference: { ...reference, label: `${reference.label} on ${devices.join("/")}` }, values, p10: quantile(values, .1), p25: quantile(values, .25), median: median(values), p75: quantile(values, .75), p90: quantile(values, .9) });
    }
  }
  const empty = $("#distribution-empty");
  empty.hidden = datasets.length > 0;
  chart.hidden = datasets.length === 0;
  if (!datasets.length) return;

  const width = 1120;
  const left = 385;
  const right = 1040;
  const top = 58;
  const rowHeight = 28;
  const bottom = top + datasets.length * rowHeight + 20;
  const values = datasets.flatMap((dataset) => [dataset.p10, dataset.p90, 1]).filter((value) => value > 0);
  const lowPower = Math.min(-2, Math.floor(Math.log2(Math.min(...values))));
  const highPower = Math.max(1, Math.ceil(Math.log2(Math.max(...values))));
  const low = 2 ** lowPower;
  const high = 2 ** highPower;
  const x = (value) => left + (Math.log2(value) - Math.log2(low)) / (Math.log2(high) - Math.log2(low)) * (right - left);
  chart.setAttribute("viewBox", `0 0 ${width} ${bottom + 50}`);
  chart.setAttribute("height", String(bottom + 50));

  const tickStep = Math.max(1, Math.ceil((highPower - lowPower) / 7));
  for (let power = lowPower; power <= highPower; power++) {
    if (power !== 0 && power % tickStep !== 0) continue;
    const value = 2 ** power;
    const px = x(value);
    chart.append(svg("line", { x1: px, x2: px, y1: top - 18, y2: bottom, class: value === 1 ? "chart-parity" : "chart-grid" }));
    const label = svg("text", { x: px, y: bottom + 17, "text-anchor": "middle", class: "chart-label" });
    const [factor, direction] = formatAxisComparison(value).split(" ");
    const factorLine = svg("tspan", { x: px });
    factorLine.textContent = factor;
    label.append(factorLine);
    if (direction) {
      const directionLine = svg("tspan", { x: px, dy: 12 });
      directionLine.textContent = direction;
      label.append(directionLine);
    }
    chart.append(label);
  }
  const heading = svg("text", { x: left, y: 18, class: "chart-label" });
  heading.textContent = "bqn-gpu slower";
  chart.append(heading);
  const rightHeading = svg("text", { x: right, y: 18, "text-anchor": "end", class: "chart-label" });
  rightHeading.textContent = "bqn-gpu faster";
  chart.append(rightHeading);

  datasets.forEach((dataset, index) => {
    const y = top + index * rowHeight;
    const label = svg("text", { x: left - 14, y: y + 4, "text-anchor": "end", class: "chart-label" });
    label.textContent = `${dataset.condition} · vs ${dataset.reference.label}`;
    chart.append(label);
    const context = `${dataset.condition} · bqn-gpu compared with ${dataset.reference.label}`;
    const range = svg("line", { x1: x(dataset.p10), x2: x(dataset.p90), y1: y, y2: y, stroke: dataset.reference.color, class: "distribution-range" });
    appendSvgTitle(range, `${context}\nThin line: p10 ${formatComparison(dataset.p10, dataset.reference.label)} through p90 ${formatComparison(dataset.p90, dataset.reference.label)}`);
    chart.append(range);
    const iqr = svg("line", { x1: x(dataset.p25), x2: x(dataset.p75), y1: y, y2: y, stroke: dataset.reference.color, class: "distribution-iqr" });
    appendSvgTitle(iqr, `${context}\nThick line: middle 50%, from ${formatComparison(dataset.p25, dataset.reference.label)} through ${formatComparison(dataset.p75, dataset.reference.label)}`);
    chart.append(iqr);
    const dotLabel = `${context}\nMedian: ${formatComparison(dataset.median, dataset.reference.label)} across ${dataset.values.length} matched programs`;
    const dot = svg("circle", { cx: x(dataset.median), cy: y, r: 5, fill: dataset.reference.color, class: "distribution-median", tabindex: 0, role: "img", "aria-label": dotLabel });
    appendSvgTitle(dot, dotLabel);
    chart.append(dot);
    const count = svg("text", { x: right + 12, y: y + 4, class: "chart-label" });
    count.textContent = `n=${dataset.values.length}`;
    chart.append(count);

    const row = document.createElement("tr");
    const wins = dataset.values.filter((value) => value > 1).length / dataset.values.length;
    row.innerHTML = `<td class="py-2 pr-5 font-sans">${escapeHtml(dataset.condition)}</td><td class="py-2 pr-5">${escapeHtml(dataset.reference.label)}</td><td class="py-2 pr-5 text-right">${dataset.values.length}</td><td class="py-2 pr-5 text-right">${comparisonMarkup(dataset.p10, dataset.reference.label)}</td><td class="py-2 pr-5 text-right">${comparisonMarkup(dataset.median, dataset.reference.label)}</td><td class="py-2 pr-5 text-right">${comparisonMarkup(dataset.p90, dataset.reference.label)}</td><td class="py-2 text-right">${formatPercent(wins)}</td>`;
    table.append(row);
  });
}

function distributionConditions(rows) {
  const sizes = unique(rows.map((row) => row.input_size).filter(Number.isFinite)).sort((a, b) => a - b);
  const definitions = [
    { label: "All", matches: () => true },
    ...sizes.map((size) => ({ label: `${formatInteger(size)} elements`, matches: (_program, row) => row.input_size === size })),
    { label: "Elementwise", matches: (program) => tags(program).includes("elementwise") },
    { label: "Reductions", matches: (program) => tags(program).includes("reduction") },
    { label: "Naive", matches: (program) => program?.variant === "naive" },
    { label: "Idiomatic", matches: (program) => program?.variant === "idiomatic" },
    ...complexityConditions(rows),
  ];
  return definitions.filter((definition) => unique(rows.filter((row) => definition.matches(programById(row.program_id), row)).map((row) => row.program_id)).length > 0);
}

function complexityConditions(rows) {
  const buckets = [
    { label: "≤8 primitives", min: 0, max: 8 },
    { label: "9–12 primitives", min: 9, max: 12 },
    { label: "13–24 primitives", min: 13, max: 24 },
    { label: "≥25 primitives", min: 25, max: Infinity },
  ];
  return buckets.map((bucket) => ({
    label: bucket.label,
    matches: (program) => {
      const count = primitiveCount(program?.source ?? "");
      return count >= bucket.min && count <= bucket.max;
    },
  })).filter((definition) => rows.some((row) => definition.matches(programById(row.program_id))));
}

function renderConditionTables(rows) {
  renderAggregateTable($("#complexity-table"), rows, complexityConditions(rows));
  const sizes = unique(rows.map((row) => row.input_size).filter(Number.isFinite)).sort((a, b) => a - b);
  renderAggregateTable($("#size-table"), rows, sizes.map((size) => ({ label: formatInteger(size), matches: (_program, row) => row.input_size === size })));
  const workloadDefinitions = [
    { label: "All", matches: () => true },
    { label: "Elementwise", matches: (program) => tags(program).includes("elementwise") },
    { label: "Reduction", matches: (program) => tags(program).includes("reduction") },
    { label: "Naive", matches: (program) => program?.variant === "naive" },
    { label: "Idiomatic", matches: (program) => program?.variant === "idiomatic" },
  ].filter((definition) => rows.some((row) => definition.matches(programById(row.program_id), row)));
  renderWorkloadTable($("#workload-table"), rows, workloadDefinitions);
}

function renderAggregateTable(target, rows, definitions) {
  target.replaceChildren();
  for (const definition of definitions) {
    const subset = rows.filter((row) => definition.matches(programById(row.program_id), row));
    const stats = aggregateStats(subset);
    const row = document.createElement("tr");
    row.innerHTML = `<td class="py-3 pr-5 font-sans font-medium">${escapeHtml(definition.label)}</td><td class="py-3 pr-5 text-right">${formatInteger(stats.programs)}</td><td class="py-3 pr-5 text-right">${formatNs(stats.bqnGpu)}</td><td class="py-3 pr-5 text-right">${comparisonMarkup(stats.vsCbqn, "cBQN")}</td><td class="py-3 pr-5 text-right">${comparisonMarkup(stats.vsNativeTinygrad, "native tinygrad")}</td><td class="py-3 text-right">${comparisonMarkup(stats.vsNativeTorch, "native Torch")}</td>`;
    target.append(row);
  }
  if (!definitions.length) target.innerHTML = '<tr><td colspan="6" class="py-8 text-center text-slate-500">No matching data.</td></tr>';
}

function renderWorkloadTable(target, rows, definitions) {
  target.replaceChildren();
  for (const definition of definitions) {
    const subset = rows.filter((row) => definition.matches(programById(row.program_id), row));
    const stats = aggregateStats(subset);
    const row = document.createElement("tr");
    row.innerHTML = `<td class="py-3 pr-5 font-sans font-medium">${escapeHtml(definition.label)}</td><td class="py-3 pr-5 text-right">${formatInteger(stats.programs)}</td><td class="py-3 pr-5 text-right">${formatNs(stats.cbqn)}</td><td class="py-3 pr-5 text-right">${formatNs(stats.bqnGpu)}</td><td class="py-3 pr-5 text-right">${formatNs(stats.nativeTinygrad)}</td><td class="py-3 pr-5 text-right">${formatNs(stats.nativeTorch)}</td><td class="py-3 pr-5 text-right">${comparisonMarkup(stats.vsCbqn, "cBQN")}</td><td class="py-3 pr-5 text-right">${comparisonMarkup(stats.vsNativeTinygrad, "native tinygrad")}</td><td class="py-3 text-right">${comparisonMarkup(stats.vsNativeTorch, "native Torch")}</td>`;
    target.append(row);
  }
}

function aggregateStats(rows) {
  const time = (backend) => median(rows.filter((row) => backendKey(row) === backend).map((row) => row.median_ns).filter(Number.isFinite));
  return {
    programs: unique(rows.map((row) => row.program_id)).length,
    cbqn: time("cbqn"),
    bqnGpu: time(BQN_GPU_BACKEND),
    nativeTinygrad: time("native-tinygrad"),
    nativeTorch: time("native-torch"),
    vsCbqn: median(pairedRatios(rows, BQN_GPU_BACKEND, "cbqn").map((item) => item.ratio)),
    vsNativeTinygrad: median(pairedRatios(rows, BQN_GPU_BACKEND, "native-tinygrad").map((item) => item.ratio)),
    vsNativeTorch: median(pairedRatios(rows, BQN_GPU_BACKEND, "native-torch").map((item) => item.ratio)),
  };
}

function pairedRatios(rows, candidateBackend, referenceBackend) {
  const groups = new Map();
  for (const row of rows) {
    const key = [row.run_id, row.program_id, row.timing_scope, row.input_size, row.execution_mode === "unused" ? row.execution_mode : ""].join("\u0000");
    if (!groups.has(key)) groups.set(key, new Map());
    groups.get(key).set(backendKey(row), row);
  }
  const output = [];
  for (const backends of groups.values()) {
    const candidate = backends.get(candidateBackend);
    const reference = backends.get(referenceBackend);
    if (candidate?.median_ns > 0 && reference?.median_ns > 0) {
      output.push({ candidate, reference, ratio: reference.median_ns / candidate.median_ns });
    }
  }
  return output;
}

function renderCapability() {
  const selectedCommit = $("#commit-filter").value;
  const snapshots = state.capability.filter((snapshot) => !selectedCommit || snapshot.project_commit === selectedCommit)
    .sort((a, b) => b.started_at.localeCompare(a.started_at));
  const snapshot = snapshots[0] ?? [...state.capability].sort((a, b) => b.started_at.localeCompare(a.started_at))[0];
  const values = $("#capability-values");
  values.replaceChildren();
  if (!snapshot) {
    values.innerHTML = '<p class="col-span-2 text-sm text-slate-500">No capability snapshot.</p>';
    return;
  }
  $("#capability-context").textContent = `${backendLabel(snapshot)} · ${shortCommit(snapshot.project_commit)} · ${formatInteger(snapshot.corpus_programs)} corpus programs`;
  const insertCount = (snapshot.manifest?.inserts ?? []).filter((item) => item.status === "supported").length;
  const scanCount = (snapshot.manifest?.scans ?? []).filter((item) => item.status === "supported").length;
  const combinatorCount = (snapshot.manifest?.combinators ?? []).filter((item) => item.status === "supported").length;
  const mappingCount = (snapshot.manifest?.mapping_modifiers ?? []).filter((item) => item.status === "supported").length;
  const trainCount = (snapshot.manifest?.trains ?? []).filter((item) => item.status === "supported").length;
  const iterationCount = (snapshot.manifest?.iteration_modifiers ?? []).filter((item) => item.status === "supported").length;
  const metadata = snapshot.metadata ?? {};
  const primitiveClaims = snapshot.manifest?.primitives ?? [];
  const monadicTotal = metadata.monadic_forms_defined
    ?? primitiveClaims.filter((item) => item.monadic?.defined !== false).length;
  const dyadicTotal = metadata.dyadic_forms_defined
    ?? primitiveClaims.filter((item) => item.dyadic?.defined !== false).length;
  const items = [
    ["Monadic forms", countOf(snapshot.monadic_supported, monadicTotal)],
    ["Dyadic forms", countOf(snapshot.dyadic_supported, dyadicTotal)],
    ["Fold operands", countOf(snapshot.folds_supported, metadata.folds_total ?? (snapshot.manifest?.folds ?? []).length)],
    ["Insert operands", countOf(insertCount, metadata.inserts_total ?? (snapshot.manifest?.inserts ?? []).length)],
    ["Scan operands", countOf(scanCount, metadata.scans_total ?? (snapshot.manifest?.scans ?? []).length)],
    ["Combinators", countOf(combinatorCount, metadata.combinators_total ?? (snapshot.manifest?.combinators ?? []).length)],
    ["Mapping modifiers", countOf(mappingCount, metadata.mapping_modifiers_total ?? (snapshot.manifest?.mapping_modifiers ?? []).length)],
    ["Function train forms", countOf(trainCount, metadata.train_forms_total ?? (snapshot.manifest?.trains ?? []).length)],
    ["Iteration modifiers", countOf(iterationCount, metadata.iteration_modifiers_total ?? (snapshot.manifest?.iteration_modifiers ?? []).length)],
    ["Tests", `${snapshot.tests_passed} passed`],
  ];
  for (const [label, value] of items) {
    const item = document.createElement("div");
    item.className = "rounded-md bg-slate-50 p-3";
    item.innerHTML = `<dt class="text-xs text-slate-500">${escapeHtml(label)}</dt><dd class="mt-1 font-mono text-lg font-semibold">${escapeHtml(value)}</dd>`;
    values.append(item);
  }
}

function countOf(supported, total) {
  return total > 0 ? `${formatInteger(supported)} / ${formatInteger(total)}` : formatInteger(supported);
}

function renderMeasurementConditions(rows) {
  const target = $("#measurement-conditions");
  target.replaceChildren();
  const first = [...rows].sort((a, b) => b.started_at.localeCompare(a.started_at))[0];
  const conditions = [
    ["CPU", unique(rows.map((row) => row.cpu_model).filter(Boolean)).join(", ") || "—"],
    ["Accelerator", unique(rows.map((row) => row.accelerator_models).filter(Boolean)).join(", ") || "—"],
    ["Input sizes", unique(rows.map((row) => row.input_size).filter(Number.isFinite)).sort((a, b) => a - b).map(formatInteger).join(", ") || "—"],
    ["Timing", unique(rows.map((row) => row.timing_scope).filter(Boolean)).join(", ") || "—"],
    ["Implementations", unique(rows.map((row) => `${backendLabel(row)} on ${resultDevice(row)}${row.backend_version ? ` (${shortVersion(row.backend_version)})` : ""}`)).join("; ") || "—"],
    ["Dtype", first?.dtype ?? "—"],
  ];
  for (const [label, value] of conditions) {
    const item = document.createElement("div");
    item.innerHTML = `<dt class="text-xs font-medium uppercase tracking-wide text-slate-500">${escapeHtml(label)}</dt><dd class="mt-1 break-words font-mono text-xs text-slate-800">${escapeHtml(value)}</dd>`;
    target.append(item);
  }
}

function renderRuns() {
  const body = $("#runs-body");
  body.replaceChildren();
  if (!state.runs.length) {
    body.innerHTML = '<tr><td colspan="7" class="px-4 py-10 text-center text-slate-500">No runs have been recorded.</td></tr>';
    return;
  }
  for (const run of state.runs) {
    const hardware = run.accelerator_models || run.cpu_model || run.environment_label || "unlabeled";
    const row = document.createElement("tr");
    row.innerHTML = `<td class="whitespace-nowrap px-4 py-3">${escapeHtml(formatDate(run.started_at))}</td><td class="px-4 py-3 font-mono text-xs"><a class="text-blue-700 hover:underline" href="https://github.com/spencerhhubert/bqn-gpu/commit/${encodeURIComponent(run.project_commit)}">${shortCommit(run.project_commit)}</a></td><td class="px-4 py-3 font-mono text-xs">${escapeHtml(run.suite)}</td><td class="max-w-64 truncate px-4 py-3" title="${escapeHtml(hardware)}">${escapeHtml(hardware)}</td><td class="px-4 py-3 text-right font-mono text-xs">${formatInteger(run.result_count)}</td><td class="px-4 py-3"><span class="${statusClasses(run.status)}">${escapeHtml(run.status)}</span></td><td class="px-4 py-3 text-right"><button class="run-details rounded-md border border-slate-300 px-2.5 py-1.5 text-xs hover:bg-slate-50" data-run-id="${escapeHtml(run.id)}">Details</button></td>`;
    body.append(row);
  }
  for (const button of body.querySelectorAll(".run-details")) button.addEventListener("click", () => openRun(button.dataset.runId));
}

async function openRun(id) {
  const dialog = $("#run-dialog");
  const body = $("#run-dialog-body");
  $("#run-dialog-title").textContent = id;
  body.innerHTML = '<p class="text-sm text-slate-500">Loading run…</p>';
  dialog.showModal();
  try {
    const bundle = await get(`/api/v1/runs/${encodeURIComponent(id)}`);
    const run = bundle.run;
    const profile = run.profile ?? {};
    const accelerators = (profile.accelerators ?? []).map((device) => `${device.count ?? 1}× ${device.model}${device.driver ? ` · driver ${device.driver}` : ""}`).join("; ") || "—";
    const software = Object.entries(profile.software ?? {}).map(([name, version]) => `${name} ${version}`).join(" · ") || "—";
    body.innerHTML = `
      <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">${detailCell("Status", run.status)}${detailCell("Commit", shortCommit(run.project_commit))}${detailCell("Suite", run.suite)}${detailCell("Timing", run.timing_scope)}${detailCell("Input", formatInputProfile(run.input_profile))}${detailCell("Warmups / repeats", `${run.warmups} / ${run.repetitions}`)}${detailCell("Device", run.device)}${detailCell("Dtype", run.dtype ?? "—")}</div>
      <h3 class="mt-6 text-sm font-semibold">Command</h3><pre class="mt-2 overflow-x-auto rounded-md bg-slate-950 p-4 font-mono text-xs leading-5 text-slate-100">${escapeHtml(run.command ?? "—")}</pre>
      <h3 class="mt-6 text-sm font-semibold">Machine</h3><dl class="mt-2 grid gap-3 rounded-md border border-slate-200 p-4 text-sm sm:grid-cols-2">${detailTerm("CPU", profile.cpu?.model ?? "—")}${detailTerm("Accelerator", accelerators)}${detailTerm("OS / kernel", `${profile.operating_system ?? "—"} · ${profile.kernel ?? "—"}`)}${detailTerm("Software", software)}</dl>
      <div class="mt-6 flex items-center justify-between gap-3"><h3 class="text-sm font-semibold">Results (${formatInteger(bundle.results.length)})</h3><a class="text-sm text-blue-700 hover:underline" href="/api/v1/runs/${encodeURIComponent(id)}">Raw JSON ↗</a></div>
      <div class="mt-2 max-h-96 overflow-auto rounded-md border border-slate-200"><table class="w-full text-left text-xs"><thead class="sticky top-0 border-b border-slate-200 bg-slate-50 text-slate-500"><tr><th class="px-3 py-2">Program</th><th class="px-3 py-2 text-right">Input</th><th class="px-3 py-2">Implementation</th><th class="px-3 py-2">Device</th><th class="px-3 py-2">Mode</th><th class="px-3 py-2 text-right">Median</th><th class="px-3 py-2">Correct</th></tr></thead><tbody class="divide-y divide-slate-100">${bundle.results.map((result) => `<tr><td class="px-3 py-2 font-mono">${escapeHtml(result.program_id)}</td><td class="px-3 py-2 text-right font-mono">${formatInteger(result.input_size)}</td><td class="px-3 py-2 font-mono">${escapeHtml(backendLabel(result))}</td><td class="px-3 py-2 font-mono font-semibold">${escapeHtml(resultDevice(result))}</td><td class="px-3 py-2 font-mono">${escapeHtml(result.execution_mode)}</td><td class="px-3 py-2 text-right font-mono">${formatNs(result.median_ns)}</td><td class="px-3 py-2">${result.correct ? "yes" : "no"}</td></tr>`).join("")}</tbody></table></div>`;
  } catch (error) {
    console.error(error);
    body.innerHTML = '<p class="text-sm text-red-700">This run could not be loaded.</p>';
  }
}

function renderPrograms() {
  const search = $("#program-search").value.trim().toLowerCase();
  const programs = state.programs.filter((program) => [program.id, program.category, program.variant, ...tags(program)].join(" ").toLowerCase().includes(search));
  const capabilityPrograms = state.summary.latest_capability?.corpus_programs;
  $("#program-count-label").textContent = `${formatInteger(state.programs.length)} measured sources stored${capabilityPrograms ? ` · ${formatInteger(capabilityPrograms)} programs in the conformance corpus` : ""}`;
  const list = $("#program-list");
  list.replaceChildren();
  for (const program of programs) {
    const button = document.createElement("button");
    button.className = program.id === state.selectedProgram
      ? "block w-full border-b border-slate-100 bg-blue-50 px-4 py-3 text-left"
      : "block w-full border-b border-slate-100 px-4 py-3 text-left hover:bg-slate-50";
    button.innerHTML = `<span class="block break-all font-mono text-xs font-medium">${escapeHtml(program.id)}</span><span class="mt-1 block text-xs text-slate-500">${escapeHtml(program.category)} · ${escapeHtml(program.variant ?? "unlabeled")} · ${primitiveCount(program.source)} primitives</span>`;
    button.addEventListener("click", () => {
      state.selectedProgram = program.id;
      renderPrograms();
      renderProgramDetail(program.id);
    });
    list.append(button);
  }
  if (!programs.length) list.innerHTML = '<p class="p-5 text-sm text-slate-500">No programs match that filter.</p>';
  if (state.selectedProgram) renderProgramDetail(state.selectedProgram);
}

function renderProgramDetail(id) {
  const program = programById(id);
  if (!program) return;
  const rows = filteredRows().filter((row) => row.program_id === id).sort((a, b) => b.started_at.localeCompare(a.started_at) || a.backend.localeCompare(b.backend));
  const tagMarkup = tags(program).map((tag) => `<span class="rounded bg-slate-100 px-2 py-1 text-xs text-slate-600">${escapeHtml(tag)}</span>`).join("");
  const nativeSources = program.metadata?.native_implementations ?? {};
  const nativeSourceMarkup = nativeSources.tinygrad || nativeSources.torch
    ? `<details class="mt-6 rounded-md border border-slate-200" open><summary class="cursor-pointer px-4 py-3 text-sm font-medium">Independent native comparison programs</summary><div class="grid gap-4 border-t border-slate-200 p-4"><div><p class="text-xs font-medium text-slate-500">Native tinygrad source</p><pre class="mt-2 overflow-x-auto rounded-md bg-slate-950 p-4 font-mono text-xs leading-5 text-slate-100">${escapeHtml(nativeSources.tinygrad ?? "—")}</pre></div><div><p class="text-xs font-medium text-slate-500">Native Torch source</p><pre class="mt-2 overflow-x-auto rounded-md bg-slate-950 p-4 font-mono text-xs leading-5 text-slate-100">${escapeHtml(nativeSources.torch ?? "—")}</pre></div></div></details>`
    : '<p class="mt-6 text-sm text-slate-500">Native comparison sources were not stored with this older record.</p>';
  $("#program-detail").innerHTML = `
    <div class="flex flex-wrap items-start justify-between gap-3"><div><h3 class="break-all font-mono text-sm font-semibold">${escapeHtml(program.id)}</h3><p class="mt-1 text-xs text-slate-500">${escapeHtml(program.category)} · ${escapeHtml(program.variant ?? "unlabeled")} · ${primitiveCount(program.source)} static primitives · ${Array.from(program.source ?? "").length} source glyphs</p></div><div class="flex flex-wrap gap-1.5">${tagMarkup}</div></div>
    <h4 class="mt-6 text-xs font-medium uppercase tracking-wide text-slate-500">BQN source</h4><pre class="mt-2 overflow-x-auto rounded-md bg-slate-950 p-4 font-mono text-sm leading-6 text-slate-100">${escapeHtml(program.source)}</pre>
    <h4 class="mt-6 text-xs font-medium uppercase tracking-wide text-slate-500">Measurements matching current filters</h4>
    <div class="mt-2 overflow-x-auto"><table class="w-full text-left text-xs"><thead class="border-b border-slate-200 text-slate-500"><tr><th class="py-2 pr-4">Implementation</th><th class="py-2 pr-4">Device</th><th class="py-2 pr-4">Mode</th><th class="py-2 pr-4">Input</th><th class="py-2 pr-4 text-right">Median</th><th class="py-2">Commit</th></tr></thead><tbody class="divide-y divide-slate-100">${rows.length ? rows.map((row) => `<tr><td class="py-2 pr-4 font-mono">${escapeHtml(backendLabel(row))}</td><td class="py-2 pr-4 font-mono font-semibold">${escapeHtml(resultDevice(row))}</td><td class="py-2 pr-4 font-mono">${escapeHtml(row.execution_mode)}</td><td class="py-2 pr-4 font-mono">${formatInteger(row.input_size)}</td><td class="py-2 pr-4 text-right font-mono">${formatNs(row.median_ns)}</td><td class="py-2 font-mono">${shortCommit(row.project_commit)}</td></tr>`).join("") : '<tr><td colspan="6" class="py-6 text-center text-slate-500">No measurements match the current filters.</td></tr>'}</tbody></table></div>
    ${nativeSourceMarkup}
    <details class="mt-6 rounded-md border border-slate-200"><summary class="cursor-pointer px-4 py-3 text-sm font-medium">Input generator and comparison policy</summary><div class="grid gap-4 border-t border-slate-200 p-4 lg:grid-cols-2"><div><p class="text-xs text-slate-500">Input generator</p><pre class="mt-2 overflow-x-auto text-xs leading-5">${escapeHtml(JSON.stringify(program.input_generator ?? {}, null, 2))}</pre></div><div><p class="text-xs text-slate-500">Comparison policy</p><pre class="mt-2 overflow-x-auto text-xs leading-5">${escapeHtml(JSON.stringify(program.comparison_policy ?? {}, null, 2))}</pre></div></div></details>`;
}

function programById(id) { return state.programs.find((program) => program.id === id); }
function tags(program) { return Array.isArray(program?.tags) ? program.tags : []; }
function primitiveCount(source) { return Array.from(source ?? "").filter((glyph) => PRIMITIVE_GLYPHS.has(glyph)).length; }
function hardwareName(row) { return row.accelerator_models || row.cpu_model || "unlabeled machine"; }
function backendKey(row) {
  if (row.backend === "tinygrad") return "bqn-gpu-tinygrad";
  if (row.backend === "torch") return "bqn-gpu-torch";
  return row.backend;
}
function backendLabel(row) {
  const labels = {
    cbqn: "cBQN reference",
    "bqn-gpu-tinygrad": "bqn-gpu → tinygrad",
    "bqn-gpu-torch": "bqn-gpu → Torch",
    "native-tinygrad": "native tinygrad",
    "native-torch": "native Torch",
  };
  return labels[backendKey(row)] ?? row.backend;
}
function resultDevice(row) { return row.metadata?.device ?? (backendKey(row) === "cbqn" ? "CPU" : row.device ?? "—"); }
function unique(values) { return [...new Set(values)]; }
function median(values) { return quantile(values, .5); }
function quantile(values, q) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position);
  const fraction = position - lower;
  return sorted[lower + 1] === undefined ? sorted[lower] : sorted[lower] + fraction * (sorted[lower + 1] - sorted[lower]);
}
function shortCommit(value) { return value ? value.slice(0, 8) : "unknown"; }
function shortVersion(value) { return String(value).length > 14 ? `${String(value).slice(0, 8)}…` : value; }
function formatDate(value) { return new Intl.DateTimeFormat("en", { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(value)); }
function formatInteger(value) { return new Intl.NumberFormat("en-US").format(value ?? 0); }
function formatPercent(value) { return Number.isFinite(value) ? `${(value * 100).toFixed(value === 1 ? 0 : 1)}%` : "—"; }
function comparisonDirection(value) {
  if (!Number.isFinite(value) || value <= 0) return "unavailable";
  const factor = value >= 1 ? value : 1 / value;
  if (factor < 1.05) return "same";
  return value > 1 ? "faster" : "slower";
}
function formatComparison(value, reference, includeReference = true) {
  const direction = comparisonDirection(value);
  if (direction === "unavailable") return "—";
  const suffix = includeReference ? ` ${direction === "same" ? "as" : "than"} ${reference}` : "";
  if (direction === "same") return `about the same${suffix}`;
  const factor = value >= 1 ? value : 1 / value;
  if (factor < 1.5) return `slightly ${direction}${suffix}`;
  return `${formatInteger(Math.round(factor))}× ${direction}${suffix}`;
}
function formatAxisComparison(value) {
  if (value === 1) return "same speed";
  return formatComparison(value, "", false);
}
function comparisonClasses(direction) {
  if (direction === "faster") return "text-emerald-700";
  if (direction === "slower") return "text-rose-700";
  return "text-slate-600";
}
function comparisonMarkup(value, reference) {
  const direction = comparisonDirection(value);
  const label = formatComparison(value, reference);
  return `<span class="whitespace-nowrap font-semibold ${comparisonClasses(direction)}" title="${escapeHtml(`bqn-gpu is ${label}`)}">${escapeHtml(label)}</span>`;
}
function renderComparisonMetric(element, value, reference) {
  const direction = comparisonDirection(value);
  const label = formatComparison(value, reference);
  element.textContent = label;
  element.title = label === "—" ? "No matched programs" : `bqn-gpu is ${label}`;
  element.classList.remove("text-emerald-700", "text-rose-700", "text-slate-600");
  element.classList.add(comparisonClasses(direction));
}
function appendSvgTitle(element, label) {
  const title = svg("title");
  title.textContent = label;
  element.append(title);
}
function formatNs(value) {
  if (!Number.isFinite(value)) return "—";
  if (value < 1e3) return `${Math.round(value)} ns`;
  if (value < 1e6) return `${(value / 1e3).toFixed(value < 1e4 ? 2 : 1)} µs`;
  if (value < 1e9) return `${(value / 1e6).toFixed(value < 1e7 ? 2 : 1)} ms`;
  return `${(value / 1e9).toFixed(2)} s`;
}
function formatInputProfile(profile) { return profile?.size ? `${formatInteger(profile.size)} elements` : JSON.stringify(profile ?? {}); }
function truncate(value, length) { return value.length <= length ? value : `${value.slice(0, length - 1)}…`; }
function capitalize(value) { return value ? value[0].toUpperCase() + value.slice(1) : value; }
function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]); }
function statusClasses(status) {
  if (status === "pass") return "rounded bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700";
  if (status === "fail") return "rounded bg-red-50 px-2 py-1 text-xs font-medium text-red-700";
  return "rounded bg-amber-50 px-2 py-1 text-xs font-medium text-amber-700";
}
function detailCell(label, value) { return `<div class="rounded-md bg-slate-50 p-3"><p class="text-xs text-slate-500">${escapeHtml(label)}</p><p class="mt-1 break-words font-mono text-xs">${escapeHtml(value)}</p></div>`; }
function detailTerm(label, value) { return `<div><dt class="text-xs text-slate-500">${escapeHtml(label)}</dt><dd class="mt-1 break-words font-mono text-xs">${escapeHtml(value)}</dd></div>`; }

for (const selector of ["#commit-filter", "#size-filter", "#category-filter", "#hardware-filter", "#timing-filter"]) {
  $(selector).addEventListener("change", renderAll);
}
$("#reset-filters").addEventListener("click", () => { initializeFilters(); renderAll(); });
$("#program-search").addEventListener("input", renderPrograms);
$("#run-dialog-close").addEventListener("click", () => $("#run-dialog").close());
$("#run-dialog").addEventListener("click", (event) => { if (event.target === $("#run-dialog")) $("#run-dialog").close(); });
load();
