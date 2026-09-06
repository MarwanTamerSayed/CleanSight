/* ============================================================
   FinChat · front-end logic (vanilla JS, no build step)
   ============================================================ */

"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (id) => document.getElementById(id);

const state = {
  backend: "openrouter",
  datasets: [],
  datasetId: null,
  interactive: false,
  pollers: new Map(),
  jobs: new Map(),
  view: "ask",
  sessionId: "sess_" + Math.random().toString(36).slice(2, 10),
};

/* ----------------------------- helpers ----------------------------- */
async function api(path, opts = {}) {
  const res = await fetch(path, {
    method: opts.method || "GET",
    headers: { "Content-Type": "application/json" },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail || res.statusText; } catch (_) { detail = res.statusText; }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

function toast(msg, kind = "info") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = msg;
  el("toasts").appendChild(t);
  setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity .4s"; setTimeout(() => t.remove(), 400); }, 3600);
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* Format numbers with commas: 144630 → "144,630" */
function fmtNum(v) {
  const n = Number(v);
  if (Number.isNaN(n)) return String(v);
  if (Number.isInteger(n)) return n.toLocaleString("en-US");
  return n.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

/* Detect if a string looks like a tabular dataset (whitespace-aligned columns) */
function detectTable(text) {
  const lines = text.trim().split("\n").filter((l) => l.trim());
  if (lines.length < 2) return null;
  const sep = /\s{2,}/;
  const headerParts = lines[0].split(sep).map((s) => s.trim()).filter(Boolean);
  if (headerParts.length < 2) return null;
  const dataLines = lines.slice(1);
  const rows = [];
  for (const line of dataLines) {
    const parts = line.split(sep).map((s) => s.trim()).filter(Boolean);
    if (parts.length < 2) return null;
    rows.push(parts);
  }
  if (rows.length < 1) return null;
  return { headers: headerParts, rows };
}

/* Format a detected table as a styled HTML table */
function renderTable(table) {
  const numCols = new Set();
  table.headers.forEach((h, i) => {
    const allNum = table.rows.every((r) => r[i] !== undefined && !isNaN(Number(String(r[i]).replace(/[,₹$%]/g, ""))));
    if (allNum) numCols.add(i);
  });
  let html = '<div class="result-table-wrap"><table class="result-table"><thead><tr>';
  table.headers.forEach((h, i) => {
    html += `<th${numCols.has(i) ? ' class="num"' : ""}>${esc(h)}</th>`;
  });
  html += "</tr></thead><tbody>";
  for (const row of table.rows) {
    html += "<tr>";
    row.forEach((cell, i) => {
      const raw = cell.replace(/[,₹$%]/g, "");
      const isNum = numCols.has(i) && !isNaN(Number(raw));
      const cls = isNum ? ' class="num"' : "";
      html += `<td${cls}>${esc(cell)}</td>`;
    });
    html += "</tr>";
  }
  html += "</tbody></table></div>";
  return html;
}

/* Render markdown text to HTML with proper structure */
function renderMarkdown(text) {
  const raw = String(text);
  const blocks = raw.split(/```/);
  let html = "";
  for (let i = 0; i < blocks.length; i++) {
    if (i % 2 === 1) {
      const lang = blocks[i].split("\n")[0].trim();
      const code = lang ? blocks[i].slice(lang.length).trimStart() : blocks[i];
      const id = "code-" + Math.random().toString(36).slice(2, 8);
      html += `<pre><button class="copy-code" onclick="navigator.clipboard.writeText(document.getElementById('${id}').textContent);this.textContent='copied ✓';setTimeout(()=>this.textContent='copy',1200)">copy</button><code id="${id}">${esc(code)}</code></pre>`;
      continue;
    }
    let seg = blocks[i];
    /* tables: detect pipe-delimited markdown tables */
    seg = seg.replace(/(?:^|\n)((?:\|.+\|(?:\n|$))+)/g, (match, tableBlock) => {
      const tLines = tableBlock.trim().split("\n").filter((l) => l.trim());
      if (tLines.length < 2) return match;
      const parseRow = (l) => l.split("|").map((s) => s.trim()).filter((s) => s !== "");
      const headers = parseRow(tLines[0]);
      if (tLines.length >= 3 && /^[\s|:-]+$/.test(tLines[1])) {
        const rows = tLines.slice(2).map(parseRow);
        const numCols = new Set();
        headers.forEach((h, ci) => {
          if (rows.every((r) => r[ci] !== undefined && !isNaN(Number(String(r[ci]).replace(/[,₹$%]/g, ""))))) numCols.add(ci);
        });
        let t = '<div class="result-table-wrap"><table class="result-table"><thead><tr>';
        headers.forEach((h, ci) => { t += `<th${numCols.has(ci) ? ' class="num"' : ""}>${esc(h)}</th>`; });
        t += "</tr></thead><tbody>";
        for (const row of rows) {
          t += "<tr>";
          row.forEach((cell, ci) => {
            const raw = String(cell).replace(/[,₹$%]/g, "");
            t += `<td${numCols.has(ci) && !isNaN(Number(raw)) ? ' class="num"' : ""}>${esc(cell)}</td>`;
          });
          t += "</tr>";
        }
        t += "</tbody></table></div>";
        return "\n" + t;
      }
      return match;
    });
    /* headings */
    seg = seg.replace(/^#### (.+)$/gm, "<h4>$1</h4>");
    seg = seg.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    seg = seg.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    seg = seg.replace(/^# (.+)$/gm, "<h1>$1</h1>");
    /* horizontal rule */
    seg = seg.replace(/^---+$/gm, "<hr/>");
    /* bold and italic */
    seg = seg.replace(/\*\*\*(.+?)\*\*\*/g, "<b><i>$1</i></b>");
    seg = seg.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
    seg = seg.replace(/\*(.+?)\*/g, "<i>$1</i>");
    /* inline code */
    seg = seg.replace(/`([^`]+)`/g, "<code>$1</code>");
    /* unordered list items */
    seg = seg.replace(/^\s*[-*]\s+(.+)$/gm, "<li>$1</li>");
    /* ordered list items */
    seg = seg.replace(/^\s*\d+\.\s+(.+)$/gm, "<oli>$1</oli>");
    /* wrap consecutive li */
    seg = seg.replace(/((?:<li>[\s\S]*?<\/li>\s*)+)/g, (m) => "<ul>" + m + "</ul>");
    seg = seg.replace(/((?:<oli>[\s\S]*?<\/oli>\s*)+)/g, (m) => {
      const inner = m.replace(/<oli>([\s\S]*?)<\/oli>/g, "<li>$1</li>");
      return "<ol>" + inner + "</ol>";
    });
    /* paragraphs: wrap remaining bare text */
    seg = seg.replace(/^(?!<[hulo]|<li|<div|<table|<pre|<hr)(.+)$/gm, "<p>$1</p>");
    html += seg;
  }
  return html;
}

function fmtTime(t) {
  const d = new Date(t * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function fmtDur(sec) {
  const m = Math.floor(sec / 60);
  const s = String(sec % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function stopPolling(jobId) {
  const iv = state.pollers.get(jobId);
  if (iv) { clearInterval(iv); state.pollers.delete(jobId); }
}

/* ------------------------------ backend ---------------------------- */
async function probeHealth() {
  try {
    const h = await api("/api/health");
    setBackendDot(h.ollama ? "ok" : "warn", h.ollama ? "available" : "offline");
  } catch (_) {
    setBackendDot("err", "server unreachable");
  }
}

function setBackendDot(stateCls, text) {
  const dot = el("backend-dot");
  dot.className = `dot ${stateCls}`;
  el("backend-state").textContent = text;
}

async function refreshBackendName() {
  const m = state.backend === "openrouter"
    ? { name: "OpenRouter", sub: "minimax-m3 · cloud" }
    : { name: "Local Ollama", sub: "qwen2.5:7b · local" };
  el("backend-name").textContent = m.name;
  el("backend-state").textContent = m.sub;
  probeHealth();
}

/* ------------------------------ datasets --------------------------- */
async function loadDatasets() {
  try {
    const { datasets } = await api("/api/datasets");
    state.datasets = datasets;
    renderDatasetSelect();
    renderDatasetGrid();
    renderCleanSelect();
  } catch (e) { toast(`Could not load datasets: ${e.message}`, "err"); }
}

function renderDatasetSelect() {
  const sel = el("dataset-select");
  const clean = state.datasets.length > 0 && state.datasets.filter((d) => d.folder === "Cleaning_Phase")[0];
  const prev = state.datasetId;
  sel.innerHTML = "";
  const raw = state.datasets.filter((d) => d.folder === "Handling_Data");
  const cleaned = state.datasets.filter((d) => d.folder === "Cleaning_Phase");
  const group = (label, arr) => {
    if (!arr.length) return;
    const og = document.createElement("optgroup");
    og.label = label;
    arr.forEach((d) => {
      const o = document.createElement("option");
      o.value = d.id;
      o.textContent = `${d.name}  (${d.rows ?? "?"} rows)`;
      og.appendChild(o);
    });
    sel.appendChild(og);
  };
  group("Cleaned", cleaned);
  group("Raw", raw);
  if (prev && state.datasets.some((d) => d.id === prev)) sel.value = prev;
  else if (clean) sel.value = clean.id;
  state.datasetId = sel.value || null;
  updateComposerHint();
}

function updateComposerHint() {
  const ds = state.datasets.find((d) => d.id === state.datasetId);
  el("composer-hint").textContent = ds
    ? `Analyzing: ${ds.name} — ${ds.columns.length} columns · ${fmtNum(ds.rows ?? "?")} rows`
    : `No cleaned dataset — open Data Prep to clean a raw file first.`;
}

/* ------------------------- view switching -------------------------- */
function switchView(view) {
  state.view = view;
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  el("view-ask").hidden = view !== "ask";
  el("view-data").hidden = view !== "data";
  el("view-title").textContent = view === "ask" ? "Ask the Finance Assistant" : "Data Preparation";
}

/* --------------------------- pipeline UI --------------------------- */
const STEP_LABELS = {
  analysis: [
    { key: "planner", name: "Planning", sub: "designs the analysis" },
    { key: "coder", name: "Data Analysis", sub: "writes & executes code" },
    { key: "reviser", name: "Validation", sub: "validates & revises" },
    { key: "synthesizer", name: "Final Answer", sub: "synthesize response" },
  ],
  cleaning: [
    { key: "load_and_profile", name: "Load & Profile", sub: "reads raw CSV" },
    { key: "planner", name: "Planning", sub: "cleaning strategy" },
    { key: "coder", name: "Execution", sub: "writes & runs code" },
    { key: "reviser", name: "Validation", sub: "validates & revises" },
  ],
};

function buildStepper(type) {
  const ol = document.createElement("ol");
  ol.className = "stepper";
  ol.id = "stepper";
  ol.style.display = "flex";
  STEP_LABELS[type].forEach((s) => {
    const li = document.createElement("li");
    li.className = "step";
    li.id = `step-${s.key}`;
    li.innerHTML = `<span class="step-dot"></span>
      <div class="step-info"><span class="step-name">${s.name}</span><span class="step-sub">${s.sub}</span></div>`;
    ol.appendChild(li);
  });
  return ol;
}

function setStepState(type, progress, seen) {
  const order = STEP_LABELS[type].map((s) => s.key);
  order.forEach((key) => {
    const node = document.getElementById(`step-${key}`);
    if (!node) return;
    node.className = "step";
    if (seen.includes(key)) node.classList.add("done");
    if (progress && progress.node === key) node.classList.add("active");
  });
}

function applyStepStates(ui, progress, allDone = false) {
  const all = STEP_LABELS[ui.type].map((s) => s.key);
  const seen = allDone ? all : [...ui.seen];
  setStepState(ui.type, allDone ? null : progress, seen);
  if (progress && progress.revision_count != null) {
    const rev = el("rev-count");
    if (rev) rev.textContent = progress.revision_count;
    const rmax = el("rev-max");
    if (rmax && progress.max_revisions != null) rmax.textContent = progress.max_revisions;
  }
}

function renderPipelinePanel(ui) {
  const empty = el("stepper-empty");
  if (empty) empty.hidden = true;
  const container = el("stepper");
  const fresh = buildStepper(ui.type);
  if (container) container.replaceWith(fresh);
  else el("pipeline-card").insertBefore(fresh, el("rev-track"));
  el("rev-track").hidden = false;
  el("rev-max").textContent = "3";
  const pill = el("pipeline-status");
  pill.className = "pill run";
  pill.textContent = "running";
  /* auto-expand pipeline when analysis starts */
  const card = el("pipeline-card");
  if (card) card.classList.remove("collapsed");
  if (ui.current) setStepState(ui.type, { node: ui.current }, ui.seen);
}

/* ------------------------------ thread ----------------------------- */
function addUserMessage(text) {
  const wrap = document.createElement("div");
  wrap.className = "msg msg-user";
  wrap.innerHTML = `<div class="bubble">${esc(text)}</div><div class="msg-meta">you · ${fmtTime(Date.now() / 1000)}</div>`;
  const thread = el("thread");
  el("empty-state")?.remove();
  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;
  return wrap;
}

function jobCard(jobId, type, backend, question) {
  const card = document.createElement("div");
  card.className = "msg-card";
  card.dataset.job = jobId;
  card.innerHTML = `
    <div class="card-head"><span class="spark"></span><span>${type === "analysis" ? "Analysis" : "Cleaning"} · ${backend === "openrouter" ? "OpenRouter" : "Local"}</span>
      <span class="pill run" id="status-pill">running</span></div>
    <div class="card-body" id="card-body">
      <div class="chip-row" id="chip-row">
        <span class="chip" id="chip-time">starting…</span>
        <span class="chip cyan" id="chip-elapsed">0:00</span>
        <span class="chip">${esc(question)}</span>
      </div>
      <div class="skeleton" id="inner-skeleton">${"<div class='skel-line w90'></div>".repeat(4)}</div>
    </div>`;
  return card;
}

function renderAnalysisResult(card, job, ui) {
  const r = job.result || {};
  const head = card.querySelector(".card-head");
  const body = card.querySelector(".card-body");
  const statusPill = card.querySelector("#status-pill");
  head.classList.add("done");
  statusPill.className = "pill doneg";
  statusPill.textContent = "validated";

  const chips = [];
  chips.push(`<span class="chip ok">✓ validated${r.validated ? "" : " (soft)"}</span>`);
  chips.push(`<span class="chip cyan">revisions ${r.revision_count ?? 0}</span>`);
  chips.push(`<span class="chip">backend ${job.backend}</span>`);
  if (ui && ui.elapsed != null) chips.push(`<span class="chip">took ${fmtDur(ui.elapsed)}</span>`);

  /* charts */
  let chartsHtml = "";
  if (job.charts && job.charts.length) {
    chartsHtml = `<div class="charts">`;
    job.charts.forEach((c) => {
      const dl = `<a class="dl-btn" href="${c.url}" download="${esc(c.filename)}" title="Download ${esc(c.filename)}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke-linecap="round" stroke-linejoin="round"/><path d="M7 10l5 5 5-5" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 15V3" stroke-linecap="round" stroke-linejoin="round"/></svg></a>`;
      const open = `<a class="dl-btn" href="${c.url}" target="_blank" rel="noopener" title="Open full size">↗</a>`;
      if (c.kind === "html") {
        chartsHtml += `<div class="chart-card"><div class="chart-frame"><iframe src="${c.url}" loading="lazy" title="${esc(c.filename)}" sandbox="allow-scripts"></iframe></div><div class="chart-label"><span>${esc(c.filename)}</span><span class="chart-actions">${open}${dl}</span></div></div>`;
      } else {
        chartsHtml += `<div class="chart-card"><div class="chart-frame"><img src="${c.url}" alt="${esc(c.filename)}" loading="lazy" /></div><div class="chart-label"><span>${esc(c.filename)}</span><span class="chart-actions">${open}${dl}</span></div></div>`;
      }
    });
    chartsHtml += `</div>`;
  }

  /* result text: detect tables or render as pre-formatted */
  let resultHtml = "";
  if (r.result_text) {
    const tbl = detectTable(r.result_text);
    if (tbl) {
      resultHtml = renderTable(tbl);
    } else {
      resultHtml = `<div class="creturn">${esc(r.result_text)}</div>`;
    }
  }

  /* answer: render as markdown */
  const answerHtml = r.answer
    ? `<div class="answer">${renderMarkdown(r.answer)}</div>` : "";

  body.innerHTML = `
    <div class="chip-row">${chips.join("")}</div>
    ${answerHtml}
    ${resultHtml}
    ${chartsHtml}`;
  card.offsetHeight;

  /* keep Plan & Code side panel in sync */
  const details = el("details-body");
  if (details) {
    const parts = [];
    if (r.plan) parts.push(`<h4>Analysis Plan</h4><pre>${esc(r.plan)}</pre>`);
    if (r.code) parts.push(`<h4>Generated Code</h4><pre>${esc(r.code)}</pre>`);
    if (parts.length) details.innerHTML = parts.join("");
  }
}

function renderErrorCard(card, job, ui) {
  const head = card.querySelector(".card-head");
  const statusPill = card.querySelector("#status-pill");
  head.classList.add("err");
  statusPill.className = "pill errc";
  statusPill.textContent = "failed";
  const dur = ui && ui.elapsed != null ? ` · ${fmtDur(ui.elapsed)}` : "";
  card.querySelector(".card-body").innerHTML = `
    <div class="chip-row"><span class="chip" style="color:var(--red);border-color:rgba(248,113,113,0.4)">failed${dur}</span></div>
    <div class="err-box">
      <div style="margin-bottom:6px;font-weight:600">Something went wrong</div>
      <div style="margin-bottom:8px;opacity:0.8">${esc(job.error || "Unknown error.")}</div>
      <details><summary style="cursor:pointer;color:var(--text-dim);font-size:11px">Technical details</summary><pre style="margin-top:6px;font-size:11px;max-height:150px;overflow:auto">${esc(job.error || "No details available.")}</pre></details>
    </div>`;
  toast("The job failed — see the card for details.", "err");
}

/* ---------------------------- analysis ----------------------------- */
async function sendQuestion(question) {
  if (!state.datasetId) { toast("Select a dataset first.", "err"); return; }
  if (!question.trim()) return;

  el("empty-state")?.remove();
  addUserMessage(question.trim());

  let jobId;
  try {
    const resp = await api("/api/analysis", {
      method: "POST",
      body: {
        dataset: state.datasetId,
        question: question.trim(),
        backend: state.backend,
        session_id: state.sessionId,
        ...(state.interactive ? { chart_kind: "interactive" } : {}),
      },
    });
    jobId = resp.job_id;
  } catch (e) {
    toast(`Failed to start job: ${e.message}`, "err");
    return;
  }

  const card = jobCard(jobId, "analysis", state.backend, question.trim());
  el("thread").appendChild(card);
  el("thread").scrollTop = el("thread").scrollHeight;

  const ui = { type: "analysis", seen: [], current: null, card, started: Date.now() };
  state.jobs.set(jobId, ui);
  state.currentAssistantId = jobId;
  state.pipelineJobId = jobId;

  renderPipelinePanel(ui);
  pollJob(jobId);
}

function pollJob(jobId) {
  const ui = state.jobs.get(jobId);
  if (!ui) return;
  const url = `/api/analysis/${jobId}`;
  const iv = setInterval(async () => {
    let job;
    try { job = await api(url); }
    catch (_) { return; }

    const p = job.progress || {};
    if (p.node && p.node !== ui.current) {
      if (ui.current) ui.seen.push(ui.current);
      ui.current = p.node;
    }
    applyStepStates(ui, p);

    const now = Date.now();
    ui.elapsed = Math.round((now - ui.started) / 1000);
    if (ui.elapsed >= 60 && !ui.noteShown) {
      ui.noteShown = true;
      const nb = document.createElement("div");
      nb.className = "long-note";
      nb.textContent = "Still working — agent pipelines make several model calls, so a run can take a minute or two.";
      ui.card.querySelector(".card-body").appendChild(nb);
    }

    const chipElapsed = document.querySelector(`[data-job="${jobId}"] #chip-elapsed`);
    if (chipElapsed) chipElapsed.textContent = `${fmtDur(ui.elapsed)}${job.queue_position ? ` · queued #${job.queue_position + 1}` : ""}`;

    const chipTime = document.querySelector(`[data-job="${jobId}"] #chip-time`);
    if (chipTime) chipTime.textContent = `running · ${p.label || p.node || "starting"}`;

    const pill = el("pipeline-status");
    if (pill && state.pipelineJobId === jobId) pill.textContent = `running · ${fmtDur(ui.elapsed)}`;

    if (job.status === "done" || job.status === "error") {
      stopPolling(jobId);
      const finished = Date.now();
      ui.elapsed = Math.round((finished - ui.started) / 1000);
      ui.card.querySelector("#inner-skeleton")?.remove();
      if (job.status === "error") {
        ui.seen = STEP_LABELS[ui.type].map((s) => s.key);
        applyStepStates(ui, null, true);
        MarkError();
        renderErrorCard(ui.card, job, ui);
      } else {
        applyStepStates(ui, null, true);
        renderAnalysisResult(ui.card, job, ui);
        ui.card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }
  }, 1200);
  state.pollers.set(jobId, iv);
}

function MarkError() {
  document.querySelectorAll("#stepper .step").forEach((s) => s.classList.add("err"));
}

/* --------------------------- answer copy --------------------------- */
document.addEventListener("click", (e) => {
  if (e.target.classList?.contains("copy-btn")) {
    navigator.clipboard?.writeText(e.target.dataset.copy || "");
    e.target.textContent = "copied ✓";
    setTimeout(() => (e.target.textContent = "copy"), 1200);
  }
});

/* --------------------------- composer ------------------------------ */
function autosize() {
  const t = el("question");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 140) + "px";
}

el("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const t = el("question");
  const q = t.value.trim();
  if (!q) return;
  t.value = "";
  autosize();
  el("send-btn").disabled = true;
  try { await sendQuestion(q); }
  finally { el("send-btn").disabled = false; }
});

el("question").addEventListener("input", autosize);
el("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); el("ask-form").requestSubmit(); }
});
el("interactive-toggle").addEventListener("change", (e) => {
  state.interactive = e.target.checked;
});

el("new-session-btn").addEventListener("click", () => {
  state.sessionId = "sess_" + Math.random().toString(36).slice(2, 10);
  const ind = el("session-indicator");
  if (ind) {
    ind.textContent = "memory on";
    ind.style.animation = "none";
    ind.offsetHeight; // reflow
    ind.style.animation = "flash 0.6s ease";
  }
  toast("New session started — previous context cleared.", "ok");
});

el("suggest-chips").addEventListener("click", (e) => {
  const b = e.target.closest(".chips button");
  if (!b) return;
  el("question").value = b.textContent;
  el("ask-form").requestSubmit();
});

/* --------------------------- backend UI ---------------------------- */
el("backend-switch").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  state.backend = b.dataset.backend;
  el("backend-switch").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  refreshBackendName();
});

el("dataset-select").addEventListener("change", (e) => {
  state.datasetId = e.target.value;
  updateComposerHint();
});

/* --------------------------- collapsible cards --------------------- */
document.addEventListener("click", (e) => {
  const toggle = e.target.closest("[data-toggle]");
  if (!toggle) return;
  const cardId = toggle.dataset.toggle;
  const card = el(cardId);
  if (card) card.classList.toggle("collapsed");
});

/* ------------------------------ upload ----------------------------- */
async function uploadFile(file) {
  if (!file) return null;
  if (!file.name.toLowerCase().endsWith(".csv")) {
    toast("Please choose a .csv file.", "err");
    return null;
  }
  if (file.size > 60 * 1024 * 1024) {
    toast("File is too large (max 60 MB).", "err");
    return null;
  }
  const fd = new FormData();
  fd.append("file", file, file.name);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      throw new Error(d.detail || "upload failed");
    }
    return (await res.json()).dataset;
  } catch (e) {
    toast(`Upload failed: ${e.message}`, "err");
    return null;
  }
}

function selectUploaded(entry) {
  if (!entry) return;
  stopPollingAll();
  toast(`Uploaded ${entry.name} — ready to use.`, "ok");
  switchView("ask");
  setTimeout(() => {
    state.datasetId = entry.id;
    const sel = el("dataset-select");
    [...sel.options].find((o) => o.value === entry.id)?.setAttribute("selected", "");
    sel.value = entry.id;
    updateComposerHint();
  }, 60);
}

function stopPollingAll() {
  state.pollers.forEach((iv) => clearInterval(iv));
  state.pollers.clear();
}

el("upload-topbar").addEventListener("click", () => el("file-input").click());
el("file-input").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (f) {
    const entry = await uploadFile(f);
    await loadDatasets();
    selectUploaded(entry);
  }
});

const dropzone = el("dropzone");
if (dropzone) {
  dropzone.addEventListener("click", () => el("upload-input").click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
    const f = e.dataTransfer.files[0];
    if (f) {
      const entry = await uploadFile(f);
      await loadDatasets();
      selectUploaded(entry);
      if (entry) {
        el("clean-dataset").value = entry.id;
        el("clean-run").disabled = false;
        switchView("data");
      }
    }
  });
}
el("upload-input")?.addEventListener("change", async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (f) {
    const entry = await uploadFile(f);
    await loadDatasets();
    if (entry) {
      toast(`Uploaded ${entry.name}.`, "ok");
      renderCleanSelect();
      el("clean-dataset").value = entry.id;
      el("clean-run").disabled = false;
    }
  }
});

/* ----------------------------- navigation -------------------------- */
document.querySelectorAll(".nav-item").forEach((b) =>
  b.addEventListener("click", () => switchView(b.dataset.view)));

/* --------------------------- data prep view ------------------------ */
function renderDatasetGrid() {
  const grid = el("dataset-grid");
  grid.innerHTML = "";
  if (!state.datasets.length) { grid.innerHTML = `<p class="muted">No CSVs found.</p>`; return; }
  state.datasets.forEach((d) => {
    const cardEl = document.createElement("div");
    cardEl.className = "ds-card";
    cardEl.innerHTML = `
      <div class="ds-head"><span class="ds-name">${esc(d.name)}</span>
        <span class="pill ${d.folder === "Cleaning_Phase" ? "doneg" : "cyan"}">${d.folder === "Cleaning_Phase" ? "cleaned" : "raw"}</span></div>
      <div class="ds-stats"><span>${fmtNum(d.rows ?? "?")} rows</span><span>${d.columns.length} cols</span><span>${d.size_kb} KB</span></div>
      <div class="ds-cols">${d.columns.slice(0, 10).map((c) => `<span class="col-chip">${esc(c)}</span>`).join("")}${d.columns.length > 10 ? `<span class="col-chip">+${d.columns.length - 10}</span>` : ""}</div>
      <div class="ds-actions">
        ${
          d.folder === "Cleaning_Phase"
            ? `<a class="btn ghost" href="/api/datasets/download?path=${encodeURIComponent(d.id)}" title="Download ${esc(d.name)}">Download</a><button class="btn ghost" data-use="${d.id}">Use in chat</button>`
            : `<button class="btn ghost" data-clean="${d.id}">Clean this</button>`
        }`;
    grid.appendChild(cardEl);
  });
}

async function renderCleanSelect() {
  const sel = el("clean-dataset");
  const raw = state.datasets.filter((d) => d.folder === "Handling_Data");
  sel.innerHTML = "";
  if (!raw.length) {
    const o = document.createElement("option");
    o.value = "";
    o.textContent = "No raw datasets found";
    sel.appendChild(o);
    el("clean-run").disabled = true;
    return;
  }
  raw.forEach((d) => {
    const o = document.createElement("option");
    o.value = d.id;
    o.textContent = `${d.name} (${fmtNum(d.rows ?? "?")} rows)`;
    sel.appendChild(o);
  });
  el("clean-run").disabled = false;
}

el("dataset-grid").addEventListener("click", async (e) => {
  const use = e.target.closest("[data-use]");
  const clean = e.target.closest("[data-clean]");
  if (use) {
    state.datasetId = use.dataset.use;
    el("dataset-select").value = state.datasetId;
    updateComposerHint();
    switchView("ask");
    toast("Dataset selected — ask away!", "ok");
  }
  if (clean) {
    el("clean-dataset").value = clean.dataset.clean;
    el("clean-run").click();
  }
});

el("refresh-datasets").addEventListener("click", async () => {
  el("refresh-datasets").textContent = "Refreshing…";
  await loadDatasets();
  el("refresh-datasets").textContent = "Refresh";
  toast("Datasets refreshed.", "ok");
});

async function runCleaning() {
  const dsId = el("clean-dataset").value;
  if (!dsId) { toast("No dataset to clean.", "err"); return; }
  const btn = el("clean-run");
  btn.disabled = true;
  btn.textContent = "Cleaning…";
  el("clean-pill").className = "pill run";
  el("clean-pill").textContent = "running";
  el("clean-result").hidden = true;

  let jobId;
  try {
    jobId = (await api("/api/cleaning", {
      method: "POST",
      body: { dataset: dsId, backend: state.backend },
    })).job_id;
  } catch (e) {
    toast(`Failed: ${e.message}`, "err");
    btn.disabled = false; btn.textContent = "Clean & Save"; return;
  }

  const ui = { type: "cleaning", seen: [], current: null, clean: true, started: Date.now() };
  state.jobs.set(jobId, ui);
  state.pipelineJobId = jobId;
  renderPipelinePanel(ui);
  pollCleaning(jobId, btn);
}

function pollCleaning(jobId, btn) {
  const ui = state.jobs.get(jobId);
  const iv = setInterval(async () => {
    let job;
    try { job = await api(`/api/cleaning/${jobId}`); } catch (_) { return; }
    const p = job.progress || {};
    if (p.node && p.node !== ui.current) {
      if (ui.current) ui.seen.push(ui.current);
      ui.current = p.node;
    }
    applyStepStates(ui, p);

    const chip = document.getElementById("pipeline-status");
    chip.textContent = `running · ${p.label || p.node || ""}`;
    const rev = el("rev-count");
    if (rev) rev.textContent = p.revision_count ?? 0;
    ui.elapsed = Math.round((Date.now() - ui.started) / 1000);
    el("clean-pill").textContent = `running · ${p.label || p.node || "…"} · ${fmtDur(ui.elapsed)}`;

    if (job.status === "done" || job.status === "error") {
      clearInterval(iv);
      state.pollers.delete(jobId);
      ui.elapsed = Math.round((Date.now() - ui.started) / 1000);
      applyStepStates(ui, null, true);
      el("pipeline-status").className = "pill doneg";
      el("pipeline-status").textContent = job.status === "done" ? "done" : "failed";
      btn.disabled = false;
      btn.textContent = "Clean & Save";

      if (job.status === "error") {
        MarkError();
        el("pipeline-status").className = "pill errc";
        el("clean-pill").className = "pill errc";
        el("clean-pill").textContent = "failed";
        toast("Cleaning failed.", "err");
        return;
      }
      el("clean-pill").className = "pill doneg";
      el("clean-pill").textContent = "done";
      renderCleaningResult(job);
      await loadDatasets();
    }
  }, 1200);
  state.pollers.set(jobId, iv);
}

function renderCleaningResult(job) {
  const r = job.result || {};
  const box = el("clean-result");
  const loss = Number(r.lost_ratio);
  box.hidden = false;

  const changes = r.column_changes || {};
  const issueEntries = (r.detected_issues || []).slice(0, 8);
  const checks = r.checks || [];
  const warnings = r.warnings || [];

  const changesHtml = Object.keys(changes).length
    ? `<details class="clean-details"><summary>Per-column changes</summary><ul class="clean-list">${
        Object.entries(changes).map(([c, d]) => `<li><b>${esc(c)}</b> — ${esc(String(d))}</li>`).join("")
      }</ul></details>`
    : "";

  const issuesHtml = issueEntries.length
    ? `<details class="clean-details"><summary>Issues found (${issueEntries.length})</summary><ul class="clean-list">${
        issueEntries.map((i) => `<li>${esc(String(i))}</li>`).join("")
      }</ul></details>`
    : "";

  const checksHtml = checks.length || warnings.length
    ? `<details class="clean-details" open><summary>Validation</summary><ul class="clean-list">${
        [...checks.map((c) => `<li class="ok">✓ ${esc(String(c))}</li>`),
         ...warnings.map((w) => `<li class="warn">⚠ ${esc(String(w))}</li>`),
         ...(r.failures || []).map((f) => `<li class="bad">✗ ${esc(String(f))}</li>`)]
          .join("")
      }</ul></details>`
    : "";

  const planHtml = r.cleaning_plan
    ? `<details class="clean-details"><summary>Cleaning plan</summary><div class="plan-text">${esc(r.cleaning_plan)}</div></details>`
    : "";

  box.innerHTML = `
    <div class="chip-row">
      <span class="chip ok">✓ saved</span>
      <span class="chip">${esc(r.output_rel || r.output_csv || "")}</span>
      <span class="chip cyan">revisions ${r.revision_count ?? 0}</span>
      ${
        r.output_rel
          ? `<a class="btn primary dl-csv" href="/api/datasets/download?path=${encodeURIComponent(r.output_rel)}">Download CSV</a>`
          : ""
      }
    </div>
    <div class="clean-stats">
      <div class="stat ${loss > 10 ? "bad" : "good"}"><b>${fmtNum(r.rows_before ?? "?")}</b><span>rows before</span></div>
      <div class="stat ${loss > 10 ? "bad" : "good"}"><b>${fmtNum(r.rows_after ?? "?")}</b><span>rows after</span></div>
      <div class="stat ${loss > 10 ? "bad" : "good"}"><b>${loss}%</b><span>rows lost</span></div>
      <div class="stat"><b>${(r.columns || []).length}</b><span>columns</span></div>
    </div>
    <h4 class="clean-sub">What happened to your data</h4>
    ${planHtml}
    ${changesHtml}
    ${issuesHtml}
    ${checksHtml}
    ${(r.preview || []).length ? `<h4 class="clean-sub">Preview</h4><table class="gen-table"><thead><tr>${Object.keys(r.preview[0]).map((k) => `<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>${r.preview.map((row) => `<tr>${Object.values(row).map((v) => `<td>${esc(v == null ? "" : v)}</td>`).join("")}</tr>`).join("")}</tbody></table>` : ""}`;
}

el("clean-run").addEventListener("click", runCleaning);

/* ------------------------------- boot ------------------------------ */
(async function init() {
  refreshBackendName();
  el("backend-switch").querySelector(`[data-backend="${state.backend}"]`)?.classList.add("on");
  await loadDatasets();
  window.addEventListener("beforeunload", () => state.pollers.forEach((iv) => clearInterval(iv)));
})();
