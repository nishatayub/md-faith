import { barChart, card, el, heatmap, legend, lineChart, seriesColors } from "./charts.js";

// ------------------------------------------------------------------------------------------ api
async function api(path, opts = {}) {
  const r = await fetch("/api" + path, {
    headers: opts.body ? { "Content-Type": "application/json" } : undefined,
    method: opts.method || (opts.body ? "POST" : "GET"),
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* keep statusText */ }
    throw new Error(`${r.status}: ${msg}`);
  }
  return r.status === 204 ? null : r.json();
}
const memo = new Map();
const cached = (key, fn) => (memo.has(key) ? Promise.resolve(memo.get(key)) : fn().then((v) => (memo.set(key, v), v)));

const put = (node, ...kids) => node.replaceChildren(...kids.flat().filter(Boolean)); // native replaceChildren stringifies null
const view = document.getElementById("view");
const modalRoot = document.getElementById("modal-root");
const COND = ["image_only", "table_only", "tool_agent", "qc_gated"];
const COND_LABEL = { image_only: "Image only", table_only: "Table only", tool_agent: "Tool agent", qc_gated: "QC-gated agent" };
const ART_LABEL = { none: "Clean", pbc_split: "Wrapped domain", shuffle_frames: "Shuffled frames", rigid_jitter: "Unfitted frames" };
const ART_SHORT = { none: "Clean", pbc_split: "Wrapped", shuffle_frames: "Shuffled", rigid_jitter: "Unfitted" };
const STATUS = {
  supported: { icon: "✓", label: "Supported" }, contradicted: { icon: "✕", label: "Contradicted" },
  artifact: { icon: "⚠", label: "Artifact misread" }, unsupported: { icon: "?", label: "Unsupported claim" },
  unchecked: { icon: "–", label: "Not checkable" },
};
const SEV_ICON = { critical: "✕", warning: "⚠", info: "i" };
const colorOf = (c) => seriesColors()[Math.max(0, COND.indexOf(c))];

const chip = (status, n) => el("span", { class: `chip ${status}` }, el("span", { class: "ic", "aria-hidden": "true" }, STATUS[status].icon), STATUS[status].label, n != null ? ` ${n}` : "");
const pct = (v) => (v == null ? "n/a" : `${(v * 100).toFixed(0)}%`);
const when = (ts) => new Date(ts * 1000).toLocaleString();
const spinner = (msg = "Working…") => el("span", {}, el("span", { class: "spinner" }), " ", msg);
function errorBanner(e) { return el("div", { class: "banner err", role: "alert" }, el("div", {}, el("b", {}, "Something went wrong. "), String(e.message || e))); }
const demoBanner = () => el("div", { class: "banner demo", role: "note" }, el("div", {}, el("b", {}, "Demo data. "), "These results come from simulated explainers whose error rates are set by hand to exercise the pipeline. They are not findings about real language models."));

// ------------------------------------------------------------------------------------------ router
const routes = { home, lab, verify, runs, results, about };
async function route() {
  const [name, arg] = location.hash.replace(/^#\/?/, "").split("/");
  const key = name || "home";
  document.querySelectorAll("#nav a").forEach((a) => (a.dataset.r === key ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  view.replaceChildren(el("div", { class: "empty" }, spinner("Loading…")));
  try {
    const node = await (routes[key] || home)(arg);
    view.replaceChildren(node);
    window.scrollTo(0, 0);
  } catch (e) { view.replaceChildren(errorBanner(e)); }
}
window.addEventListener("hashchange", route);

// theme toggle (persisted); charts re-render because they read CSS variables at draw time
const root = document.documentElement;
if (localStorage.theme) root.dataset.theme = localStorage.theme;
document.getElementById("theme").addEventListener("click", () => {
  const dark = (root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark";
  root.dataset.theme = dark ? "light" : "dark"; localStorage.theme = root.dataset.theme; route();
});

// ------------------------------------------------------------------------------------------ views
async function home() {
  const [health, meta] = await Promise.all([api("/health"), api("/meta")]);
  const seed = el("button", { class: "btn", type: "button" }, "Load demo data");
  seed.addEventListener("click", async () => {
    seed.disabled = true; seed.replaceChildren(spinner("Seeding demo run…"));
    try { const run = await api("/demo/seed", { body: { seeds: 8 } }); localStorage.lastRun = run.id; location.hash = `#/results/${run.id}`; }
    catch (e) { seed.disabled = false; seed.textContent = "Load demo data"; view.prepend(errorBanner(e)); }
  });
  return el("div", {},
    el("section", { class: "hero" },
      el("h1", {}, "Fact-check AI explanations of molecular dynamics."),
      el("p", {}, "MD-Faith computes ground truth from a trajectory, plants known processing problems, and checks every claim an AI makes against the numbers. Paste an explanation to see which sentences hold up, which don't, and why."),
      el("div", { class: "row" },
        el("a", { class: "btn", href: "#/verify", style: "text-decoration:none" }, "Fact-check an explanation"),
        el("a", { class: "btn btn-ghost", href: "#/lab", style: "text-decoration:none" }, "Explore trajectories"),
        health.demo_present ? el("a", { class: "btn btn-ghost", href: "#/results", style: "text-decoration:none" }, "See benchmark results") : seed)),
    el("div", { class: "grid cols-4", style: "margin:18px 0" },
      stat(health.n_runs, "stored runs"), stat(meta.conditions.length, "conditions compared"),
      stat(meta.artifacts.length - 1, "planted artifacts"), stat(meta.claim_kinds.length, "claim types verified")),
    el("div", { class: "grid cols-2" },
      el("section", { class: "card" }, el("h2", {}, "How it works"),
        el("ol", { class: "steps" },
          el("li", {}, el("div", {}, el("b", {}, "Compute ground truth."), " RMSD, RMSF, radius of gyration, contacts and hydrogen bonds, cross-checked against MDAnalysis.")),
          el("li", {}, el("div", {}, el("b", {}, "Plant traps."), " Periodic-boundary wrapping, shuffled frames and unfitted frames, each with a record of the affected frames.")),
          el("li", {}, el("div", {}, el("b", {}, "Split into claims."), " Numeric, ranking, temporal and causal statements are extracted from an explanation.")),
          el("li", {}, el("div", {}, el("b", {}, "Verify with code."), " Every claim is checked against computed values. No LLM judge decides what is true.")))),
      el("section", { class: "card" }, el("h2", {}, "What the verdicts mean"),
        el("div", { class: "grid", style: "gap:10px" },
          verdictRow("supported", "Matches the computed value within tolerance."),
          verdictRow("contradicted", "Checkable and wrong; the real value is shown."),
          verdictRow("artifact", "Presents a processing artifact (e.g. wrapped frames) as physical motion."),
          verdictRow("unsupported", "A mechanism or cause stated without cited computed evidence.")))));
}
const stat = (v, l) => el("div", { class: "card stat" }, el("div", { class: "v" }, v), el("div", { class: "l" }, l));
const verdictRow = (s, t) => el("div", { class: "row" }, chip(s), el("span", { class: "muted" }, t));

async function lab() {
  const tasks = await cached("tasks", () => api("/tasks"));
  const sel = el("select", { id: "task", "aria-label": "Trajectory variant" }, tasks.map((t) => el("option", { value: t.task_id }, `${t.system} · ${ART_LABEL[t.artifact]}`)));
  sel.value = localStorage.labTask && tasks.some((t) => t.task_id === localStorage.labTask) ? localStorage.labTask : "adk_dims:none";
  const out = el("div", {});
  const load = async () => {
    localStorage.labTask = sel.value;
    out.replaceChildren(el("div", { class: "empty" }, spinner("Computing ground truth for this trajectory…")));
    try { out.replaceChildren(labBody(await cached("t:" + sel.value, () => api(`/tasks/${sel.value}`)), tasks.find((t) => t.task_id === sel.value))); }
    catch (e) { out.replaceChildren(errorBanner(e)); }
  };
  sel.addEventListener("change", load);
  load();
  return el("div", {}, el("h1", {}, "Trajectory lab"),
    el("p", { class: "muted" }, "Ground truth and automated QC for the bundled AdK trajectory and its planted-artifact variants. Switch variants to see how each problem changes the numbers and what QC catches."),
    el("div", { class: "card", style: "margin-bottom:16px" }, el("label", { for: "task" }, "Trajectory variant"), sel), out);
}
function labBody(d, task) {
  const bands = d.artifact.frames.length && d.artifact.kind === "pbc_split" ? [{ x0: Math.min(...d.artifact.frames), x1: Math.max(...d.artifact.frames), label: "wrapped", note: "planted artifact: domain wrapped across the boundary" }] : [];
  const frames = d.rmsd.map((_, i) => i);
  const [c1] = seriesColors();
  const qc = d.qc;
  return el("div", { class: "grid", style: "gap:16px" },
    el("div", { class: "split" },
      el("section", { class: "card" },
        el("div", { class: "row between" }, el("h3", {}, "Automated trajectory QC"), el("span", { class: `chip ${qc.passed ? "supported" : "contradicted"}` }, el("span", { class: "ic" }, qc.passed ? "✓" : "✕"), qc.passed ? "Passed" : "Problems found")),
        el("p", { class: "muted" }, task.description),
        qc.flags.length ? qc.flags.map((f) => el("div", { class: "flag" }, el("span", { class: `sev ${f.severity}` }, SEV_ICON[f.severity] + " " + f.severity),
          el("div", {}, el("b", { class: "mono" }, f.code), el("div", {}, f.message), f.frames.length ? el("div", { class: "muted" }, `frames: ${f.frames.slice(0, 12).join(", ")}${f.frames.length > 12 ? ", …" : ""}`) : null))) : el("p", {}, "No problems detected.")),
      el("section", { class: "card" }, el("h3", {}, "Computed quantities"),
        el("div", { class: "scroll" }, el("table", { class: "table" }, el("thead", {}, el("tr", {}, el("th", {}, "Quantity"), el("th", {}, "Value"))),
          el("tbody", {}, Object.entries(d.scalars).map(([k, v]) => el("tr", {}, el("td", { class: "mono" }, k), el("td", {}, typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : v)))))))),
    card({ title: "Backbone RMSD to frame 0", subtitle: bands.length ? "Shaded frames are the planted wrapping artifact, not physical motion." : "Superposed on frame 0.",
      tableHead: ["Frame", "RMSD (Å)"], tableRows: d.rmsd.map((v, i) => [i, v.toFixed(2)]),
      render: (h) => lineChart(h, { series: [{ name: "RMSD", color: c1, values: frames.map((x, i) => ({ x, y: d.rmsd[i] })) }], xLabel: "frame", yLabel: "RMSD (Å)", bands, yZero: true }) }),
    el("div", { class: "grid cols-2" },
      card({ title: "CA RMSF per residue", tableHead: ["Residue", "RMSF (Å)"], tableRows: d.resids.map((r, i) => [r, d.rmsf[i].toFixed(2)]),
        render: (h) => lineChart(h, { series: [{ name: "RMSF", color: c1, values: d.resids.map((x, i) => ({ x, y: d.rmsf[i] })) }], xLabel: "residue", yLabel: "RMSF (Å)", yZero: true, height: 210 }) }),
      card({ title: "Hydrogen bonds per frame", subtitle: "Donor–acceptor < 3.0 Å, angle > 150°.", tableHead: ["Frame", "H-bonds"], tableRows: d.hbond_counts.map((v, i) => [i, v]),
        render: (h) => lineChart(h, { series: [{ name: "H-bonds", color: seriesColors()[2], values: frames.map((x, i) => ({ x, y: d.hbond_counts[i] })) }], xLabel: "frame", yLabel: "count", bands, height: 210 }) })));
}

// ----- sentence-level annotated report (used by fact-check and the run explorer)
function reportView(res) {
  const s = res.summary;
  const chips = ["supported", "contradicted", "artifact", "unsupported"].filter((k) => s[k]).map((k) => chip(k, s[k]));
  return el("div", {},
    el("div", { class: "row", style: "margin-bottom:6px" }, chips.length ? chips : el("span", { class: "muted" }, "No checkable claims found in this text."),
      s.unchecked ? el("span", { class: "muted" }, `${s.unchecked} sentence(s) without a checkable claim`) : null),
    res.sentences.map((x) => sentence(x)));
}
function sentence(x) {
  const ev = el("div", { class: "evidence", hidden: true });
  const toggle = el("button", { class: "toggle", type: "button", "aria-expanded": "false" }, "Evidence");
  toggle.addEventListener("click", () => { ev.hidden = !ev.hidden; toggle.setAttribute("aria-expanded", String(!ev.hidden)); toggle.textContent = ev.hidden ? "Evidence" : "Hide"; });
  for (const e of x.evidence) {
    ev.append(el("div", {}, el("b", {}, e.kind + " claim · " + e.label.replace("_", " "))),
      el("dl", {}, el("dt", {}, "Verifier"), el("dd", {}, e.detail),
        e.computed != null ? [el("dt", {}, "Computed"), el("dd", {}, JSON.stringify(e.computed))] : null,
        e.tool ? [el("dt", {}, "Reproduce with"), el("dd", { class: "mono" }, e.tool)] : null,
        e.definition ? [el("dt", {}, "Definition"), el("dd", {}, e.definition)] : null));
  }
  return el("div", { class: `sentence ${x.status}` }, el("div", { class: "head" },
    el("div", {}, el("div", { style: "margin-bottom:4px" }, chip(x.status)), x.text), x.evidence.length ? toggle : null), x.evidence.length ? ev : null);
}

async function verify() {
  const [tasks] = await Promise.all([cached("tasks", () => api("/tasks"))]);
  const sel = el("select", { id: "vtask" }, tasks.map((t) => el("option", { value: t.task_id }, `${t.system} · ${ART_LABEL[t.artifact]}`)));
  const ta = el("textarea", { id: "text", placeholder: "Paste an AI-written explanation of the trajectory here…", maxlength: "20000" });
  const out = el("div", {}, el("div", { class: "empty" }, "Results appear here."));
  const samples = el("div", { class: "row" });
  const loadSamples = async () => {
    samples.replaceChildren();
    try { (await api(`/tasks/${sel.value}/samples`)).forEach((s) => samples.append(el("button", { class: "btn btn-ghost btn-sm", type: "button", onclick: () => { ta.value = s.text; run(); } }, s.label))); } catch { /* samples are optional */ }
  };
  const go = el("button", { class: "btn", type: "button" }, "Fact-check");
  async function run() {
    if (!ta.value.trim()) { out.replaceChildren(el("div", { class: "empty" }, "Paste some text first, or load a sample.")); return; }
    go.disabled = true; out.replaceChildren(el("div", { class: "empty" }, spinner("Checking claims against ground truth…")));
    try {
      const res = await api("/verify", { body: { task_id: sel.value, text: ta.value } });
      const crit = res.qc.flags.filter((f) => f.severity === "critical");
      put(
        out,
        crit.length ? el("div", { class: "banner", role: "note" }, el("div", {}, el("b", {}, "Trajectory QC: "), crit.map((f) => f.code).join(", "), ". Numbers from the affected frames are not physical.")) : null,
        res.n_claims === 0 ? el("p", { class: "muted" }, "The rule-based extractor understands a controlled set of sentence patterns (see Methods). Free-form text needs the LLM extractor.") : null,
        reportView(res));
    } catch (e) { out.replaceChildren(errorBanner(e)); } finally { go.disabled = false; }
  }
  go.addEventListener("click", run); sel.addEventListener("change", loadSamples); loadSamples();
  return el("div", {}, el("h1", {}, "Fact-check an explanation"),
    el("p", { class: "muted" }, "Each sentence is tagged with a verdict and the computed evidence. The checker extracts claims with a deterministic rule-based parser for a documented set of sentence patterns; try a sample to see the format."),
    el("div", { class: "split" },
      el("section", { class: "card" }, el("label", { for: "vtask" }, "Trajectory the explanation is about"), sel,
        el("label", { for: "text", style: "margin-top:12px" }, "Explanation"), ta,
        el("div", { class: "row between", style: "margin-top:10px" }, go, el("span", { class: "muted" }, "Samples:")), samples),
      el("section", { class: "card" }, el("h3", {}, "Verdicts"), out)));
}

async function runs() {
  const [list, meta] = await Promise.all([api("/runs"), cached("meta", () => api("/meta"))]);
  const seed = el("button", { class: "btn btn-ghost", type: "button" }, "Seed demo run");
  seed.addEventListener("click", async () => { seed.disabled = true; seed.replaceChildren(spinner("Seeding…")); try { const r = await api("/demo/seed", { body: { seeds: 8 } }); localStorage.lastRun = r.id; location.hash = `#/results/${r.id}`; } catch (e) { view.prepend(errorBanner(e)); seed.disabled = false; seed.textContent = "Seed demo run"; } });
  const form = el("form", { class: "card" }, el("h3", {}, "New simulated run"),
    el("p", { class: "muted" }, "Runs the full grid with simulated explainers. Real-model runs are started from the CLI (mdfaith run) so that API credit is only spent deliberately."),
    el("div", { class: "grid cols-2" },
      el("fieldset", { class: "fieldset" }, el("legend", {}, "Simulated explainers"), meta.simulated_models.map((m) => el("label", { class: "check", title: m.description }, el("input", { type: "checkbox", name: "model", value: m.name, checked: true }), m.name))),
      el("fieldset", { class: "fieldset" }, el("legend", {}, "Conditions"), meta.conditions.map((c) => el("label", { class: "check", title: c.description }, el("input", { type: "checkbox", name: "cond", value: c.name, checked: true }), COND_LABEL[c.name])))),
    el("div", { class: "grid cols-2" },
      el("fieldset", { class: "fieldset" }, el("legend", {}, "Trajectory variants"), meta.artifacts.map((a) => el("label", { class: "check", title: a.description }, el("input", { type: "checkbox", name: "art", value: a.kind, checked: true }), ART_LABEL[a.kind]))),
      el("div", {}, el("label", { for: "seeds" }, "Seeds (replicates)"), el("input", { type: "number", id: "seeds", min: 1, max: 20, value: 4 }), el("label", { for: "rname", style: "margin-top:10px" }, "Name"), el("input", { type: "text", id: "rname", value: "web run", maxlength: 120 }))),
    el("div", { class: "row end" }, el("button", { class: "btn", type: "submit" }, "Run")));
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const pick = (n) => [...form.querySelectorAll(`input[name=${n}]:checked`)].map((i) => i.value);
    const body = { name: form.querySelector("#rname").value, models: pick("model"), conditions: pick("cond"), artifacts: pick("art"), seeds: +form.querySelector("#seeds").value };
    if (!body.models.length || !body.conditions.length || !body.artifacts.length) return view.prepend(errorBanner(new Error("Pick at least one explainer, condition and variant.")));
    const btn = form.querySelector("button[type=submit]"); btn.disabled = true; btn.replaceChildren(spinner("Running…"));
    try { const r = await api("/runs", { body }); localStorage.lastRun = r.id; location.hash = `#/results/${r.id}`; } catch (err) { view.prepend(errorBanner(err)); btn.disabled = false; btn.textContent = "Run"; }
  });
  const table = list.length ? el("div", { class: "scroll" }, el("table", { class: "table" },
    el("thead", {}, el("tr", {}, ["Run", "Created", "Explanations", "Status", ""].map((h) => el("th", {}, h)))),
    el("tbody", {}, list.map((r) => el("tr", { class: "click", onclick: () => { localStorage.lastRun = r.id; location.hash = `#/results/${r.id}`; } },
      el("td", {}, r.name, " ", r.is_demo ? el("span", { class: "chip pill-demo" }, "demo") : null), el("td", {}, when(r.created_at)), el("td", {}, r.n_explanations), el("td", {}, r.status),
      el("td", {}, el("button", { class: "btn btn-danger btn-sm", type: "button", onclick: async (ev) => { ev.stopPropagation(); if (confirm("Delete this run?")) { await api(`/runs/${r.id}`, { method: "DELETE" }); if (localStorage.lastRun === r.id) delete localStorage.lastRun; route(); } } }, "Delete"))))))) : el("div", { class: "empty" }, "No runs yet. Seed the demo run or start a simulated one below.");
  return el("div", {}, el("div", { class: "row between" }, el("h1", {}, "Runs"), seed), el("section", { class: "card", style: "margin-bottom:16px" }, table), form);
}

async function results(runId) {
  const list = await api("/runs");
  if (!list.length) return el("div", { class: "empty card" }, el("p", {}, "No runs yet."), el("a", { class: "btn", href: "#/runs", style: "text-decoration:none" }, "Go to Runs to seed the demo"));
  const id = runId || (list.some((r) => r.id === localStorage.lastRun) ? localStorage.lastRun : list[0].id);
  const run = list.find((r) => r.id === id) || list[0];
  localStorage.lastRun = run.id;
  const [byCond, byCondArt, byModelCond] = await Promise.all([
    api(`/runs/${run.id}/summary?by=condition`), api(`/runs/${run.id}/summary?by=condition,artifact_kind`), api(`/runs/${run.id}/summary?by=model,condition`)]);
  const conds = COND.filter((c) => byCond.rows.some((r) => r.condition === c));
  const arts = Object.keys(ART_LABEL).filter((a) => byCondArt.rows.some((r) => r.artifact_kind === a));
  const models = [...new Set(byModelCond.rows.map((r) => r.model))];
  const find = (rows, f) => rows.find(f);

  const picker = el("select", { "aria-label": "Run" }, list.map((r) => el("option", { value: r.id }, `${r.name} · ${when(r.created_at)}`)));
  picker.value = run.id; picker.addEventListener("change", () => (location.hash = `#/results/${picker.value}`));

  const c1 = card({ title: "Hallucination rate by condition", subtitle: "Share of checkable claims that were wrong or misread an artifact. Whiskers: 95% bootstrap interval over explanations.",
    tableHead: ["Condition", "Rate", "95% CI", "Claims"], tableRows: conds.map((c) => { const r = find(byCond.rows, (x) => x.condition === c); return [COND_LABEL[c], r.hallucination_rate.toFixed(3), `${r.ci_lo.toFixed(3)} – ${r.ci_hi.toFixed(3)}`, r.n_claims]; }),
    render: (h) => barChart(h, { items: conds.map((c) => { const r = find(byCond.rows, (x) => x.condition === c); return { label: COND_LABEL[c], value: r.hallucination_rate, lo: r.ci_lo, hi: r.ci_hi, color: colorOf(c), detail: [`${r.n_explanations} explanations, ${r.n_claims} claims`, `support rate ${pct(r.support_rate)}`, `95% CI ${r.ci_lo.toFixed(2)}–${r.ci_hi.toFixed(2)}`] }; }), valueLabel: "hallucination rate" }) });

  const c2 = card({ title: "Hallucination rate by condition and trajectory problem", subtitle: "Darker is worse. Pooled across explainers.",
    tableHead: ["Condition", ...arts.map((a) => ART_LABEL[a])], tableRows: conds.map((c) => [COND_LABEL[c], ...arts.map((a) => { const r = find(byCondArt.rows, (x) => x.condition === c && x.artifact_kind === a); return r ? r.hallucination_rate.toFixed(2) : "n/a"; })]),
    render: (h) => heatmap(h, { rows: conds.map((c) => COND_LABEL[c]), cols: arts.map((a) => ART_SHORT[a]),
      values: conds.map((c) => arts.map((a) => find(byCondArt.rows, (x) => x.condition === c && x.artifact_kind === a)?.hallucination_rate ?? null)),
      detail: (i, j) => { const r = find(byCondArt.rows, (x) => x.condition === conds[i] && x.artifact_kind === arts[j]); return r ? [`${r.n_explanations} explanations, ${r.n_claims} claims`] : []; } }) });

  const c3 = card({ title: "Hallucination rate by explainer and condition", tableHead: ["Explainer", ...conds.map((c) => COND_LABEL[c])],
    tableRows: models.map((m) => [m, ...conds.map((c) => { const r = find(byModelCond.rows, (x) => x.model === m && x.condition === c); return r ? r.hallucination_rate.toFixed(2) : "n/a"; })]),
    render: (h) => heatmap(h, { rows: models, cols: conds.map((c) => COND_LABEL[c]), values: models.map((m) => conds.map((c) => find(byModelCond.rows, (x) => x.model === m && x.condition === c)?.hallucination_rate ?? null)),
      detail: (i, j) => { const r = find(byModelCond.rows, (x) => x.model === models[i] && x.condition === conds[j]); return r ? [`${r.n_explanations} explanations`, `unsupported mechanisms ${pct(r.unsupported_mechanism_rate)}`] : []; } }) });

  const artRows = conds.map((c) => ({ c, r: find(byCond.rows, (x) => x.condition === c) })).filter((x) => x.r.artifact_misread_share != null);
  const c4 = card({ title: "Explanations that misread a planted artifact", subtitle: "Share of explanations on artifact variants with at least one artifact-misread claim.",
    tableHead: ["Condition", "Share"], tableRows: artRows.map((x) => [COND_LABEL[x.c], pct(x.r.artifact_misread_share)]),
    render: (h) => barChart(h, { items: artRows.map((x) => ({ label: COND_LABEL[x.c], value: x.r.artifact_misread_share, color: colorOf(x.c), detail: [`${x.r.n_explanations} explanations`] })) }) });

  const conditionLegend = legend(conds.map((c) => ({ name: COND_LABEL[c], color: colorOf(c) })));
  const explorer = explorerView(run, models);
  return el("div", {}, run.is_demo ? demoBanner() : null,
    el("div", { class: "row between" }, el("h1", {}, "Results"), picker),
    el("p", { class: "muted" }, `${run.name} · ${run.n_explanations} explanations · ${run.status}`), conditionLegend,
    el("div", { class: "grid", style: "gap:16px;margin-top:10px" }, c1, el("div", { class: "grid cols-2" }, c2, c4), c3, explorer));
}

function explorerView(run, models) {
  const f = { condition: "", model: "", artifact: "" };
  const body = el("div", {});
  const sel = (key, opts, label) => { const s = el("select", { "aria-label": label }, el("option", { value: "" }, `All ${label}`), opts.map(([v, t]) => el("option", { value: v }, t))); s.addEventListener("change", () => { f[key] = s.value; load(); }); return s; };
  const load = async () => {
    body.replaceChildren(el("div", { class: "empty" }, spinner("Loading…")));
    const q = new URLSearchParams({ limit: 25, ...Object.fromEntries(Object.entries(f).filter(([, v]) => v)) });
    try {
      const d = await api(`/runs/${run.id}/explanations?${q}`);
      body.replaceChildren(el("p", { class: "muted" }, `${d.total} explanations${d.total > 25 ? " (showing the first 25)" : ""}. Click one to see its evidence-linked report.`),
        el("div", { class: "scroll" }, el("table", { class: "table" }, el("thead", {}, el("tr", {}, ["#", "Explainer", "Condition", "Variant", "Claims", "Wrong"].map((h) => el("th", {}, h)))),
          el("tbody", {}, d.items.map((r) => el("tr", { class: "click", onclick: () => openReport(r.explanation_id) }, el("td", {}, r.explanation_id), el("td", {}, r.model), el("td", {}, COND_LABEL[r.condition]), el("td", {}, ART_LABEL[r.artifact_kind]), el("td", {}, r.n), el("td", {}, r.wrong)))))));
    } catch (e) { body.replaceChildren(errorBanner(e)); }
  };
  load();
  return el("section", { class: "card" }, el("h3", {}, "Explanation explorer"),
    el("div", { class: "row", style: "margin:8px 0" }, sel("condition", COND.map((c) => [c, COND_LABEL[c]]), "conditions"), sel("model", models.map((m) => [m, m]), "explainers"), sel("artifact", Object.entries(ART_LABEL), "variants")), body);
}
async function openReport(id) {
  const close = () => modalRoot.replaceChildren();
  modalRoot.replaceChildren(el("div", { class: "modal-bg", onclick: (e) => e.target.classList.contains("modal-bg") && close() }, el("div", { class: "card modal", role: "dialog", "aria-modal": "true" }, el("div", { class: "empty" }, spinner("Loading report…")))));
  try {
    const d = await api(`/explanations/${id}`);
    const m = modalRoot.querySelector(".modal");
    put(m, el("div", { class: "row between" }, el("h2", {}, `Explanation #${id}`), el("button", { class: "btn btn-ghost btn-sm", type: "button", onclick: close }, "Close")),
      el("p", { class: "muted" }, `${d.explanation.model} · ${COND_LABEL[d.explanation.condition]} · ${ART_LABEL[d.explanation.artifact_kind]} · seed ${d.explanation.seed}`),
      d.is_demo ? demoBanner() : null, reportView(d));
  } catch (e) { modalRoot.querySelector(".modal").replaceChildren(errorBanner(e)); }
}
document.addEventListener("keydown", (e) => e.key === "Escape" && modalRoot.replaceChildren());

async function about() {
  const meta = await cached("meta", () => api("/meta"));
  return el("div", {}, el("h1", {}, "Methods"),
    el("div", { class: "grid cols-2" },
      el("section", { class: "card" }, el("h3", {}, "Question"), el("p", {}, "When an LLM explains an MD analysis, which claims are faithful to quantities computed from the trajectory, and does giving the model tools instead of a picture or a table reduce unfaithful claims, including when the trajectory has a processing problem?")),
      el("section", { class: "card" }, el("h3", {}, "Conditions"), el("dl", {}, meta.conditions.map((c) => [el("dt", { style: "font-weight:700" }, COND_LABEL[c.name]), el("dd", { style: "margin:0 0 8px" }, c.description)])))),
    el("div", { class: "grid cols-2", style: "margin-top:16px" },
      el("section", { class: "card" }, el("h3", {}, "Planted artifacts"), el("dl", {}, meta.artifacts.map((a) => [el("dt", { style: "font-weight:700" }, ART_LABEL[a.kind]), el("dd", { style: "margin:0 0 8px" }, a.description)]))),
      el("section", { class: "card" }, el("h3", {}, "Rule-based extractor grammar"), el("p", { class: "muted" }, "The offline extractor understands only these sentence patterns. Real model output needs the LLM extractor."), el("ul", {}, meta.extractor_grammar.map((g) => el("li", {}, g))))),
    el("section", { class: "card", style: "margin-top:16px" }, el("h3", {}, "Limits you should know"),
      el("ul", {}, [
        "Demo results use simulated explainers; their error rates are set by hand and are not findings.",
        "One public system (AdK) so far. Conclusions about other systems need more systems.",
        "Artifacts are synthetic. Tolerances (10% numeric, 5-frame window, 60% residue overlap) are defaults and need a sensitivity analysis.",
        "Claim extraction by an LLM must be validated by hand on a random subset before any result is reported.",
        "Causal statements are not checkable from a trajectory alone; they are reported separately as unsupported mechanisms."].map((t) => el("li", {}, t)))));
}

route();
