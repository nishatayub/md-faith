// Small dependency-free SVG chart helpers for the MD-Faith app.
// Rules followed: one axis per chart, thin marks, legend for >= 2 series plus direct end labels, hover tooltips,
// text in ink tokens (never series colour), a table view for every chart.

const NS = "http://www.w3.org/2000/svg";
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) n.append(c instanceof Node ? c : document.createTextNode(String(c ?? "")));
  return n;
}
function svg(tag, attrs = {}, ...children) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  for (const c of children.flat()) if (c) n.append(c);
  return n;
}
const txt = (x, y, s, attrs = {}) => {
  const t = svg("text", { x, y, ...attrs });
  t.textContent = s;
  return t;
};

function niceTicks(lo, hi, n = 5) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const start = Math.ceil(lo / step) * step, out = [];
  for (let v = start; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
const fmt = (v, d = 2) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(d));

function tooltipFor(host) {
  let tip = host.querySelector(".viz-tip");
  if (!tip) { tip = el("div", { class: "viz-tip", role: "tooltip" }); host.append(tip); }
  return {
    show(html, px, py) {
      tip.replaceChildren(...html);
      tip.style.display = "block";
      const w = host.clientWidth, tw = tip.offsetWidth;
      tip.style.left = Math.max(4, Math.min(w - tw - 4, px + 12)) + "px";
      tip.style.top = Math.max(0, py - tip.offsetHeight - 8) + "px";
    },
    hide() { tip.style.display = "none"; },
  };
}

export function legend(items) {
  return el("div", { class: "viz-legend" }, items.map((i) =>
    el("span", { class: "viz-legend-item" }, el("i", { style: `background:${i.color}` }), i.name)));
}

// Wrap a chart with a title, optional legend and a "table view" toggle that shows the same numbers.
export function card({ title, subtitle, legendItems, render, tableRows, tableHead }) {
  const body = el("div", { class: "viz-body" });
  const tableWrap = el("div", { class: "viz-table", hidden: true });
  const btn = el("button", { class: "btn btn-ghost btn-sm", type: "button", "aria-pressed": "false" }, "Table view");
  btn.addEventListener("click", () => {
    const on = tableWrap.hidden;
    tableWrap.hidden = !on; body.hidden = on;
    btn.setAttribute("aria-pressed", String(on)); btn.textContent = on ? "Chart view" : "Table view";
  });
  if (tableRows) {
    tableWrap.append(el("table", { class: "table" },
      el("thead", {}, el("tr", {}, tableHead.map((h) => el("th", {}, h)))),
      el("tbody", {}, tableRows.map((r) => el("tr", {}, r.map((c) => el("td", {}, c)))))));
  }
  const root = el("section", { class: "card viz-card" },
    el("header", { class: "viz-head" }, el("div", {}, el("h3", {}, title), subtitle ? el("p", { class: "muted" }, subtitle) : null), tableRows ? btn : null),
    legendItems && legendItems.length > 1 ? legend(legendItems) : null, body, tableWrap);
  const draw = () => { body.replaceChildren(); render(body); };
  new ResizeObserver(() => { if (!body.hidden) draw(); }).observe(root);
  draw();
  root.redraw = draw;
  return root;
}

// ---------------------------------------------------------------------------------------------- line chart
export function lineChart(host, { series, xLabel, yLabel, bands = [], height = 230, yZero = false, xIsCategory = false }) {
  const W = Math.max(280, host.clientWidth || 600), H = height;
  const m = { l: 52, r: 78, t: 14, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xs = series.flatMap((s) => s.values.map((p) => p.x)), ys = series.flatMap((s) => s.values.map((p) => p.y));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = yZero ? Math.min(0, ...ys) : Math.min(...ys), y1 = Math.max(...ys);
  const pad = (y1 - y0 || 1) * 0.08; y1 += pad; if (!yZero) y0 -= pad;
  const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * iw, Y = (v) => m.t + ih - ((v - y0) / (y1 - y0 || 1)) * ih;
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": `${yLabel} by ${xLabel}` });
  const grid = css("--grid"), axis = css("--axis"), muted = css("--muted");
  for (const t of niceTicks(y0, y1, 4)) {
    root.append(svg("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: grid, "stroke-width": 1 }));
    root.append(txt(m.l - 8, Y(t) + 4, fmt(t), { fill: muted, "text-anchor": "end", "font-size": 11 }));
  }
  for (const t of niceTicks(x0, x1, 6)) root.append(txt(X(t), H - m.b + 16, fmt(t, 0), { fill: muted, "text-anchor": "middle", "font-size": 11 }));
  root.append(svg("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, stroke: axis, "stroke-width": 1 }));
  root.append(txt(m.l + iw / 2, H - 6, xLabel, { fill: css("--text-2"), "text-anchor": "middle", "font-size": 12 }));
  const yl = txt(14, m.t + ih / 2, yLabel, { fill: css("--text-2"), "text-anchor": "middle", "font-size": 12 });
  yl.setAttribute("transform", `rotate(-90 14 ${m.t + ih / 2})`); root.append(yl);
  for (const b of bands) {
    const bx0 = X(b.x0 - 0.5), bx1 = X(b.x1 + 0.5);
    root.append(svg("rect", { x: bx0, y: m.t, width: Math.max(2, bx1 - bx0), height: ih, fill: css("--status-serious"), opacity: 0.16 }));
    if (b.label) root.append(txt(bx0 + 3, m.t + 11, b.label, { fill: css("--text-2"), "font-size": 10 }));
  }
  const labelYs = [];
  series.forEach((s) => {
    const d = s.values.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
    root.append(svg("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-dasharray": s.dash || "" }));
    if (series.length > 1) labelYs.push({ s, y: Y(s.values[s.values.length - 1].y) });
  });
  labelYs.sort((a, b) => a.y - b.y).forEach((l, i, arr) => { if (i && l.y - arr[i - 1].y < 13) l.y = arr[i - 1].y + 13; });
  labelYs.forEach((l) => root.append(txt(W - m.r + 6, l.y + 4, l.s.name, { fill: css("--text-1"), "font-size": 11 })));
  const cross = svg("line", { y1: m.t, y2: m.t + ih, stroke: css("--text-2"), "stroke-width": 1, opacity: 0, "stroke-dasharray": "3 3" });
  root.append(cross);
  const dots = series.map((s) => svg("circle", { r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 2, opacity: 0 }));
  dots.forEach((d) => root.append(d));
  const hit = svg("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
  root.append(hit);
  host.style.position = "relative";
  host.append(root);
  const tip = tooltipFor(host);
  hit.addEventListener("mousemove", (e) => {
    const r = root.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * W;
    const xv = x0 + ((px - m.l) / iw) * (x1 - x0);
    const ref = series[0].values.reduce((a, b) => (Math.abs(b.x - xv) < Math.abs(a.x - xv) ? b : a));
    cross.setAttribute("x1", X(ref.x)); cross.setAttribute("x2", X(ref.x)); cross.setAttribute("opacity", 1);
    const rows = [el("div", { class: "tip-title" }, `${xLabel} ${ref.x}`)];
    series.forEach((s, i) => {
      const p = s.values.find((q) => q.x === ref.x);
      if (!p) return;
      dots[i].setAttribute("cx", X(p.x)); dots[i].setAttribute("cy", Y(p.y)); dots[i].setAttribute("opacity", 1);
      rows.push(el("div", {}, el("i", { class: "tip-dot", style: `background:${s.color}` }), `${s.name}: `, el("b", {}, fmt(p.y))));
    });
    const inBand = bands.find((b) => ref.x >= b.x0 && ref.x <= b.x1);
    if (inBand) rows.push(el("div", { class: "tip-note" }, inBand.note || "artifact frame"));
    tip.show(rows, ((X(ref.x)) / W) * r.width, e.clientY - r.top);
  });
  hit.addEventListener("mouseleave", () => { tip.hide(); cross.setAttribute("opacity", 0); dots.forEach((d) => d.setAttribute("opacity", 0)); });
}

// ---------------------------------------------------------------------------------------------- horizontal bars with CI
export function barChart(host, { items, valueLabel = "rate", max = 1, height }) {
  const rowH = 38, W = Math.max(280, host.clientWidth || 600), H = height || items.length * rowH + 30;
  const m = { l: 110, r: 56, t: 6, b: 24 };
  const iw = W - m.l - m.r;
  const X = (v) => m.l + (v / max) * iw;
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": valueLabel });
  for (const t of niceTicks(0, max, 4)) {
    root.append(svg("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, stroke: css("--grid") }));
    root.append(txt(X(t), H - 8, fmt(t), { fill: css("--muted"), "text-anchor": "middle", "font-size": 11 }));
  }
  root.append(svg("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, stroke: css("--axis") }));
  host.style.position = "relative";
  const tip = tooltipFor(host);
  items.forEach((it, i) => {
    const y = m.t + i * rowH + 6, bh = 18, w = Math.max(2, X(it.value) - m.l);
    root.append(txt(m.l - 10, y + bh / 2 + 4, it.label, { fill: css("--text-1"), "text-anchor": "end", "font-size": 12 }));
    // 4px rounded data end, square at the baseline
    const r = Math.min(4, w / 2);
    root.append(svg("path", { d: `M${m.l},${y}H${m.l + w - r}Q${m.l + w},${y} ${m.l + w},${y + r}V${y + bh - r}Q${m.l + w},${y + bh} ${m.l + w - r},${y + bh}H${m.l}Z`, fill: it.color }));
    if (it.lo != null && it.hi != null) {
      const cy = y + bh / 2;
      root.append(svg("line", { x1: X(it.lo), x2: X(it.hi), y1: cy, y2: cy, stroke: css("--text-1"), "stroke-width": 1.5 }));
      for (const v of [it.lo, it.hi]) root.append(svg("line", { x1: X(v), x2: X(v), y1: cy - 4, y2: cy + 4, stroke: css("--text-1"), "stroke-width": 1.5 }));
    }
    root.append(txt(X(it.hi ?? it.value) + 8, y + bh / 2 + 4, fmt(it.value), { fill: css("--text-1"), "font-size": 12, "font-weight": 600 }));
    const hit = svg("rect", { x: 0, y: y - 6, width: W, height: rowH, fill: "transparent" });
    hit.addEventListener("mousemove", (e) => {
      const r2 = root.getBoundingClientRect();
      tip.show([el("div", { class: "tip-title" }, it.label), ...it.detail.map((d) => el("div", {}, d))], e.clientX - r2.left, e.clientY - r2.top);
    });
    hit.addEventListener("mouseleave", () => tip.hide());
    root.append(hit);
  });
  host.append(root);
}

// ---------------------------------------------------------------------------------------------- heatmap
const RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
export function heatmap(host, { rows, cols, values, formatter = (v) => v.toFixed(2), detail = () => [], max = 1 }) {
  const cw = Math.max(70, Math.min(140, ((host.clientWidth || 600) - 140) / cols.length)), ch = 38;
  const W = 130 + cw * cols.length, H = 44 + ch * rows.length + 30;
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "heatmap" });
  host.style.position = "relative";
  const tip = tooltipFor(host);
  cols.forEach((c, j) => root.append(txt(130 + j * cw + cw / 2, 28, c, { fill: css("--text-1"), "text-anchor": "middle", "font-size": 12 })));
  rows.forEach((r, i) => {
    root.append(txt(122, 44 + i * ch + ch / 2 + 4, r, { fill: css("--text-1"), "text-anchor": "end", "font-size": 12 }));
    cols.forEach((c, j) => {
      const v = values[i][j];
      const x = 130 + j * cw, y = 44 + i * ch;
      if (v == null || Number.isNaN(v)) {
        root.append(svg("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, rx: 3, fill: css("--grid") }));
        root.append(txt(x + cw / 2, y + ch / 2 + 4, "n/a", { fill: css("--muted"), "text-anchor": "middle", "font-size": 11 }));
        return;
      }
      const idx = Math.max(0, Math.min(RAMP.length - 1, Math.round((v / max) * (RAMP.length - 1))));
      root.append(svg("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, rx: 3, fill: RAMP[idx] }));
      root.append(txt(x + cw / 2, y + ch / 2 + 4, formatter(v), { fill: idx >= 7 ? "#ffffff" : "#0b0b0b", "text-anchor": "middle", "font-size": 12, "font-weight": 600 }));
      const hit = svg("rect", { x, y, width: cw, height: ch, fill: "transparent" });
      hit.addEventListener("mousemove", (e) => {
        const r2 = root.getBoundingClientRect();
        tip.show([el("div", { class: "tip-title" }, `${r} · ${c}`), el("div", {}, `value: ${formatter(v)}`), ...detail(i, j).map((d) => el("div", {}, d))], e.clientX - r2.left, e.clientY - r2.top);
      });
      hit.addEventListener("mouseleave", () => tip.hide());
      root.append(hit);
    });
  });
  // sequential key
  const ky = 44 + rows.length * ch + 12;
  RAMP.forEach((c, k) => root.append(svg("rect", { x: 130 + k * 14, y: ky, width: 14, height: 8, fill: c })));
  root.append(txt(126, ky + 8, "0", { fill: css("--muted"), "text-anchor": "end", "font-size": 10 }));
  root.append(txt(130 + RAMP.length * 14 + 4, ky + 8, String(max), { fill: css("--muted"), "font-size": 10 }));
  host.append(root);
}

export const seriesColors = () => [css("--series-1"), css("--series-2"), css("--series-3"), css("--series-4")];
