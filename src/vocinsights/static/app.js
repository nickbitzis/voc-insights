"use strict";
// VoC Insights UI: plain JS, SVG charts, hash routes (#/ and #/theme/<name>).

const view = document.getElementById("view");
const tip = document.getElementById("tip");
const cache = {};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
const fmt = (n) => Number(n).toLocaleString("en-US");

async function get(url) {
  if (!cache[url]) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    cache[url] = await r.json();
  }
  return cache[url];
}

// Tooltip: any element with data-tip shows it on hover or focus.
document.addEventListener("mouseover", (e) => {
  const el = e.target.closest("[data-tip]");
  if (!el) { tip.hidden = true; return; }
  tip.innerHTML = el.getAttribute("data-tip");
  tip.hidden = false;
});
document.addEventListener("mousemove", (e) => {
  if (tip.hidden) return;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + w > window.innerWidth - 8) x = e.clientX - w - 14;
  if (y + h > window.innerHeight - 8) y = e.clientY - h - 14;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
});

function niceMax(v) {
  const p = 10 ** Math.floor(Math.log10(Math.max(1, v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

// One small multiple: stacked sentiment bars per quarter on a y scale shared by every card.
function multiple(theme, yMax) {
  const W = 170, H = 104, L = 30, B = 16, T = 8;
  const bw = (W - L - 6) / theme.quarters.length;
  let bars = "";
  theme.quarters.forEach((q, i) => {
    let y = H - B;
    const x = L + i * bw + 3;
    for (const [k, col] of [["negative", "var(--neg)"], ["neutral", "var(--neu)"], ["positive", "var(--pos)"]]) {
      const h = (q[k] / yMax) * (H - B - T);
      if (h <= 0) continue;
      y -= h;
      bars += `<rect x="${x}" y="${y + 1}" width="${bw - 6}" height="${Math.max(0, h - 2)}" rx="2" fill="${col}"/>`;
    }
    const tipText = `<b>${esc(theme.label)}, ${q.quarter}</b><br>${fmt(q.total)} mentions<br>${fmt(q.negative)} negative, ${fmt(q.neutral)} neutral, ${fmt(q.positive)} positive`;
    bars += `<rect x="${L + i * bw}" y="${T}" width="${bw}" height="${H - B - T}" fill="transparent" data-tip="${esc(tipText)}"/>`;
    bars += `<text x="${x + (bw - 6) / 2}" y="${H - 3}" text-anchor="middle">Q${i + 1}</text>`;
  });
  const axis = `<line x1="${L - 2}" x2="${W - 2}" y1="${H - B}" y2="${H - B}" stroke="var(--axis)"/>
    <text x="${L - 5}" y="${T + 4}" text-anchor="end">${yMax >= 1000 ? `${yMax / 1000}k` : yMax}</text><text x="${L - 5}" y="${H - B}" text-anchor="end">0</text>`;
  return `<a class="mult" href="#/theme/${theme.theme}"><div class="t" title="${esc(theme.label)}">${esc(theme.label)}</div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(theme.label)} mentions by quarter">${axis}${bars}</svg></a>`;
}

// Weekly line chart, one shared y axis. series: [{name, values, color}]; events: [{date,label}].
function lineChart(weeks, series, events = [], height = 220) {
  const W = 760, H = height, L = 40, R = 12, T = 12, B = 26;
  const max = niceMax(Math.max(1, ...series.flatMap((s) => s.values)));
  const x = (i) => L + (i / Math.max(1, weeks.length - 1)) * (W - L - R);
  const y = (v) => H - B - (v / max) * (H - B - T);
  let g = "";
  for (const f of [0, 0.5, 1]) {
    g += `<line x1="${L}" x2="${W - R}" y1="${y(max * f)}" y2="${y(max * f)}" stroke="var(--border)"/>
      <text x="${L - 6}" y="${y(max * f) + 4}" text-anchor="end">${fmt(max * f)}</text>`;
  }
  weeks.forEach((w, i) => {
    if (w.slice(8, 10) <= "07" && ["01", "04", "07", "10"].includes(w.slice(5, 7))) {
      g += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${w.slice(0, 7)}</text>`;
    }
  });
  events.forEach((ev, k) => {
    const i = weeks.findIndex((w) => w > ev.date) - 1;
    if (i < 0) return;
    g += `<line x1="${x(i)}" x2="${x(i)}" y1="${T}" y2="${H - B}" stroke="var(--muted)" stroke-dasharray="3 3"/>
      <text x="${x(i) + 4}" y="${T + 10 + (k % 2) * 14}" class="lbl">${esc(ev.label)}</text>`;
  });
  for (const s of series) {
    const d = s.values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`;
    const peak = s.values.indexOf(Math.max(...s.values));
    if (series.length <= 4) g += `<text x="${x(peak) - 8}" y="${y(s.values[peak]) + 4}" text-anchor="end" class="lbl">${esc(s.name)}</text>`;
  }
  const step = (W - L - R) / Math.max(1, weeks.length - 1);
  weeks.forEach((w, i) => {
    const rows = series.map((s) => `${esc(s.name)}: ${fmt(s.values[i])}`).join("<br>");
    g += `<rect x="${x(i) - step / 2}" y="${T}" width="${step}" height="${H - B - T}" fill="transparent" data-tip="<b>Week of ${w}</b><br>${esc(rows)}"/>`;
  });
  g += `<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="var(--axis)"/>`;
  return `<svg viewBox="0 0 ${W} ${H}" role="img">${g}</svg>`;
}

// NPS drivers: coefficient bars from zero with 95% CI whiskers and the planted value as a tick.
function driverChart(drivers) {
  const rowH = 22, L = 150, R = 16, W = 560, T = 8;
  const H = T + rowH * drivers.length + 26;
  const lo = Math.min(-1, ...drivers.map((d) => d.ci_low)), hi = Math.max(1, ...drivers.map((d) => d.ci_high));
  const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
  let g = `<line x1="${x(0)}" x2="${x(0)}" y1="${T}" y2="${H - 22}" stroke="var(--axis)"/>`;
  for (let v = Math.ceil(lo); v <= Math.floor(hi); v++) {
    g += `<text x="${x(v)}" y="${H - 6}" text-anchor="middle">${v > 0 ? "+" : ""}${v}</text>`;
  }
  drivers.forEach((d, i) => {
    const yc = T + i * rowH + rowH / 2;
    const col = d.coef < 0 ? "var(--neg)" : "var(--pos)";
    const x0 = Math.min(x(0), x(d.coef)), w = Math.abs(x(d.coef) - x(0));
    g += `<text x="${L - 8}" y="${yc + 4}" text-anchor="end" class="lbl">${esc(d.label)}</text>
      <rect x="${x0}" y="${yc - 6}" width="${Math.max(1, w)}" height="12" rx="3" fill="${col}"/>
      <line x1="${x(d.ci_low)}" x2="${x(d.ci_high)}" y1="${yc}" y2="${yc}" stroke="var(--text-2)" stroke-width="1.5"/>`;
    if (d.planted != null) g += `<line x1="${x(d.planted)}" x2="${x(d.planted)}" y1="${yc - 9}" y2="${yc + 9}" stroke="var(--text)" stroke-width="2"/>`;
    const t = `<b>${esc(d.label)}</b><br>${d.coef > 0 ? "+" : ""}${d.coef.toFixed(2)} points (95% CI ${d.ci_low.toFixed(2)} to ${d.ci_high.toFixed(2)})<br>mentioned in ${pct(d.prevalence)} of survey comments<br>planted in the generator: ${d.planted?.toFixed(2)}`;
    g += `<rect x="0" y="${yc - rowH / 2}" width="${W}" height="${rowH}" fill="transparent" data-tip="${esc(t)}"/>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="NPS driver coefficients">${g}</svg>`;
}

function kpi(v, l) { return `<div class="panel kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`; }

const ROWS = { rules_only: "Keyword rules only", classifier_only: "Classifier only", llm_all: "Claude Haiku for everything (simulated)", cascade: "Cascade, chosen point" };

async function dashboard() {
  const [s, th, em, dr, ev] = await Promise.all([get("/api/summary"), get("/api/themes"), get("/api/emerging"), get("/api/drivers"), get("/api/eval?per_theme=false")]);
  const q = Object.entries(s.nps_by_quarter);
  const [q4, q3] = [q[q.length - 1], q[q.length - 2]];
  const det = s.detection;
  const yMax = niceMax(th.y_max);
  const ems = em.alerts.map((a) => `<div class="alert"><b><a href="#/theme/${a.theme}">${esc(a.label)}</a></b> <span class="chip">${a.kind.replace("_", " ")}</span>
      <div class="small">${a.first_day} to ${a.last_day}, ${a.kind === "volume" ? `peak ${a.peak_ratio.toFixed(2)} times baseline` : `negative share up ${pct(a.peak_lift)}`}, p = ${a.p_value.toExponential(1)}</div></div>`).join("");
  const cls = em.clusters.map((c) => `<div class="alert"><b>New cluster: ${esc(c.name)}</b> <span class="chip">${fmt(c.size)} comments</span>
      <div class="small">First flagged ${c.first_day}, closest theme ${esc(c.nearest_theme)}. Terms: ${c.terms.slice(0, 6).map(esc).join(", ")}</div>
      <div class="quote negative small">${esc(c.examples[0])}</div></div>`).join("");
  const series = [];
  if (em.series.app_stability) series.push({ name: "Crashes and freezes (theme)", values: em.series.app_stability, color: "var(--neg)" });
  for (const [name, v] of Object.entries(em.cluster_series).slice(0, 1)) series.push({ name: `New cluster: ${name}`, values: v, color: "var(--violet)" });
  const evRows = Object.entries(ROWS).map(([k, name]) => {
    const r = ev.rows[k];
    return `<tr><td>${name}</td><td class="num">${r.micro_f1.toFixed(3)}</td><td class="num">${r.macro_f1.toFixed(3)}</td><td class="num hide-sm">${pct(r.sentiment.aspect_accuracy)}</td><td class="num">$${r.cost_per_1k.toFixed(4)}</td><td class="num hide-sm">${pct(r.llm_share, 0)}</td></tr>`;
  }).join("");
  view.innerHTML = `
    ${kpi(fmt(s.texts), "texts across 4 quarters")}
    ${kpi(`${q4[1].toFixed(1)}`, `NPS ${q4[0]} (${q3[0]}: ${q3[1].toFixed(1)})`)}
    ${kpi(s.eval.cascade.micro_f1.toFixed(3), `cascade micro F1, ${pct(s.eval.cascade.llm_share, 0)} sent to the LLM`)}
    ${kpi(det.cluster_alert_lag_days != null ? `${det.cluster_alert_lag_days} days` : "missed", "to flag the release 6.0 bug")}
    <section class="panel wide">
      <h2>Volume and sentiment by theme, per quarter</h2>
      <div class="legend"><span><span class="sw" style="background:var(--neg)"></span>negative</span><span><span class="sw" style="background:var(--neu)"></span>neutral</span><span><span class="sw" style="background:var(--pos)"></span>positive</span><span class="muted">Every chart uses the same y scale (0 to ${fmt(yMax)} mentions). Click a theme for detail.</span></div>
      <div class="multiples">${th.themes.map((t) => multiple(t, yMax)).join("")}</div>
    </section>
    <section class="panel three">
      <h2>Emerging issues</h2>
      <div class="legend">${series.map((x) => `<span><span class="sw" style="background:${x.color}"></span>${esc(x.name)}</span>`).join("")}</div>
      ${lineChart(em.weeks, series, em.events)}
      <p class="small muted">Mentions per week. Release 6.0 shipped ${det.release}. The new-cluster detector flagged it on ${det.cluster_alert_day} (${det.cluster_alert_lag_days} days), the theme volume test on ${det.theme_alert_day} (${det.theme_alert_lag_days} days). Alerts not tied to a planted event: ${s.false_alarms}.</p>
    </section>
    <section class="panel"><h2>Alerts</h2>${cls}${ems || '<p class="muted">None.</p>'}</section>
    <section class="panel half">
      <h2>NPS drivers</h2>
      <div class="legend"><span><span class="sw" style="background:var(--neg)"></span>lowers score</span><span><span class="sw" style="background:var(--pos)"></span>raises score</span><span>whisker: 95% CI</span><span>black tick: planted effect</span></div>
      ${driverChart(dr.drivers)}
      <p class="small muted">Change in the 0 to 10 score when a theme is mentioned, from ${fmt(dr.n)} survey responses (R squared ${dr.r2.toFixed(2)}). ${esc(dr.caveats[0])}</p>
    </section>
    <section class="panel half">
      <h2>Theme classification, labeled test set</h2>
      <div class="scroll"><table><tr><th>Policy</th><th class="num">Micro F1</th><th class="num">Macro F1</th><th class="num hide-sm">Aspect sent.</th><th class="num">Cost / 1k</th><th class="num hide-sm">To LLM</th></tr>${evRows}</table></div>
      <p class="small muted">${fmt(ev.n_test)} held-out labeled texts, ${fmt(ev.n_train)} for training. Operating point: ${esc(ev.operating_point.label)}. LLM numbers are simulated.</p>
      <h2 style="margin-top:16px">NPS by negative aspects</h2>
      <div class="scroll"><table><tr><th>Negative aspects</th><th class="num">Responses</th><th class="num">Mean score</th><th class="num">NPS</th></tr>
      ${dr.nps_vs_sentiment.by_negative_aspects.map((r) => `<tr><td>${r.negative_aspects}</td><td class="num">${fmt(r.n)}</td><td class="num">${r.mean_score.toFixed(2)}</td><td class="num">${r.nps.toFixed(1)}</td></tr>`).join("")}</table></div>
    </section>
    <section class="panel wide">
      <h2>Try the classifier</h2>
      <textarea id="txt" rows="2">Since the update the widget crashes the app, but support was really helpful.</textarea>
      <button id="go">Classify</button><div class="out" id="out"></div>
    </section>`;
  document.getElementById("go").onclick = classify;
}

async function classify() {
  const out = document.getElementById("out");
  const r = await fetch("/api/classify", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text: document.getElementById("txt").value }) });
  const d = await r.json();
  if (!r.ok) { out.textContent = JSON.stringify(d.detail); return; }
  out.innerHTML = `Tier: <b>${d.tier}</b>${d.simulated ? ' <span class="badge sim">simulated</span>' : ""}, confidence ${d.confidence.toFixed(2)}, cost $${d.cost.toFixed(6)}<br>` +
    (d.themes.length ? d.themes.map((t) => `<span class="chip">${esc(t)}: ${d.sentiments[t]}</span> <span class="small muted">"${esc(d.evidence[t])}"</span>`).join("<br>") : "No theme found. Overall: " + d.overall);
}

async function themePage(name) {
  const t = await get(`/api/themes/${encodeURIComponent(name)}?quotes=8`);
  const em = await get("/api/emerging");
  const src = Object.entries(t.by_source).map(([s, c]) => `<tr><td>${s}</td><td class="num">${fmt(c.negative || 0)}</td><td class="num">${fmt(c.neutral || 0)}</td><td class="num">${fmt(c.positive || 0)}</td></tr>`).join("");
  const quotes = t.quotes.map((q) => {
    const txt = esc(q.text), ev = esc(q.evidence || "");
    const body = ev && txt.includes(ev) ? txt.replace(ev, `<mark>${ev}</mark>`) : txt;
    return `<div class="quote ${q.sentiment}">${body}<div class="meta">${q.source}, ${q.date}, ${q.sentiment}, tagged by ${q.tier}</div></div>`;
  }).join("");
  const d = t.driver, e = t.eval;
  view.innerHTML = `
    <section class="panel wide"><a href="#/">&larr; all themes</a><h3 style="margin-top:8px">${esc(t.label)}</h3>
      <span class="muted">${fmt(t.mentions)} mentions over the year</span></section>
    ${kpi(e ? e.f1.toFixed(3) : "-", "classification F1 on the test set")}
    ${kpi(e ? `${pct(e.precision)} / ${pct(e.recall)}` : "-", "precision / recall")}
    ${kpi(d ? `${d.coef > 0 ? "+" : ""}${d.coef.toFixed(2)}` : "-", "NPS points when mentioned")}
    ${kpi(fmt(t.alerts.length), "alerts this year")}
    <section class="panel three"><h2>Weekly mentions</h2>
      <div class="legend"><span><span class="sw" style="background:var(--accent)"></span>all mentions</span><span><span class="sw" style="background:var(--neg)"></span>negative mentions</span></div>
      ${lineChart(t.weeks, [{ name: "all", values: t.count, color: "var(--accent)" }, { name: "negative", values: t.negative, color: "var(--neg)" }], em.events)}
    </section>
    <section class="panel"><h2>By source</h2><div class="scroll"><table><tr><th>Source</th><th class="num">Neg</th><th class="num">Neu</th><th class="num">Pos</th></tr>${src}</table></div>
      ${t.alerts.map((a) => `<p class="small">Alert (${a.kind.replace("_", " ")}): ${a.first_day} to ${a.last_day}</p>`).join("")}</section>
    <section class="panel wide"><h2>What customers said (random sample, verbatim)</h2><p class="small muted">The bold span is the evidence the tagging tier pointed to.</p>${quotes}</section>`;
}

async function route() {
  tip.hidden = true;
  const m = location.hash.match(/^#\/theme\/([a-z_]+)/);
  try {
    if (m) await themePage(m[1]); else await dashboard();
    window.scrollTo(0, 0);
  } catch (err) {
    view.innerHTML = `<p class="pad">Could not load: ${esc(err.message)}</p>`;
  }
}

get("/api/summary").then((s) => { if (!s.simulated_llm) document.getElementById("simbadge").hidden = true; }).catch(() => {});
window.addEventListener("hashchange", route);
route();
