/* FilingLens UI — vanilla JS, no build step. All rendering escapes user/API
   text via esc() to avoid HTML injection from filing snippets. */
const API = location.protocol === "file:" ? "http://localhost:8000" : "";

// Example questions per corpus, so the demo always asks something answerable.
const CHIPS = {
  sec: [
    "What are the risk factors disclosed by Apple?",
    "What foundry strategy does Intel describe?",
    "How did Microsoft's cloud revenue grow year over year?",
    "What does Eli Lilly say about Mounjaro?",
    "How does S&P Global describe its Ratings segment?",
  ],
  emerging: [
    "How does Snowflake describe its consumption-based revenue model?",
    "What does Datadog say about customer growth and net retention?",
    "How does Coinbase describe regulatory risks around crypto assets?",
  ],
};

const $ = id => document.getElementById(id);

function esc(s) {
  return (s == null ? "" : String(s)).replace(/[&<>"]/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function fmtNum(n) {
  if (typeof n !== "number" || !isFinite(n)) return "—";
  return Math.abs(n) >= 1e9 ? (n / 1e9).toFixed(1) + "B"
    : Math.abs(n) >= 1e6 ? (n / 1e6).toFixed(1) + "M"
    : Math.abs(n) >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(n);
}

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => {
      b.classList.toggle("active", b === btn);
      b.setAttribute("aria-selected", b === btn ? "true" : "false");
    });
    document.querySelectorAll(".view").forEach(v => v.classList.add("hidden"));
    $("view-" + btn.dataset.view).classList.remove("hidden");
    if (btn.dataset.view === "graph") {
      gResize();                     // canvas has 0 size while hidden
      if (!graphLoaded) loadGraph();
    }
  });
});

/* ---------- tenants ---------- */
function chipsFor(t) { return CHIPS[t] || CHIPS.sec; }

function renderChips() {
  const t = $("tenant").value;
  $("chips").innerHTML = chipsFor(t)
    .map(q => '<button class="chip" onclick="useq(this)">' + esc(q) + "</button>").join("");
}

function fillTenantSelects(slugs, current) {
  for (const id of ["tenant", "g-tenant"]) {
    const sel = $(id);
    sel.innerHTML = slugs.map(s =>
      '<option value="' + esc(s) + '"' + (s === current ? " selected" : "") + ">" +
      esc(s) + "</option>").join("");
  }
}

async function loadTenants() {
  let slugs = ["sec"];
  try {
    const res = await fetch(API + "/tenants");
    if (!res.ok) throw new Error("HTTP " + res.status);
    tenantsCache = await res.json();
    slugs = tenantsCache.map(t => t.slug);
    // annotate names into option labels
    for (const id of ["tenant", "g-tenant"]) {
      const sel = $(id);
      // Default to the main corpus, not whichever slug sorts first.
      const cur = sel.value || (slugs.includes("sec") ? "sec" : slugs[0]);
      sel.innerHTML = tenantsCache.map(t =>
        '<option value="' + esc(t.slug) + '"' + (t.slug === cur ? " selected" : "") + ">" +
        esc(t.slug) + (t.name && t.name !== t.slug ? " — " + esc(t.name) : "") + "</option>").join("");
    }
    $("tcount").textContent = slugs.length + " corpora";
  } catch (e) {
    fillTenantSelects(slugs, "sec");
    $("tcount").textContent = "(tenant list unavailable)";
  }
  renderChips();
}

let tenantsCache = [];
let graphLoaded = false;
$("tenant").addEventListener("change", () => { renderChips(); $("out").innerHTML = ""; });
$("g-tenant").addEventListener("change", () => { graphLoaded = false; });
$("g-focus").addEventListener("keydown", e => { if (e.key === "Enter") loadGraph(); });

/* ---------- ASK ---------- */
function useq(el) { $("q").value = el.textContent.trim(); ask(); }

async function ask() {
  const out = $("out"), btn = $("ask"), status = $("status");
  const question = $("q").value.trim();
  if (!question) { status.textContent = "Type a question first."; return; }
  btn.disabled = true;
  status.innerHTML = '<span class="spin"></span>retrieving &middot; reranking &hellip;';
  out.innerHTML = "";
  const t0 = performance.now();
  try {
    const res = await fetch(API + "/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        tenant_slug: $("tenant").value,
        question,
        company: $("company").value.trim() || null,
        form_type: $("form").value || null,
        debug: true,   // drives the Explain + Scout panels below
      }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status + " — " + (await res.text()).slice(0, 240));
    const d = await res.json();
    d._ms = Math.round(performance.now() - t0);
    renderAnswer(out, d);
    status.textContent = "";
  } catch (e) {
    out.innerHTML = '<div class="card err">Error: ' + esc(String(e)) +
      "<br><span class='muted'>Is the API running at " + esc(API || location.origin) + " ?</span></div>";
    status.textContent = "";
  }
  btn.disabled = false;
}

/* ---------- ANSWER RENDERING ---------- */
function renderAnswer(out, d) {
  document.getElementById("mode").textContent = d.generation_mode || "";
  const cited = new Set(d.citation_ids || []);
  let h = '<div class="card"><h2>Answer</h2><pre class="answer">' + esc(d.answer) + "</pre>";

  let points = (d.key_points && d.key_points.length) ? d.key_points
    : (d.answer || "").split("\n")
        .filter(s => /^\s*[•\-]\s/.test(s))
        .map(s => s.replace(/^\s*[•\-]\s*/, "").trim())
        .slice(0, 6);
  if (points.length) {
    h += '<h2 style="margin-top:14px">Key points</h2><ul class="points">' +
      points.map(p => "<li>" + esc(p) + "</li>").join("") + "</ul>";
  }

  // "Why this answer" — the plan the engine followed, in one line.
  if (d.plan) {
    const p = d.plan;
    const bits = [];
    if (p.mentioned_companies && p.mentioned_companies.length)
      bits.push("companies: " + p.mentioned_companies.join(", "));
    if (p.wants_temporal) bits.push("temporal");
    if (p.wants_peer_comparison) bits.push("peer comparison");
    if (p.wants_graph_hop) bits.push("graph hop");
    if (bits.length)
      h += '<div class="plan">Why this answer: ' + esc(bits.join(" Â· ")) + "</div>";
  }

  // Retrieval detail (debug=true is always sent by this UI).
  if (d.debug) {
    const t = d.debug.timings_ms || {};
    const rows = [
      ["planner+filters", t["plan+filters"]], ["retrieval", t.retrieval],
      ["fuse+rerank+dedup", t["fuse+rerank+dedup"]], ["graph", t.graph],
      ["answer", t.answer], ["total (server)", t.total],
      ["round-trip (browser)", d._ms],
    ].map(([k, v]) => "<dt>" + esc(k) + "</dt><dd>" +
      (v == null ? "—" : esc(String(v)) + " ms") + "</dd>").join("");
    h += '<details class="dbg"><summary>Explain — plan, filters, stage timings</summary>' +
      '<dl class="debug-grid">' + rows +
      "<dt>filters</dt><dd>" + esc(JSON.stringify(d.debug.filters || {})) + "</dd>" +
      "<dt>retrieval</dt><dd>" + esc(
        "lexical " + d.debug.lexical_hits + " Â· vector " + d.debug.vector_hits +
        " Â· pool " + d.debug.pool_size) + "</dd>" +
      "<dt>multi-query</dt><dd>" + esc(d.debug.variant_query_used ? "on" : "off") + "</dd>" +
      "</dl></details>";
  }
  h += "</div>";

  // Scout — where did the ranked list come from? top-2 evidence cards.
  const sampled = sampledScout(d);
  if (sampled.length) {
    h += '<div class="card"><h2>Scout — strongest supporting evidence</h2>' +
      '<div class="proof">From a fused pool of ' +
      esc(d.debug && d.debug.pool_size != null ? String(d.debug.pool_size) : "…") +
      " candidates, the reranker kept these as the strongest support:</div>" +
      '<div class="cites">' + sampled.map(scoutCard).join("") + "</div></div>";
  }

  if (d.graph_facts && d.graph_facts.length) {
    h += '<div class="card"><h2>Graph evidence <span class="muted">(' +
      d.graph_facts.length + " path" + (d.graph_facts.length > 1 ? "s" : "") +
      ')</span></h2><ul class="graph">' +
      d.graph_facts.slice(0, 8).map(g => "<li>" + esc(g) + "</li>").join("") +
      "</ul></div>";
  }

  if (d.citations && d.citations.length) {
    h += '<div class="card"><h2>All citations (' + d.citations.length +
      ') <span class="muted">— green = cited by the answer</span></h2><div class="cites">' +
      d.citations.map((c, i) => citeCard(c, i + 1, cited.has(i + 1))).join("") +
      "</div></div>";
  }
  out.innerHTML = h;
}

// The top cited cards first, then the rest — max 2, as a "proof trail".
function sampledScout(d) {
  if (!d.citations || !d.citations.length) return [];
  const have = new Set(d.citation_ids || []);
  const citedIdx = d.citations.map((c, i) => i + 1).filter(n => have.has(n));
  const rest = d.citations.map((c, i) => i + 1).filter(n => !have.has(n));
  return (citedIdx.concat(rest)).slice(0, 2).map(n => ({ n, c: d.citations[n - 1] }));
}

function scoutCard(s) {
  return '<div class="cite cited"><span class="idx">rank #' + s.n + "</span>" +
    "<b>" + esc(s.c.company + " " + s.c.form_type) + "</b>" +
    '<div class="meta">' + esc(s.c.section) + "</div>" +
    '<div class="snip">' + esc(String(s.c.snippet || "").slice(0, 180)) + "…</div></div>";
}

function citeCard(c, n, isCited) {
  const title = esc(c.company + " " + c.form_type + " (" + c.filing_date + ")");
  const link = c.filing_url
    ? '<a href="' + esc(c.filing_url) + '" target="_blank" rel="noopener">' + title + "</a>"
    : title;
  return '<div class="cite' + (isCited ? " cited" : "") + '">' +
    '<span class="idx">' + (isCited ? "✓ [" + n + "]" : "#" + n) + "</span>" +
    "<b>" + link + "</b>" +
    '<div class="meta">' + esc(c.section) + " &middot; p." + c.page + "</div>" +
    '<div class="snip">' + esc(c.snippet) + "</div></div>";
}

document.getElementById("q").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask();
});


/* ---------- GRAPH ----------
   A hand-rolled force layout drawn straight onto <canvas>: no library, no
   build step, nothing to install -- matching the rest of this UI. */
const TYPE_COLORS = {
  Company: "#2563eb", "Peer mention": "#93c5fd", SecurityClass: "#7c3aed",
  Filing: "#0d9488", Person: "#db2777", Executive: "#ea580c",
  BoardMember: "#ca8a04", GovernanceEvent: "#dc2626",
};
const REL_COLORS = {
  DESCRIBES: "#7c3aed", FILED: "#0d9488", HOLDS_ROLE: "#db2777",
  PEER_OF: "#2563eb", HAS_VOTING_RIGHTS: "#dc2626",
};

const G = {
  nodes: [], edges: [], byId: {},
  scale: 1, tx: 0, ty: 0, dpr: 1, w: 0, h: 0,
  alpha: 0, raf: 0, fitted: false,
  selected: null, hover: null,
  drag: null, panning: false, moved: false, lastX: 0, lastY: 0,
};

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function gRadius(n) { return 6 + Math.min(16, 4 * Math.sqrt(n.degree || 0)); }
// A Company node you actually ingested vs. one only named in a filing's peer
// group -- both belong in the graph, but only the first is one of "your" N.
function gKind(n) {
  return (n.type === "Company" && n.in_corpus === false) ? "Peer mention" : n.type;
}
function gKick() { if (!G.raf) G.raf = requestAnimationFrame(gLoop); }

function gResize() {
  const c = $("g-canvas");
  const w = c.clientWidth || 640, h = c.clientHeight || 480;
  G.dpr = window.devicePixelRatio || 1;
  G.w = w; G.h = h;
  c.width = Math.round(w * G.dpr);
  c.height = Math.round(h * G.dpr);
  gDraw();
}

function fillFilter(sel, counts) {
  const cur = sel.value;
  let h = '<option value="">all</option>';
  for (const [k, n] of Object.entries(counts || {})) {
    h += '<option value="' + esc(k) + '">' + esc(k) + " (" + n + ")</option>";
  }
  sel.innerHTML = h;
  sel.value = (cur && counts && cur in counts) ? cur : "";
}

async function loadGraph() {
  const status = $("g-status"), btn = $("g-go");
  btn.disabled = true;
  status.innerHTML = '<span class="spin"></span>loading graph…';
  const params = new URLSearchParams({ tenant_slug: $("g-tenant").value || "sec" });
  const type = $("g-type").value, rel = $("g-relation").value, focus = $("g-focus").value.trim();
  if (type) params.set("node_type", type);
  if (rel) params.set("relation", rel);
  if (focus) params.set("company", focus);
  params.set("limit", $("g-limit").value);
  try {
    const res = await fetch(API + "/graph?" + params);
    if (!res.ok) throw new Error("HTTP " + res.status + " — " + (await res.text()).slice(0, 200));
    const d = await res.json();
    fillFilter($("g-type"), d.node_type_counts);
    fillFilter($("g-relation"), d.relation_counts);

    G.byId = {};
    G.nodes = (d.nodes || []).map(n => (G.byId[n.id] = n));
    G.edges = (d.edges || []).map(e => ({
      ...e, s: G.byId[e.source], t: G.byId[e.target],
    })).filter(e => e.s && e.t);
    G.selected = null; G.hover = null; G.drag = null;
    graphLoaded = true;
    $("g-empty").classList.toggle("hidden", G.nodes.length > 0);

    gSeed();
    gFit();
    G.fitted = false;      // auto re-fit once the layout has settled
    G.alpha = 1;
    gLegend();
    renderNodeDetail(null);

    const t = d.totals || {}, r = d.returned || {};
    status.textContent = r.nodes + " nodes · " + r.edges + " edges shown" +
      (t.nodes ? "  (of " + fmtNum(t.nodes) + " / " + fmtNum(t.edges) + " in corpus)" : "");
  } catch (e) {
    status.innerHTML = '<span class="err">Error: ' + esc(String(e)) + "</span>";
  }
  btn.disabled = false;
  gKick();
}

function gLegend() {
  const types = {}, rels = {};
  for (const n of G.nodes) types[gKind(n)] = (types[gKind(n)] || 0) + 1;
  for (const e of G.edges) rels[e.relation] = (rels[e.relation] || 0) + 1;
  const dot = (c, label, n) =>
    '<span class="lg" title="' + esc(label) + '"><i style="background:' + c + '"></i>' +
    esc(label) + ' <b>' + n + "</b></span>";
  let h = Object.entries(types).sort((a, b) => b[1] - a[1])
    .map(([t, n]) => dot(TYPE_COLORS[t] || "#64748b", t, n)).join("");
  h += '<span class="lg-sep"></span>' + Object.entries(rels).sort((a, b) => b[1] - a[1])
    .map(([r, n]) => dot(REL_COLORS[r] || "#94a3b8", r, n)).join("");
  $("g-legend").innerHTML = h;
}

function neighboursOf(node) {
  const set = new Set([node.id]);
  for (const e of G.edges) {
    if (e.s === node) set.add(e.t.id);
    else if (e.t === node) set.add(e.s.id);
  }
  return set;
}

function renderNodeDetail(node) {
  const out = $("g-detail");
  if (!node) {
    out.innerHTML = '<div class="proof">Click a node to inspect its properties and the ' +
      "edges that carry evidence for them.</div>";
    return;
  }
  const props = Object.entries(node.properties || {})
    .map(([k, v]) => "<div><span>" + esc(k) + "</span> " + esc(String(v)) + "</div>").join("");
  const mine = G.edges.filter(e => e.s === node || e.t === node)
    .sort((a, b) => (b.count || 1) - (a.count || 1));
  const rows = mine.slice(0, 25).map(e => {
    const other = e.s === node ? e.t : e.s;
    const dir = e.s === node ? "→" : "←";
    return "<tr><td>" + esc(e.relation) + "</td><td>" + dir + " " + esc(other.label) +
      " <span class='muted'>(" + esc(other.type) + ")</span></td>" +
      "<td>×" + (e.count || 1) + "</td><td>" +
      (e.evidence ? '<span class="ok">evidence</span>' : '<span class="muted">none</span>') +
      "</td></tr>";
  }).join("");
  out.innerHTML = '<div class="card"><h2>' + esc(node.label) +
    ' <span class="muted" style="font-weight:400">' + esc(gKind(node)) + "</span></h2>" +
    (gKind(node) === "Peer mention"
      ? '<p class="muted">Named in a filing\'s peer group, but no filing of its own in this corpus.</p>'
      : "") +
    '<div class="g-meta">degree <b>' + (node.degree || 0) + "</b> · connected edges <b>" +
    mine.length + "</b>" +
    (node.valid_from ? " · valid from <b>" + esc(node.valid_from) + "</b>" : "") +
    (node.valid_to ? " to <b>" + esc(node.valid_to) + "</b>" : "") + "</div>" +
    (props ? '<div class="g-props">' + props + "</div>" : "") +
    (rows ? '<table class="g-edges"><tr><th>relation</th><th>endpoint</th><th>asserts</th><th>source</th></tr>' +
      rows + "</table>" : '<p class="muted">No edges in the current view.</p>') +
    (mine.length > 25 ? '<p class="muted">… ' + (mine.length - 25) + " more not shown</p>" : "") +
    "</div>";
}

/* --- layout: seed positions, fit-to-view, and a plain force simulation --- */
function gSeed() {
  const N = G.nodes.length;
  if (!N) return;
  const R = 46 * Math.sqrt(N);
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    const w1 = ((i * 7919) % 100) / 100, w2 = ((i * 104729) % 100) / 100;
    const rr = R * (0.55 + 0.45 * w1);
    const n = G.nodes[i];
    n.x = Math.cos(a) * rr;
    n.y = Math.sin(a) * rr * (0.55 + 0.45 * w2);
    n.vx = 0; n.vy = 0; n.fixed = false;
  }
}

function gFit() {
  const N = G.nodes.length;
  if (!N || !G.w) return;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const n of G.nodes) {
    const r = gRadius(n);
    x0 = Math.min(x0, n.x - r); y0 = Math.min(y0, n.y - r);
    x1 = Math.max(x1, n.x + r); y1 = Math.max(y1, n.y + r);
  }
  const bw = Math.max(1, x1 - x0), bh = Math.max(1, y1 - y0), pad = 34;
  G.scale = Math.max(0.05, Math.min((G.w - pad * 2) / bw, (G.h - pad * 2) / bh, 1.6));
  G.tx = G.w / 2 - (x0 + bw / 2) * G.scale;
  G.ty = G.h / 2 - (y0 + bh / 2) * G.scale;
}

function gStep() {
  const nodes = G.nodes, edges = G.edges, N = nodes.length;
  if (!N) return;
  const a = G.alpha, rep = 1200, spring = 0.02, rest = 78, pull = 0.004;
  const clamp = v => (v > 40 ? 40 : v < -40 ? -40 : v);

  for (let i = 0; i < N; i++) {
    const p = nodes[i];
    for (let j = i + 1; j < N; j++) {
      const q = nodes[j];
      let dx = q.x - p.x, dy = q.y - p.y;
      let d2 = dx * dx + dy * dy;
      if (d2 < 1) { dx = ((i * 31 + j * 17) % 13) - 6 || 1; dy = ((i * 17 + j * 29) % 11) - 5 || 1; d2 = dx * dx + dy * dy; }
      const d = Math.sqrt(d2);
      // Floor the distance so overlapping nodes can't produce a runaway force.
      const f = a * rep / Math.max(d2, 64);
      const ux = dx / d, uy = dy / d;
      p.vx -= ux * f; p.vy -= uy * f;
      q.vx += ux * f; q.vy += uy * f;
    }
  }
  for (const e of edges) {
    const p = e.s, q = e.t;
    const dx = q.x - p.x, dy = q.y - p.y;
    const d = Math.hypot(dx, dy) || 1;
    const f = a * spring * (d - rest);
    const ux = dx / d, uy = dy / d;
    p.vx += ux * f; p.vy += uy * f;
    q.vx -= ux * f; q.vy -= uy * f;
  }
  let cx = 0, cy = 0;
  for (const n of nodes) { cx += n.x; cy += n.y; }
  cx /= N; cy /= N;
  for (const n of nodes) {
    if (n.fixed) { n.vx = 0; n.vy = 0; continue; }
    n.vx += (cx - n.x) * a * pull;
    n.vy += (cy - n.y) * a * pull;
    n.vx *= 0.86; n.vy *= 0.86;
    n.x += clamp(n.vx); n.y += clamp(n.vy);
  }
  G.alpha *= 0.985;
}

function gLoop() {
  G.raf = 0;
  const N = G.nodes.length;
  if (G.alpha > 0.02 && N) {
    const iters = N > 600 ? 1 : 2;
    for (let i = 0; i < iters; i++) gStep();
    if (!G.fitted && G.alpha < 0.35) { gFit(); G.fitted = true; }
  }
  gDraw();
  if (G.alpha > 0.02) G.raf = requestAnimationFrame(gLoop);
}

/* --- rendering --- */
function gDraw() {
  const c = $("g-canvas");
  if (!c || !G.w) return;
  const ctx = c.getContext("2d");
  const bg = cssVar("--card") || "#fff";
  const txt = cssVar("--txt") || "#111";
  const accent = cssVar("--accent") || "#2563eb";
  const dim = G.selected ? neighboursOf(G.selected) : null;

  ctx.setTransform(G.dpr, 0, 0, G.dpr, 0, 0);
  ctx.clearRect(0, 0, G.w, G.h);
  ctx.save();
  ctx.translate(G.tx, G.ty);
  ctx.scale(G.scale, G.scale);

  for (const e of G.edges) {
    const p = e.s, q = e.t;
    const dx = q.x - p.x, dy = q.y - p.y;
    const d = Math.hypot(dx, dy);
    if (d < 0.5) continue;
    const ux = dx / d, uy = dy / d;
    const x1 = p.x + ux * (gRadius(p) + 2), y1 = p.y + uy * (gRadius(p) + 2);
    const x2 = q.x - ux * (gRadius(q) + 4), y2 = q.y - uy * (gRadius(q) + 4);
    const on = !dim || (dim.has(p.id) && dim.has(q.id));
    const col = REL_COLORS[e.relation] || "#94a3b8";
    const cnt = e.count || 1;
    ctx.globalAlpha = dim ? (on ? 0.95 : 0.06) : 0.5;
    ctx.strokeStyle = col;
    ctx.lineWidth = 1 + Math.min(3.5, Math.log10(1 + cnt) * 1.4);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();

    const ah = 4 + Math.min(5, Math.log10(1 + cnt) * 2.5);   // arrowhead
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.moveTo(x2 + ux * 2, y2 + uy * 2);
    ctx.lineTo(x2 - ux * ah - uy * ah * 0.55, y2 - uy * ah + ux * ah * 0.55);
    ctx.lineTo(x2 - ux * ah + uy * ah * 0.55, y2 - uy * ah - ux * ah * 0.55);
    ctx.closePath(); ctx.fill();
  }

  const showAll = G.nodes.length <= 140;
  const px = 1 / G.scale;
  for (const n of G.nodes) {
    const r = gRadius(n);
    const on = !dim || dim.has(n.id);
    ctx.globalAlpha = on ? 1 : 0.16;
    ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
    ctx.fillStyle = TYPE_COLORS[gKind(n)] || "#64748b";
    ctx.fill();
    if (n === G.selected || n === G.hover) {
      ctx.lineWidth = 2.5 * px;
      ctx.strokeStyle = n === G.selected ? txt : accent;
      ctx.beginPath(); ctx.arc(n.x, n.y, r + 2.5 * px, 0, Math.PI * 2); ctx.stroke();
    }
    if (on && (showAll || gKind(n) === "Company" || n === G.selected ||
               n === G.hover || r >= 13)) {
      const label = n.label.length > 26 ? n.label.slice(0, 25) + "…" : n.label;
      ctx.font = (11 * px) + "px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.lineWidth = 3 * px;
      ctx.strokeStyle = bg; ctx.globalAlpha = 0.95;
      ctx.strokeText(label, n.x, n.y + r + 3 * px);
      ctx.fillStyle = txt;
      ctx.fillText(label, n.x, n.y + r + 3 * px);
    }
  }
  ctx.restore();
  ctx.globalAlpha = 1;
}

/* --- interaction --- */
function gXY(e) {
  const r = $("g-canvas").getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function gPick(x, y) {
  const wx = (x - G.tx) / G.scale, wy = (y - G.ty) / G.scale;
  let best = null, bd = Infinity;
  for (const n of G.nodes) {
    const d = Math.hypot(n.x - wx, n.y - wy);
    if (d <= gRadius(n) + 5 / G.scale && d < bd) { bd = d; best = n; }
  }
  return best;
}

function selectNode(n) { G.selected = n; renderNodeDetail(n); gDraw(); }

function gInit() {
  const c = $("g-canvas");
  gResize();

  c.addEventListener("mousedown", e => {
    const { x, y } = gXY(e);
    const n = gPick(x, y);
    G.moved = false; G.lastX = x; G.lastY = y;
    if (n) { G.drag = n; n.fixed = true; } else { G.panning = true; }
    c.style.cursor = "grabbing";
  });

  window.addEventListener("mousemove", e => {
    const { x, y } = gXY(e);
    if (G.drag) {
      G.moved = true;
      G.drag.x = (x - G.tx) / G.scale;
      G.drag.y = (y - G.ty) / G.scale;
      G.drag.vx = 0; G.drag.vy = 0;
      G.alpha = Math.max(G.alpha, 0.2);
      gKick(); gDraw();
      return;
    }
    if (G.panning) {
      G.moved = true;
      G.tx += x - G.lastX; G.ty += y - G.lastY;
      G.lastX = x; G.lastY = y;
      gDraw();
      return;
    }
    const h = gPick(x, y);
    if (h !== G.hover) {
      G.hover = h;
      c.style.cursor = h ? "pointer" : "grab";
      gDraw();
    }
  });

  window.addEventListener("mouseup", () => {
    if (G.drag) {
      const wasClick = !G.moved;
      G.drag.fixed = false;
      const n = G.drag;
      G.drag = null;
      if (wasClick) selectNode(n); else gKick();
    } else if (G.panning) {
      if (!G.moved && G.selected) selectNode(null);
      G.panning = false;
    }
    c.style.cursor = G.hover ? "pointer" : "grab";
  });

  c.addEventListener("wheel", e => {
    e.preventDefault();
    const { x, y } = gXY(e);
    const ns = Math.min(6, Math.max(0.05, G.scale * Math.exp(-e.deltaY * 0.0015)));
    G.tx = x - (x - G.tx) * (ns / G.scale);
    G.ty = y - (y - G.ty) * (ns / G.scale);
    G.scale = ns;
    gDraw();
  }, { passive: false });

  window.addEventListener("resize", gResize);
}

gInit();
loadTenants();
