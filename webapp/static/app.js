"use strict";

const AXES = ["x", "y", "z"];
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const state = {
  config: [],
  trajectory: null,       // 6 arrays of nT values
  trajectoryPending: false,
  trajectoryStats: null,
  sims: [null, null, null],   // saved simulations: {payload, trajectory, stats, shape, spacing, metrics, original, corrupted, visible}
  target: 0,                  // slot receiving the next simulation
  display: "coreg",           // image shown for simulations that have a coregistered version: "coreg" or "raw"
  slicePosition: {},          // slider position (0..1) of each view, kept when the viewer is rebuilt
  requestId: 0,
  mode: "simple",             // "simple" or "advanced" (shows the fields flagged `advanced`)
  fov: [181, 217, 181],       // field of view of the template (mm), for the ghost spacing hint
  controlOf: {},              // field key -> key of the preset select that fills it
};
const SLOT_LABELS = ["A", "B", "C"];
const COREG_SHORT = { elastix: "Elastix", torchreg: "torchreg" };

// ------------------------------------------------------------------------------------------
// form
// ------------------------------------------------------------------------------------------
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key in node && key !== "list") node[key] = value;
    else node.setAttribute(key, value);
  }
  node.append(...children);
  return node;
}

function numberInput(field, value, label) {
  return el("input", {
    type: "number", value, min: field.min, max: field.max, step: field.step || "any",
    "aria-label": label,
  });
}

function scopesOf(field) {
  const scope = field.scope || "trajectory";
  return Array.isArray(scope) ? scope : [scope];
}

function allFields() { return state.config.flatMap((group) => group.fields); }
function fieldByKey(key) { return allFields().find((field) => field.key === key); }

function buildRow(field) {
  const id = `f-${field.key}`;
  const row = el("div", { class: "row", "data-key": field.key });
  if (field.advanced) row.setAttribute("data-advanced", "");
  const unit = field.unit ? el("span", { class: "unit" }, `(${field.unit})`) : "";
  if (field.kind === "number") {
    const control = el("div", { class: "inputs" }, numberInput(field, field.default, field.label));
    control.firstChild.id = id;
    row.append(el("label", { htmlFor: id }, field.label, unit), control);
  } else if (field.kind === "select") {
    const select = el("select", { id });
    for (const [value, text] of field.options) select.append(el("option", { value: String(value) }, text));
    select.value = String(field.default);
    row.append(el("label", { htmlFor: id }, field.label), select);
  } else {
    const lo = numberInput(field, field.default[0], `${field.label}, minimum`);
    const hi = numberInput(field, field.default[1], `${field.label}, maximum`);
    let toggle = null;
    if (field.optional) {
      toggle = el("input", { type: "checkbox", checked: true, "aria-label": `Activer ${field.label}` });
      toggle.addEventListener("change", () => { lo.disabled = hi.disabled = !toggle.checked; });
    }
    row.append(el("div", { class: "label" }, ...(toggle ? [toggle] : []), field.label, unit),
               el("div", { class: "inputs" }, lo, el("span", {}, "–"), hi));
    row._range = { lo, hi, toggle };
  }
  if (field.help) row.append(el("div", { class: "help" }, field.help));
  if (field.hint) row.append(el("div", { class: "help live", id: `hint-${field.hint}` }));
  return row;
}

function buildForm() {
  const form = $("form");
  form.replaceChildren();
  state.controlOf = {};
  for (const field of allFields()) {
    for (const target of Object.keys(Object.values(field.presets || {}).reduce((all, p) => ({ ...all, ...p }), {}))) {
      state.controlOf[target] = field.key;
    }
  }
  for (const group of state.config) {
    const toggleField = group.toggle ? group.fields.find((field) => field.key === group.toggle) : null;
    let legend;
    if (toggleField) {
      const checkbox = el("input", { type: "checkbox", id: `f-${toggleField.key}`, checked: toggleField.default });
      legend = el("legend", {}, el("label", { class: "toggle", "data-key": toggleField.key }, checkbox, group.group));
    } else {
      legend = el("legend", {}, group.group);
    }
    const set = el("fieldset", {}, legend);
    if (group.advanced) set.setAttribute("data-advanced", "");
    if (toggleField && !toggleField.default) set.setAttribute("data-off", "");
    const body = el("div", { class: "group-body" });
    for (const field of group.fields) if (field !== toggleField) body.append(buildRow(field));
    set.append(body);
    form.append(set);
  }
}

function rowOf(field) { return document.querySelector(`[data-key="${field.key}"]`); }

/** Write a value in the form control of a field (programmatically: no event is fired). */
function setFieldValue(field, value) {
  const row = rowOf(field);
  if (field.kind === "boolean") {
    const input = row.querySelector("input");
    input.checked = value;
    input.closest("fieldset").toggleAttribute("data-off", !value);
  } else if (field.kind === "number") row.querySelector("input").value = value;
  else if (field.kind === "select") row.querySelector("select").value = String(value);
  else {
    const { lo, hi, toggle } = row._range;
    if (toggle) toggle.checked = value !== null;
    lo.disabled = hi.disabled = value === null;
    if (value !== null) { lo.value = value[0]; hi.value = value[1]; }
  }
}

/** Read the form into a payload; returns null (and flags the inputs) if something is not a number. */
function readForm() {
  const payload = {};
  let valid = true;
  const parse = (input) => {
    const value = input.value.trim() === "" ? NaN : Number(input.value);
    input.setAttribute("aria-invalid", Number.isFinite(value) ? "false" : "true");
    if (!Number.isFinite(value)) valid = false;
    return value;
  };
  for (const field of allFields()) {
    const row = rowOf(field);
    if (field.kind === "boolean") payload[field.key] = row.querySelector("input").checked;
    else if (field.kind === "number") payload[field.key] = parse(row.querySelector("input"));
    else if (field.kind === "select") {
      const chosen = row.querySelector("select").value;
      payload[field.key] = field.options.find(([value]) => String(value) === chosen)[0];
    } else {
      const { lo, hi, toggle } = row._range;
      payload[field.key] = toggle && !toggle.checked ? null : [parse(lo), parse(hi)];
    }
  }
  return valid ? payload : null;
}

/** Choosing a preset fills the fields it controls; editing one of them switches the select to "custom". */
function applyPreset(field) {
  const select = rowOf(field).querySelector("select");
  const preset = field.presets[select.value];
  if (!preset) return false;
  for (const [key, value] of Object.entries(preset)) setFieldValue(fieldByKey(key), value);
  return Object.keys(preset).some((key) => scopesOf(fieldByKey(key)).includes("trajectory"));
}

function markCustom(key) {
  const control = state.controlOf[key];
  if (!control) return;
  const select = rowOf(fieldByKey(control)).querySelector("select");
  if (select.value !== "custom") select.value = "custom";
}

function setMode(mode) {
  state.mode = mode;
  $("form").dataset.mode = mode;
  document.querySelectorAll(".mode button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
  $("mode-help").textContent = mode === "simple"
    ? "Réglages essentiels. Le mode avancé donne accès à tous les paramètres."
    : "Tous les paramètres. Les niveaux et sources du mode simple sont des préréglages de ces valeurs.";
  try { localStorage.setItem("mode", mode); } catch (_) { /* storage may be blocked */ }
}

/** Spacing of the ghosts: the field of view (with the padding) divided by the number of cycles. */
function updateGhostHint() {
  const node = $("hint-ghost_spacing");
  if (!node) return;
  const cycles = Number(rowOf(fieldByKey("ghost_cycles")).querySelector("input").value);
  const axis = Number(rowOf(fieldByKey("phase_axis")).querySelector("select").value);
  const padding = Number(rowOf(fieldByKey("oversampling_pct")).querySelector("input").value) || 0;
  if (!(cycles > 0)) { node.textContent = ""; return; }
  const fov = state.fov[axis] * (1 + padding / 100);
  node.textContent = `→ un fantôme tous les ≈ ${(fov / cycles).toFixed(0)} mm le long de l'axe ${AXES[axis]} `
    + `(champ de vue ${fov.toFixed(0)} mm avec le padding).`;
}

/** The fields of the form that only concern the volume computation (not the trajectory). */
function simulationOptions(payload) {
  const options = {};
  for (const group of state.config) {
    for (const field of group.fields) if (scopesOf(field).includes("simulation")) options[field.key] = payload[field.key];
  }
  return options;
}

function isSimulationField(key) {
  const field = fieldByKey(key);
  return Boolean(field) && !scopesOf(field).includes("trajectory");
}

// ------------------------------------------------------------------------------------------
// charts
// ------------------------------------------------------------------------------------------
function niceTicks(min, max, target = 5) {
  if (!(max > min)) { min -= 1; max += 1; }
  const span = max - min;
  const rough = span / target;
  const power = Math.pow(10, Math.floor(Math.log10(rough)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= rough);
  const first = Math.ceil(min / step) * step;
  const ticks = [];
  for (let v = first; v <= max + step * 1e-6; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { ticks, step };
}

class Chart {
  constructor(canvas, unit) {
    this.canvas = canvas;
    this.unit = unit;
    this.series = null;
    this.hover = null;
    canvas.addEventListener("mousemove", (e) => this.onMove(e));
    canvas.addEventListener("mouseleave", () => { this.hover = null; this.draw(); });
    canvas.addEventListener("touchmove", (e) => { this.onMove(e.touches[0]); }, { passive: true });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  setData(series) { this.series = series; this.draw(); }

  get margin() { return { left: 46, right: 24, top: 10, bottom: 28 }; }

  onMove(event) {
    if (!this.series) return;
    const rect = this.canvas.getBoundingClientRect();
    const { left, right } = this.margin;
    const frac = (event.clientX - rect.left - left) / (rect.width - left - right);
    const n = this.series[0].length;
    this.hover = Math.min(n - 1, Math.max(0, Math.round(frac * (n - 1))));
    this.draw();
  }

  draw() {
    const canvas = this.canvas;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!this.series) return;

    const { left, right, top, bottom } = this.margin;
    const plotW = width - left - right, plotH = height - top - bottom;
    const n = this.series[0].length;
    let lo = Infinity, hi = -Infinity;
    for (const s of this.series) for (const v of s) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = (hi - lo) * 0.08 || 1;
    lo -= pad; hi += pad;
    const { ticks } = niceTicks(lo, hi);
    const X = (i) => left + (i / (n - 1)) * plotW;
    const Y = (v) => top + (1 - (v - lo) / (hi - lo)) * plotH;

    ctx.font = "12px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 1;
    // horizontal grid + y labels
    ctx.textAlign = "right";
    for (const t of ticks) {
      ctx.strokeStyle = t === 0 ? css("--axis") : css("--grid");
      ctx.beginPath(); ctx.moveTo(left, Math.round(Y(t)) + 0.5); ctx.lineTo(left + plotW, Math.round(Y(t)) + 0.5); ctx.stroke();
      ctx.fillStyle = css("--muted");
      ctx.fillText(String(+t.toFixed(2)), left - 6, Y(t));
    }
    // x ticks: 0 -> 100 % of the acquisition
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let p = 0; p <= 100; p += 25) {
      const x = left + (p / 100) * plotW;
      ctx.strokeStyle = css("--grid");
      ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, top); ctx.lineTo(Math.round(x) + 0.5, top + plotH); ctx.stroke();
      ctx.fillStyle = css("--muted");
      ctx.fillText(`${p} %`, x, top + plotH + 7);
    }
    // k-space centre
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = css("--axis");
    ctx.beginPath(); ctx.moveTo(left + plotW / 2, top); ctx.lineTo(left + plotW / 2, top + plotH); ctx.stroke();
    ctx.restore();

    // series
    const colors = [css("--series-1"), css("--series-2"), css("--series-3")];
    ctx.lineWidth = 1.8; ctx.lineJoin = "round";
    this.series.forEach((s, k) => {
      ctx.strokeStyle = colors[k];
      ctx.beginPath();
      s.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
      ctx.stroke();
    });

    // hover read-out
    if (this.hover !== null) {
      const i = this.hover, x = X(i);
      ctx.strokeStyle = css("--text"); ctx.lineWidth = 1; ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotH); ctx.stroke();
      ctx.globalAlpha = 1;
      this.series.forEach((s, k) => {
        ctx.fillStyle = colors[k];
        ctx.beginPath(); ctx.arc(x, Y(s[i]), 3.5, 0, 2 * Math.PI); ctx.fill();
      });
      const lines = [`${(100 * i / (n - 1)).toFixed(0)} % de l'acquisition`,
                     ...this.series.map((s, k) => `${AXES[k]} : ${s[i].toFixed(2)} ${this.unit}`)];
      ctx.font = "12px system-ui, sans-serif";
      const boxW = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 24;
      const boxH = lines.length * 17 + 8;
      const bx = x + 12 + boxW > width ? x - 12 - boxW : x + 12;
      ctx.fillStyle = css("--surface"); ctx.strokeStyle = css("--border");
      ctx.beginPath(); ctx.roundRect(bx, top + 4, boxW, boxH, 6); ctx.fill(); ctx.stroke();
      ctx.textAlign = "left"; ctx.textBaseline = "middle";
      lines.forEach((l, j) => {
        const y = top + 4 + 4 + 8.5 + j * 17;
        if (j) { ctx.fillStyle = colors[j - 1]; ctx.fillRect(bx + 8, y - 4, 8, 8); }
        ctx.fillStyle = css("--text");
        ctx.fillText(l, bx + (j ? 22 : 8), y);
      });
    }
  }
}

let charts;
function buildLegends() {
  document.querySelectorAll("[data-legend]").forEach((legend) => {
    legend.replaceChildren(...AXES.map((axis, k) =>
      el("span", {}, el("i", { style: `background:var(--series-${k + 1})` }), axis)));
  });
}

// ------------------------------------------------------------------------------------------
// step 1: trajectory
// ------------------------------------------------------------------------------------------
let debounceTimer = null;
function scheduleTrajectory() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(fetchTrajectory, 300);
}

function setTrajStatus(text, isError = false) {
  const node = $("traj-status");
  node.textContent = text;
  node.classList.toggle("error", isError);
}

/** The curves are flat when neither the random motion nor the ghosting is on. */
function flatNote(stats) {
  return stats.max_translation === 0 && stats.max_rotation === 0
    ? "Aucun mouvement : la trajectoire est nulle, seuls les autres artefacts agiront." : "";
}

function chip(label, value) { return el("span", { class: "chip" }, `${label} `, el("b", {}, value)); }

function showTrajectory(params, stats) {
  state.trajectory = params;
  state.trajectoryStats = stats;
  charts.trans.setData(params.slice(0, 3));
  charts.rot.setData(params.slice(3));
  $("traj-stats").replaceChildren(
    chip("Déplacement max", `${stats.max_translation.toFixed(1)} mm`),
    chip("Rotation max", `${stats.max_rotation.toFixed(1)}°`),
    chip("Points", String(params[0].length)));
}

async function fetchTrajectory() {
  const payload = readForm();
  if (!payload) { setTrajStatus("Un des paramètres n'est pas un nombre valide.", true); return; }
  const id = ++state.requestId;
  state.trajectoryPending = true;
  updateRunButton();
  try {
    const response = await fetch("/api/trajectory", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (id !== state.requestId) return;   // a more recent request is in flight
    if (!response.ok) throw new Error(body.error || "Erreur du serveur");
    showTrajectory(body.params, body.stats);
    setTrajStatus(flatNote(body.stats));
  } catch (error) {
    if (id !== state.requestId) return;
    setTrajStatus(error.message, true);
  } finally {
    if (id === state.requestId) { state.trajectoryPending = false; updateRunButton(); }
  }
}

// ------------------------------------------------------------------------------------------
// step 2: saved simulations (up to three, to compare)
// ------------------------------------------------------------------------------------------
let running = false;

function firstFree() { return state.sims.findIndex((sim) => sim === null); }

function updateRunButton() {
  const button = $("run");
  button.disabled = running || state.trajectoryPending || !state.trajectory;
  const label = SLOT_LABELS[state.target];
  button.textContent = state.sims[state.target]
    ? `Lancer la simulation et remplacer ${label}`
    : `Lancer la simulation (${label})`;
  const count = state.sims.filter(Boolean).length;
  $("sim-count").replaceChildren(chip("Simulations", `${count} / ${SLOT_LABELS.length}`));
}

function decode(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function fieldText(field, value) {
  if (field.kind === "select") return (field.options.find(([v]) => v === value) || [null, String(value)])[1].toLowerCase();
  if (field.kind === "range") return value ? `${value[0]}–${value[1]}` : "—";
  return String(value);
}

/** One line per saved simulation: the seed, the cards that are on, the acquisition options. */
function summarise(payload) {
  const parts = [`graine ${payload.seed}`];
  for (const group of state.config) {
    if (!group.summary || (group.toggle && !payload[group.toggle])) continue;
    parts.push(group.summary.replace(/\{(\w+)\}/g, (_, key) => fieldText(fieldByKey(key), payload[key])));
  }
  parts.push(`axe de phase ${AXES[payload.phase_axis]}`, `${payload.resolution} mm`);
  if (payload.oversampling_pct > 0) parts.push(`padding ${payload.oversampling_pct} %`);
  if (payload.coregistration !== "none") parts.push(`corég. ${COREG_SHORT[payload.coregistration]}`);
  return parts.join(" · ");
}

function renderSlots() {
  const container = $("slots");
  container.replaceChildren();
  state.sims.forEach((sim, i) => {
    const label = SLOT_LABELS[i];
    const radio = el("input", { type: "radio", name: "target", value: String(i), checked: state.target === i,
                                "aria-label": `Enregistrer la prochaine simulation dans l'emplacement ${label}` });
    radio.addEventListener("change", () => { state.target = i; renderSlots(); updateRunButton(); });
    const head = el("div", { class: "slot-head" }, el("span", { class: "slot-badge" }, label),
                    el("label", { class: "slot-target" }, radio, "Cible"));
    const card = el("div", { class: "slot" + (state.target === i ? " selected" : "") + (sim ? "" : " empty") }, head);
    if (!sim) {
      card.append(el("p", { class: "slot-empty" }, "Emplacement libre"));
    } else {
      const show = el("input", { type: "checkbox", checked: sim.visible, "aria-label": `Afficher la simulation ${label}` });
      show.addEventListener("change", () => { sim.visible = show.checked; buildViews(); });
      head.insertBefore(el("label", { class: "slot-show" }, show, "Afficher"), head.lastChild);
      const reload = el("button", { type: "button", class: "ghost" }, "Recharger les paramètres");
      reload.addEventListener("click", () => restoreParams(sim));
      const remove = el("button", { type: "button", class: "ghost" }, "Supprimer");
      remove.addEventListener("click", () => {
        state.sims[i] = null;
        if (state.sims[state.target]) state.target = i;
        renderSlots(); updateRunButton(); buildViews();
      });
      card.append(
        el("div", { class: "chips" },
          chip("Corrélation", sim.coregistration
            ? `${sim.metrics.correlation.toFixed(4)} → ${sim.coregistration.correlation.toFixed(4)}`
            : sim.metrics.correlation.toFixed(4)),
          chip("NRMSE", sim.coregistration
            ? `${(100 * sim.metrics.nrmse).toFixed(1)} → ${(100 * sim.coregistration.nrmse).toFixed(1)} %`
            : `${(100 * sim.metrics.nrmse).toFixed(1)} %`)),
        el("div", { class: "chips" },
          chip("Déplacement max", `${sim.stats.max_translation.toFixed(1)} mm`),
          chip("Rotation max", `${sim.stats.max_rotation.toFixed(1)}°`)),
        el("p", { class: "slot-params" }, summarise(sim.payload)),
        el("p", { class: "slot-params" }, `Volume ${sim.shape.join("×")} · calcul ${sim.metrics.seconds.toFixed(1)} s`
          + (sim.coregistration ? ` + corégistration ${sim.coregistration.method} ${sim.coregistration.seconds.toFixed(1)} s` : "")),
        el("div", { class: "slot-actions" }, reload, remove));
    }
    container.append(card);
  });
}

/** Put the parameters (and curves) of a saved simulation back in the form. */
function restoreParams(sim) {
  clearTimeout(debounceTimer);
  state.requestId++;                       // drop any trajectory request still in flight
  state.trajectoryPending = false;
  for (const field of allFields()) setFieldValue(field, sim.payload[field.key]);
  document.querySelectorAll('#form input[aria-invalid="true"]').forEach((i) => i.setAttribute("aria-invalid", "false"));
  showTrajectory(sim.trajectory, sim.stats);
  setTrajStatus(flatNote(sim.stats));
  updateGhostHint();
  updateRunButton();
}

async function runSimulation() {
  const payload = readForm();
  if (!payload || !state.trajectory) return;
  const slot = state.target;
  const trajectory = state.trajectory, stats = state.trajectoryStats;
  running = true;
  updateRunButton();
  const started = performance.now();
  const status = $("sim-status");
  status.classList.remove("error");
  const tick = () => {
    status.replaceChildren(el("span", { class: "spinner" }),
      `Simulation ${SLOT_LABELS[slot]} en cours${payload.coregistration !== "none" ? " (puis corégistration)" : ""}… `
      + `${((performance.now() - started) / 1000).toFixed(0)} s`);
  };
  tick();
  const timer = setInterval(tick, 500);
  try {
    const response = await fetch("/api/simulate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trajectory, ...simulationOptions(payload) }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "Erreur du serveur");
    state.sims[slot] = {
      payload: JSON.parse(JSON.stringify(payload)), trajectory, stats, visible: true,
      shape: body.shape, spacing: body.spacing, metrics: body.metrics,
      original: decode(body.original), corrupted: decode(body.corrupted),
      coregistered: body.coregistered ? decode(body.coregistered) : null, coregistration: body.coregistration,
    };
    const free = firstFree();
    if (free !== -1) state.target = free;
    status.textContent = "";
    if (body.coregistration_error) { status.classList.add("error"); status.textContent = body.coregistration_error; }
    renderSlots();
    buildViews();
  } catch (error) {
    status.classList.add("error");
    status.textContent = error.message;
  } finally {
    clearInterval(timer);
    running = false;
    updateRunButton();
  }
}

// ------------------------------------------------------------------------------------------
// slice viewer: original + every displayed simulation side by side, difference underneath
// ------------------------------------------------------------------------------------------
const VIEWS = [
  { id: "axial", title: "Coupe axiale", letter: "z" },
  { id: "coronal", title: "Coupe coronale", letter: "y" },
  { id: "sagittal", title: "Coupe sagittale", letter: "x" },
];

function makeLut() {
  const stops = [[0, [10, 10, 24]], [0.4, [140, 40, 110]], [0.75, [240, 120, 40]], [1, [255, 236, 160]]];
  const lut = new Uint8ClampedArray(256 * 3);
  for (let v = 0; v < 256; v++) {
    const t = v / 255;
    let k = 1;
    while (k < stops.length - 1 && t > stops[k][0]) k++;
    const [t0, c0] = stops[k - 1], [t1, c1] = stops[k];
    const f = (t - t0) / (t1 - t0);
    for (let c = 0; c < 3; c++) lut[v * 3 + c] = c0[c] + (c1[c] - c0[c]) * f;
  }
  return lut;
}
const LUT = makeLut();

/** Orientation: array axes are (R, A, S). Axial: anterior up. Coronal/sagittal: superior up, sagittal anterior left. */
function sliceGeometry(view, shape) {
  const [X, Y, Z] = shape;
  if (view === "axial") return { w: X, h: Y, at: (c, r, i) => (c * Y + (Y - 1 - r)) * Z + i, max: Z - 1 };
  if (view === "coronal") return { w: X, h: Z, at: (c, r, i) => (c * Y + i) * Z + (Z - 1 - r), max: Y - 1 };
  return { w: Y, h: Z, at: (c, r, i) => (i * Y + (Y - 1 - c)) * Z + (Z - 1 - r), max: X - 1 };
}

function paintSlice(canvas, geometry, index, mode, gain, sim, moved) {
  const { w, h, at } = geometry;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(w, h);
  const original = sim.original, corrupted = moved;
  let p = 0;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const k = at(c, r, index);
      if (mode === "diff") {
        const d = Math.min(255, Math.abs(original[k] - corrupted[k]) * gain);
        image.data[p] = LUT[d * 3]; image.data[p + 1] = LUT[d * 3 + 1]; image.data[p + 2] = LUT[d * 3 + 2];
      } else {
        const v = (mode === "original" ? original : corrupted)[k];
        image.data[p] = image.data[p + 1] = image.data[p + 2] = v;
      }
      image.data[p + 3] = 255;
      p += 4;
    }
  }
  ctx.putImageData(image, 0, 0);
}

function buildViews() {
  const container = $("views");
  container.replaceChildren();
  const shown = state.sims.map((sim, i) => (sim && sim.visible ? { sim, label: SLOT_LABELS[i] } : null)).filter(Boolean);
  $("results").hidden = shown.length === 0;
  if (!shown.length) return;
  const reference = shown[0].sim;   // the original is displayed once, at the resolution of the first simulation
  const anyCoregistered = shown.some(({ sim }) => sim.coregistered);
  $("display-row").hidden = !anyCoregistered;
  const useCoregistered = anyCoregistered && state.display === "coreg";
  const imageOf = (sim) => (useCoregistered && sim.coregistered ? sim.coregistered : sim.corrupted);
  const titleOf = (sim, label) => `Simulation ${label}` + (useCoregistered && sim.coregistered ? " · corégistrée" : "");

  for (const view of VIEWS) {
    const slider = el("input", { type: "range", min: 0, max: 1000, step: 1,
                                 value: Math.round((state.slicePosition[view.id] ?? 0.5) * 1000),
                                 id: `slice-${view.id}`, "aria-label": `${view.title}, position` });
    slider.dataset.view = view.id;
    const out = el("output", { htmlFor: slider.id }, "");

    const cells = [{ caption: "Original", sim: reference, mode: "original", image: reference.corrupted },
                   ...shown.map(({ sim, label }) => ({ caption: titleOf(sim, label), sim, mode: "corrupted", image: imageOf(sim) }))];
    const diffs = shown.map(({ sim, label }) => ({ caption: `|Original − ${label}|`, sim, mode: "diff", image: imageOf(sim) }));
    const grid = el("div", { class: "compare-grid", style: `--cols:${cells.length}` });
    const figure = (cell) => {
      cell.canvas = el("canvas", {});
      return el("figure", {}, el("figcaption", {}, cell.caption), cell.canvas);
    };
    cells.forEach((cell) => grid.append(figure(cell)));
    grid.append(el("div", { class: "diff-note" }, "Différence absolue avec l'original"));
    diffs.forEach((cell) => grid.append(figure(cell)));

    const render = () => {
      const fraction = Number(slider.value) / 1000;
      state.slicePosition[view.id] = fraction;
      const gain = Number($("gain").value);
      for (const cell of [...cells, ...diffs]) {
        const geometry = sliceGeometry(view.id, cell.sim.shape);
        paintSlice(cell.canvas, geometry, Math.round(fraction * geometry.max), cell.mode, gain, cell.sim, cell.image);
      }
      const index = Math.round(fraction * sliceGeometry(view.id, reference.shape).max);
      out.textContent = `${view.letter} = ${(index * reference.spacing).toFixed(0)} mm (${Math.round(fraction * 100)} %)`;
    };
    slider.addEventListener("input", render);
    slider._render = render;
    container.append(el("div", { class: "view" },
      el("div", { class: "view-head" }, el("b", {}, view.title), slider, out),
      el("div", { class: "grid-scroll" }, grid)));
    render();
  }
}

// ------------------------------------------------------------------------------------------
// start
// ------------------------------------------------------------------------------------------
async function main() {
  if (matchMedia("(max-width: 900px)").matches) $("params").open = false;   // curves first on small screens
  charts = { trans: new Chart($("chart-trans"), "mm"), rot: new Chart($("chart-rot"), "°") };
  buildLegends();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    charts.trans.draw(); charts.rot.draw();
  });

  const response = await fetch("/api/config");
  const config = await response.json();
  state.config = config.config;
  state.fov = config.fov_mm;
  buildForm();
  let mode = "simple";
  try { mode = localStorage.getItem("mode") === "advanced" ? "advanced" : "simple"; } catch (_) { /* blocked */ }
  setMode(mode);
  document.querySelectorAll(".mode button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  updateGhostHint();
  renderSlots();
  updateRunButton();

  // the curves only depend on the trajectory fields: the others (resolution, artefacts ...) keep them
  const concernsTrajectory = (target) => {
    const key = target.closest("[data-key]")?.dataset.key;
    return !(key && isSimulationField(key));
  };
  $("form").addEventListener("input", (event) => {
    const key = event.target.closest("[data-key]")?.dataset.key;
    if (key) markCustom(key);
    updateGhostHint();
    if (concernsTrajectory(event.target)) scheduleTrajectory();
  });
  $("form").addEventListener("change", (event) => {
    const target = event.target;
    const key = target.closest("[data-key]")?.dataset.key;
    const field = key && fieldByKey(key);
    if (target.type === "checkbox") {
      const set = target.closest("fieldset");
      if (field && field.kind === "boolean") set.toggleAttribute("data-off", !target.checked);
      if (concernsTrajectory(target)) scheduleTrajectory();
    }
    if (field && field.presets && applyPreset(field)) scheduleTrajectory();
    updateGhostHint();
  });
  $("reseed").addEventListener("click", () => {
    const input = $("f-seed");
    input.value = Math.floor(Math.random() * 1000000);
    scheduleTrajectory();
  });
  $("run").addEventListener("click", runSimulation);
  document.querySelectorAll('input[name="display"]').forEach((radio) => radio.addEventListener("change", () => {
    state.display = radio.value;
    buildViews();
  }));
  $("gain").addEventListener("input", () => {
    $("gain-out").textContent = `×${$("gain").value}`;
    document.querySelectorAll("input[data-view]").forEach((s) => s._render());
  });

  setTrajStatus("Simulation de la trajectoire…");
  await fetchTrajectory();
}

main().catch((error) => setTrajStatus(`Impossible de charger l'application : ${error.message}`, true));
