/* UI for the pgmol web interface. All computation happens in worker.js. */

const $ = (id) => document.getElementById(id);
const statusBox = $("status");
const statusText = $("status-text");
const dropZone = $("drop");
const fileInput = $("file");
const textArea = $("text");
const exampleSelect = $("example");
const toleranceInput = $("tolerance");
const runButton = $("run");
const resultPanel = $("result");
const resultBody = $("result-body");
const viewerPanel = $("viewer-panel");
const showAxes = $("show-axes");
const showPlanes = $("show-planes");
const legend = $("legend");

let worker = null;
let ready = false;
let pending = 0;
let buildInfo = null;

/* ---------- rendering ---------- */

const escapeHtml = (value) =>
  String(value).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

/** Render "Dinfh" as D∞h and "C2v" as C<sub>2</sub>v for display only. */
function prettyLabel(label) {
  if (label === "Kh") return "K<sub>h</sub>";
  const infinite = label.match(/^([CD])inf([vh])$/);
  if (infinite) return `${infinite[1]}<sub>∞</sub>${infinite[2]}`;
  const axial = label.match(/^([CDS])(\d+)([vhd]?)$/);
  if (axial) return `${axial[1]}<sub>${axial[2]}</sub>${axial[3]}`;
  return escapeHtml(label);
}

const prettyOperation = (name) =>
  name === "sigma" ? "σ" : name.replace(/^([CS])(\d+)$/, "$1<sub>$2</sub>");

function showError(message, unexpected) {
  viewerPanel.hidden = true;
  lastResult = null;
  resultPanel.hidden = false;
  resultBody.innerHTML =
    `<div class="error"><strong>${unexpected ? "Unexpected error" : "Could not read that geometry"}</strong>` +
    `<pre>${escapeHtml(message)}</pre></div>`;
}

function showResult(r) {
  const facts = [
    ["Atoms", r.natoms],
    ["Group order", r.order === null ? "infinite" : r.order],
    ["Tolerance", `${r.tolerance} Å`],
  ];
  if (r.effective_tolerance !== r.tolerance) {
    facts.push(["Effective tolerance", `${Number(r.effective_tolerance).toPrecision(4)} Å`]);
  }

  const operations = Object.entries(r.counts || {})
    .map(([name, count]) => `${count}×${prettyOperation(name)}`)
    .join("&nbsp; ");

  let html = `<div class="label">${prettyLabel(r.label)}</div>`;
  if (r.comment && r.comment.trim()) {
    html += `<p style="color:var(--muted);margin:.4rem 0 0">${escapeHtml(r.comment.trim())}</p>`;
  }
  html += '<dl class="facts">';
  for (const [term, value] of facts) {
    html += `<div class="fact"><dt>${term}</dt><dd>${escapeHtml(value)}</dd></div>`;
  }
  html += "</dl>";

  if (operations) html += `<div class="ops"><strong>Operations:</strong> ${operations}</div>`;

  if (r.effective_tolerance !== r.tolerance) {
    html +=
      '<div class="note">The requested tolerance was capped because two atoms of the same ' +
      "element are close enough that matching them would be ambiguous.</div>";
  }
  // Never quieter than the CLI, which always warns in this case.
  if (!r.consistent) {
    html +=
      '<div class="note"><strong>Only approximate.</strong> The operations found do not form a ' +
      `group, so <code>${escapeHtml(r.label)}</code> is not reliable for this geometry at this ` +
      "tolerance. Try a smaller one.</div>";
  }
  resultPanel.hidden = false;
  resultBody.innerHTML = html;
  lastResult = r;
  renderViewer(r);
}


/* ---------- 3D viewer ---------- */

let viewer = null;
let lastResult = null;

// Axis colours by rotation order; improper-only axes get their own.
const AXIS_COLOURS = { 2: "#2f81f7", 3: "#1a7f37", 4: "#bf8700", 5: "#8250df", 6: "#cf222e" };
const IMPROPER_ONLY = "#8250df";
const PLANE_COLOUR = "#d98324";

const axisColour = (element) =>
  element.rotation_order > 1 ? AXIS_COLOURS[element.rotation_order] || "#2f81f7" : IMPROPER_ONLY;

/** Two unit vectors spanning the plane perpendicular to `n`. */
function perpendicularBasis(n) {
  const away = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  let u = [
    away[1] * n[2] - away[2] * n[1],
    away[2] * n[0] - away[0] * n[2],
    away[0] * n[1] - away[1] * n[0],
  ];
  const len = Math.hypot(...u);
  u = u.map((v) => v / len);
  const v = [
    n[1] * u[2] - n[2] * u[1],
    n[2] * u[0] - n[0] * u[2],
    n[0] * u[1] - n[1] * u[0],
  ];
  return [u, v];
}

/** A mirror plane, drawn as a circle outline through the origin.
 *
 * A filled translucent disc via addCustom renders invisible here -- it comes
 * out the colour of the background whatever colour is asked for -- so the
 * plane is outlined with short cylinders, which do render reliably.
 */
function addPlane(normal, radius, segments) {
  const [u, v] = perpendicularBasis(normal);
  const point = (i) => {
    const a = (2 * Math.PI * i) / segments;
    const c = Math.cos(a) * radius, s = Math.sin(a) * radius;
    return { x: u[0] * c + v[0] * s, y: u[1] * c + v[1] * s, z: u[2] * c + v[2] * s };
  };
  for (let i = 0; i < segments; i++) {
    viewer.addCylinder({
      start: point(i), end: point(i + 1), radius: 0.025,
      color: PLANE_COLOUR, fromCap: 1, toCap: 1,
    });
  }
}

function renderViewer(result) {
  if (!window.$3Dmol || window.__no3dmol) {
    viewerPanel.hidden = true;
    return;
  }
  viewerPanel.hidden = false;
  if (!viewer) {
    viewer = $3Dmol.createViewer($("viewer"), { backgroundAlpha: 0 });
    // A deliberate handle for debugging and for the browser-driven checks.
    window.pgmolViewer = viewer;
  }
  viewer.clear();
  viewer.addModel(result.xyz, "xyz");
  viewer.setStyle({}, { stick: { radius: 0.09 }, sphere: { scale: 0.22 } });

  const reach = result.radius * 1.45;
  const elements = result.elements || [];
  // Each plane costs one cylinder per segment, so coarsen the outline when
  // a high-symmetry group brings a lot of them (Ih has fifteen).
  const planeCount = elements.filter((e) => e.kind === "plane").length;
  const planeSegments = planeCount > 8 ? 24 : 36;
  const shown = { axes: 0, planes: 0, inversion: false };

  for (const element of elements) {
    if (element.kind === "axis" && showAxes.checked) {
      const d = element.direction;
      const end = (sign) => ({ x: d[0] * reach * sign, y: d[1] * reach * sign, z: d[2] * reach * sign });
      viewer.addCylinder({
        start: end(-1), end: end(1), radius: 0.035,
        color: axisColour(element), fromCap: 1, toCap: 1,
      });
      viewer.addLabel(element.label, {
        position: end(1), fontSize: 11, fontColor: axisColour(element),
        backgroundColor: "black", backgroundOpacity: 0.45, borderThickness: 0, inFront: true,
      });
      shown.axes += 1;
    } else if (element.kind === "plane" && showPlanes.checked) {
      addPlane(element.direction, reach * 0.85, planeSegments);
      shown.planes += 1;
    } else if (element.kind === "inversion") {
      viewer.addSphere({ center: { x: 0, y: 0, z: 0 }, radius: 0.09, color: "#8250df" });
      shown.inversion = true;
    }
  }

  viewer.zoomTo();
  viewer.render();

  if (!elements.length) {
    legend.innerHTML = result.order === null
      ? "Drag to rotate, scroll to zoom. This group is infinite, so it has no finite set of symmetry elements to draw."
      : "Drag to rotate, scroll to zoom. This molecule has no symmetry elements beyond the identity.";
    return;
  }

  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  const drawn = [];
  if (shown.axes) drawn.push(plural(shown.axes, "axis").replace("axiss", "axes"));
  if (shown.planes) drawn.push(plural(shown.planes, "mirror plane"));
  if (shown.inversion) drawn.push("the inversion centre");

  const key = [];
  if (shown.axes) {
    for (const n of [...new Set(elements.filter((e) => e.kind === "axis").map((e) => e.rotation_order))].sort((a, b) => b - a)) {
      key.push(`<span class="swatch" style="background:${AXIS_COLOURS[n] || IMPROPER_ONLY}"></span>C<sub>${n}</sub>`);
    }
  }
  if (shown.planes) {
    key.push(`<span class="swatch" style="background:${PLANE_COLOUR}"></span>mirror plane`);
  }

  const total = `${elements.length} element${elements.length === 1 ? "" : "s"} found`;
  legend.innerHTML =
    "Drag to rotate, scroll to zoom. " +
    (drawn.length ? `Showing ${drawn.join(", ")} of ${total}. ` : `${total}; none shown. `) +
    key.join("&nbsp; ");
}

for (const toggle of [showAxes, showPlanes]) {
  toggle.addEventListener("change", () => { if (lastResult) renderViewer(lastResult); });
}

/* ---------- worker plumbing ---------- */

function analyse() {
  const text = textArea.value;
  if (!text.trim()) {
    showError("Nothing to analyse — upload, paste or pick a geometry first.");
    return;
  }
  const tolerance = Number(toleranceInput.value);
  if (!Number.isFinite(tolerance) || tolerance <= 0) {
    showError("Tolerance must be a positive number of ångströms.");
    return;
  }
  const id = ++pending;
  runButton.disabled = true;
  runButton.textContent = "Working…";
  worker.postMessage({ type: "analyse", id, text, tolerance });
}

function startWorker(wheelUrl) {
  worker = new Worker("worker.js");
  worker.onmessage = ({ data }) => {
    if (data.type === "status") {
      statusText.textContent = data.message;
    } else if (data.type === "ready") {
      ready = true;
      statusBox.dataset.state = "ready";
      statusText.textContent =
        `Ready — pgmol ${data.version} on CPython ${data.python}, NumPy ${data.numpy}`;
      runButton.disabled = false;
    } else if (data.type === "failed") {
      statusBox.dataset.state = "failed";
      statusText.textContent = `Could not start: ${data.message}`;
    } else if (data.type === "result") {
      if (data.id !== pending) return; // a newer request superseded this one
      runButton.disabled = false;
      runButton.textContent = "Find point group";
      if (data.payload.ok) showResult(data.payload);
      else showError(data.payload.error, data.payload.unexpected);
    }
  };
  worker.onerror = (err) => {
    statusBox.dataset.state = "failed";
    statusText.textContent = `Worker error: ${err.message || "failed to start"}`;
  };
  worker.postMessage({ type: "init", wheelUrl });
}

/* ---------- input handling ---------- */

function loadFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    textArea.value = reader.result;
    exampleSelect.value = "";
    if (ready) analyse();
  };
  reader.onerror = () => showError(`Could not read ${file.name}.`);
  reader.readAsText(file);
}

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => loadFile(fileInput.files[0]));
for (const event of ["dragenter", "dragover"]) {
  dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.add("over"); });
}
for (const event of ["dragleave", "drop"]) {
  dropZone.addEventListener(event, () => dropZone.classList.remove("over"));
}
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  loadFile(e.dataTransfer.files[0]);
});
runButton.addEventListener("click", analyse);
textArea.addEventListener("input", () => { exampleSelect.value = ""; });

exampleSelect.addEventListener("change", async () => {
  const path = exampleSelect.value;
  if (!path) return;
  try {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    textArea.value = await response.text();
    if (ready) analyse();
  } catch (err) {
    showError(`Could not load the example: ${err.message}`);
  }
});

/* ---------- boot ---------- */

(async function init() {
  try {
    const response = await fetch("build.json", { cache: "no-cache" });
    buildInfo = await response.json();
  } catch {
    statusBox.dataset.state = "failed";
    statusText.textContent = "Could not read build.json — the site is incomplete.";
    return;
  }
  $("build").textContent = `pgmol ${buildInfo.version} (${buildInfo.commit})`;

  // The commit is a cache-buster: the wheel filename alone does not change
  // between builds of the same version.
  startWorker(`${buildInfo.wheel}?v=${encodeURIComponent(buildInfo.commit)}`);

  try {
    const examples = await (await fetch("examples.json", { cache: "no-cache" })).json();
    exampleSelect.innerHTML = '<option value="">— choose an example —</option>';
    for (const group of examples) {
      const optgroup = document.createElement("optgroup");
      optgroup.label = group.label;
      for (const item of group.items) {
        const option = document.createElement("option");
        option.value = item.path;
        option.textContent = `${item.name}  —  expected ${item.expected}`;
        optgroup.appendChild(option);
      }
      exampleSelect.appendChild(optgroup);
    }
  } catch {
    exampleSelect.innerHTML = '<option value="">(examples unavailable)</option>';
  }
})();
