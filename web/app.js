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
