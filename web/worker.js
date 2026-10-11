/* Hosts Pyodide off the main thread.
 *
 * Booting Pyodide takes seconds and analysing a large molecule takes seconds
 * more; on the main thread both would freeze the tab with no way to show
 * progress. The worker installs the pgmol wheel that CI built from the
 * repository source, so the page runs exactly the code the test suite covers
 * rather than a reimplementation of it.
 */

const PYODIDE_VERSION = "0.29.5"; // CPython 3.13.2, NumPy 2.2.5
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let analyse = null;

// Only PGMolError is caught: that is the package's documented contract for
// bad input. Anything else is a genuine bug and must not be dressed up as a
// malformed file.
const GLUE = `
import json
from pgmol import PGMolError, analyze, parse_xyz
from pgmol.geometry import centered

def _analyse(text, tolerance):
    try:
        molecule = parse_xyz(text)
        result = analyze(molecule, tolerance=tolerance)
    except PGMolError as exc:
        return json.dumps({"ok": False, "error": str(exc)})

    # The viewer draws symmetry elements through the origin, so it needs the
    # geometry in the same frame the analysis used: centred on the centroid.
    coordinates = centered(molecule.coordinates)
    radius = float(max(1.0, (coordinates ** 2).sum(axis=1).max() ** 0.5))
    # chr(10), not an escape: this Python is embedded in a JS template
    # literal, so any backslash here would be read differently by the
    # browser than by the test that extracts it from the file.
    newline = chr(10)
    lines = [str(molecule.natoms), "centred"]
    lines += [
        "%s %.9f %.9f %.9f" % (symbol, row[0], row[1], row[2])
        for symbol, row in zip(molecule.symbols, coordinates)
    ]

    return json.dumps({
        "ok": True,
        "label": result.label,
        "natoms": molecule.natoms,
        "comment": molecule.comment,
        "order": result.order,
        "consistent": result.consistent,
        "tolerance": result.tolerance,
        "effective_tolerance": result.effective_tolerance,
        "counts": result.operation_counts(),
        "radius": radius,
        "xyz": newline.join(lines) + newline,
        "elements": [
            {
                "kind": element.kind,
                "label": element.label,
                "rotation_order": element.rotation_order,
                "improper_order": element.improper_order,
                "direction": (
                    None if element.direction is None
                    else [float(v) for v in element.direction]
                ),
            }
            for element in result.elements
        ],
    })

_analyse
`;

function status(message) {
  self.postMessage({ type: "status", message });
}

async function boot(wheelUrl) {
  status("Downloading the Python runtime…");
  self.importScripts(`${PYODIDE_BASE}pyodide.js`);
  const pyodide = await self.loadPyodide({ indexURL: PYODIDE_BASE });

  status("Loading NumPy…");
  await pyodide.loadPackage(["numpy", "micropip"]);

  status("Installing pgmol…");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(wheelUrl);

  analyse = pyodide.runPython(GLUE);
  const version = pyodide.runPython("import pgmol; pgmol.__version__");
  self.postMessage({
    type: "ready",
    version,
    python: pyodide.runPython("import sys; '%d.%d.%d' % sys.version_info[:3]"),
    numpy: pyodide.runPython("import numpy; numpy.__version__"),
  });
}

self.onmessage = async (event) => {
  const { type } = event.data;
  if (type === "init") {
    try {
      await boot(event.data.wheelUrl);
    } catch (err) {
      self.postMessage({ type: "failed", message: String(err && err.message ? err.message : err) });
    }
    return;
  }
  if (type === "analyse") {
    const { id, text, tolerance } = event.data;
    try {
      // The glue returns JSON rather than a Python object, which keeps proxy
      // lifetimes out of the message boundary entirely.
      self.postMessage({ type: "result", id, payload: JSON.parse(analyse(text, tolerance)) });
    } catch (err) {
      self.postMessage({
        type: "result",
        id,
        payload: { ok: false, unexpected: true, error: String(err && err.message ? err.message : err) },
      });
    }
  }
};
