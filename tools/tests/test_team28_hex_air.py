#!/usr/bin/env python3
"""
Tests fuer tools/gmsh/team28_hex_air.py (Hexaedernetz mit Luft und Zonen).

    python3 tools/tests/test_team28_hex_air.py        # oder: pytest tools/tests

Braucht nur numpy. Netz: --scale 0.5 (rund 9 300 Hexaeder, unter einer Sekunde).
Jeder Test hat eine Gegenprobe, die einen absichtlich eingebauten Fehler
erkennen muss.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..")
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "gmsh"))
import meshcheck as mc  # noqa: E402
import team28_hex_air as g  # noqa: E402

GEN = os.path.join(TOOLS, "gmsh", "team28_hex_air.py")
_CACHE = {}


def make(extra=()):
    key = tuple(extra)
    if key not in _CACHE:
        d = tempfile.mkdtemp(prefix="hexair_")
        out = os.path.join(d, "m.vtk")
        p = subprocess.run([sys.executable, GEN, "--scale", "0.5", "--plate-z0", "11.3",
                            "-o", out, *extra], capture_output=True, text=True)
        _CACHE[key] = (out, p)
    return _CACHE[key]


def test_generator_runs_and_checks_itself():
    out, p = make()
    assert p.returncode == 0, p.stdout + p.stderr
    assert "Netz in Ordnung." in p.stdout


def test_mesh_properties():
    out, _ = make()
    m = mc.read_vtk(out)
    assert set(m.types.tolist()) == {12}, "nur Hexaeder"
    res = {r.name.split(" (")[0]: r for r in mc.run_checks(
        m, require_hex=True, air_enclosed=True,
        expected_volumes=g.ngon_volumes(4 * 6), names=g.NAMES)}
    assert all(r.ok for r in res.values()), [r for r in res.values() if not r.ok]
    pts = m.points
    conn = np.array(m.cells)
    sj = g.scaled_jacobians(pts, conn)
    assert sj.min() >= 0.5, sj.min()
    assert mc.check_conformity(m).ok
    # Volumen je Leiter exakt (rel. 1e-9 steckt in check_volumes)
    for r in mc.check_volumes(m, g.ngon_volumes(24), names=g.NAMES):
        assert r.ok, r


def test_zones_cover_every_air_cell_once():
    out, _ = make()
    m = mc.read_vtk(out)
    zone = mc.read_zones(os.path.splitext(out)[0] + ".zones.json", len(m.comp))
    air = m.comp == 0
    assert (zone[air] > 0).all() and (zone[~air] == 0).all()
    for k in (10, 11, 12, 13):
        assert (zone == k).any(), f"Zone {k} leer"
    assert set(np.unique(zone[air])) >= {1, 2, 3, 4, 10, 11, 12, 13}


def test_zone_geometry():
    """Zonen liegen dort, wo die Tabelle im Auftrag sie verlangt."""
    out, _ = make()
    m = mc.read_vtk(out)
    zone = mc.read_zones(os.path.splitext(out)[0] + ".zones.json")
    P = m.points
    ctr = np.array([P[c].mean(axis=0) for c in m.cells])
    r = np.hypot(ctr[:, 0], ctr[:, 1])
    z = ctr[:, 2]
    z0 = 11.3
    assert (r[zone == 11] < 27).all() and (z[zone == 11] < 0).all() and (z[zone == 11] > -52).all()
    assert ((r[zone == 12] > 55) & (r[zone == 12] < 80)).all()
    assert ((r[zone == 13] > 65) & (r[zone == 13] < 95)).all()
    assert ((z[zone == 13] > z0) & (z[zone == 13] < z0 + 3)).all()
    assert ((z[zone == 10] > 0) & (z[zone == 10] < z0) & (r[zone == 10] < 95)).all()
    shell = (zone >= 1) & (zone <= 4)
    outside = (r > 95) | (z < -52) | (z > z0 + 3)
    assert outside[shell].all(), "Schalen nur ausserhalb des Leiterzylinders"
    # Die Zonen 1..S sind ineinander geschachtelt: die Ausdehnung der Vereinigung
    # der Zonen 1..k waechst mit k in r und in z (je Achse nicht abnehmend, insgesamt wachsend).
    ext = []
    for k in (1, 2, 3, 4):
        sel = (zone >= 1) & (zone <= k)
        ext.append((r[sel].max(), z[sel].max(), z[sel].min()))
    for lo, hi in zip(ext[:-1], ext[1:]):
        # nicht streng je Achse: unten gibt es weniger Schichten als Gruppen
        assert hi[0] >= lo[0] and hi[1] >= lo[1] and hi[2] <= lo[2], ext
        assert hi != lo, ext


def test_conductor_nodes_independent_of_truncation():
    a, _ = make(("--truncation-radial", "2", "--truncation-axial", "2"))
    b, _ = make(("--truncation-radial", "3", "--truncation-axial", "4"))
    ma, mb = mc.read_vtk(a), mc.read_vtk(b)

    def nodes(m):
        cond = np.flatnonzero((m.comp > 0))
        gap = np.flatnonzero(np.isin(mc.read_zones(
            os.path.join(os.path.dirname(a if m is ma else b), "m.zones.json")), (10,)))
        idx = np.concatenate([cond, gap])
        n = np.unique(np.concatenate([m.cells[i] for i in idx]))
        return {tuple(p) for p in m.points[n]}
    assert nodes(ma) == nodes(mb), "Leiter- und Spaltknoten haengen vom Abschneidefaktor ab"


def test_counterprobe_wrong_zone_length_rejected():
    out, _ = make()
    m = mc.read_vtk(out)
    path = os.path.splitext(out)[0] + ".zones.json"
    data = json.load(open(path))
    bad = dict(data, zone=data["zone"][:-1], n_cells=len(data["zone"]) - 1)
    bp = path + ".kurz.json"
    json.dump(bad, open(bp, "w"))
    try:
        mc.read_zones(bp, len(m.comp))
        raise AssertionError("zu kurze zones.json wurde nicht abgelehnt")
    except ValueError:
        pass
    res = g.check_zones(m, bp, 4)
    assert not all(r.ok for r in res)
    # Gegenprobe der Gegenprobe: die richtige Datei besteht
    assert all(r.ok for r in g.check_zones(m, path, 4))


def test_counterprobe_leiterzone_nonzero_rejected():
    out, _ = make()
    m = mc.read_vtk(out)
    zone = mc.read_zones(os.path.splitext(out)[0] + ".zones.json").copy()
    zone[np.flatnonzero(m.comp == 1)[0]] = 5          # Leiterzelle mit Zone
    bp = os.path.splitext(out)[0] + ".falsch.json"
    json.dump({"version": 1, "n_cells": len(zone), "zone": zone.tolist()}, open(bp, "w"))
    assert not all(r.ok for r in g.check_zones(m, bp, 4))


def test_counterprobe_flat_core_detected():
    """Kern-Ecken zu nah am Kreis -> kleine Jacobi-Determinante wird gemeldet."""
    out, _ = make()
    m = mc.read_vtk(out)
    pts = m.points.copy()
    conn = np.array(m.cells)
    ok = g.scaled_jacobians(pts, conn).min()
    # Kernquadrat-Knoten (|x|,|y| < 14) in x um 90 % stauchen -> flache Zellen
    mask = (np.abs(pts[:, 0]) < 14) & (np.abs(pts[:, 1]) < 14)
    pts[mask, 0] *= 0.1
    assert g.scaled_jacobians(pts, conn).min() < 0.5 <= ok


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok             {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FEHLGESCHLAGEN {name}: {e!r}")
    print(f"{len(tests) - fails} von {len(tests)} Tests bestanden")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
