#!/usr/bin/env python3
"""
Tests fuer tools/strip_air.py und die Randpruefungen in tools/meshcheck.py.

    python3 tools/tests/test_strip_air.py        # oder: pytest tools/tests

Netz: team28_hex_air.py --scale 0.5. Gegenproben: nur Zone 11 (Kaverne), zwei
Wuerfel mit gemeinsamer Kante und eine Diagonalzelle (nicht mannigfaltig), ein
Netz mit umgedrehter Zelle (Orientierung).
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
import strip_air as sa  # noqa: E402

# AP 8: Stufe -> entfernte Zonen (S = 4 Schalengruppen)
STAGES = {
    "B1": "4", "B2": "3-4", "B3": "2-4", "B_S": "1-4",
    "B_rille": "1-4,12", "B_kern": "1-4,11,12", "B_platte": "1-4,11,12,13",
    "B_end": "1-4,10-13",
}
_FULL = {}


def full():
    if not _FULL:
        d = tempfile.mkdtemp(prefix="strip_")
        out = os.path.join(d, "full.vtk")
        p = subprocess.run([sys.executable, os.path.join(TOOLS, "gmsh", "team28_hex_air.py"),
                            "--scale", "0.5", "--plate-z0", "11.3", "-o", out],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stdout
        _FULL.update(dir=d, path=out, mesh=mc.read_vtk(out),
                     zone=mc.read_zones(os.path.splitext(out)[0] + ".zones.json"))
    return _FULL


def hex_box(n, comp=1):
    """Strukturierter Quader n x n x n aus Einheitswuerfeln; Zellreihenfolge i + n*(j + n*k)."""
    idx = lambda i, j, k: i + (n + 1) * (j + (n + 1) * k)
    pts = np.array([(i, j, k) for k in range(n + 1) for j in range(n + 1)
                    for i in range(n + 1)], dtype=float)
    cells = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                cells.append(np.array([idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k),
                                       idx(i, j + 1, k), idx(i, j, k + 1), idx(i + 1, j, k + 1),
                                       idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)]))
    return mc.Mesh(pts, cells, np.full(len(cells), mc.VTK_HEX), np.full(len(cells), comp))


def subset(mesh, keep):
    conn = np.array([mesh.cells[i] for i in keep])
    used = np.unique(conn)
    ren = np.full(len(mesh.points), -1)
    ren[used] = np.arange(len(used))
    return mc.Mesh(mesh.points[used], [ren[c] for c in conn],
                   np.full(len(conn), mc.VTK_HEX), mesh.comp[keep])


def test_every_stage_passes_all_checks():
    f = full()
    ok, res, info = sa.check_stage(f["mesh"], f["zone"], None, None)
    assert ok, [r for r in res if not r.ok]
    for name, spec in STAGES.items():
        mesh, zone, removed = sa.strip(f["mesh"], f["zone"], sa.parse_zones(spec))
        ok, res, info = sa.check_stage(mesh, zone, f["mesh"], removed)
        assert ok, (name, [r for r in res if not r.ok])
        if name == "B_end":
            assert info["n_components"] == 3
            assert sorted(c["chi"] for c in info["components"]) == [0, 0, 2]
            assert not (zone > 0).any()
            assert any(r.name == "Koerper getrennt" and r.ok for r in res)


def test_conductor_nodes_bitwise_and_order_kept():
    f = full()
    parent = f["mesh"]
    for spec in ("1-4,12", "1-4,10-13"):
        mesh, zone, removed = sa.strip(parent, f["zone"], sa.parse_zones(spec))
        keep = np.flatnonzero(~removed)
        assert len(mesh.cells) == len(keep)
        # je uebrige Zelle: gleiche Koordinaten (bitgleich) in derselben Reihenfolge
        for new_i in (0, len(keep) // 2, len(keep) - 1):
            assert np.array_equal(mesh.points[mesh.cells[new_i]],
                                  parent.points[parent.cells[keep[new_i]]])
        assert np.array_equal(mesh.comp, parent.comp[keep])
        # alle Leiterknoten des Elternnetzes sind enthalten
        cond = np.unique(np.concatenate([parent.cells[i] for i in np.flatnonzero(parent.comp > 0)]))
        have = {tuple(p) for p in mesh.points}
        assert all(tuple(p) in have for p in parent.points[cond])


def test_round_trip_through_files():
    f = full()
    out = os.path.join(f["dir"], "stufe.vtk")
    rc = sa.main([f["path"], "--remove-zones", "1-4,12", "-o", out])
    assert rc == 0
    m = mc.read_vtk(out)
    z = mc.read_zones(os.path.splitext(out)[0] + ".zones.json", len(m.comp))
    assert set(np.unique(z)) == {0, 10, 11, 13}
    # Koordinaten ueberleben das Schreiben bitgleich
    parent = f["mesh"]
    pset = {tuple(p) for p in parent.points}
    assert all(tuple(p) in pset for p in m.points)


def test_counterprobe_zone_11_only_is_cavity():
    f = full()
    mesh, zone, removed = sa.strip(f["mesh"], f["zone"], [11])
    ok, res, info = sa.check_stage(mesh, zone, f["mesh"], removed)
    assert not ok
    failed = {r.name for r in res if not r.ok}
    assert "Keine Kaverne" in failed
    assert any(c["volume"] < 0 for c in info["components"]), "Kavernenrand hat negatives Volumen"
    out = os.path.join(f["dir"], "kaverne.vtk")
    assert sa.main([f["path"], "--remove-zones", "11", "-o", out]) == 1


def test_counterprobe_two_cubes_sharing_an_edge():
    box = hex_box(2)                 # Zellen 0..7, i + 2*(j + 2*k)
    two = subset(box, [0, 3])        # (0,0,0) und (1,1,0): gemeinsame Kante
    res, info = mc.check_surface(two)
    failed = {r.name for r in res if not r.ok}
    assert "Randkanten: je zwei Randflaechen" in failed
    # Gegenprobe der Gegenprobe: zwei Wuerfel mit gemeinsamer FLAECHE bestehen
    ok, _ = mc.check_surface(subset(box, [0, 1]))
    assert all(r.ok for r in ok)


def test_counterprobe_diagonal_cell_left_standing():
    box = hex_box(3)
    zone = np.ones(len(box.cells), dtype=np.int64)
    idx = lambda i, j, k: i + 3 * (j + 3 * k)
    for c in (idx(1, 1, 1), idx(2, 2, 1)):           # Diagonalnachbarn (gemeinsame Kante)
        zone[c] = 5
    mesh, z, removed = sa.strip(box, zone, [1])
    assert len(mesh.cells) == 2
    ok, res, info = sa.check_stage(mesh, z, box, removed)
    assert not ok
    assert any(not r.ok and "Randkanten" in r.name for r in res)


def test_counterprobe_flipped_cell_orientation():
    box = hex_box(2)
    cells = [c.copy() for c in box.cells]
    cells[0] = cells[0][[0, 3, 2, 1, 4, 7, 6, 5]]    # Zelle umgedreht
    bad = mc.Mesh(box.points, cells, box.types, box.comp)
    assert not mc.check_orientation(bad).ok


def test_unknown_zone_and_zone_zero_rejected():
    f = full()
    try:
        sa.strip(f["mesh"], f["zone"], [99])
        raise AssertionError("unbekannte Zone akzeptiert")
    except ValueError:
        pass
    try:
        sa.parse_zones("0,1")
        raise AssertionError("Zone 0 akzeptiert")
    except ValueError:
        pass
    assert sa.parse_zones("1-4,12") == [1, 2, 3, 4, 12]


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
