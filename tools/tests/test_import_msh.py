#!/usr/bin/env python3
"""
Tests fuer tools/gmsh/import_msh.py (Phase 4 des Gmsh-Plans).

    python3 tools/tests/test_import_msh.py

Braucht gmsh und numpy. Endet mit 0, wenn alles bestanden ist.

GEPRUEFT
  * Rundreise: team28_hex.py schreibt dasselbe Netz einmal als .msh und einmal
    direkt als VTK; der Import der .msh muss dieselben Zellen (Mittelpunkte),
    component_ids und Volumina liefern und --bem/--check-map bestehen.
  * Eigene Modelle: Tetraeder mit Luft (--fem-direct), Prismen aus einer
    .geo (wird vernetzt), Zuordnung ueber --map, Gruppe ohne Namen.
  * Prismen-Orientierung: ein von Hand geschriebenes, umgekehrt orientiertes
    Prisma (so wie Gmsh es schreibt) wird gedreht und besteht meshcheck.
    KONTROLLFALL: derselbe Import ohne Drehung muss an meshcheck scheitern;
    sonst koennte der Test den Fehler gar nicht finden.
  * Fehlerfaelle mit Exit-Code 1: zweite Ordnung, Volumen ohne Gruppe, keine
    Gruppe, unbekannter Name, Name aus --map fehlt, verlangte component_id
    fehlt, Pyramide, Volumen in zwei Gruppen, doppelte Knoten (nicht
    verschmolzen, meshcheck meldet sie).
  * Knoten ohne Volumenzelle werden entfernt und umnummeriert.
"""
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "gmsh"))
import meshcheck as mc  # noqa: E402
import import_msh  # noqa: E402
import solver_vtk  # noqa: E402

FAILS = []
TMP = tempfile.mkdtemp(prefix="fembem_import_msh_")


def expect(name, cond):
    print(f"  {'ok' if cond else 'FEHLGESCHLAGEN':14s} {name}")
    if not cond:
        FAILS.append(name)


def tmp(name):
    return os.path.join(TMP, name)


def run(*argv):
    """import_msh.main im selben Prozess; liefert (Exit-Code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = import_msh.main([str(a) for a in argv])
    return rc, out.getvalue(), err.getvalue()


def write(name, text):
    path = tmp(name)
    with open(path, "w", newline="\n") as f:
        f.write(text)
    return path


def msh22(nodes, elements, names):
    """Kleine .msh im Format 2.2 von Hand: names = {(dim, Nummer): Name}."""
    s = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
    if names:
        s += f"$PhysicalNames\n{len(names)}\n"
        s += "".join(f'{d} {t} "{n}"\n' for (d, t), n in names.items())
        s += "$EndPhysicalNames\n"
    s += f"$Nodes\n{len(nodes)}\n"
    s += "".join(f"{i + 1} {x:.17g} {y:.17g} {z:.17g}\n" for i, (x, y, z) in enumerate(nodes))
    s += f"$EndNodes\n$Elements\n{len(elements)}\n"
    # Element: (Gmsh-Typ, physikalische Gruppe, Elementar-Entitaet, Knoten 1-basiert)
    s += "".join(f"{k + 1} {t} 2 {p} {e} " + " ".join(map(str, nn)) + "\n"
                 for k, (t, p, e, nn) in enumerate(elements))
    return s + "$EndElements\n"


PRISM = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)]
# Umgekehrt orientiert in der Konvention des Loesers (0, 2, 1, 3, 5, 4) --
# genau so, wie Gmsh Prismen schreibt (gmsh-meshing-plan.md, Ergebnis Phase 1).
PRISM_REVERSED = [1, 3, 2, 4, 6, 5]


def cell_signature(mesh):
    """Zellmittelpunkte mit component_id, sortiert: unabhaengig von Nummerierung."""
    rows = [tuple(np.round(mesh.points[c].mean(axis=0), 9)) + (int(k),)
            for c, k in zip(mesh.cells, mesh.comp)]
    return sorted(rows)


def test_roundtrip_hex():
    print("Rundreise team28_hex.py: .msh importieren gegen direkt geschriebenes VTK:")
    gen = os.path.join(TOOLS, "gmsh", "team28_hex.py")
    msh, ref = tmp("hex.msh"), tmp("hex_ref.vtk")
    for out, extra in ((msh, ["--no-check"]), (ref, [])):
        r = subprocess.run([sys.executable, gen, "--scale", "0.5", "-o", out] + extra,
                           capture_output=True, text=True)
        expect(f"team28_hex.py -o {os.path.basename(out)} laeuft (Exit {r.returncode})",
               r.returncode == 0)
    vtk = tmp("hex_imp.vtk")
    rc, so, se = run(msh, "-o", vtk, "--bem", "--check-map")
    expect(f"Import mit --bem --check-map besteht (Exit {rc})", rc == 0)
    if rc != 0:
        print(so, se)
        return
    a, b = mc.read_vtk(vtk), mc.read_vtk(ref)
    expect(f"gleiche Zellzahl ({len(a.cells)} / {len(b.cells)})", len(a.cells) == len(b.cells))
    expect("nur Hexaeder", set(a.types.tolist()) == {mc.VTK_HEX})
    expect("gleiche Knotenzahl", len(a.points) == len(b.points))
    expect("gleiche Zellen mit gleicher component_id (Mittelpunkte)",
           cell_signature(a) == cell_signature(b))
    va, vb = mc.volumes(a), mc.volumes(b)
    expect(f"gleiche Volumina je component_id {sorted(va)}",
           set(va) == {1, 2, 3} and all(abs(va[k] - vb[k]) <= 1e-12 * vb[k] for k in vb))
    expect("Hexaeder aus Gmsh nicht gedreht", "Hexaeder 0 von" in so)


def test_tet_with_air():
    print("Eigenes Modell: Tetraeder, Platte in Luft, --fem-direct:")
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("luftkasten")
        occ = gmsh.model.occ
        air = occ.addBox(-2, -2, -2, 4, 4, 4)
        plate = occ.addBox(-0.5, -0.5, -0.1, 1, 1, 0.2)
        out, _ = occ.fragment([(3, air)], [(3, plate)])
        occ.synchronize()
        vols = [t for d, t in out if d == 3]
        # Nach dem Fragmentieren: das kleinere Volumen ist die Platte
        mass = {t: occ.getMass(3, t) for t in vols}
        p = min(mass, key=mass.get)
        gmsh.model.addPhysicalGroup(3, [p], 3, "Scheibe")
        gmsh.model.addPhysicalGroup(3, [t for t in vols if t != p], 1, "Air")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.6)
        gmsh.model.mesh.generate(3)
        gmsh.write(tmp("tet.msh"))
    finally:
        gmsh.finalize()
    rc, so, se = run(tmp("tet.msh"), "--require", "3", "--fem-direct")
    expect("unbekannter Name 'Scheibe' abgelehnt (Exit 1)", rc == 1 and "'Scheibe'" in se)
    rc, so, se = run(tmp("tet.msh"), "--require", "3", "--fem-direct", "--map", "Scheibe=3",
                     "-o", tmp("tet.vtk"))
    expect(f"mit --map Scheibe=3 besteht (Exit {rc})", rc == 0)
    if rc == 0:
        m = mc.read_vtk(tmp("tet.vtk"))
        v = mc.volumes(m)
        expect("nur Tetraeder", set(m.types.tolist()) == {mc.VTK_TET})
        expect(f"Volumen Platte 0.2 ({v.get(3, 0):.12g})", abs(v.get(3, 0) - 0.2) < 1e-12)
        expect(f"Volumen Luft 64 - 0.2 ({v.get(0, 0):.12g}, 'Air' -> 0)",
               abs(v.get(0, 0) - 63.8) < 1e-10)
    rc, so, se = run(tmp("tet.msh"), "--require", "3", "--map", "Scheibe=3", "--map", "Air=2",
                     "-o", tmp("tet2.vtk"))
    expect("--map ueberschreibt die Vorgabe (Air -> 2)",
           rc == 0 and set(mc.volumes(mc.read_vtk(tmp("tet2.vtk")))) == {2, 3})
    rc, so, se = run(tmp("tet.msh"), "--require", "3", "--map", "Scheibe=3", "--no-default-names")
    expect("--no-default-names: 'Air' dann unbekannt (Exit 1)", rc == 1 and "'Air'" in se)
    rc, so, se = run(tmp("tet.msh"), "--require", "3", "--map", "Scheibe=3",
                     "--map", "Scheibbe=3")
    expect("Name aus --map nicht in der Datei (Exit 1)", rc == 1 and "'Scheibbe'" in se)
    rc, so, se = run(tmp("tet.msh"), "--map", "Scheibe=3")
    expect("verlangte component_id 1, 2 fehlt (Vorgabe --require 1,2,3; Exit 1)",
           rc == 1 and "1 (Aussenspule)" in se and "2 (Innenspule)" in se)
    rc, so, se = run(tmp("tet.msh"), "--map", "Scheibe=7")
    expect("component_id 7 abgelehnt (Exit 1)", rc == 1 and "7" in se)


GEO_PRISMS = """// Platte als extrudiertes Dreiecksnetz -> Prismen
SetFactory("Built-in");
Mesh.ElementOrder = %d;
Mesh.MeshSizeMax = 0.4;
Point(1) = {0, 0, 0}; Point(2) = {2, 0, 0}; Point(3) = {2, 1, 0}; Point(4) = {0, 1, 0};
Point(9) = {5, 5, 5};   // Punkt ohne Volumen: sein Knoten muss entfallen
Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 4}; Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4}; Plane Surface(1) = {1};
// Ohne Recombine teilt Gmsh die Prismen in Tetraeder
out[] = Extrude {0, 0, 0.3} { Surface{1}; Layers{3}; Recombine; };
Physical Volume("plate") = {out[1]};
"""


def test_geo_prisms():
    print("Eigene .geo: Prismen, wird mit Gmsh vernetzt:")
    geo = write("prismen.geo", GEO_PRISMS % 1)
    rc, so, se = run(geo, "--require", "3", "-o", tmp("prismen.vtk"))
    expect(f".geo vernetzt und importiert (Exit {rc})", rc == 0)
    if rc == 0:
        m = mc.read_vtk(tmp("prismen.vtk"))
        expect("nur Prismen", set(m.types.tolist()) == {mc.VTK_PRISM})
        expect(f"Volumen 0.6 ({mc.volumes(m)[3]:.12g})", abs(mc.volumes(m)[3] - 0.6) < 1e-12)
        expect("Knoten ohne Volumenzelle entfernt (Punkt (5,5,5) nicht im Netz)",
               not np.any(np.all(np.isclose(m.points, [5, 5, 5]), axis=1))
               and mc.check_orphan_nodes(m).ok)
        expect("alle Prismen positiv orientiert", mc.check_orientation(m).ok)
    print("Fehlerfall zweite Ordnung (.geo mit Mesh.ElementOrder = 2):")
    geo2 = write("prismen2.geo", GEO_PRISMS % 2)
    rc, so, se = run(geo2, "--require", "3")
    expect("abgelehnt (Exit 1, Meldung nennt die Ordnung)", rc == 1 and "Ordnung 2" in se)


def test_prism_orientation():
    print("Prismen-Orientierung (von Hand, umgekehrt wie aus Gmsh):")
    names = {(3, 5): "Platte", (2, 9): "Deckel"}
    # Knoten 7 haengt nur an einem Dreieck einer Flaechengruppe: muss entfallen
    nodes = PRISM + [(9, 9, 9)]
    elems = [(6, 5, 1, PRISM_REVERSED), (2, 9, 2, [4, 5, 7])]
    path = write("prisma.msh", msh22(nodes, elems, names))
    vtk = tmp("prisma.vtk")
    rc, so, se = run(path, "--require", "3", "-o", vtk)
    expect(f"Import besteht (Exit {rc})", rc == 0)
    m = mc.read_vtk(vtk)
    r = mc.check_orientation(m)
    expect(f"Prisma nach dem Import positiv ({r.detail})", r.ok)
    expect("als gedreht gemeldet", "Prismen 1 von 1" in so)
    expect(f"Knoten ohne Volumenzelle entfernt ({len(m.points)} Knoten)", len(m.points) == 6)
    expect(f"Volumen 0.5 ({mc.volumes(m)[3]})", abs(mc.volumes(m)[3] - 0.5) < 1e-15)
    expect("Knotenfolge (0, 2, 1, 3, 5, 4) angewendet: Ergebnis = richtig orientiertes Prisma",
           np.allclose(m.points[m.cells[0]], PRISM))

    print("KONTROLLFALL: derselbe Import OHNE Drehung muss scheitern:")
    original = solver_vtk.orient_to_solver
    solver_vtk.orient_to_solver = lambda pts, t, conn, rel_tol=1e-12: (np.asarray(conn), 0)
    try:
        rc, so, se = run(path, "--require", "3", "-o", tmp("prisma_ohne.vtk"))
    finally:
        solver_vtk.orient_to_solver = original
    caught = rc == 1 and any(l.split()[:4] == ["Jacobi-Determinante", ">", "0", "FEHLER"]
                             for l in so.splitlines())
    print(f"  {'DURCHGEFALLEN (erwartet)' if caught else 'NICHT ERKANNT'}: Exit {rc}")
    expect("ohne Drehung meldet meshcheck das Prisma (Exit 1)", caught)

    print("Gemischt orientierte Tetraeder und Hexaeder werden einzeln gedreht:")
    tet = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    hexa = [(3, 0, 0), (4, 0, 0), (4, 1, 0), (3, 1, 0), (3, 0, 1), (4, 0, 1), (4, 1, 1), (3, 1, 1)]
    # Zellen ohne gemeinsame Knoten, damit nur die Orientierung zaehlt
    nodes = (tet + [(x, y + 5, z) for x, y, z in tet]
             + hexa + [(x, y + 2, z) for x, y, z in hexa])
    elems = [(4, 1, 1, [1, 2, 3, 4]), (4, 1, 1, [5, 7, 6, 8]),           # richtig, gespiegelt
             (5, 2, 2, [9, 10, 11, 12, 13, 14, 15, 16]),                  # richtig
             (5, 2, 2, [21, 22, 23, 24, 17, 18, 19, 20])]                 # oben/unten vertauscht
    path = write("gemischt.msh", msh22(nodes, elems, {(3, 1): "outer", (3, 2): "inner"}))
    rc, so, se = run(path, "--require", "1,2", "-o", tmp("gemischt.vtk"))
    expect(f"Import besteht (Exit {rc})", rc == 0)
    expect("je ein Tetraeder und ein Hexaeder gedreht",
           "Tetraeder 1 von 2" in so and "Hexaeder 1 von 2" in so)
    expect("alle Ecken positiv", mc.check_orientation(mc.read_vtk(tmp("gemischt.vtk"))).ok)


def test_errors():
    print("Fehlerfaelle (Exit 1 mit Meldung):")
    prism = [(6, 5, 1, PRISM_REVERSED)]
    path = write("ohne_gruppe.msh", msh22(PRISM, prism, {}))
    # Ohne $PhysicalNames und mit Gruppe 5 -> Gruppe ohne Namen, Nummer 5
    rc, so, se = run(path, "--require", "3")
    expect("Gruppe ohne Namen nicht zugeordnet", rc == 1 and "Nummer 5" in se)
    rc, so, se = run(path, "--require", "3", "--map", "5=3", "-o", tmp("nr5.vtk"))
    expect("Gruppe ohne Namen ueber die Nummer zugeordnet (--map 5=3)", rc == 0)

    # Physikalische Gruppe 0 heisst in msh 2.2: keine Gruppe
    path = write("keine.msh", msh22(PRISM, [(6, 0, 1, PRISM_REVERSED)], {}))
    rc, so, se = run(path, "--require", "3")
    expect("keine Volumen-Gruppe", rc == 1 and "keine Volumen-Gruppe" in se)

    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        a = occ.addBox(0, 0, 0, 1, 1, 1)
        b = occ.addBox(3, 0, 0, 1, 1, 1)
        occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [a], 3, "plate")
        gmsh.option.setNumber("Mesh.SaveAll", 1)      # auch Elemente ohne Gruppe schreiben
        gmsh.model.mesh.generate(3)
        gmsh.write(tmp("luft_vergessen.msh"))
        gmsh.model.addPhysicalGroup(3, [a, b], 1, "outer_coil")
        gmsh.write(tmp("zwei_gruppen.msh"))
    finally:
        gmsh.finalize()
    rc, so, se = run(tmp("luft_vergessen.msh"), "--require", "3")
    expect("Volumenelemente ohne Gruppe", rc == 1 and "ohne Gruppe" in se)
    rc, so, se = run(tmp("zwei_gruppen.msh"), "--require", "3")
    expect("Volumen in zwei Gruppen mit verschiedener component_id",
           rc == 1 and "liegt in den Gruppen" in se)

    pyr = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 1)]
    path = write("pyramide.msh", msh22(pyr, [(7, 3, 1, [1, 2, 3, 4, 5])], {(3, 3): "plate"}))
    rc, so, se = run(path, "--require", "3")
    expect("Pyramide abgelehnt", rc == 1 and "nur" in se and "Prismen" in se)

    tet10 = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (.5, 0, 0), (.5, .5, 0),
             (0, .5, 0), (0, 0, .5), (0, .5, .5), (.5, 0, .5)]
    path = write("tet10.msh", msh22(tet10, [(11, 3, 1, list(range(1, 11)))], {(3, 3): "plate"}))
    rc, so, se = run(path, "--require", "3")
    expect("Tetraeder zweiter Ordnung (.msh) abgelehnt", rc == 1 and "Ordnung 2" in se)

    print("Doppelte Knoten werden nicht verschmolzen, meshcheck meldet sie:")
    hexa = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    nodes = hexa + [(x + 1, y, z) for x, y, z in hexa]       # gemeinsame Flaeche x = 1, doppelt
    elems = [(5, 3, 1, list(range(1, 9))), (5, 3, 2, list(range(9, 17)))]
    path = write("doppelt.msh", msh22(nodes, elems, {(3, 3): "plate"}))
    rc, so, se = run(path, "--require", "3", "-o", tmp("doppelt.vtk"))
    m = mc.read_vtk(tmp("doppelt.vtk"))
    expect(f"Exit 1, alle 16 Knoten erhalten ({len(m.points)})", rc == 1 and len(m.points) == 16)
    expect("meshcheck: 4 doppelte Knoten", "Doppelte Knoten" in so and not mc.check_duplicate_nodes(m).ok)

    rc, so, se = run(tmp("fehlt.msh"))
    expect("fehlende Datei", rc == 1 and "nicht gefunden" in se)


def main():
    try:
        test_roundtrip_hex()
        test_tet_with_air()
        test_geo_prisms()
        test_prism_orientation()
        test_errors()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print()
    print("Alle Tests bestanden." if not FAILS else f"{len(FAILS)} Test(s) fehlgeschlagen.")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
