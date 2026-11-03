#!/usr/bin/env python3
"""
Luftzonen aus einem Netz von tools/gmsh/team28_hex_air.py entfernen.

    python tools/strip_air.py NETZ.vtk --remove-zones 1-4,12 -o stufe.vtk

Liest NETZ.vtk und NETZ.zones.json, loescht die Zellen der genannten Zonen,
entfernt verwaiste Knoten (solver_vtk.compact_nodes) und schreibt stufe.vtk
und stufe.zones.json (die uebrigen Zonen). Die Koordinaten der verbleibenden
Knoten werden bitgleich uebernommen (%.17g laesst Gleitkommazahlen exakt
umlaufen), die Zellreihenfolge der uebrigen Zellen bleibt erhalten. Es wird
nie neu vernetzt: Leiterknoten, Quellterm und Plattenmaske sind in allen
Stufen identisch (docs/design/reference-harmonic-air.md).

Pruefungen nach jedem Strip (Exit-Code 1, die Datei bleibt zur Ansicht stehen):
  * Netzpruefung (nur Hexaeder, Duplikate, Konformitaet, Jacobi > 0)
  * Rand geschlossen und mannigfaltig: jede Randkante in genau zwei
    Randflaechen, um jeden Randknoten ein Faecher (meshcheck.check_surface)
  * keine Kaverne: das Komplement reicht bis ins Unendliche
    (meshcheck.check_cavities, Flutfuellung ueber die geloeschten Zellen
    gegen die Aussenflaeche des Elternnetzes) und kein Randkomponente mit
    negativem Volumen
  * Orientierung der Randflaechen, Summe der Flaechenvektoren null
  * ist keine Luft uebrig (Leiter-only-Netz): zusaetzlich check_separate_bodies
    python tools/strip_air.py NETZ.vtk --check-only      # nur die Pruefungen (volles Netz)

Kennzahlen: Randkomponenten, Randknoten, Randflaechen, Flaeche und
Euler-Charakteristik je Komponente, dichte Algebra in GiB.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "gmsh"))

import meshcheck as mc  # noqa: E402
from solver_vtk import compact_nodes, write_legacy_vtk  # noqa: E402


def parse_zones(spec):
    """'1-4,12' -> [1, 2, 3, 4, 12]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if hi < lo:
                raise ValueError(f"Zonenbereich {part} ist rueckwaerts")
            out += list(range(lo, hi + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError("keine Zonen angegeben")
    if 0 in out:
        raise ValueError("Zone 0 (Leiter) darf nicht entfernt werden")
    return sorted(set(out))


def zones_path(vtk):
    return os.path.splitext(vtk)[0] + ".zones.json"


def strip(parent, zone, remove):
    """Gibt (Netz ohne die Zonen, Zonen der uebrigen Zellen, Maske removed) zurueck."""
    present = set(np.unique(zone).tolist())
    unknown = [z for z in remove if z not in present]
    if unknown:
        raise ValueError(f"Zone(n) {unknown} kommen im Netz nicht vor (vorhanden: "
                         f"{sorted(z for z in present if z)})")
    removed = np.isin(zone, remove)
    keep = np.flatnonzero(~removed)
    conn = np.array([parent.cells[i] for i in keep], dtype=np.int64)
    pts, (conn,) = compact_nodes(parent.points, [conn])
    mesh = mc.Mesh(pts, [c for c in conn], np.full(len(keep), mc.VTK_HEX), parent.comp[keep])
    return mesh, zone[keep], removed


def check_stage(mesh, zone, parent, removed, verbose=True):
    """Alle Pruefungen aus AP 3. Gibt (ok, Ergebnisse, Kennzahlen) zurueck."""
    res = [mc.check_types(mesh, require_hex=True), mc.check_duplicate_nodes(mesh),
           mc.check_orphan_nodes(mesh), mc.check_conformity(mesh), mc.check_orientation(mesh)]
    surf, info = mc.check_surface(mesh)
    res += surf
    if parent is not None:
        res.append(mc.check_cavities(parent, removed))
    if not (zone > 0).any():
        res.append(mc.check_separate_bodies(mesh))
    ok = all(r.ok for r in res)
    return ok, res, info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Luftzonen aus einem Hexaedernetz mit Zonenkennung entfernen.")
    ap.add_argument("mesh", help="NETZ.vtk (NETZ.zones.json daneben)")
    ap.add_argument("--remove-zones", help="z. B. 1-4,12")
    ap.add_argument("-o", "--out", help="Ausgabe-VTK (daneben .zones.json)")
    ap.add_argument("--check-only", action="store_true",
                    help="nichts entfernen, nur die Randpruefungen auf NETZ.vtk (volles Netz)")
    ap.add_argument("--zones", help="Zonendatei, wenn nicht NETZ.zones.json")
    a = ap.parse_args(argv)
    if a.check_only:
        try:
            mesh = mc.read_vtk(a.mesh)
            zp = a.zones or zones_path(a.mesh)
            zone = mc.read_zones(zp, len(mesh.comp)) if os.path.exists(zp) else \
                np.zeros(len(mesh.comp), dtype=np.int64)
        except (ValueError, OSError) as e:
            print(f"FEHLER: {e}")
            return 1
        ok, res, info = check_stage(mesh, zone, None, None)
        print(f"{a.mesh}: {len(mesh.cells)} Zellen, {len(mesh.points)} Knoten (nur Pruefung)")
        mc.print_results(res)
        print("\n".join(mc.surface_lines(info)))
        print("Netz in Ordnung." if ok else "NETZ FEHLERHAFT.")
        return 0 if ok else 1
    if not a.remove_zones or not a.out:
        ap.error("--remove-zones und -o sind noetig (oder --check-only)")
    try:
        remove = parse_zones(a.remove_zones)
        parent = mc.read_vtk(a.mesh)
        zone = mc.read_zones(a.zones or zones_path(a.mesh), len(parent.comp))
        mesh, zone_new, removed = strip(parent, zone, remove)
    except (ValueError, OSError) as e:
        print(f"FEHLER: {e}")
        return 1
    write_legacy_vtk(a.out, mesh.points, [(mc.VTK_HEX, np.array(mesh.cells), mesh.comp)],
                     f"FEMBEM strip_air {os.path.basename(a.mesh)} -zones {a.remove_zones}")
    import json
    with open(zones_path(a.out), "w") as f:
        json.dump({"version": 1, "generator": "tools/strip_air.py",
                   "parent": os.path.basename(a.mesh), "removed_zones": remove,
                   "n_cells": int(len(zone_new)), "zone": [int(z) for z in zone_new]}, f)
    print(f"{a.out}: {len(mesh.cells)} von {len(parent.cells)} Zellen, "
          f"{len(mesh.points)} von {len(parent.points)} Knoten (Zonen {a.remove_zones} entfernt)")
    ok, res, info = check_stage(mesh, zone_new, parent, removed)
    mc.print_results(res)
    print("\n".join(mc.surface_lines(info)))
    print("Stufe in Ordnung." if ok else "STUFE FEHLERHAFT -- nicht verwenden.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
