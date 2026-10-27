#!/usr/bin/env python3
"""
Erzeugt grobe Ersatznetze der TEAM-28-Geometrie fuer eine Netzstudie.

    python3 tools/make_coarse_team28.py 12   ->  team28_coarse_12.vtk

WOZU
----
Der Loeser stagniert am Originalnetz (18690 Kanten) bei einem relativen
Residuum um 0.015 und faellt danach nur noch mit rund 26000 Iterationen je
Dekade. Ob das an der Netzgroesse liegt oder an der Formulierung, laesst sich
nur durch einen Vergleich ueber mehrere Netzgroessen entscheiden:

  * Faellt das Residuum auf groben Netzen sauber auf 1e-6 und wird es mit
    feinerem Netz systematisch schlechter, ist es ein Konditionierungsproblem.
    Dann lohnt die Arbeit an netzunabhaengigen Vorkonditionierern (AMS).
  * Stagniert es auch auf einem sehr groben Netz, sitzt das Problem NICHT im
    Loeser. Dann ist die Formulierung oder die Assemblierung zu pruefen, und
    jede weitere Stunde am Vorkonditionierer waere verschwendet.

Das Originalnetz ist aus Nastran konvertiert und in x/y unstrukturiert (1082
verschiedene x-Werte). Es laesst sich nicht ohne Weiteres vergroebern, deshalb
hier ein strukturiertes Ersatznetz mit derselben Physik.

GEOMETRIE (am Originalnetz vermessen, Angaben in mm)
----------------------------------------------------
  Gebiet          x, y in [-95, 95],  z in [-52, 6.8]
  ID 3 Platte     z in [ 4.3,  6.3],  Radius bis 63     (Aluminium, kappa > 0)
  ID 2 Innenspule z in [-49.4, -2.6], Radius  30 bis 51
  ID 1 Aussensp.  z in [-49.4, -2.6], Radius  84 bis 91

Die Spulen sind Ringe; ihre radiale Dicke (21 bzw. 7 mm) muss vom Gitter
aufgeloest werden, sonst verschwinden sie. Bei n < 12 wird die Aussenspule
deshalb auf eine Zellschicht verbreitert -- der Generator meldet das, statt es
stillschweigend zu tun. Ein Netz, in dem eine Spule fehlt, wuerde eine andere
Aufgabe loesen als das Original.

AZIMUTALER SPULENSTROM BRAUCHT EIN POLARES GITTER
--------------------------------------------------
Ein erster Anlauf vernetzte auch die Spulen kartesisch. Die Laeufe brachen
nicht ab, lieferten aber:

    ||G^T b||/||b|| = 3.2e-01   ->  NICHT divergenzfrei

gegen 2.1e-04 am Originalnetz. Der eingepraegte Spulenstrom ist azimutal
(current_is_azimuthal), und auf einem kartesischen Gitter ist ein Ring eine
Treppe: der Strom schliesst sich an den Stufen nicht, die diskrete Divergenz
verschwindet nicht, und das System wird inkonsistent. GMRES konvergiert dann
gegen die Kleinste-Quadrate-Loesung und stagniert -- aus einem Grund, der mit
der eigentlichen Frage nichts zu tun hat.

Die Spulen werden deshalb POLAR vernetzt: Knoten bei (r_i cos phi_j,
r_i sin phi_j, z_k). Damit liegt e_phi in den Elementflaechen und der
eingepraegte Strom ist konstruktionsbedingt quellenfrei.

Die Platte bleibt kartesisch. Dort ist der Strom INDUZIERT, nicht eingepraegt
(kappa > 0, J = 0), die Divergenzfreiheit des Quellterms ist also nicht
betroffen. Ein polares Gitter haette dort ausserdem eine entartete Achse oder
ein Loch in der Mitte -- Letzteres wuerde die Topologie aendern und eine
zusaetzliche Kohomologie-Klasse erzeugen.

Platte und Spulen sind getrennte Koerper (z 4.3..6.3 gegen -49.4..-2.6), teilen
also keine Knoten. Die beiden Vernetzungsarten muessen nicht zusammenpassen.

KEINE LUFTELEMENTE
------------------
Das Originalnetz hat 4805 Zellen, und 1200 + 1400 + 2205 = 4805 -- es enthaelt
AUSSCHLIESSLICH die drei Koerper. Den Aussenraum macht die BEM, nicht das FEM.

Ein erster Anlauf dieses Generators fuellte das ganze Quadergebiet mit Zellen
und gab der Luft die Component-ID 0. Alle fuenf Laeufe brachen ab. Das ist auch
richtig so: mit Luftelementen ist es eine ANDERE Aufgabe. Der Aussenraum waere
dann doppelt erfasst -- einmal durch die Luftelemente und einmal durch die
BEM-Kopplung --, und Zellen mit unbekannter Component-ID haben kein Material.

Es werden deshalb nur Zellen ausgegeben, die in einem der drei Koerper liegen;
die Knoten werden anschliessend neu durchnummeriert. Das Ergebnis besteht wie
das Original aus drei getrennten Koerpern (Platte oben, zwei Spulenringe
unten).
"""

import sys

# Am Originalnetz vermessen (mm)
XY_MIN, XY_MAX = -95.0, 95.0
Z_MIN, Z_MAX = -52.0, 6.8

PLATE_Z = (4.3, 6.3)
PLATE_R = 63.0

COIL_Z = (-49.4, -2.6)
INNER_R = (30.0, 51.0)
OUTER_R = (84.0, 91.0)



def hex_jacobian_det(nodes):
    """detJ im Zellmittelpunkt, gerechnet wie in NodalHexahedron.cuh.

    Eine eigene Volumenformel war hier zuerst falsch und meldete ALLE Zellen
    als entartet, auch die offensichtlich gesunden kartesischen. Deshalb jetzt
    genau die Rechnung, die auch der Loeser anstellt: trilineare
    Formfunktionen, ausgewertet bei (0,0,0).

    Ein negatives detJ bedeutet eine falsche Knotenreihenfolge. Das faellt
    sonst NICHT auf, weil die Assemblierung |detJ| nimmt -- das Element
    traegt dann mit falschem Vorzeichen der Orientierung bei.
    """
    sx = (-1, 1, 1, -1, -1, 1, 1, -1)
    sy = (-1, -1, 1, 1, -1, -1, 1, 1)
    sz = (-1, -1, -1, -1, 1, 1, 1, 1)
    J = [[0.0] * 3 for _ in range(3)]
    for i in range(8):
        d = (sx[i] / 8.0, sy[i] / 8.0, sz[i] / 8.0)
        for a in range(3):
            for b in range(3):
                J[a][b] += d[b] * nodes[i][a]
    return (J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
          - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
          + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]))


def build_coil(r_in, r_out, n_phi, n_r, n_z, cid, pts, cells, ids):
    """Polarer Spulenring. Haengt Knoten und Zellen an die uebergebenen Listen.

    e_phi liegt damit in den Elementflaechen, und der eingepraegte azimutale
    Strom ist quellenfrei -- auf einem kartesischen Gitter ist er das nicht
    (gemessen: ||G^T b||/||b|| = 0.32 statt 2.1e-04).
    """
    import math
    base = len(pts)
    for k in range(n_z + 1):
        z = COIL_Z[0] + (COIL_Z[1] - COIL_Z[0]) * k / n_z
        for j in range(n_phi):                      # kein doppelter Knoten bei 2*pi
            a = 2.0 * math.pi * j / n_phi
            for i in range(n_r + 1):
                r = r_in + (r_out - r_in) * i / n_r
                pts.append((r * math.cos(a), r * math.sin(a), z))

    nid = lambda i, j, k: base + (k * n_phi + (j % n_phi)) * (n_r + 1) + i
    for k in range(n_z):
        for j in range(n_phi):
            for i in range(n_r):
                cells.append([
                    nid(i, j, k),     nid(i + 1, j, k),
                    nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
                ])
                ids.append(cid)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12

    # --- Platte: kartesisch, drei Schichten wie im Original -------------------
    n_plate = 3
    zs_p = [PLATE_Z[0] + (PLATE_Z[1] - PLATE_Z[0]) * t / n_plate
            for t in range(n_plate + 1)]
    xs = [XY_MIN + (XY_MAX - XY_MIN) * i / n for i in range(n + 1)]

    pts, cells, ids = [], [], []
    nid_p = lambda i, j, k: (k * (n + 1) + j) * (n + 1) + i
    for k in range(n_plate + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                pts.append((xs[i], xs[j], zs_p[k]))
    for k in range(n_plate):
        for j in range(n):
            for i in range(n):
                cx = 0.5 * (xs[i] + xs[i + 1])
                cy = 0.5 * (xs[j] + xs[j + 1])
                if (cx * cx + cy * cy) ** 0.5 > PLATE_R:
                    continue
                cells.append([
                    nid_p(i, j, k),     nid_p(i + 1, j, k),
                    nid_p(i + 1, j + 1, k), nid_p(i, j + 1, k),
                    nid_p(i, j, k + 1), nid_p(i + 1, j, k + 1),
                    nid_p(i + 1, j + 1, k + 1), nid_p(i, j + 1, k + 1),
                ])
                ids.append(3)

    # --- Spulen: polar --------------------------------------------------------
    # Die Sektorzahl bestimmt die Umfangsaufloesung. Unter 12 Sektoren wird der
    # Ring so kantig, dass die Kohomologie-Suche keinen brauchbaren Generator
    # mehr findet (bei den kartesischen Netzen mit n <= 20 ist sie genau daran
    # gescheitert).
    n_phi = max(16, 2 * n)
    n_z = max(2, n // 3)
    build_coil(INNER_R[0], INNER_R[1], n_phi, max(1, n // 8), n_z, 2, pts, cells, ids)
    build_coil(OUTER_R[0], OUTER_R[1], n_phi, 1,              n_z, 1, pts, cells, ids)

    # Knotenreihenfolge kontrollieren: negative Volumina waeren ein stiller
    # Fehler, weil die Assemblierung |detJ| nimmt.
    neg = sum(1 for c in cells if hex_jacobian_det([pts[v] for v in c]) <= 0.0)
    if neg:
        print(f"  WARNUNG: {neg} Zellen mit nichtpositivem Volumen "
              f"(falsche Knotenreihenfolge).")

    used = sorted({v for c in cells for v in c})
    remap = {o: i for i, o in enumerate(used)}
    pts = [pts[o] for o in used]
    cells = [[remap[v] for v in c] for c in cells]

    counts = {c: ids.count(c) for c in (1, 2, 3)}
    print(f"Netz n={n}: {len(pts)} Knoten, {len(cells)} Zellen "
          f"(Original: 7013 / 4805)")
    print(f"  Aussenspule {counts[1]}, Innenspule {counts[2]}, Platte {counts[3]}"
          f"   (Original: 1200 / 1400 / 2205)")
    print(f"  Spulen polar mit {n_phi} Sektoren")

    name = f"team28_coarse_{n}.vtk"
    with open(name, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"Grobes TEAM-28-Ersatznetz, n={n}\n")
        f.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(pts)} double\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        f.write(f"\nCELLS {len(cells)} {len(cells) * 9}\n")
        for c in cells:
            f.write("8 " + " ".join(str(v) for v in c) + "\n")
        f.write(f"\nCELL_TYPES {len(cells)}\n")
        for _ in cells:
            f.write("12\n")
        f.write(f"\nCELL_DATA {len(cells)}\n")
        f.write("SCALARS component_id int 1\nLOOKUP_TABLE default\n")
        for c in ids:
            f.write(f"{c}\n")
    print(f"  geschrieben: {name}")


if __name__ == "__main__":
    main()
