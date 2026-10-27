# Gmsh-Vernetzung: Bericht zu Phase 0

Stand: 27.10.2026. Der Prototyp dieser Phase ist inzwischen durch
`tools/gmsh/team28_hex.py` ersetzt (Phase 2); die Messungen unten stammen vom
Prototyp mit Kerngröße 0,3.
Plan und feste Entscheidungen: [gmsh-meshing-plan.md](gmsh-meshing-plan.md).

## Umgebung

Gmsh 4.15.2 aus pip in einer virtuellen Umgebung (Ubuntu 24.04, Python 3.12):

    python3 -m venv ~/venv-gmsh
    ~/venv-gmsh/bin/pip install gmsh numpy
    ~/venv-gmsh/bin/python -c "import gmsh; print(gmsh.__version__)"

Auch im Batch-Betrieb lädt Gmsh die GL-Bibliotheken. Unter Ubuntu fehlte
`libGLU.so.1`; nötig waren `libglu1-mesa`, `libgl1`, `libxrender1`,
`libxcursor1`, `libxft2`, `libxinerama1`, `libfontconfig1`. Unter RHEL heißt
das Paket für `libGLU` meist `mesa-libGLU` (nicht geprüft).

## Ergebnisse

Prototyp mit Vorgaben (12 Zellen je Viertelkreis, Spulen 4 × 8, Platte 10
radial × 3 in z, Kerngröße 0,3):

| Prüfung | Ergebnis |
|---|---|
| Elementtypen | ausschließlich Hexaeder (VTK-Typ 12) |
| Zellen | Außenspule 1536, Innenspule 1536, Platte 1872 — wie berechnet |
| Doppelte Knoten (Naht bei 360°) | 0 |
| Konformität | keine Fläche gehört zu mehr als zwei Zellen |
| Jacobi-Determinante | positiv an allen 8 Ecken aller Zellen |
| Volumen gegen exaktes 48-Eck | Abweichung ≤ 3,4·10⁻¹⁵ relativ, alle drei Körper |
| Min. skalierte Jacobi-Determinante | Außenspule 0,977, Innenspule 0,885, Platte 0,770 |
| Platte | Knoten auf der Achse: Vollscheibe |
| `tools/check_map.py` | besteht |
| Leser des Projekts (`VtkReader`) | liest die Datei ohne Anpassung |
| Kohomologie-Generatoren im Löser | **2** (Standardnetz: 2) |
| Quellterm | divergenzfrei, ‖Gᵀb‖/‖b‖ = 1,1·10⁻¹⁵ |

Vergleich mit dem Standardnetz `data/team28/team28.vtk`:

| | Standardnetz | Gmsh-Prototyp |
|---|---|---|
| Hexaeder | 4805 | 4944 |
| Kanten | 18690 | 18651 |
| Randflächen | 4136 | 3696 |
| Mitte der Platte | Mittelzelle über der Achse, Ecken bei r = 1,84 mm | Knoten auf der Achse |
| Kohomologie-Generatoren | 2 | 2 |
| GMRES nach etwa 1000 Iterationen (1. Zeitschritt, CPU) | rel. Residuum ≈ 0,059 | rel. Residuum ≈ 0,054 |

Der Löser stagniert auf beiden Netzen gleich. Das neue Netz verursacht das
bekannte Konvergenzproblem nicht und behebt es nicht; für Vergleiche ist das
erwünscht. Die dabei ausgegebene Kraft ist bedeutungslos (nicht konvergierter
erster Zeitschritt).

## Erkenntnisse

1. **BEM nur bei reinen Hexaedernetzen.** `src/fem.cu` baut die BEM nur auf,
   wenn `mesh.is_pure_hex()`; schon ein Prisma schaltet sie ab.
2. **Querschnitt bis zur Achse gedreht ergibt Prismen.** Gmsh erzeugt dann
   keine entarteten Hexaeder, sondern einen Prismenkern (Versuch: 24 Prismen,
   168 Hexaeder). Wegen 1 für den Hauptpfad unbrauchbar; daher der
   Hexaederkern.
3. **Loch.** Ein Netz mit Loch (r 5…65) wird von `check_map.py` jetzt
   abgelehnt. Ein Loch ist ausgeschlossen (Entscheidung E1 im Plan).
4. **Gmsh schreibt die Nummer der Physical Group** in `CellEntityIds`, nicht
   die Nummer des geometrischen Volumens (geprüft mit Gruppen 11, 12, 13).
5. **Falle im Geo-Kern:** Netzvorgaben (transfinit, Vierecke) müssen über
   `gmsh.model.geo.mesh.*` gesetzt werden. Über `gmsh.model.mesh.*` gehen sie
   beim nächsten `synchronize()` verloren, und aus Vierecken werden Dreiecke —
   das Ergebnis sind Prismen statt Hexaeder, ohne Fehlermeldung.
6. **Der Geo-Kern dreht nur um weniger als 180°;** viermal 90° ergibt einen
   konformen Ring ohne Naht.
7. **Der kleinste Knotenradius ist kein Lochkriterium.** Im Standardnetz ist er
   1,84 mm, obwohl die Platte eine Vollscheibe ist. Der Kommentar in
   `check_map.py` ist korrigiert.

## Offen für die nächsten Phasen

- Kerngröße und radiale Verteilung so wählen, dass die Ecken des Kerns nicht
  zu stark verzerren; Abstufung zum Plattenrand hin (Eindringtiefe).
- Plattenlage als Parameter (Start- und Gleichgewichtslage).
- Gmsh auf dem Server prüfen (Befehle oben).
- Prüfungen aus diesem Bericht als Werkzeug (Phase 1) und als Test (Phase 5).
