# Gmsh-Vernetzung: Bericht zu Phase 6 (FEM-direkt mit Vollscheibe)

Stand: 28.10.2026. Gerechnet auf einem Server mit NVIDIA RTX 6000 Ada (sm_89),
CUDA 13.1, cuDSS 0.8.0, Branch `feature/gmsh-meshing`. Netze aus
`tools/gmsh/team28_fem_air.py`, Lauf `./fembem --fem-direct`
(ε = 1e-8, 50 Hz, Quelle auf der Amplitude). Plan:
[gmsh-meshing-plan.md](gmsh-meshing-plan.md).

Sollwert im Schwebegleichgewicht (Unterkante 11,3 mm): Gewicht der Platte,
0,107 kg · 9,81 = **1,0497 N**. Die Platte ist eine Vollscheibe (TEAM 28 hat
kein Loch).

## Ergebnis

**Der FEM-direkt-Pfad konvergiert am Gleichgewichtspunkt gegen den Sollwert.**
Mit dem Gmsh-Netz hängt F_z kaum von der Netzfeinheit ab; der verbleibende
Fehler kommt fast ganz aus dem abgeschnittenen Luftgebiet, und zwar aus seiner
**axialen** Ausdehnung. Bei Abschneidefaktor 10 liegt F_z innerhalb von 1 % am
Sollwert. Mit den BDF-Netzen (Platte mit Loch, siehe Korrekturhinweis in
[fem-direct-plan.md](fem-direct-plan.md)) schwankte F_z dagegen mit der
Auflösung (0,842 → 1,250 → 1,020 N).

Alle Läufe: relatives Residuum ≤ 4e-14, keine Pivotierungen, Scheinkraft aus
dem regularisierten Luftgebiet ~1e-7 N. Die Kraft ist auf drei Stellen
ausgegeben (Ausgabe inzwischen auf sechs Stellen umgestellt).

### Netzfeinheit (Abschneidefaktor 3)

| `--scale` | 0,75 | 1,0 | 1,25 |
|---|---|---|---|
| nnz(LU) | 5,7e8 | 1,46e9 | 3,27e9 |
| F_z | 1,31 N | 1,30 N | 1,29 N |

Über eine Verdopplung der Elementzahl ändert sich F_z um 1,5 %.

### Größe des Luftgebiets, gemeinsam (radial = axial)

| Abschneidefaktor | 2 | 3 | 5 | 7 | 10 |
|---|---|---|---|---|---|
| `--scale` | 1,0 | 1,0 | 1,0 / 0,75 | 0,75 | 0,75 |
| nnz(LU) | 9,6e8 | 1,46e9 | 3,05e9 / 1,09e9 | 2,05e9 | 4,08e9 |
| F_z | 1,49 N | 1,30 N | 1,10 / 1,11 N | 1,06 N | **1,04 N** |

### Radial gegen axial (`--scale 0,75`)

| radial / axial | 3 / 3 | 5 / 3 | 3 / 5 | 5 / 5 |
|---|---|---|---|---|
| nnz(LU) | 5,7e8 | 9,3e8 | 6,5e8 | 1,09e9 |
| F_z | 1,31 N | 1,30 N | 1,13 N | 1,11 N |

**Der axiale Abstand bestimmt das Ergebnis**, der radiale ab Faktor 3 kaum.
Bei Faktor 3 bleiben über der Platte nur rund 60 mm Luft, seitlich der
Außenspule 190 mm. Die Randbedingung A × n = 0 hält den Fluss im Luftgebiet
fest, verdichtet das Rückfeld an der Platte und erhöht die Kraft; deshalb
nähert sich F_z dem Sollwert von oben.

## Folgerungen

- Für FEM-direkt genügt radial Faktor 3; axial braucht es deutlich mehr. Ein
  Luftgebiet radial 3 / axial groß ist erheblich billiger als ein großes
  gleichmäßiges (3 / 5: nnz(LU) 6,5e8 gegen 1,09e9 bei 5 / 5, gleiche Kraft
  bis auf 2 %).
- Die Vorgabe des Generators (3 / 3) überschätzt F_z am Gleichgewicht um rund
  24 %. Die Vorgabe sollte axial größer werden, sobald die Reihe radial 3 /
  axial 7 und 10 vorliegt.
- Eine Hochrechnung auf ein unendliches Luftgebiet aus 5 / 7 / 10 ergibt etwa
  1,01 bis 1,03 N; die Spanne kommt vor allem von der Rundung auf drei
  Stellen.

## Nächste Messungen

Mit sechs Stellen in der Ausgabe, `--plate-z0 11.3 --scale 0.75`:

- radial 3, axial 7, 10, 15 — trägt radial 3 auch bei großem axialem Abstand?
- `--scale 1.0` bei radial 3 / axial 10 — Netzfeinheit am großen Luftgebiet.
- Startlage (`--plate-z0 3.8`) mit dem gefundenen Luftgebiet.
