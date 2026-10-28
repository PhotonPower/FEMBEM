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
Mit dem Gmsh-Netz hängt F_z kaum von der Netzfeinheit ab (rund 1 % je
Verfeinerungsstufe); der verbleibende Fehler kommt aus dem abgeschnittenen
Luftgebiet, **axial und radial**. Axial genügt Faktor 10; radial bleibt bei
Faktor 3 noch rund +5 %. Bei Abschneidefaktor 10 in beiden Richtungen liegt
F_z innerhalb von 1 % am Sollwert. Mit den BDF-Netzen (Platte mit Loch, siehe Korrekturhinweis in
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

Bei axial 3 bestimmt der **axiale Abstand** das Ergebnis; der radiale scheint
ab Faktor 3 kaum zu zählen. Das täuscht, siehe die nächste Reihe: der viel
größere axiale Fehler überdeckt hier den radialen. Bei Faktor 3 bleiben über
der Platte nur rund 60 mm Luft, seitlich der Außenspule 190 mm. Die Randbedingung A × n = 0 hält den Fluss im Luftgebiet
fest, verdichtet das Rückfeld an der Platte und erhöht die Kraft; deshalb
nähert sich F_z dem Sollwert von oben.

### Axial bei radial 3 (`--scale 0,75`, Ausgabe mit sechs Stellen)

| axial | 5 | 7 | 10 | 15 |
|---|---|---|---|---|
| nnz(LU) | 6,5e8 | 8,1e8 | 9,0e8 | 9,6e8 |
| F_z | 1,13214 N | 1,10438 N | 1,09881 N | 1,09792 N |

Axial ist bei Faktor 10 ausgeschöpft (10 → 15: −0,08 %). F_z bleibt aber bei
**1,099 N** stehen, während radial = axial = 10 (Tabelle oben) 1,04 N ergab:
der **radiale** Abstand macht bei Faktor 3 noch rund 5 % aus. Er war in der
vorigen Reihe nur vom axialen Fehler verdeckt.

### Netzfeinheit am großen Luftgebiet (radial 3 / axial 10)

| `--scale` | 0,75 | 1,0 |
|---|---|---|
| nnz(LU) | 9,0e8 | 2,42e9 |
| F_z | 1,09881 N | 1,08688 N |

−1,1 % je Stufe, wie bei kleinem Luftgebiet (−0,8 %): kleiner, aber nicht
verschwundener Beitrag.

### Startlage (Unterkante 3,8 mm, radial 3 / axial 10, `--scale 0,75`)

F_z = **3,48226 N** (mit radial 3 / axial 3: 3,84 N). Wegen radial 3 noch um
einige Prozent zu hoch. Nebenbei: Startlage und radial 3 / axial 7 haben
dieselbe nnz(LU), weil beide Netze gleich viele Luftschichten haben (je 10;
die Schichten wachsen nach außen geometrisch).

## Folgerungen

- Axial genügt Faktor 10. **Radial genügt Faktor 3 nicht** (eine frühere
  Fassung dieses Berichts behauptete das; der radiale Fehler war nur vom
  axialen verdeckt). Wie viel radial nötig ist, zeigt die nächste Reihe.
- Die Vorgabe des Generators (3 / 3) überschätzt F_z am Gleichgewicht um rund
  24 %. Sie sollte axial 10 werden, radial nach der nächsten Reihe.
- Eine Hochrechnung auf ein unendliches Luftgebiet aus 5 / 7 / 10 ergibt etwa
  1,01 bis 1,03 N; die Spanne kommt vor allem von der Rundung auf drei
  Stellen.

## Nächste Messungen

`--plate-z0 11.3 --scale 0.75`, axial 10:

- radial 5, 7, 10, 15 — wie viel radial ist nötig? (radial 10 / axial 10
  zur Kontrolle neu mit sechs Stellen)
- danach das gefundene Luftgebiet mit `--scale 1.0` und `1.25`, um den
  Beitrag der Netzfeinheit abzuschätzen, und die Startlage.
