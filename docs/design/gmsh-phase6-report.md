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

### Radial bei axial 10 (`--scale 0,75`)

| radial | 3 | 5 | 7 | 10 |
|---|---|---|---|---|
| nnz(LU) | 9,0e8 | 1,49e9 | 2,35e9 | 4,08e9 |
| F_z | 1,09881 N | 1,04469 N | 1,03822 N | **1,03707 N** |

Radial ist bei Faktor 7 ausgeschöpft (7 → 10: −0,1 %); 5 liegt 0,7 % daneben.
Radial 15 (nnz(LU) 8,4e9) lieferte nach der Zerlegung keine Kraft mehr
(Abbruch, vermutlich Speicher oder Laufzeit).

**Mit ausgeschöpftem Luftgebiet ist F_z = 1,037 N bei `--scale 0,75`, 1,2 %
unter dem Sollwert 1,0497 N.** Die Netzfeinheit senkt die Kraft je Stufe um
etwa 1 % (gemessen an kleineren Luftgebieten); voll konvergiert dürfte F_z
daher bei etwa 1,02–1,03 N liegen, 2–3 % unter dem Sollwert — abgeschätzt, noch
nicht gemessen. Der Sollwert selbst ist nur auf etwa 0,5 % genau (Masse
0,107 kg auf drei Stellen).

### Netzfeinheit am ausgeschöpften Luftgebiet (radial 7 / axial 10)

| `--scale` | 0,75 | 1,0 |
|---|---|---|
| nnz(LU) | 2,35e9 | 6,22e9 |
| F_z | 1,03822 N | **1,02702 N** |

−1,1 % je Stufe, wie zuvor abgeschätzt. Bei `--scale 1.0` liegt F_z **2,2 %
unter dem Sollwert** 1,0497 N. `--scale 1.25` (nnz(LU) über 8e9) lief nicht
durch.

Hochrechnung auf unendliche Feinheit aus diesen zwei Punkten: 0,99 N bei
angenommener Konvergenzordnung 1, 1,01 N bei Ordnung 2. Die Ordnung ist aus
zwei Punkten nicht bestimmbar; dazu fehlt ein dritter.

### Startlage (Unterkante 3,8 mm, radial 7 / axial 10, `--scale 0,75`)

F_z = **3,39048 N** (radial 3 / axial 3: 3,84 N; radial 3 / axial 10:
3,48 N).

### Abweichung als Schwebehöhe

Aus Start- und Gleichgewichtslage folgt bei exponentiellem Abfall der Kraft
eine Abklinglänge von 6,3 mm, also |dF/dz| ≈ 0,16 N/mm bei 11,3 mm. Das
Kraftdefizit von 0,023 N (2,2 %) entspricht damit einer um **etwa 0,14 mm
tieferen Schwebehöhe** (rund 11,16 statt 11,3 mm). Die Abschätzung beruht auf
zwei Punkten und der Annahme exponentiellen Abfalls; eine lineare Verbindung
der beiden Punkte gäbe die doppelte Steigung und die halbe Höhenabweichung.

## Folgerungen

- Axial genügt Faktor 10. **Radial genügt Faktor 3 nicht** (eine frühere
  Fassung dieses Berichts behauptete das; der radiale Fehler war nur vom
  axialen verdeckt). Radial ist bei 7 ausgeschöpft, 5 liegt 0,7 % daneben.
- Die frühere Vorgabe des Generators (3 / 3) überschätzte F_z am Gleichgewicht
  um rund 24 %. **Neue Vorgabe: radial 5, axial 10** (0,7 % vom ausgeschöpften
  Wert, deutlich billiger als 7 / 10).
- Eine Hochrechnung auf ein unendliches Luftgebiet aus 5 / 7 / 10 ergibt etwa
  1,01 bis 1,03 N; die Spanne kommt vor allem von der Rundung auf drei
  Stellen.

## Nächste Messungen

Offen ist die Konvergenzordnung in der Netzfeinheit. `--scale` verfeinert
auch das Luftgebiet, das den Speicher bestimmt; `--scale 1.25` passt nicht
mehr. Günstiger ist, gezielt die Platte zu verfeinern (`--plate-nz`, `--h`)
bei grobem Außenraum, um zu sehen, welcher Teil der Auflösung die −1,1 % je
Stufe trägt.
