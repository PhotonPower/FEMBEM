# Gmsh-Vernetzung: Bericht zu Phase 9 (Luftgebiet, zweite Höhe, Ringmodell)

Stand: 31.10.2026. Gerechnet auf einer NVIDIA GeForce RTX 3090 (sm_86, 24 GB),
Windows 11, CUDA 13.4, cuDSS 0.8.0. Programm aus `main` 694b521. Netze aus
`tools/gmsh/team28_fem_air.py`, Lauf `fembem --fem-direct --mesh=…`, wie in
[gmsh-phase8-report.md](gmsh-phase8-report.md).

## Frage

Phase 8 hat gezeigt, dass FEM-direkt hochgerechnet rund 1 % über dem
Ringmodell (`tools/team28_axisymmetric.py`) liegt. Unsicher waren noch das
Luftgebiet und die Frage, ob der Abstand von der Höhe abhängt. Drei
Messungen:

1. Abschneiden radial 5 / 7 / 10 bei 12 Spulenschichten.
2. Startlage 3,8 mm: Basisnetz, kombiniertes Netz und je ein Richtungsschritt.
3. BEM-FEM-Pfad bei 11,3 mm gegen das Ringmodell.

## Ringmodell: Stromfäden der Spulen

Die Spulen sind im Ringmodell Stromfäden nach der Mittelpunktsregel. Sie
konvergieren mit Ordnung 2 (Verhältnis der Schritte 3,9 und 4,0):

| Fäden (innen r × außen r × z) | 11,3 mm | 3,8 mm |
|---|---|---|
| 14 × 8 × 26 (bisherige Vorgabe) | 1,00611 N | 3,28804 N |
| 28 × 16 × 52 (neue Vorgabe) | 1,00665 N | 3,28925 N |
| 56 × 32 × 104 | 1,00679 N | 3,28955 N |
| Grenzwert | **1,00684 N** | **3,28965 N** |

Die bisherige Vorgabe lag 0,07 % zu tief; die neue liegt 2e-4 N darunter.
Am Befund aus Phase 8 ändert das nichts: das Gewicht der Platte (1,0497 N)
wird bei 11,3 mm um 4,1 % verfehlt.

## 1. Luftgebiet: Abschneiden addiert sich

`--plate-z0 11.3 --scale 0.75 --coil-nz 16` (12 Spulenschichten):

| radial | 5 | 7 | 10 |
|---|---|---|---|
| nnz(LU) | 2,14e9 | 3,30e9 | 5,89e9 |
| F_z | 1,00062 N | 0,994254 N | 0,993122 N |

Radial 5 → 10: **−0,749 %**; in Phase 6 mit 4 Spulenschichten −0,729 %. Der
Beitrag des Randes hängt also nicht von der Spulenauflösung ab, und die
Addition gilt auch für das Luftgebiet. Die drei Punkte ergeben Ordnung 5,2
im Abschneidefaktor und einen Grenzwert 0,021 % unter radial 10; radial 5
liegt **0,77 %** über dem ausgeschöpften Wert (Phase 8 setzte 0,7 % ein).
Der Lauf mit 5,9e9 nnz(LU) passte noch in die 24 GB (1081 s).

Hochrechnung bei 11,3 mm mit diesem Wert:

    1,04469 N · (1 − 4,75 % + 0,44 % + 0,72 % + 1,30…1,45 % + 0,2 %) · (1 − 0,77 %)
      ≈ 1,0150 … 1,0166 N

Das liegt **+0,81 … +0,97 %** über dem Ringmodell (1,00684 N).

## 2. Zweite Höhe: 3,8 mm

| Lauf | F_z | gegen Ringmodell (3,28965 N) |
|---|---|---|
| Basis (`--coil-nz 6`, 4 Schichten) | 3,39907 N | +3,33 % |
| kombiniert (Spule 16 / Platte 6 / Spalt 4) | 3,27726 N | −0,38 % |
| Basis, h 6,67 → 5 mm | 3,42158 N | Schritt +0,66 % (11,3 mm: +0,86 %) |
| Basis, radial 5 → 7 | 3,39048 N | Schritt −0,25 % (11,3 mm: −0,62 %) |

Zum Vergleich bei 11,3 mm: Basis +3,76 %, kombiniert +0,02 % gegen das
Ringmodell.

Für 3,8 mm sind die beiden fehlenden Beiträge nur je einen Schritt weit
gemessen. Skaliert man die Grenzwerte aus 11,3 mm mit dem Verhältnis der
Schritte (Netzweite 0,77, Rand 0,41), ergibt sich vom kombinierten Netz aus
3,306 … 3,310 N, also **+0,50 … +0,61 %** über dem Ringmodell. Dabei ist
angenommen, dass die Reste von Spule, Platte und Spalt oberhalb des
kombinierten Netzes sich wie bei 11,3 mm aufheben (dort −0,03 %).

## 3. BEM-FEM: nicht konvergiert

`team28_hex.py --plate-z0 11.3 --scale 1` (4944 Hexaeder),
`fembem --harmonic --fixed-plate`. Nach einer Stunde stand GMRES nach 1540
Iterationen bei einem relativen Residuum von 0,02. Das Programm meldet
selbst, dass der innere (W+T)-Löser seine Toleranz von 1e-4 nicht erreicht
und der äußere deshalb nicht konvergieren wird; das ist die bekannte Grenze
aus [bem-block-preconditioning.md](bem-block-preconditioning.md) (Abschnitt
„Der innere Löser (W+T) = K + ωC“). Beide Läufe (`--scale 1` und `2`) sind
abgebrochen. Ein Vergleich des BEM-FEM-Pfads mit dem Ringmodell setzt einen
besseren Vorkonditionierer für (W+T) voraus.

## Nebenbefund: Meldung der Maxwell-Nullprobe

Bei radial 10 meldete das Programm „die Nullprobe ist klein … dann steht das
Volumenintegral zur Debatte“, obwohl die Nullprobe 0 N um 0,194 N verfehlte
(19,5 % der Kraft). Die Grenze lag bei 20 % der Kraft; der Text im anderen
Zweig behauptete zudem, der Nullwert werde „um mehr als das gesuchte
Ergebnis“ verfehlt, und nannte fest „rund 200 N“. Jetzt gilt die Probe ab 5 %
als nicht tragfähig, und die Meldung nennt den gemessenen Anteil. Auf allen
Netzen dieser Reihe liegt die Maxwell-Kraft bei 0,18 N gegen 1 N (Lorentz);
der Flächenpfad bleibt unbrauchbar.

## Ergebnis

- **Luftgebiet geklärt:** Der Rand trägt −0,77 % bis zum Grenzwert, gemessen
  bei der heutigen Spulenvorgabe. Er erklärt das verbleibende Prozent nicht.
- **FEM-direkt liegt bei beiden Höhen über dem Ringmodell:** +0,8 … +1,0 %
  bei 11,3 mm, +0,5 … +0,6 % bei 3,8 mm. Der Abstand ist klein, gleichsinnig
  und nicht ganz fest.
- **Größte offene Unsicherheit ist die Netzweite in der Ebene.** Sie trägt
  bei 11,3 mm +1,30 … +1,45 % bei, mit einer Ordnung zwischen 2,8 und 4,2 auf
  unstrukturierten Netzen; bei 3,8 mm ist sie nur einen Schritt weit
  gemessen. Ein Fehler von einem halben Prozent in diesem Beitrag reicht für
  den ganzen Abstand.

Mit dem Gewicht der Platte verglichen läge FEM-direkt 3 % zu tief, mit dem
Ringmodell verglichen 0,5 … 1 % zu hoch. Maßgeblich ist das Ringmodell
(Phase 8).

## Nächste Schritte

1. Netzweite in der Ebene an einem Netz mit 12 Spulenschichten und radial 7,
   bei beiden Höhen, mit zwei Schritten (der Speicher reicht bis etwa 5e9
   nnz(LU)).
2. Vorkonditionierer für den inneren (W+T)-Löser des harmonischen BEM-FEM,
   damit der Hauptpfad gegen das Ringmodell geprüft werden kann.
