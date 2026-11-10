# Auftrag Phase 1b: Operatorbibliothek der Maxwell-Spur-Kopplung (dicht, Host) mit Tests T1 bis T9

Stand: 10.11.2026. Basis: `main` nach diesem Auftrag. Zweig: `maxwell-trace-operators`, ein Commit je
Schritt, nach jedem Commit pushen. Bericht: `docs/design/maxwell-trace-operators.md`.

## Anlass und Grundlage

Phase 1a ist abgeschlossen: `docs/theory/maxwell-trace-coupling.md` (im Folgenden „Theorie“) und
`docs/design/maxwell-trace-formulation.md`. Gewählt sind die symmetrische Kopplung (S1) bis (S3) in
(a, ψ, c), Galerkin für alle Blöcke und W = Gᵀ V_rot G. Fahrplan: `docs/design/maxwell-trace-coupling-plan.md`.

Diese Phase baut die Randoperatoren und das gekoppelte Blocksystem **dicht, in Double, auf dem Host** und
prüft sie diskret mit dem Testplan aus Abschnitt 7 der Theorie, ergänzt um die Punkte unten.

**Der bestehende Lösungspfad bleibt unverändert und bitgleich.** Der neue Code lebt in eigenen Dateien
(Vorschlag: `include/TraceBem*.h`, `src/TraceBem*.cpp`) und wird nur von den neuen Tests benutzt.

## Schritt 0: Korrekturen an der Theorie (zuerst, eigener Commit)

Beim Gegenlesen sind vier Punkte aufgefallen. In der Theorie korrigieren oder ergänzen, jeweils mit
kurzer Begründung:

1. **Abschnitt 4.2, Längs-Zirkulation.** Die Aussage „die Luftseite erzwingt c_Längs = 0“ gilt nur,
   wenn der Längskreis eine Scheibe im Luftraum berandet. Geht ein anderer Körper durch das Loch
   (Eisenkern durch Spule, leitender Kern mit Wirbelströmen, Joch), ist c_Längs der Strom durch diesen
   Körper. Bei TEAM 28 ist c_Längs = 0, weil alle Ströme azimutal sind.
   Ausdrücklich festhalten: c_Längs ist eine Unbekannte und wird im Code **nie** auf null gesetzt.
2. **Abschnitt 5.9 (iii), Eisen.** Das Verhältnis ist umgekehrt. K skaliert im Eisen mit 1/(μ₀μ_r),
   der Randterm (1/μ₀) B_Rᵀ V₀ B_R mit 1/μ₀. Bei μ_r ≫ 1 ist der Randterm also um den Faktor μ_r
   **größer**. In die Risiken aufnehmen (Skalierung des Vorkonditionierers, TEAM 13).
3. **Abschnitt 5.6 und T4, Eichinvarianz von Q.** Exakt (kombinatorisch) eichinvariant sind die Blöcke,
   die a über B_R a sehen. Im Block Q bzw. Qᵀ, ⟨h_l, (½ − 𝒞n×)a_t⟩, verschwindet der ½-Anteil auf
   Gradienten exakt (H_l ∈ ker R, also ⟨n×u_H, ∇_Γψ⟩ = ⟨curl_Γ u_H, ψ⟩ = 0). Der 𝒞-Anteil verschwindet
   dagegen nur über eine Identität, die nach Quadratur bloß näherungsweise gilt. Behauptung und
   Testschwelle entsprechend aufteilen.
4. **Risiko R4, kleine Frequenz.** Im Leiter wird nicht geeicht; die Gradienten fixiert nur jωσ. Bei
   kleiner Frequenz (0,01 Hz) wird das System deshalb schlecht konditioniert, und die Randblöcke helfen
   nicht, weil sie nur den Fluss sehen. Als Risiko für den harmonischen Pfad vermerken.

## Schritt 1: Oberflächenstrukturen

Aus einem Hexaedernetz (bzw. Testflächen aus Quads) aufbauen:
- Randquads, Randknoten, Randkanten mit Orientierung (`boundary_edge_signs`, `BemCouplingMap`
  wiederverwenden),
- G (Knoten → Kante), R (Kante → Quad), B_R (in alle FEM-Kanten eingebettet),
- Zusammenhangskomponenten Γ_j und ihr Geschlecht g_j,
- die harmonischen Felder H_k (Theorie, Abschnitt 4.1; an `CohomologyGauge.cpp` angelehnt).

Tests (T9): R G = 0 exakt; dim ker R − dim range G = 2Σg; Euler-Charakteristik; harmonische Felder in
ker R und orthogonal zu range G; Zirkulationsmatrix regulär. Netze: Würfel, Torus, zwei Tori,
Quader + Torus, dazu die Randfläche von `data/team28/team28.vtk` (erwartet: g = 0, 1, 1).

## Schritt 2: Randoperatoren, dicht

Die Matrizen V₀ (P0×P0), D̃ (P0×Q1), M̃, V_rot (E×E, mit (n_x×W_e)·(n_y×W_f)), daraus W = Gᵀ V_rot G,
Z = Gᵀ V_rot H, Y = Hᵀ V_rot H, und Q (Spalten je harmonischem Feld). Alle Galerkin.

- **Quadratur:** den vorhandenen Sauter-Schwab-Rahmen nutzen (`SauterSchwab.cu`, `SingularQuadPair.cuh`,
  `AdjacentIntegration.cu`; die Integratoren sind `__host__ __device__`). Neu sind die Integranden aus
  Theorie, Abschnitt 6.1: das Skalarprodukt mit n_x, n_y, der Q-Kern, P0- und Q1-Formfunktionen.
- Für jeden Block die Quadraturordnung und den erreichten Quadraturfehler angeben (Selbstkonvergenz).

Tests T1, T2, T3 aus der Theorie (Kugel, Würfel, Dipolfeld), jeweils mit Konvergenzordnung über
mindestens drei Netze und mit den Gegenproben aus der Tabelle. Bei T3 zusätzlich die Gegenprobe
„V_rot nur mit koplanarem Skalarprodukt“: Sie muss auf dem Würfel an den Kanten abweichen.

## Schritt 3: harmonische Felder und Torus (T5)

Torus mit Kreisstrom auf dem Kernkreis, Felder analytisch (Theorie, Abschnitt 7). Zu messen:
- c_Meridian gegen I,
- c_Längs → 0 mit Verfeinerung (bei diesem Feld; siehe Schritt 0, Punkt 1),
- Residuen (S2)/(S3) mit exakten Spuren, mit Konvergenzordnung.

Gegenprobe: c_Meridian := 0 muss ein Residuum O(1) ergeben.

## Schritt 4: gekoppeltes System auf kleinen Netzen (T4, T6, T7, T8)

Das Blocksystem aus Theorie, Abschnitt 5.2, dicht assemblieren. Den FEM-Block K über die vorhandene
Assemblierung bilden, Baum-Kobaum-Eichung im σ = 0-Gebiet wie in Abschnitt 5.5. Dazu die Variante (J)
als Vergleich.

- **T4, Eichtest:** a → a + Gψ_h mit nicht harmonischem ψ_h. Die Flussblöcke müssen exakt
  (≤ 1e-12 relativ) invariant sein, Q/Qᵀ auf Quadraturniveau (Schwelle aus der Selbstkonvergenz von
  Schritt 2 ableiten und angeben). Gegenprobe wie in der Theorie.
- **T6:** Symmetrie von (S), Unsymmetrie von (J).
- **T7:** Nullraum per SVD (Spule als Torus mit σ = 0 plus Platte mit σ > 0, K = 2, g = 1), mit den
  Gegenproben aus der Theorie. Ausdrücklich prüfen, ob die Kohomologie-Eichung der Spule durch (S3)
  ersetzt ist (Theorie, Abschnitt 5.5): Mit Baum-Kobaum allein und ohne Kohomologie darf kein Nullvektor
  bleiben.
- **T8, Ende zu Ende:** Kugel mit Azimutalstrom, FEM-Kugel mit BEM-Außenraum, Lösung gegen die exakte
  (ψ, g) mit Konvergenzordnung. Ein Hexaeder-Kugelnetz ist dafür nötig. Falls keines verfügbar ist,
  einen Würfel mit demselben Strom verwenden und gegen den Volumenstrom-Biot-Savart bzw. eine
  feine Referenzrechnung prüfen; die Wahl begründen.
- **Diskret divergenzfreie Quelle (R3):** Für die rechte Seite f im σ = 0-Gebiet Gᵀf = 0 prüfen und
  ausgeben. Die vorhandene Gradientenprojektion des Quellterms wiederverwenden (siehe
  `--fem-direct-no-project` in `Backend.cu` und ihre Umsetzung). Zeigen, dass c_Meridian ohne
  Projektion von der Wahl der Schnittscheibe abhängt und mit Projektion nicht.

## Schritt 5: Konditionierung (nur messen)

Für die Netze aus Schritt 4 und das gröbste Netz der TEAM-28-Randfläche, soweit dicht machbar:
- Konditionszahl bzw. Eigenwertspanne der Blöcke K, W, V₀ und des Schurkomplements auf (a_Γ, ψ, c),
- je für μ_r = 1 und μ_r = 1000 in einem Testkörper (Schritt 0, Punkt 2),
- und für ω von 0,01 Hz bis 50 Hz im Leiter (Schritt 0, Punkt 4).

Keine Gegenmaßnahmen umsetzen, nur messen und einordnen.

## Nicht Teil dieses Auftrags

- ACA, GPU-Apply, Vorkonditionierer, iterative Löser für große Systeme.
- Einbau in den Lösungspfad, TEAM-28-Rechnungen mit der neuen Kopplung (Phase 2 und 3).
- Der transiente Pfad.
- Änderungen am alten BEM-FEM-Pfad oder am Referenzpfad.

## Form

- Bericht wie bisher: Hardware und Basis-Commit im Kopf, „Ergebnis in Kürze“, „Gemessen“ klar getrennt
  von „Interpretation (nicht geprüft)“, Befehle am Ende.
- Alle Tests als ctest mit Marke host; Laufzeit je Test unter 120 s, sonst begründen.
- Testzählung vorher und nachher (`ctest -L host`, vollständiges `ctest`). Abweichungen vom Auftrag
  begründen.
