# Phase 1a: Formulierung der Kopplung in Maxwell-Spuren (Bericht)

Stand: 09.11.2026. Gerechnet auf NVIDIA RTX 6000 Ada (sm_89, 48 GB), 32 CPU-Kerne, Linux; die Skripte laufen auf der CPU (Python 3.13, numpy, scipy). Basis: `main` 4ca78be, Zweig
`maxwell-trace-formulation`. Ergebnis: `docs/theory/maxwell-trace-coupling.md`; Prüfskripte `tools/theory/`. **Am Lösungspfad, an Operatoren und an der
Eichung wurde nichts geändert**; kein C++/CUDA-Code geändert.

## Ergebnis in Kürze

Das Formulierungsdokument ist vollständig bis auf die unten genannten offenen Punkte. Die Kopplung wird **symmetrisch** in den Unbekannten (a, ψ, c) formuliert (Hiptmair/Costabel-Typ, Abschnitt 5.2 der
Theorie); ψ ist Q1 auf den Randquads, c sind die 2g Koeffizienten der harmonischen Oberflächenfelder. Alle Randoperatoren sind skalare Laplace-Operatoren (V₀, D̃, V_rot, daraus W = Gᵀ V_rot G),
dazu eine Doppelschichtspalte je harmonischem Feld. Der Fall 4 aus `bem-normal-components.md` (nichtharmonisches ψ_h) ist **per Konstruktion** kein Problem, weil
die Blöcke a nur über R a und die harmonischen Zirkulationen sehen und R G = 0 kombinatorisch exakt gilt.

## Gemessen (Skripte, `tools/theory/`)

Alle Skripte laufen in wenigen Sekunden (außer `check_scalar_reduction.py`, ~5 s) und enden mit `ERGEBNIS: OK`. Jede Identität hat eine Gegenprobe.

| Identität | Skript | Ergebnis |
|---|---|---|
| Darstellungsformel A und B, Eichverschiebung, innen = 0 | `check_representation.py` | ≤ 4·10⁻¹⁵ (Kugel, Ellipsoid) |
| Calderón-Zeilen (R-T), (R-N), Hauptwert | `check_calderon_rows.py` | ≤ 1,3·10⁻¹² |
| Potentialidentität, Nédélec-Identität W, K' auf dem Fluss | `check_scalar_reduction.py` | 5·10⁻¹⁴ / 3·10⁻¹⁰ / 3·10⁻¹⁴ / 1·10⁻¹³ |
| Adjungiertheit der Maxwell-Doppelschicht | `check_adjointness.py` | 2,7·10⁻¹¹ |
| Topologie: R G = 0, dim ℋ = 2g, Zirkulationsmatrix, RT-Zählung | `check_surface_topology.py` | alle Fälle (Würfel, Torus, 2 Tori, Quader + Torus) |
| Ende-zu-Ende-Vorzeichen, Eigenwerte, Kugel mit Azimutalstrom | `check_sphere_coupling.py` | Maschinengenauigkeit |
| Volumenstrom-Biot-Savart, Rechteckring | `check_volume_biot_savart.py` | Achse 9·10⁻¹³; Selbstkonvergenz ≤ 4,6·10⁻⁸ |

Weitere Messwerte: Fadenmodell (0,5 mm) weicht im |B| am Spulenrand um 0,1 bis 0,5 % vom Volumenstrom ab (3 mm: 0,7 bis 3 %). Das ist **nicht** die im Plan genannte Abweichung von 24 bis 69 %; welche
Größe diese Zahl beschreibt, wurde hier nicht geklärt (offen, siehe unten). Die Literaturangaben sind gegen Verlagsseiten geprüft; die DOI der Hiptmair-Arbeit, die ich zunächst aus dem Gedächtnis
angesetzt hatte, war falsch und ist korrigiert.

## Getroffene Wahlen

1. **λ-Raum:** Λ_h = ker(div_Γ) auf RT niedrigster Ordnung = curl_ΓQ1 ⊕ ℋ_h; Basis n×W_e (gedrehte Rooftops), also keine neue Basisfunktion.
2. **Symmetrische Kopplung** (S) statt Johnson–Nédélec (J); (J) bleibt als Rückfall (ohne V₀, unsymmetrisch).
3. **Galerkin für alle Blöcke**, keine Kollokation.
4. **Hypersingulär nur als W = Gᵀ V_rot G**, nie als eigener Integrand.
5. **Konstanten in ψ:** je Randkomponente ein Knoten gestrichen; sie sind kein physikalischer Freiheitsgrad.
6. **Eichung:** nur im σ = 0-Gebiet des FEM-Teils (Baum-Kobaum); Kohomologie-Eichung der Spulen soll durch die harmonische Zeile (S3) ersetzt werden (Behauptung, Test T7).
7. **Referenzfeld für Phase 2:** Rechteckring aus Kreisschleifen mit elliptischen Integralen, Produkt-Gauss auf zum Aufpunkt verdichteten Panels.

## Interpretation (nicht geprüft)

* Dass (S) auf realen Netzen ein gut konditioniertes Schurkomplement hat und der Nullraum nach Baum-Kobaum und Streichen je eines ψ-Knotens leer ist.
* Dass (S3) die Kohomologie-Eichung der Spulen ersetzt (es ist hergeleitet, aber diskret nicht gemessen).
* Dass c_Längs bei TEAM 28 verschwindet (nur wenn kein Körper durch das Loch geht, sonst Unbekannte) und c_Meridian = N·I ist; das Letztere hängt an diskret divergenzfreiem J_s.
* Übertragbarkeit der Stabilität von Hiptmair 2002 (E-basiert) auf die A-Formulierung.

## Offene Punkte

1. **Diskrete Lösbarkeit/Stabilität von (S)** (Phase 1b, Tests T6, T7, T8).
2. **Zusammenhang mit der Plan-Zahl 24 bis 69 %** für das Fadenmodell: nicht reproduziert; die Größe ist zu klären (vermutlich Tangential-/Normalanteile oder K1/K2, nicht |B|).
3. **Bewegungsterm σ v × curl a** im bewegten Leiter: Randkopplung unabhängig, aber nicht ausgearbeitet.
4. **Eisen mit hohem μ_r:** Verhalten des Beitrags (1/μ₀) B_Rᵀ V₀ B_R gegenüber K (Auslöschung?).
5. **Beweis** der Eindeutigkeit im Kontinuum für (S) mit dem hier benutzten Raum (nicht geführt; nur Verweis auf die Literatur für verwandte Modelle).

## Risiken für Phase 1b

* **R1** Vorzeichen und Normale: drei von vier Fehlern in #23 lagen hier. Gegenmaßnahme: Tests T1 bis T3, T8 jeweils mit Vorzeichen-Gegenprobe; Normale ist die der Quads (`evaluate_quad_normal`).
* **R2** Neue Integranden: Skalarprodukt (n_x×W_e)·(n_y×W_f) auf nicht koplanaren Paaren (Würfelkanten), Q-Kern (Ordnung −2). Nicht von Sauter–Schwab als solche getestet; Test T3 mit Würfel.
* **R3** J_s diskret divergenzfrei: Sonst hängt c_Meridian von der Wahl der Schnittscheibe ab und das Gradientenprojektionsproblem im Baum-Kobaum-System wird inkonsistent.
* **R7** (Phase 1b, Schritt 0) Eisen: der Randterm (1/μ₀)B_RᵀV₀B_R ist bei μ_r ≫ 1 um μ_r **größer** als K im Eisen (nicht kleiner); Skalierung des Vorkonditionierers (TEAM 13).
* **R4'** (Phase 1b, Schritt 0) Kleine Frequenz (0,01 Hz): Im Leiter fixiert nur jωσ die Gradienten; die Randblöcke sehen nur den Fluss und helfen nicht; schlechte Konditionierung im Gradientenraum.
* **R4** Konditionierung: W hat Kern (Konstanten); V₀ hat große Eigenwertspanne; Block-Vorkonditionierer für den Schurkomplement nicht entworfen.
* **R5** Harmonische Felder auf realen Netzen (Spulen mit Ecken): Konstruktion per CG-Projektion wie `CohomologyGauge.cpp`; Randflächen mit nicht-manifoldem Netz sind ein Risiko (TEAM 28 hat keine).
* **R6** Das Abklingverhalten (Coulomb-Eichung außen) ist nur für die Herleitung gebraucht; im Code wird sie nie erzwungen; die Darstellung gilt aber nur für abklingende Felder. Bei Netzen mit Quellen im Außenraum (nicht Teil des Plans) wäre die Formulierung zu erweitern.

## Befehle

```
python3 tools/theory/check_representation.py
python3 tools/theory/check_calderon_rows.py
python3 tools/theory/check_scalar_reduction.py
python3 tools/theory/check_adjointness.py
python3 tools/theory/check_surface_topology.py
python3 tools/theory/check_sphere_coupling.py
python3 tools/theory/check_volume_biot_savart.py
```

## Tests

Es wurde kein Code des Projekts geändert; `ctest -L host` (43/43) und `ctest` (76/76) sind gegenüber dem Stand vor dem Auftrag unverändert (gelaufen am 09.11.2026: 43/43 bzw. 76/76).
