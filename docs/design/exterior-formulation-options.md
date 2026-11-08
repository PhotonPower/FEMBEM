# Außenraumformulierung: Optionen für eine eichinvariante FEM-BEM-Kopplung

Stand: 08.11.2026. Entwurfsskizze zur Entscheidung, kein Auftrag. Grundlage: `bem-trace-consistency.md` und
`bem-normal-components.md`. Code-Stand `main` 3db24f2 und Zweig `bem-normal-components`.

## 1. Ausgangslage

Die Diagnose der letzten vier Berichte ergibt zwei Defekte der bestehenden Kopplung. Beide sind
Formulierungsfehler, keine Umsetzungsfehler.

1. **Eine Unbekannte k für zwei verschiedene Randgrößen.** Zeile 1 (curl-curl-FEM) braucht
   μ₀K2 = −n × B. Zeile 2 (komponentenweise Vektor-Laplace-BEM) braucht μ₀K1 = (∂A/∂n)_tan. Beide
   unterscheiden sich um −∇_Γ(A·n) + W·A_tan, auf den Spulenmänteln um rund 25 %. Das gilt auch für das
   exakte Feld und hängt nicht von der Eichung ab.
2. **Eichabhängigkeit.** Die komponentenweise Darstellungsformel gilt nur für alle drei Komponenten
   eines Felds in Coulomb-Klasse. Zeile 2 sieht nur Tangentialanteile, und die Baum-Kobaum-Eichung
   liefert innen ein A, das nicht divergenzfrei ist. Bei 0,01 Hz ist A·n in der Platte fast halb so
   groß wie A_tan.

**Gegenprobe im eigenen Code:** Der Referenzpfad koppelt ebenfalls komponentenweise (`NodalBem.cuh`),
aber mit knotenbasiertem A in allen drei Komponenten und einem div-div-Strafterm im Innern. Dort ist A
auch innen von Coulomb-Klasse, und die Formulierung ist konsistent. Der Referenzpfad trifft das Feld
auf 1 bis 3 %. Für Nédélec-Elemente ist dieser Weg nicht übertragbar: Sie haben keine stetige
Normalkomponente und keinen natürlichen div-Strafterm.

**Konsistenzbedingung:** Über das Interface dürfen nur eichinvariante Größen laufen. Das sind auf der
Dirichlet-Seite die Tangentialspur γ_t A, und davon wirksam nur curl_Γ(γ_t A) = n·B. Auf der
Neumann-Seite ist es λ = n × H, die natürliche Randgröße der FEM. Damit verschwinden beide Defekte
gleichzeitig. Option (a), Übersetzungsterme mit Kopplung an A·n, erfüllt das nicht und ist verworfen.

Die beiden Optionen, die die Bedingung erfüllen:

- **(b) Skalares Potential im Luftraum:** H = H_s − ∇φ außen, Kopplung über n·B und n × H.
- **(c) Symmetrische Kopplung in den Spuren γ_t A und λ = n × H** mit Laplace-Kernen (quasistatisch,
  keine Wellenzahl).

Dazu eine Vorentscheidung, die beide Optionen stark vereinfacht: **die Spulen als eingeprägte Quellen
statt als FEM-Körper** (Abschnitt 2).

## 2. Vorentscheidung: Spulen als Quellen oder als FEM-Körper?

Bei TEAM 28 sind die Spulen bewickelt (σ = 0), μ_r = 1, stromgeprägt und fest. Als FEM-Körper tragen
sie physikalisch nichts bei, was eine eingeprägte Stromdichte J_s mit ihrem Biot-Savart-Feld H_s nicht
auch liefert. Sie kosten aber:

- **Randfreiheitsgrade:** Der größte Teil der Randquads liegt auf den Spulen. Die genaue Zahl je Körper
  steht in den Row-Check-Kopfzeilen; vor der Entscheidung nachsehen.
- **Topologie:** Jede Spule ist ein Volltorus (b₁ = 1). Daher kommen die Kohomologie-Eichung im Innern
  und, bei Option (b), ein mehrfach zusammenhängender Luftraum, der Schnittflächen braucht.
- **Fehlerort:** Der Quellfeldfehler und der K1/K2-Widerspruch sitzen genau auf den Spulenmänteln.

Ohne Spulen als FEM-Körper bleibt nur die Platte (Vollscheibe, R = 65 mm, t = 3 mm,
`tools/gmsh/team28_hex.py`). Ihr Komplement ist einfach zusammenhängend.

**Voraussetzung:** H_s aus dem **Volumenstrom** der Spulen, nicht aus dem Fadenmodell. Der Row-Check hat
gezeigt, dass das Fadenmodell an den Spulen um 24 bis 69 % vom Volumenstrom abweicht. Das ist eine
Quadraturaufgabe über rechteckige Ringquerschnitte (analytisch in elliptischen Integralen oder mit
hoher Gaußordnung). `BiotSavart.cpp` ist die Ausgangsbasis. Am Plattenort ist das unkritisch, weil der
Abstand zu den Spulen groß ist gegen die Querschnittsabmessungen.

**Wann die Spulen FEM-Körper bleiben müssen:** bei massiven Spulen mit eigenen Wirbelströmen, bei
spannungsgespeisten Spulen mit Rückwirkung über einen Stromkreis, bei Eisen in der Nähe der Spulen oder
bei bewegten Spulen. Für TEAM 28 trifft nichts davon zu. Die Frage an euch ist, ob eines davon auf der
Roadmap steht. Davon hängt die Wahl zwischen (b) und (c) wesentlich ab.

## 3. Option (b): reduziertes Skalarpotential im Luftraum, A in der Platte

### Formulierung

- **Innen (Platte):** curl(ν curl A) + jωσA = 0 mit Nédélec-Elementen, wie bisher. Die Eichung im Innern
  ist frei wählbar: Baum-Kobaum, bei kleiner Frequenz nötig.
- **Außen:** H = H_s − ∇φ mit Δφ = 0 und φ → 0 im Unendlichen.
- **Interfacebedingungen:**
  - tangential: n × (ν curl A) = n × (H_s − ∇_Γφ), als natürlicher Randterm der FEM,
  - normal: n·curl A = μ₀(n·H_s − ∂φ/∂n), also Neumann-Daten für die BEM.
- **BEM:** die skalare Außengleichung, wie sie in `NodalBem.cuh` schon steht und getestet ist:
  V q = (K − ½I) φ mit q = ∂φ/∂n.

### Warum das eichinvariant ist

Von A geht nur curl_Γ(γ_t A) ein. Das ist der Fluss durch das Randquad, für Nédélec-Elemente niedrigster
Ordnung exakt die Summe der vier Kanten-DOF, also eine Inzidenzmatrix. Ein Gradient ∇_Γψ hat keine
Zirkulation und fällt heraus. Von φ geht nur ∇_Γφ in die FEM ein. Ein Q1-φ hat einen exakten
Nédélec-Gradienten (diskreter Gradient G, `DiscreteGradient.cpp`). Die Unbekannte k entfällt, damit
auch der K1/K2-Widerspruch.

### Was bleibt, was neu ist

| bleibt | neu oder geändert |
|---|---|
| FEM-Assemblierung, Baum-Kobaum-Eichung (Platte), Wirbelstromterm | Kopplungsblock ⟨∇_Γφ, γ_t v⟩ über den diskreten Gradienten |
| `NodalBem.cuh`: V, K, Kollokation, singuläre Quadratur, Test | Flussmatrix curl_Γ: Kanten-DOF → Quadfluss → q |
| Cluster-Baum, ACA, Blockpaare (für skalare Kerne) | ACA für V, K (bisher nur dicht im Referenzpfad) |
| `BiotSavart.cpp` | Volumenstrom-Biot-Savart für H_s (Abschnitt 2) |
| Row-Check-Infrastruktur, Lorentz- und Wirbelstromdiagnose | Kohomologie-Eichung entfällt (keine Spulen im FEM) |

Die Größe des Systems ist klein: φ lebt nur auf den Randknoten der Platte. Dicht ist das für die
Prototypnetze kein Problem.

### Risiken

- **Auslöschung beim reduzierten Skalarpotential:** H_s und ∇φ heben sich nahezu auf. Bekannt ist das in
  Eisen mit großem μ_r. Bei μ_r = 1 und einer nichtmagnetischen Platte ist es harmlos, zu prüfen an der
  Feldgenauigkeit bei 0,01 Hz.
- **Kollokation ist unsymmetrisch.** Das kostet beim Vorkonditionierer etwas. Galerkin bräuchte
  Sauter-Schwab für skalare Kerne. Das ist vorhanden, aber an die Rooftop-Integranden angepasst.
- **Nicht direkt verallgemeinerbar** auf Spulen als FEM-Körper. Dann wird der Luftraum mehrfach
  zusammenhängend und braucht Schnittflächen.

## 4. Option (c): symmetrische Kopplung in γ_t A und λ = n × H

### Formulierung (Skizze)

Quasistatische Wirbelstrom-Kopplung. Außen gilt curl curl A = 0 mit div A = 0. Die Eichung wird dabei
nur im Außenraum benutzt und erscheint nicht im Interface. Die Randoperatoren mit dem Laplace-Kern
G(x,y) = 1/(4π|x−y|) sind:

- **Einfachschichtoperator, vektoriell** auf λ (div-konform): ⟨λ', V λ⟩ mit dem Skalarprodukt der
  Basisfunktionen. Ein Divergenzanteil entfällt, weil div_Γ λ = −n·curl H = 0 auf der Luftseite gilt.
- **Hypersingulärer Operator in curl-Form:** ⟨curl_Γ u', V curl_Γ u⟩. Er ist von vornherein
  eichinvariant, weil curl_Γ den Anteil ∇_Γψ tötet. Für Kantenelemente niedrigster Ordnung ist
  curl_Γ u stückweise konstant pro Quad, also ein skalarer Einfachschichtoperator auf konstanten
  Dichten.
- **Doppelschichttyp** mit Kern ∇G × (·) zwischen λ und γ_t u.

Das gekoppelte System ist symmetrisch in den Unbekannten a (alle FEM-Kanten) und λ (Randquad-Kanten).
Die genaue Gestalt der Operatoren und die Vorzeichen übernimmt man aus der Literatur und prüft sie am
Calderón-Test.

Literatur, aus dem Gedächtnis zitiert und vor Verwendung zu prüfen: R. Hiptmair, „Symmetric coupling
for eddy current problems“, SIAM J. Numer. Anal., 2002; M. Kuhn, O. Steinbach, Arbeiten zur
FEM-BEM-Kopplung bei 3D-Magnetfeld- und Wirbelstromproblemen, um 2002.

### Was bleibt, was neu ist

| bleibt | neu oder geändert |
|---|---|
| FEM, Eichung, Kohomologie (wenn Spulen FEM bleiben) | Vektor-Einfachschicht auf div-konformer Basis: λ als n × (vorhandene kovariante Rooftops) |
| `BemCouplingMap` (Gather/Scatter Randkante → FEM-Kante): die Dirichlet-Unbekannten sind direkt die FEM-Randkanten | hypersingulärer Operator in curl-Form (skalarer Kern auf konstanten Quaddichten, billig) |
| Cluster-Baum, Blockpaare, ACA, GPU-Apply | Doppelschichtoperator mit ∇G ×: neue singuläre Integranden in `SauterSchwab.cu` |
| T als Massenkopplung ⟨λ, γ_t v⟩ | D und G der jetzigen Zeile 2 entfallen |
| Row-Check-Werkzeuge | Vorkonditionierer: Blockstruktur ändert sich, `--harmonic-inner-direct` bleibt möglich |

Die Zahl der Unbekannten bleibt wie heute. Mit Spulen als Quellen schrumpft sie auf die Platte.

### Risiken

- **Mehr neue singuläre Integrale** als bei (b), vor allem der Doppelschichttyp. Hier lag in #23 schon
  einmal ein Fehler (T-Geometrie, Vorzeichen, Spur).
- **Topologie:** Nach meinem Verständnis kommt die A-basierte Außenformulierung ohne Schnittflächen aus,
  auch wenn die Spulen FEM-Körper bleiben. Das ist der Hauptvorteil gegenüber (b). Ob das in der
  gewählten Variante gilt, muss der Prototyp zeigen.
- **Kern des Hypersingulären:** Er ist auf Gradienten null. Ob das System ohne zusätzliche Eichung des
  Randraums eindeutig lösbar ist, hängt davon ab, wie die Kopplung an die Innen-Eichung angebunden
  wird. Das wäre zuerst am Würfel zu prüfen.

## 5. Vergleich

| Kriterium | (b) Skalarpotential, Spulen als Quellen | (c) Maxwell-Spuren |
|---|---|---|
| eichinvariant | ja | ja |
| K1/K2-Problem | entfällt | entfällt |
| neue singuläre Integrale | keine (skalar, vorhanden) | Vektor-Einfachschicht, Doppelschicht mit ∇G × |
| Wiederverwendung | hoch (`NodalBem`, `DiscreteGradient`, ACA) | mittel (Infrastruktur ja, Kerne neu) |
| Spulen als FEM-Körper möglich | nur mit Schnittflächen | ja |
| Eisen, massive oder bewegte Spulen | eingeschränkt | ja |
| Systemgröße (TEAM 28) | klein (Platte, φ auf Plattenknoten) | wie heute oder Platte allein |
| Aufwand bis zum Prototyp | gering bis mittel | mittel bis hoch |

## 6. Empfehlung

- **Für TEAM 28 und den jetzigen Projektumfang: (b) mit Spulen als Volumenstromquellen.** Es ist der
  kürzeste Weg zu einem konsistenten Pfad, mit fast nur vorhandenen, getesteten Bausteinen.
- **Falls Spulen als FEM-Körper oder Eisen absehbar nötig sind: gleich (c).** Ein Umweg über (b) würde
  dann doppelte Arbeit bedeuten.
- **Unabhängig von der Wahl:** zuerst den Volumenstrom-Biot-Savart bauen. Er dient bei (b) als Quelle,
  bei (c) wahlweise ebenso, und verbessert in jedem Fall die Referenz des Row-Checks an den Spulen.

## 7. Abnahmekriterien für einen Prototyp (beide Optionen)

1. **Eichtest:** Das Residuum bleibt unter A + ∇ψ invariant, für beliebige, auch nicht harmonische
   abklingende ψ. Das ist der Fall 4 aus `bem-normal-components.md`, an dem die jetzige Zeile 2 scheitert.
2. **Quellfeld:** Bei 0,01 Hz liegt das Plattenfeld gegen Biot-Savart auf dem Niveau des Referenzpfads
   (1 bis 3 %), dicht und mit ACA.
3. **Kraft:** Bei 50 Hz und 11,3 mm liegt F_z (Lorentz) innerhalb weniger Prozent des Referenzpfads
   (0,9665 N) und nahe m g = 1,05 N. Konvergenz über die Skalierungen 0,5 / 1,0 zeigen.
4. **Unabhängigkeit von der Eichung:** Das Feld ändert sich mit und ohne `--nogauge` nur im Rahmen der
   Lösertoleranz.

## 8. Offene Fragen an euch

- Stehen Spulen als FEM-Körper, Eisen, massive Spulen, Spannungsspeisung oder bewegte Spulen auf der
  Roadmap? (Entscheidet zwischen b und c.)
- Soll der bestehende BEM-FEM-Pfad als Vergleich erhalten bleiben (hinter einer Option), oder wird er
  ersetzt?
- Der transiente Pfad nutzt dieselbe Kopplung. Wird er mit umgestellt oder zunächst nur der harmonische
  Pfad?
