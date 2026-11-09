# Kopplung FEM-BEM in Maxwell-Spuren (quasistatisch, κ = 0): Formulierung

Stand: 09.11.2026. Phase 1a nach `docs/design/maxwell-trace-coupling-plan.md` (Option c). Reine Theorie, kein Code am Lösungspfad.
Jede Identität mit dem Vermerk **[geprüft: Skript]** ist numerisch mit einem Skript unter `tools/theory/` nachgerechnet,
einschließlich Gegenprobe (falsches Vorzeichen muss durchfallen). Was nur hergeleitet und nicht gerechnet ist, steht als
**[hergeleitet]**, was offen ist, als **[offen]**. Zusammenfassung der Prüfungen in Abschnitt 9.

## 0. Ergebnis in Kürze

* Außen genügt die Kenntnis zweier **eichinvarianter** Randgrößen: λ = n × H (Neumann-Spur) und der Fluss g = n·B = curl_Γ(γ_t A).
  Das Außenfeld ist durch λ und g bestimmt, B = μ₀ curl Ψ(λ) − ∇Ψ(g); der Anteil n·A (und damit jede Eichung von A) tritt nicht auf.
* Auf der Luftseite gilt div_Γ λ = 0. Deshalb ist **λ = curl_Γ ψ + Σ c_k h_k** (ψ: Q1 auf den Randquads, h_k: 2g harmonische
  Felder je Randfläche). Das Einfachschichtproblem wird damit zum **skalaren** Laplace-Problem: ⟨curl_Γψ', V curl_Γψ⟩ = ⟨Wψ', ψ⟩
  (Nédélec-Identität), und die Doppelschicht mit ∇G× wird zu K' auf dem Fluss. In der Rooftop-Basis ist **W = Gᵀ V_rot G**: kein hypersingulärer
  Integrand tritt je auf.
* **Gewählt: symmetrische Kopplung** (Abschnitt 5.2) in den Unbekannten (a, ψ, c). Operatoren: V₀ (P0×P0), D̃ (P0×Q1), V_rot (E×E,
  daraus W, Z, Y) und eine Spalte je harmonischem Feld für die Doppelschicht Q. Alle Kerne sind 1/r bzw. ∂ₙG; alles
  weich oder schwach singulär.
* **Eichinvarianz** (Abschnitt 5.4): Jeder Block, der auf a wirkt, sieht a nur über R a (Fluss je Quad) und über die Zirkulationen auf den
  harmonischen Feldern. Wegen R G = 0 **exakt** (Kombinatorik) ist a + Gψ_h für **beliebiges** diskretes ψ_h, auch nicht harmonisches,
  ohne Wirkung. Das ist der Fall 4 aus `bem-normal-components.md`.
* **Topologie** (Abschnitt 4): 2g harmonische Felder je Fläche, kombinatorisch **[geprüft: Skript]**. Die Meridian-Zirkulation ist der
  umschlossene Strom (Ampère); mit eingeprägtem Strom folgt c_Meridian = N·I direkt aus der FEM-Zeile. Die Längszirkulation der Luftseite
  ist 0. Die Innen-Kohomologie-Eichung der Spulen wird durch die harmonische Zeile (S3) ersetzt **[hergeleitet]**.
* Ende-zu-Ende-Vorzeichenprüfung an der Kugel mit bekannter Lösung (eingeprägter Azimutalstrom) **[geprüft: Skript]**.
* Bibliographie in Abschnitt 8, gegen Verlagsseiten geprüft; meine zunächst aus dem Gedächtnis notierte DOI der Hiptmair-Arbeit war falsch (siehe dort).

## 1. Kontinuierliches Problem

### 1.1 Gebiete und Annahmen

* Ω = Ω_c ∪ Ω_n: FEM-Gebiete. Ω_c: Leiter, σ > 0, μ = μ₀ (oder ν(x)). Ω_n: nichtleitend, σ = 0, entweder Spule mit eingeprägter
  Stromdichte J_s (μ = μ₀) oder Eisen mit ν(|B|) = 1/μ(|B|) (J_s = 0).
* Ω_e = ℝ³ ∖ Ω̄: Luft, μ = μ₀, keine Quellen. Γ = ∂Ω = ∂Ω_e besteht aus K disjunkten geschlossenen Flächen Γ_j vom Geschlecht
  g_j (TEAM 28: Platte g = 0; zwei Ringspulen g = 1). Wegen disjunkter Körper ist Ω_e zusammenhängend mit
  b₁(Ω_e) = Σ g_j. Berühren sich FEM-Körper, gehört die Berührfläche zum Inneren und nicht zu Γ.
* Es gibt keine eingeprägten Flächenströme auf Γ und keine Verschiebungsströme (quasistatisch).

### 1.2 Gleichungen

Innen, A-Formulierung ohne Skalarpotential, E = −∂ₜA (kein elektrisches Feld aus Ladungen/Elektroden):

* transient: curl(ν curl A) + σ ∂ₜA = J_s in Ω,
* harmonisch (e^{jωt}): curl(ν curl A) + jωσ A = J_s in Ω,

mit ν = 1/μ₀ bzw. ν(|curl A|). Diese Formulierung ist konsistent, weil die Außenseite div_Γ λ = 0 erzwingt (Abschnitt 3):
Aus curl H = J_s − σ ∂ₜA folgt für beliebige gradientenartige Testfunktionen ∫(σ∂ₜA − J_s)·∇w = 0, also div(σ ∂ₜA − J_s) = 0
und (σ∂ₜA − J_s)·n = 0 auf Γ. Dazu muss das eingeprägte J_s divergenzfrei sein (Risiko R3).

Außen (Ω_e): curl H = 0, div B = 0, B = μ₀H, B = curl A, mit Coulomb-Eichung div A = 0 **nur im Außenraum**; dann ist jede
kartesische Komponente von A harmonisch. Abklingen: A = O(1/|x|), B = O(1/|x|³) (Dipolverhalten; kein Monopol, weil div B = 0).

Interface Γ: n × H und n·B stetig.

### 1.3 Spuren und Konventionen

n zeigt aus Ω in die Luft (äußere Normale der FEM-Körper), exakt wie `evaluate_quad_normal` in `BemIntegration.cuh` (der Row-Check
meldete 0 von 3696 Quads mit nach innen zeigender Normale). Für ein Vektorfeld u auf Γ:

| Spur | Definition | Bedeutung |
|---|---|---|
| γ_t u | n × (u × n) = u − (u·n)n | Tangentialspur, hier A_t = γ_t A |
| γ_× u | n × u | verdrehte Spur |
| γ_N A | n × (ν curl A) = n × H =: **λ** | Neumann-Spur |
| g | n·curl A = n·B | Fluss, g = curl_Γ γ_t A |

Oberflächenoperatoren (alle auf geschlossenem Γ):

* ∇_Γ ψ: Tangentialgradient; **curl_Γ ψ := ∇_Γψ × n** (vektorielle Rotation eines Skalars);
* für tangentiale v: **curl_Γ v := n·curl v**, div_Γ(n × v) = −n·curl v, also div_Γ(n×v) = −curl_Γ v;
* Adjungiertheit (partielle Integration, Γ geschlossen): ⟨curl_Γψ, v⟩ = ⟨ψ, curl_Γ v⟩, ⟨∇_Γχ, v⟩ = −⟨χ, div_Γ v⟩.

Kerne: G(x,y) = 1/(4π|x−y|), ∇_xG = −(x−y)/(4π|x−y|³). Potentiale: Ψ(v)(x) = ∫_Γ G v dS (komponentenweise),
𝒟ψ(x) = ∫_Γ ∂_{n_y}G ψ dS mit ∂_{n_y}G = (x−y)·n_y/(4π|x−y|³).

### 1.4 Bezug zum Code und Abweichungen

* **Normale, Kernel, D, V**: `NodalBem.cuh` benutzt dasselbe G, dieselbe Außennormale und dieselbe Doppelschicht (Kern
  (x−y)·n_y/(4π|x−y|³)); die dort hergeleitete Gleichung (B) V q = (K − ½I)u mit q = ∂ₙu ist genau ½u − Du + Vq = 0 aus
  Abschnitt 2.2 (Kugel-Kontrolle: u = 1/r, V q = −1/a). Hier: K = D.
* **Abweichung 1:** Der alte BEM-FEM-Pfad hat die Unbekannte k = −n × B/μ₀ = **−λ** (Zeile 1, `TK = −T k`). Die neue Unbekannte λ trägt das
  umgekehrte Vorzeichen. Ferner hat k zwei verschiedene Größen vertreten (K2 = −n×B/μ₀ in Zeile 1, K1 = (∂A/∂n)_tan/μ₀ in Zeile 2;
  `bem-trace-consistency.md`). λ erscheint nur noch in der FEM-Zeile.
* **Abweichung 2:** Die Rooftop-Basisfunktionen W_i in `evaluate_rooftop_basis` sind die kovariante Nédélec-Spur (Kantenintegral 1,
  tangential stetig); Koeffizient = Kantenintegral von A_t. Für λ wird die **gedrehte** Basis n × W_i benutzt (Abschnitt 3.2); eine
  neue Basis wird nicht gebraucht.
* **Abweichung 3:** T (Massenmatrix auf den Rooftops) erscheint nicht mehr als Operator ½T − D; es entsteht stattdessen die lokale Kopplung
  ⟨λ, γ_t v⟩ (Abschnitt 5).

## 2. Außenraum-Darstellung und Calderón-Operatoren bei κ = 0

### 2.1 Darstellungsformel

Für x ∈ Ω_e (und = 0 für x ∈ Ω)

    A(x) = μ₀ Ψ(λ)(x) + curl Ψ(n × A)(x) − ∇ Ψ(n·A)(x)                                   (R-A)
    B(x) = μ₀ curl Ψ(λ)(x) − ∇ Ψ(n·B)(x)                                                 (R-B)

**Herleitung.** Man benutzt F = G e (e konstant) und div(F × curl A − A × curl F) = A·curl curl F − F·curl curl A auf Ω_e, mit
curl curl A = 0 und curl curl(Ge) = ∇(e·∇_yG) + δ_x e. Das Randintegral liefert mit ν_e = −n und div A = 0
(A·∇φ = div(Aφ)) die Terme n×B (also μ₀λ), n×A und n·A; Beitrag im Unendlichen entfällt wegen des Abklingens. (R-B) folgt aus
(R-A) durch curl. **[geprüft: `check_representation.py`]**: Dipolfeld A = μ₀ m×r/(4πr³) im Inneren von Kugel und Ellipsoid,
auch mit Eichverschiebung A + ∇χ (χ aussen harmonisch, abklingend); Fehler außen ≤ 4·10⁻¹⁵, innen (Wert 0) ≤ 4·10⁻¹⁴;
jedes einzelne Vorzeichen umgekehrt gibt Fehler 0,11 bis 1,5.

**Folgerung (Eichunabhängigkeit).** In (R-B) kommt A nur über n·B vor. In (R-A) kommt A_t nur über n×A vor; der Term
∇Ψ(n·A) ist ein reiner Gradient und tritt nach Testen mit div-freien Feldern aus (Abschnitt 2.3) heraus.

### 2.2 Sprungrelationen (n in die Luft, „+“ = Luftseite)

| Potential | Sprung [·] = (+) − (−) | Mittelwert |
|---|---|---|
| Ψ(φ) | 0 | Vφ |
| ∂ₙΨ(φ), φ skalar | −φ | K'φ, K'φ(x) = ∫∂_{n_x}G φ dS |
| 𝒟ψ | +ψ | Dψ, D = K (NodalBem), D 1 = −½ auf Γ_j |
| ∂ₙ𝒟ψ | 0 | −Wψ (W: Hypersingulär) |
| curl Ψ(φ), φ tangential | −n × φ | 𝒞φ := PV∫∇_xG × φ dS |
| ∇_ΓΨ(f) | 0 | – |

Mit (R-A), (R-B) folgen die **Calderón-Zeilen** der Luftseite, mit g = n·B, λ = n×H:

    (R-A)_Γ   ½ A_t = μ₀ (VΛ)_t + (𝒞(n×A))_t − ∇_Γ V(n·A)                          [tangential]
    (R-T)     ½ λ − n × 𝒞λ + (1/μ₀) n × ∇_Γ V g = 0                                  [tangential, aus n×(R-B)]
    (R-N)     ½ g + K' g − μ₀ n·𝒞λ = 0                                              [skalar, aus n·(R-B)]

**[geprüft: `check_calderon_rows.py`]** (R-T) und (R-N) mit Hauptwert-Integralen (Polarkoordinaten um den Aufpunkt, Winkelsumme
zuerst) an drei Punkten der Kugel, Dipolfeld: Residuen ≤ 1,3·10⁻¹²; jeder Summand mit umgekehrtem Vorzeichen: ≥ 0,33.
Die Sprünge von 𝒞 sind zusätzlich durch Annäherung (1 ± ε)x₀ bestätigt (Mittelwert = PV).

### 2.3 Symmetrie, Sprung, Kerne

* **V** (skalar und komponentenweise vektoriell): symmetrisch, positiv definit auf H^{-1/2}. Kern: 0.
* **K' = Dᵀ** (Galerkin-Matrizen transponiert). Kerne in dem Sinne, dass ½ ± K' auf Konstanten je Komponente Γ_j verschwindet:
  K1 = −½ auf Γ_j, 0 auf den übrigen.
* **W** (curl-Form, Abschnitt 3.3): symmetrisch, positiv semidefinit, **Kern = Konstanten je Komponente** (curl_Γψ = 0 ⇔ ψ konstant).
* **Maxwell-Doppelschicht:** Adjungiertheit ⟨v, n × 𝒞η⟩ = −⟨η, 𝒞(n×v)⟩ **[geprüft: `check_adjointness.py`]** (Kugel, zwei
  beliebige tangentiale Felder: Summe 2,7·10⁻¹¹ relativ, Gegenprobe Differenz = 2,0). Sie ist die Grundlage der Symmetrie von (S).
* **Was ½ − 𝒞n× annulliert:** Für ξ = ∇_Γχ gilt ⟨η', (½ − 𝒞n×)∇_Γχ⟩ = 0 für alle div-freien η'. (Mit n × ∇_Γχ = −curl_Γχ und
  Potentialidentität (I1): (𝒞 curl_Γχ)_t = −∇_Γ Dχ, also ½⟨η',∇χ⟩ + ⟨η',𝒞curlχ⟩ = 0 − ⟨η',∇_ΓDχ⟩ = 0.) **[hergeleitet]** Das ist die
  Eichinvarianz der Zeile (R-A) auf Γ.

## 3. Raum für λ

### 3.1 div_Γ λ = 0

Auf der Luftseite gilt div_Γ λ = div_Γ(n × H) = −n·curl H = 0, bis zum Rand. Also liegt λ im Raum
Λ = {λ tangential, div_Γ λ = 0} = curl_Γ H^{1/2}(Γ) ⊕ ℋ(Γ), ℋ = harmonische Felder (div_Γ = 0 und curl_Γ = 0), dim ℋ = 2 Σ g_j.

**Warum nicht die volle RT-Basis mit Maxwell-Einfachschicht?** Die Bilinearform des Maxwell-Einfachschichtoperators ist (Buffa–Hiptmair,
Wortlaut nicht geprüft) ⟨λ', 𝒮_κ λ⟩ = ⟨λ', V_κ λ⟩ − (1/κ²) ⟨div_Γλ', V_κ div_Γλ⟩ mit dem Helmholtz-Kern V_κ. Der zweite Term ∝ 1/κ² verschwindet nur auf
div-freien λ. In der vollen RT-Basis gibt es F − 1 Richtungen je Komponente mit div_Γ ≠ 0, auf denen er für κ → 0 divergiert; der Operator ist bei κ = 0 nicht definiert.
Die Wahl Λ_h = ker(div_Γ|RT) vermeidet das exakt und braucht nur den κ = 0-Kern 1/r.

### 3.2 Diskreter Raum Λ_h = n × ker R, Parametrisierung

Die Rotation n × bildet die kovariante Nédélec-Spur auf die **kontravariante** RT-Abbildung der gedrehten Referenzfunktion ab, für
beliebige bilineare Quads (DF^{-T} = Rot DF Rot^T / J und −Rot^T = Rot; Herleitung **[hergeleitet]**). Daher sind {n × W_e} die
Standard-RT-Funktionen niedrigster Ordnung auf den Randquads, und

* div_Γ(Σ_e c_e n × W_e) = −Σ_q (R c)_q /|q|: **div-frei ⇔ R c = 0** (R: Kante → Quad, Zirkulation um das Quad, Einträge 0, ±1),
* für ψ ∈ Q1 mit Knotenwerten ψ: **curl_Γψ = −Σ_e (Gψ)_e n × W_e** (G: Knoten → Kante, Differenz der Knotenwerte, wegen
  ∇_Γψ = Σ_e (Gψ)_e W_e).

Also ist **Λ_h = { n × Σ c_e W_e : c = −Gψ + Σ_k c_k H_k }**, H_k ∈ ker R ⊖ range G (Abschnitt 4).
Dimension je Zusammenhangskomponente: E − (F − 1) = V − 1 + 2g **[geprüft: `check_surface_topology.py`, Euler E = 2F]**.

### 3.3 Einfachschichtblock = skalares Hypersingular (Nédélec-Identität)

Für ψ, ψ' ∈ H^{1/2}(Γ):

    ⟨V curl_Γψ', curl_Γψ⟩ = ⟨Wψ', ψ⟩,    W := −∂ₙ𝒟                                         (I2)

**[geprüft: `check_scalar_reduction.py`]** An der Einheitskugel, ψ = Y₂: linke Seite 3,01592895 (Polarkoordinaten-Integration),
rechte Seite l(l+1)/(2l+1)·‖Y‖² = 3,01592895, relative Abweichung 2,7·10⁻¹⁴. Der Eigenwert l(l+1)/(2l+1) folgt unabhängig aus den
Sprungrelationen von 𝒟 (innen −(l+1)/(2l+1), außen l/(2l+1)); Gegenprobe Faktor 2 fällt durch.

Daraus ergibt sich in der Rooftop-Basis die Matrixform

    **W = Gᵀ V_rot G**,    (V_rot)_{ef} := ∫∫ G(x,y) (n_x × W_e(x))·(n_y × W_f(y)) dS_x dS_y        (E_Γ × E_Γ)

Der Integrand ist **schwach singulär** (1/r); das Hypersingulär ist als Gᵀ V_rot G dargestellt und wird nie als eigenständiger
Integrand ausgewertet. Auf koplanaren Quadpaaren ist (n_x×W_e)·(n_y×W_f) = W_e·W_f; V_rot ist dort der Rooftop-Einfachschichtblock G des
bisherigen Codes. Auf nicht koplanaren Paaren (Kanten und Ecken des Netzes) gilt
(n_x×a)·(n_y×b) = (n_x·n_y)(a·b) − (n_x·b)(n_y·a).

**Was im Code schon vorhanden ist:** `NodalBem.cuh` hat V und K (skalar, Kollokation, Q1) und die Konstantenzeile für D. Ein
Galerkin-Hypersingulär ist nicht vorhanden. Die Rooftop-Einfachschicht (`integrate_regular_quad_pair`,
Sauter–Schwab-Rahmen für gleiche/Kanten-/Eckpaare, `SingularQuadPair.cuh`, `AdjacentIntegration.cu`) ist vorhanden und liefert V_rot mit einer
geänderten Integrandenauswertung; **Kollokation statt Galerkin** ist für diese Blöcke nicht geeignet (Abschnitt 6.2).

### 3.4 Doppelschicht-Block auf dem Fluss

Für div_Γ m = −g (m = n × A_t):

    ⟨curl_Γψ', 𝒞(n×A_t)⟩ = ⟨ψ', K' div_Γ(n×A_t)⟩ = −⟨ψ', K' g⟩,   g = curl_Γ A_t                (I3)

**[geprüft: `check_scalar_reduction.py`]** Kugel, Dipolfeld in nicht-axialer Richtung, ψ' linear: 0,08000000 gegen 0,08000000
(Abweichung 1,4·10⁻¹³); Gegenprobe Vorzeichen: 2,0. Ebenfalls geprüft (I1): Ψ(curl_Γψ) = curl Ψ(ψ n) (4,9·10⁻¹⁴) und
**curl Ψ(curl_Γψ) = −∇𝒟ψ** (2,8·10⁻¹⁰ mit Zentraldifferenzen; Gegenprobe +∇𝒟ψ: 2,0).

Mit (I1) ist (R-B) für λ = curl_Γψ gleich B = −μ₀∇(𝒟ψ + (1/μ₀)Ψ(g)): die exakte skalare Darstellung φ = 𝒟ψ + (1/μ₀)S g, ψ = φ|_Γ
(H = −∇φ, λ = curl_Γ φ|_Γ ohne Vorzeichen). Ohne harmonischen Anteil ist die Maxwell-Spur-Kopplung also der skalare Außenraum in
ψ; mit harmonischen Anteilen (Zirkulation um Löcher) kommt etwas hinzu, das ein einwertiges ψ nicht darstellt.

## 4. Topologie

### 4.1 Harmonische Felder auf einer Quadfläche

Auf Γ_j vom Geschlecht g_j: dim ℋ = 2 g_j. Konstruktion (kombinatorisch, an `CohomologyGauge.cpp` angelehnt, aber auf der Oberfläche):

1. Matrizen: G (V×E_Γ-Inzidenz, Knoten → Kante, Vorzeichen aus `boundary_edge_signs`), R (E_Γ → F, Zirkulation um das Quad, die Randkanten der
   Quads in der Reihenfolge {0,1},{1,2},{2,3},{3,0}). Es gilt **R G = 0 exakt** (Kombinatorik).
2. Zufälligen Startvektor c₀ auf den Kanten der Komponente wählen; auf ker R projizieren: (R Rᵀ + ε I) y = R c₀, c₁ = c₀ − Rᵀ y.
3. Den Gradientenanteil herausprojizieren: (Gᵀ G + ε I) φ = Gᵀ c₁ (Graph-Laplace, CG), H = c₁ − G φ.
4. Wiederholung 2g mal mit gegenseitiger Orthogonalisierung.

**[geprüft: `check_surface_topology.py`]** (dichte lineare Algebra auf kleinen Netzen) für: Würfel (g = 0): dim H = 0; Torus 12×6
(g = 1): dim H = 2; zwei getrennte Tori: 4; Quader + Torus: 2. In allen Fällen gilt: max|R G| = 0; max|R G ψ| = 2·10⁻¹⁵ bei
nichtharmonischem ψ; Euler V − E + F = 2K − 2Σg; dim ker R = dim range G + 2Σg; dim ker GᵀG = K (Konstanten); die
Zirkulationsmatrix zwischen den Zyklen (Meridian, Längskreis) und den harmonischen Basisvektoren ist regulär (kleinster Singularwert 0,71,
0,71, 0,71); jeder Gradient hat auf jedem Zyklus Zirkulation exakt 0. Damit **sind die 2g Freiheitsgrade genau die Zirkulationen auf den Erzeugern von H₁(Γ)**.

### 4.2 Physikalische Bedeutung (Torus, g = 1)

* **Meridian-Zirkulation** (kleiner Kreis um den Querschnitt): ∮ H_t·dl = ∫_{Querschnittsscheibe} J = **umschlossener Strom** I_enc (Ampère, die
  Scheibe liegt im Inneren). Bei einer Spule mit eingeprägtem Strom N·I und ohne Wirbelstrom ist c_Meridian = N·I.
* **Längs-Zirkulation** (Kreis um die Achse): Der Kreis berandet in Ω_e eine Scheibe ohne Strom, also ∮ H_t·dl = 0.
  Die Luftseite erzwingt c_Längs = 0 **[hergeleitet]**. Es verbleibt ein Freiheitsgrad im λ-Raum, der von den Gleichungen auf null gesetzt wird; der
  Test „c_Längs → 0 mit Verfeinerung" gehört in die Prüfung (Abschnitt 7).
* **Fluss durch das Loch** Φ = ∮_{Längskreis} A·dl ist eichinvariant (∮∇χ = 0) und steckt in den **harmonischen Anteilen von γ_t A**, nicht in g. Er wird von
  der Außengleichung (S3) mit dem Außenfeld verknüpft (B_ext bestimmt Φ über Stokes auf der Lochscheibe).
* **Verträglichkeit mit eingeprägtem Strom:** Die Testfunktion v_cut mit Träger auf den die Querschnittsscheibe durchstoßenden Kanten (Stufenfunktion des Potentials,
  curl v_cut = 0, Zirkulation 1 längs) liefert in (S1) K(a, v_cut) = 0 und ⟨λ, v_cut⟩ = Σ c_k⟨h_k, v_cut⟩ = c_Meridian (Normierung), also c_Meridian = ∫J_s·v_cut = N·I.
  Das gilt unabhängig von der Wahl der Scheibe nur, wenn J_s **diskret divergenzfrei** ist (Risiko R3).

### 4.3 Leiter mit Loch (TEAM 7)

Zusätzlich gilt: Der Wirbelstrom kann um das Loch laufen (σ > 0). Seine Größe ist c_Meridian = I_enc (induziert, nicht eingeprägt). Faraday
verknüpft I_enc über die EMK −jωΦ mit dem Fluss Φ durch das Loch, also mit den harmonischen Anteilen von A_t. Im Leiter ist die Eichung durch den σ-Term fixiert (A-only mit V = 0 ist
konsistent, Abschnitt 1.2). **Die Innen-Kohomologie wird im Leiter nicht gebraucht**; für nichtleitende Körper mit Loch ersetzt (S3) sie [hergeleitet; Test T7 prüft den
Nullraum].

### 4.4 Mehrere Körper

ψ ist nur mod Konstanten bestimmt, eine Konstante je Γ_j, weil curl_Γ Konstanten annulliert (W 1_j = 0, ½ + K' tested mit 1_j verschwindet,
Abschnitt 2.3). Fixierung: je Komponente einen Knoten streichen (oder Mittelwert 0). Die Konstante ist **kein physikalischer Freiheitsgrad**; die
nur gradientenartig wirkenden Anteile von ψ sind in (S1) bis (S3) unsichtbar.

## 5. Gekoppeltes System

### 5.1 Schwache FEM-Zeile

Mit H = ν curl A und ∫ w·curl v = ∫ curl w·v − ∮ (n×w)·v ergibt sich für alle Testfelder v (Nédélec, Fehlerfreiheit der Vorzeichen
**[geprüft: `check_sphere_coupling.py` (V2)]**):

    ∫_Ω ν curl a·curl v + jω ∫_{Ω_c} σ a·v  +  ⟨λ, γ_t v⟩_Γ  =  ∫_Ω J_s·v                    (E1)

(Gegenprobe Vorzeichen von λ: Abweichung 0,78.) Mit λ = curl_Γψ + Σ c_k h_k wird ⟨λ, γ_t v⟩ = ⟨ψ, g_v⟩ + Σ c_k ⟨h_k, γ_t v⟩, g_v = curl_Γ γ_t v = n·curl v
(Fluss der Testfunktion; für Nédélec niedrigster Ordnung stückweise konstant, g_v = (R v)/|q| pro Quad).

### 5.2 Symmetrische Kopplung (Hiptmair/Costabel) – gewählt

Aus (E1) mit dem Ersatz ⟨λ, v_t⟩ = ⟨v_t, (½ + n×𝒞)λ⟩ + (1/μ₀)⟨g_v, V g_a⟩ (aus (R-T) mit v_t getestet; ⟨η, n×∇_ΓVg⟩ = −⟨curl_Γη, Vg⟩), zusammen mit (R-A)_Γ
getestet mit η' ∈ Λ (der Term −∇_ΓV(n·A) fällt weg, Abschnitt 2.3) und (I2), (I3):

    (S1)  K(a,v) + (1/μ₀)⟨V g_a, g_v⟩ + ⟨g_v, (½ + D)ψ⟩ + Σ_k c_k ⟨v_t, (½ + n×𝒞) h_k⟩ = F(v)
    (S2)  ⟨ψ', (½ + K') g_a⟩ − μ₀ ⟨Wψ, ψ'⟩ − μ₀ Σ_k c_k ⟨curl_Γψ', V h_k⟩ = 0                    für alle ψ'
    (S3)  ⟨h_l, (½ − 𝒞n×) a_t⟩ − μ₀ ⟨h_l, V curl_Γψ⟩ − μ₀ Σ_k c_k ⟨h_l, V h_k⟩ = 0                 für l = 1..2g

K(a,v) = ∫ν curl a·curl v + jω∫σ a·v. Matrixform (Unbekannte x = (a, ψ, c)), mit B_R = R eingebettet in alle FEM-Kanten,
V₀ = (1/|q||q'|)∫_q∫_{q'}G (P0-Einfachschicht), D̃_{qj} = (1/|q|)∫_q∫ ∂_{n_y}G φ_j, M̃_{qj} = (1/|q|)∫_q φ_j, N = ½M̃ + D̃,
W = Gᵀ V_rot G, Z = Gᵀ V_rot H, Y = Hᵀ V_rot H, Q_{e,l} = ⟨W_e, (½ + n×𝒞) h_l⟩:

    ⎡ K + (1/μ₀) B_Rᵀ V₀ B_R     B_Rᵀ N       Q       ⎤ ⎡a⎤   ⎡f⎤
    ⎢ Nᵀ B_R                     −μ₀ W      −μ₀ Z      ⎥ ⎢ψ⎥ = ⎢0⎥
    ⎣ Qᵀ                          −μ₀ Zᵀ     −μ₀ Y     ⎦ ⎣c⎦   ⎣0⎦

**Die Matrix ist symmetrisch** (komplex symmetrisch bei jω): (a,ψ)- und (ψ,a)-Block sind Transponierte, weil K' = Dᵀ; (a,c)- und (c,a)-Block, weil
⟨h_l,(½−𝒞n×)w_e⟩ = ⟨w_e,(½+n×𝒞)h_l⟩ (Adjungiertheit, Abschnitt 2.3); der Rest aus V = Vᵀ. Die Herleitung ist **[hergeleitet]**; die einzelnen Bausteine sind
**[geprüft]** in Abschnitt 2/3, das Gesamtsystem auf Matrixebene ist **[offen]** (Phase 1b).

Ende-zu-Ende-Kontrolle (Kugel, Eigenwerte, **[geprüft: `check_sphere_coupling.py`]**): V = 1/(2l+1), D = K' = −1/(2(2l+1)), W = l(l+1)/(2l+1) auf Y_l;
(½+K')g = μ₀ W ψ liefert S_ext = (½ − K')V^{-1}(½ − D) + W = l + 1 (= −∂ₙφ/φ für φ = r^{-(l+1)}Y_l) für l = 1, 2, 3, 5; mit der exakten Lösung des
Azimutalstroms (J_s = j₀ ẑ×r) gilt g = μ₀ S_ext ψ und Zeile 1 des skalaren Außenproblems (½−D)ψ + Vp = 0, p = −g/μ₀, jeweils auf Maschinengenauigkeit.

### 5.3 Unsymmetrische Kopplung (Johnson–Nédélec-Typ)

Wie (S) mit (E1) statt (S1):

    (J1)  K(a,v) + ⟨ψ, g_v⟩ + Σ_k c_k ⟨h_k, v_t⟩ = F(v),     (J2) = (S2),   (J3) = (S3)

    ⎡ K              B_Rᵀ M̃     P     ⎤
    ⎢ Nᵀ B_R          −μ₀ W    −μ₀ Z   ⎥        P_{e,k} = ⟨h_k, W_e⟩ (lokale Paarung, ohne Doppelschicht)
    ⎣ Qᵀ              −μ₀ Zᵀ   −μ₀ Y  ⎦

Unsymmetrisch (N ≠ M̃, Q ≠ P). Operatoren: nur W (aus V_rot), D̃, Q, Z, Y; **ohne V₀**.

### 5.4 Wahl (begründet) und Vergleich

| Kriterium | symmetrisch (S) | Johnson–Nédélec (J) |
|---|---|---|
| Operatoren | V₀, D̃, V_rot (+ Q, Z, Y) | D̃, V_rot (+ Q, Z, Y); ohne V₀ |
| Symmetrie | ja (Saddle: SPD-Blöcke K̃, W) | nein |
| Krylov | MINRES/GMRES, symmetrischer Block-Vorkonditionierer | GMRES |
| transient | konstante symmetrische Randblöcke, nur K̃ ändert sich | wie (S), aber ohne Symmetrieausnutzung |
| Stabilität | Literatur: Hiptmair 2002 (E-basiertes Modell), Costabel 1987 (skalar); für die A-Formulierung hier **nicht gezeigt** | Johnson–Nédélec 1980 (skalar); Bedingungen für den Maxwell-Fall nicht geprüft |
| Aufwand | ein Block mehr (V₀) | kleiner |

**Wahl: (S).** Gründe: (i) Symmetrie ist für den Vorkonditionierer und den transienten Pfad wertvoll (Plan, Frage 4); (ii) V₀ ist ein skalarer, glatter, ACA-fähiger
Kern ohne Singularitätsprobleme, der Mehraufwand ist klein; (iii) bei nichtlinearem Eisen bleibt die Newton-Jacobi-Matrix von (S) symmetrisch (Abschnitt 5.9). Offen: ob (S) auf den Netzen von TEAM 28 zu einem gut konditionierten Schurkomplement führt. **(J)** bleibt als
Rückfall, falls V₀ unerwartet Probleme macht. Vorhandene `--harmonic-inner-direct`-Struktur: Die innere Eliminierung der Platte/Spulen ohne Rand bleibt erhalten, weil
K nur über B_Rᵀ(·) an den Rand koppelt; die Randunbekannten (a_Γ, ψ, c) bilden den Schurkomplement-Block mit **dichten** BEM-Blöcken.

### 5.5 Eindeutigkeit und Eichung

* **Innen, σ = 0** (Spulen, Eisen): a ist nur bis auf a + Gχ bestimmt (χ mit Träger auf Knoten, die nur an σ = 0-Kanten hängen, einschließlich Randknoten). Baum-Kobaum-Eichung entfernt diese Zeilen und Spalten
  (bestehender Code `TreeCotreeGauge`).
* **Innen, σ > 0** (Leiter): keine Eichung nötig, der Term jωσ a löst sie auf.
* **Kohomologie-Eichung der Spulen** (`CohomologyGauge`): **entfällt in der gekoppelten Formulierung**, weil (S3) die Längs-Zirkulation (Fluss durch das Loch) festlegt:
  Der Nullraum a → a + h_int (curl-frei, nicht Gradient) hat auf Γ eine nichtverschwindende Zirkulation und wird von (S3) angezeigt **[hergeleitet; Test T7]**.
* **ψ:** Konstanten je Komponente (Abschnitt 4.4).
* **Wo geeicht wird:** nur im FEM-Teil, nur im σ = 0-Gebiet (Baum-Kobaum). Das Außenfeld braucht die Coulomb-Eichung nur in der Herleitung.
* **Warum B nicht von der Eichung abhängt:** Alles, was an a koppelt, ist (R a, Zirkulationen auf ℋ); beides ist invariant unter a → a + Gχ.

### 5.6 Eichinvarianz (ausdrücklich)

**Behauptung.** Jeder Block, der auf a wirkt, behandelt a und a + G ψ_h für beliebige diskrete ψ_h (Q1 auf Γ oder im Gebiet; auch nicht harmonisch) gleich.

**Beweis.** Die Blöcke (S1)–(S3) greifen auf a über (i) B_R a = Fluss je Quad, (ii) die Zirkulationen ⟨h_l, a_t⟩ und ⟨h_l, 𝒞(n×a_t)⟩ zu.
(i) B_R G ψ_h = 0 **exakt** (Kombinatorik, **[geprüft: `check_surface_topology.py`]**: max|R G ψ| = 1,8·10⁻¹⁵ für ψ = zufällig³). (ii) h_l ⟂ range G (harmonisch) und der
Term mit 𝒞 annulliert Gradienten (Abschnitt 2.3). Der FEM-Block K G = 0 im σ = 0-Gebiet (curl∘grad = 0); im Leiter ist jωσ G ≠ 0, das ist die gewollte Einschränkung
auf die V = 0-Konvention. ∎

**Gegenüber dem alten Operator:** Der alte Operator griff mit D, G, T auf A_t selbst (nicht nur auf den Fluss) zu; für nichtharmonisches ψ (Fall 4)
blieb das Residuum bei 62 %. Der neue Operator ist für Fall 4 per Konstruktion invariant (Test T4).

### 5.7 Eingeprägter Strom

J_s steht nur in F(v) = ∫J_s·v über FEM-Kanten im Spulenvolumen; weder im Rand noch im Außenraum. Voraussetzungen: div J_s = 0, J_s·n = 0 auf Γ (diskret: Gᵀ f = 0
auf den σ = 0-Knoten). Das erzwingt c_Meridian = N·I (Abschnitt 4.2). Für die ganze Kopplung relevant: Die Quelle ist **in** der FEM; die Außenseite sieht sie nur über λ.

### 5.8 Transient

Implizites Euler (wie bisher): (K̃ + (σ/Δt)C) a^{n+1} + … = f^{n+1} + (σ/Δt) C a^n. Die Randoperatoren (V₀, D̃, V_rot, Q) sind **zeitunabhängig**,
weil der Außenraum magnetostatisch ist (kein Gedächtnis); nur der Massenterm σC/Δt in K̃ ändert sich, und f. **Bewegte Körper:** Diagonalblöcke je Körper (V₀,jj, D̃_jj, V_rot,jj) sind
bei Starrkörperbewegung invariant (Kerne hängen nur von Abständen ab, die Skalarprodukte (n×a)·(n'×b) sind drehinvariant); neu aufzubauen sind nur die Blöcke zwischen Körpern mit
Relativbewegung (V₀,jk, D̃_jk, V_rot,jk und die Spalten Q, Z, Y mit Beteiligung beider Körper). Die harmonischen Felder je Körper bleiben unverändert.
Ein Bewegungsterm σ v × curl a in der FEM-Zeile des bewegten Leiters ist von der Randkopplung unabhängig und hier nicht Thema **[offen]**.

### 5.9 Eisen

Die Nichtlinearität steckt nur in K(a,v) = ∫ν(|curl a|) curl a·curl v. Die Randoperatoren (V₀, D̃, V_rot, Q, M̃, R, G) sind linear, geometrieabhängig und
**bleiben unberührt**. Newton: Jacobi-Matrix K_T(a) ersetzt K im (1,1)-Block, alles andere bleibt; K_T bleibt symmetrisch für isotrope ν(|B|), also bleibt (S) symmetrisch. Zu beachten: (i) K_T muss
positiv definit bleiben (ν_diff > 0); (ii) Kačanov-Fixpunkt: K(ν^k) mit festem Rand, monoton; (iii) bei μ_r ≫ 1 ist (1/μ₀)V₀ gegenüber K klein und B_Rᵀ V₀ B_R wirkt wie eine
Randkorrektur, die bei Auslöschung (H_s ≈ −∇φ) kritisch werden könnte [offen]; (iv) Eisenkanten verlangen im FEM-Teil regularisierungsfreie Nédélec-Elemente (nicht
den nodalen div-div-Referenzpfad; Costabel–Dauge zeigen das Versagen der Regularisierung an einspringenden Ecken).

## 6. Diskretisierung und Quadratur

### 6.1 Integranden

| Block | Kern, Basis | Singularität | Bestand im Code |
|---|---|---|---|
| V₀ | 1/(4πr), P0×P0 | schwach (1/r) | Rooftop-Einfachschicht mit Sauter–Schwab (gleich, Kante, Ecke, getrennt); Basisfunktion tauschen |
| V_rot | 1/(4πr)·(n_x×W_e)·(n_y×W_f) | schwach (1/r), Vektor-Skalarprodukt ändert sich auf nicht koplanaren Paaren | wie oben, **neue Integrandenauswertung** mit n_x, n_y |
| D̃ | ∂_{n_y}G, P0×Q1 | stark (1/r²) nur für nicht koplanare berührende Paare; koplanar identisch 0 | DLP mit Rooftop, `shared` 0/1/2, Selbstpatch analytisch 0 (`compute_dense_block_kernel_H`); Basisfunktion tauschen |
| Q | (W·∇_xG)(n_x·h) − (W·h)(n_x·∇_xG) | 1/r², koplanar identisch 0 | wie D̃ mit neuem Integrand (**neu**) |
| Z, Y, V_rot H | wie V_rot | schwach | wie V_rot |
| W | als Gᵀ V_rot G | – | **kein neuer Integrand** |

**Neu** also: (a) das Skalarprodukt (n×a)·(n'×b) im Einfachschicht-Integranden, (b) der Q-Kern, (c) Q1- und P0-Formfunktionen in den vorhandenen Integratoren.
**Per partieller Integration auf schwach Singuläres zurückgeführt:** das Hypersingular (über die Nédélec-Identität, Abschnitt 3.3) und die ∇G×-Doppelschicht auf Q1×Fluss (über (I3) auf K').
**Von Sauter–Schwab gedeckt** (laut Quelltext, in diesem Auftrag nicht neu getestet): 4-D-Fälle gleich / Kante / Ecke / getrennt für Kerne der Ordnung −1 und −2 (`SauterSchwab.cu`, `SingularQuadPair.cuh`,
`AdjacentIntegration.cu`; Sauter–Schwab 2011).

### 6.2 Kollokation oder Galerkin

**Galerkin für alle Blöcke.** Die Symmetrie von (S) beruht auf der Galerkin-Struktur (K' = Dᵀ in Matrixform, V_rot = V_rotᵀ). Kollokation (wie `NodalBem.cuh` für den Referenzpfad) würde diese Struktur zerstören
und das Schurkomplement unsymmetrisch machen; ferner verlangt W in Kollokation echte hypersinguläre Auswertung. Der Mehraufwand der Galerkin-Integration (Doppelintegral) ist im vorhandenen Sauter–Schwab-Rahmen
bezahlt.

### 6.3 ACA

Komprimierbar sind V₀ (Kern 1/r, glatt in der Ferne), D̃ (∂_nG), V_rot und damit W (W = Gᵀ V_rot G: ACA auf V_rot, danach Sparse-Produkte mit G), jeweils mit
asymptotisch glatten Kernen und Fernfeld-Zulässigkeitskriterium wie im bestehenden Cluster-Baum. Nicht komprimiert werden müssen: **Q, Z, Y** (2g Spalten
je harmonischem Feld: Aufwand O(g·N)), M̃, R, G (dünn).

## 7. Testplan für Phase 1b

Die kontinuierlichen Identitäten aus 2 bis 5 sind oben an Kugel/Ellipsoid numerisch bestätigt. Phase 1b muss die **diskreten** Operatoren gegen dieselben Felder prüfen.
Erwartete Ordnungen sind Erwartungen aus der Theorie (Galerkin Q1/P0, quasi-uniform), keine Messungen.

| Nr. | Identität | Geometrie, exaktes Feld | Erwartete Größe, Ordnung | Gegenprobe (muss durchfallen) |
|---|---|---|---|---|
| T1 | (R-A), (R-B): Darstellung | **Kugel** R = 1, Dipol m = ẑ im Ursprung: A = μ₀ m × r/(4π r³) (Coulomb) | Fehler des Quadraturoperators an Punkten außen ≤ 1·10⁻³ bei 6·6 Quads je Seite, Ordnung ~2 | Vorzeichen jedes Terms umkehren |
| T2 | (R-T), (R-N), Matrix-Residuum | Kugel, **Würfel** (Kanten, Ecken), dasselbe Dipolfeld (Würfel mit Dipol im Innern) | Kugel: ~h²; Würfel: ~h^{1…2} wegen Singularitäten an Kanten | Vorzeichen von ½, K', D einzeln |
| T3 | Nédélec-Identität W = Gᵀ V_rot G, K' = Dᵀ, Symmetrie | Kugel, Würfel: Eigenwerte l(l+1)/(2l+1), D/K' Spurzeilen | ‖W − Wᵀ‖ ≤ 10⁻¹² ‖W‖; Eigenwert Y_l: Fehler ~h² | falscher Faktor 2; W nur mit koplanarem Skalarprodukt (ohne n_x·n_y) auf dem Würfel muss auf Kanten abweichen |
| T4 | Eichtest | beliebige der obigen, a → a + Gψ_h mit **nichtharmonischem** ψ_h (z. B. ψ = z exp(−r²/s²) wie Fall 4 des alten Row-Checks) | alle Zeilen (S1–S3), Rechte Seiten ändern sich um ≤ 10⁻¹² relativ | a + n-Anteil (nicht Gradient, z. B. curl-Anteil) ändert um O(1); das alte Operatorresiduum 62 % |
| T5 | Torus, harmonische Felder | **Torus** (R = 3, r = 1; Quadnetz 24×12 …), **Kreisstrom** I auf dem Kernkreis im Innern: B_ext analytisch (Schleifenformel, Abschnitt 7.5), A analytisch | c_Meridian = I (Fehler ~h²), c_Längs → 0 mit Verfeinerung (Ordnung ~h²); Zeilen (S2)/(S3) mit exakten (A_t, λ): Residuum ~h² | c_Meridian := 0 setzen: Residuum O(1) |
| T6 | Symmetrie des Gesamtsystems | alle Netze | ‖M − Mᵀ‖/‖M‖ ≤ 10⁻¹²; (J) unsymmetrisch (‖M_J − M_Jᵀ‖ ≥ 10⁻²) | (J) als symmetrisch ausgeben |
| T7 | Eindeutigkeit/Nullraum | Torus-Spule + Platte (K = 2, g = 1) | SVD des Gesamtsystems nach Baum-Kobaum und Streichen je eines ψ-Knotens: kein weiterer Nullvektor | Streichen weglassen: Nullraumdimension K; Baum-Kobaum weglassen: Nullraum wächst um die Zahl der σ = 0-Gradienten |
| T8 | Ende zu Ende | **Kugel** mit Azimutalstrom J_s = j₀ ẑ × r (Abschnitt 5.2): FEM-Kugel mit BEM-Außenraum | ψ = (β/μ₀) cos θ, g = 2β cos θ, β = μ₀ j₀/15; Fehler ~h² | Vorzeichen von λ in (E1) umkehren (Abweichung 0,78 im Kontinuum) |
| T9 | Hodge-Strukturen | Würfel, Torus, zwei Körper | R G = 0 exakt; dim ker R − dim range G = 2Σg; harmonische Felder orthogonal zu range G und ker R | – (kombinatorisch, bereits geprüft) |

### 7.5 Referenz für Phase 2: Volumenstrom-Biot-Savart (Rechteckring)

**Verfahren.** Kreisschleife (Radius a, Höhe z', 1 A) hat A_φ, B_ρ, B_z in geschlossener Form mit vollständigen elliptischen Integralen K(m), E(m), m = 4aρ/((a+ρ)²+Δz²)
(identisch zu `BiotSavart.cpp` bzw. `BemRowCheck.h`). Die Ringspule mit Rechteckquerschnitt [r₀,r₁]×[z₀,z₁], N Windungen, ist das Integral
dieser Schleifen mit Dichte N/((r₁−r₀)(z₁−z₀)). Quadratur: Produkt-Gauss 8×8 auf **geometrisch zum Aufpunkt verdichteten** Panels (Faktor 0,3, 11 Stufen), der Aufpunkt-Querschnitt teilt den
Rechteckquerschnitt in bis zu 4 Teile (die integrable 1/dist-Singularität liegt an Panelecken); K wird über 1 − m = d²/s² berechnet (`ellipkm1`), damit es auch bei m → 1 keine Auslöschung gibt.

**Test [geprüft: `check_volume_biot_savart.py`]**, Innenspule TEAM 28 (r 27…55 mm, z −52…0 mm, 960 Windungen):
(B1) Schleifenformel gegen Linienintegral: ≤ 5,6·10⁻¹⁵; (B2) curl A = B: 1,2·10⁻⁹ (Gegenprobe Vorzeichen: 2,0);
(B3) Achse gegen unabhängige geschlossene 2-D-Quadratur: ≤ 9·10⁻¹³; (B4) außerhalb der Achse, Abstand 20 / 5 / 1 / 0,1 / 0 / −1 mm zum Querschnitt (auch auf dem Rand und im Inneren der Spule):
Selbstkonvergenz (q = 0,3, 8 Gauss gegen q = 0,2, 10 Gauss) 3·10⁻¹⁴ bis 4,6·10⁻⁸, gegen 900×900-Fadensumme 9·10⁻⁸ bis 1,2·10⁻³ (die Grenze dort ist die Fadenreferenz). Erwartete Genauigkeit im Einsatz: ≤ 10⁻⁷ relativ.

**Fadenmodell:** Auf den Randpunkten der Innenspule weicht das Fadenmodell (0,5 mm Raster) im B-Betrag um 0,1 bis 0,5 % ab, bei 3 mm Raster um 0,7 bis 3 %. Das ist **kleiner** als die im
Plan genannten 24 bis 69 %; diese Zahl beschreibt eine andere Größe (Row-Check, Normal-/Tangentialkomponenten bzw. K1/K2-Spurdifferenz, nicht |B|) und ist hier **nicht** reproduziert. Für Phase 2 gilt:
Volumenstrom als Referenz ist trotzdem sinnvoll (schnell, Genauigkeit 10⁻⁷).

## 8. Literatur (mit Prüfstand)

Die Angaben wurden am 09.11.2026 per Websuche gegen Verlags- bzw. Katalogseiten geprüft. Nicht geprüft sind Seiten und Bandnummern, wo es heißt.

1. R. Hiptmair, *Symmetric coupling for eddy current problems*, SIAM J. Numer. Anal. **40** (2002), DOI 10.1137/S0036142900380467
   (https://epubs.siam.org/doi/10.1137/S0036142900380467). Autor, Zeitschrift, Jahr, DOI bestätigt; Bandnummer 40 (Heft 1) und Seiten nicht bestätigt (Bezahlschranke). **Hinweis:** Die DOI, die ich zunächst aus dem Gedächtnis notiert hatte (Endung `...373208`), war falsch; die Prüfung ergab die oben genannte. Das Modell der Arbeit ist **E-basiert** (E im Leiter, H×n an der Schnittstelle); die Übertragung auf die hier gewählte A-Formulierung ist nicht Teil der Arbeit.
2. M. Kuhn, O. Steinbach, *Symmetric coupling of finite and boundary elements for exterior magnetic field problems*, Math. Methods Appl. Sci. **25**(5) (2002) 357–371, DOI 10.1002/mma.286. Bestätigt (Katalog, arXiv-Zitat). Dort wird ein Vektorpotential innen mit einem **Skalarpotential** im Außenraum gekoppelt (Option b), nicht mit Maxwell-Spuren.
3. R. Hiptmair, J. Ostrowski, *Coupled boundary-element scheme for eddy-current computation*, J. Engrg. Math. **51** (2005) 231–250, DOI 10.1007/s10665-004-2116-3. Bestätigt. Stichworte: divergenzkonforme Randelemente, Oberflächen-Stromfunktionen, Homologieräume (entspricht ψ und den harmonischen Feldern hier).
4. M. Costabel, *Symmetric methods for the coupling of finite elements and boundary elements*, in: C. A. Brebbia, W. L. Wendland, G. Kuhn (Hrsg.), *Boundary Elements IX*, Vol. 1, Springer 1987, 411–420 (DOI 10.1007/978-3-662-21908-9_26). Bestätigt (Springer-Eintrag; Seiten stammen aus einer zweiten Quelle, die Bandbezeichnung ist uneinheitlich 9/1 bzw. 1).
5. C. Johnson, J.-C. Nédélec, *On the coupling of boundary integral and finite element methods*, Math. Comp. **35** (1980) 1063–1079, DOI 10.1090/S0025-5718-1980-0583487-9. Bestätigt (AMS).
6. M. Costabel, M. Dauge, *Singularities of electromagnetic fields in polyhedral domains*, Arch. Rational Mech. Anal. **151** (2000) 221–276, DOI 10.1007/s002050050197. Bestätigt.
7. M. Costabel, M. Dauge, *Weighted regularization of Maxwell equations in polyhedral domains*, Numer. Math. **93** (2002) 239–277, DOI 10.1007/s002110100388. Bestätigt; die Aussage „die übliche Regularisierung liefert an einspringenden Ecken/Kanten falsche Lösungen" steht im Abstract des Aufsatzes, Fälle mit Materialsprung dort nicht geprüft.
8. A. Bossavit, *Computational Electromagnetism: Variational Formulations, Complementarity, Edge Elements*, Academic Press, 1998, ISBN 978-0-12-118710-1 (Druckausgabe; Erscheinungsjahr 1997/1998 je nach Katalog).
9. A. Buffa, R. Hiptmair, *Galerkin boundary element methods for electromagnetic scattering*, in: *Topics in Computational Wave Propagation*, Lecture Notes in Computational Science and Engineering **31**, Springer 2003, 83–124, DOI 10.1007/978-3-642-55483-4_3. Bestätigt (mehrere Zitierlisten); die EFIE-Form mit 1/κ²-Term in Abschnitt 3.1 stammt dort, **nicht im Wortlaut geprüft**.
10. S. A. Sauter, C. Schwab, *Boundary Element Methods*, Springer Series in Computational Mathematics **39**, Springer 2011, DOI 10.1007/978-3-540-68093-2. Bestätigt.
11. O. Steinbach, *Numerical Approximation Methods for Elliptic Boundary Value Problems: Finite and Boundary Elements*, Springer 2008, DOI 10.1007/978-0-387-68805-3 (ISBN 978-0-387-31312-2). Bestätigt; die Kurl-Form des Hypersingulären (Maue) ist dort bzw. bei Nédélec zu finden, **Seite nicht geprüft**. Die Identität (I2) ist hier selbst numerisch geprüft.

## 9. Verifikationsskripte (`tools/theory/`)

| Skript | Identität | Ergebnis |
|---|---|---|
| `check_representation.py` | (R-A), (R-B), innen = 0, mit/ohne Eichverschiebung, Kugel + Ellipsoid | ≤ 4·10⁻¹⁵ außen; Gegenproben 0,11–1,5 |
| `check_calderon_rows.py` | (R-T), (R-N) als Hauptwert-Integrale an der Kugel | ≤ 1,3·10⁻¹²; Gegenproben ≥ 0,33 |
| `check_scalar_reduction.py` | (I1) Ψ(curl_Γψ), curl Ψ(curl_Γψ) = −∇𝒟ψ; (I2) Nédélec-Identität; (I3) K' auf dem Fluss | 5·10⁻¹⁴, 3·10⁻¹⁰; 3·10⁻¹⁴; 1·10⁻¹³ |
| `check_adjointness.py` | ⟨v,n×𝒞η⟩ = −⟨η,𝒞(n×v)⟩ | 2,7·10⁻¹¹ relativ |
| `check_surface_topology.py` | R G = 0, dim ℋ = 2g, RT-Zählung, Zirkulationsmatrix, Gradient ohne Zirkulation | alle Fälle OK |
| `check_sphere_coupling.py` | (E1) mit Vorzeichen, S_ext = l + 1, Kugel-Lösung, (V1)–(V5) | Maschinengenauigkeit; Gegenproben 0,24 bis 0,78 |
| `check_volume_biot_savart.py` | Rechteckring-Verfahren, Fadenmodell | siehe 7.5 |

Offen nach diesem Auftrag (Risikoliste im Bericht `docs/design/maxwell-trace-formulation.md`): diskrete Lösbarkeit/Stabilität von (S) auf realen Netzen; die
Behauptung, dass (S3) die Kohomologie-Eichung der Spulen ersetzt; Abstimmung der diskreten Divergenzfreiheit von J_s; Konditionierung des Schurkomplements.
