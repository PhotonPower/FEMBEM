#ifndef AMS_PRECONDITIONER_CUH
#define AMS_PRECONDITIONER_CUH

// =============================================================================
// AmsPreconditioner.cuh -- Auxiliary-Space Maxwell Solver (Hiptmair und Xu)
// =============================================================================
//
// Vorkonditionierer fuer den Kantenelement-Operator
//
//     A = curl-curl  +  sigma/dt * Masse
//
// in additiver Form:
//
//     B r  =  S r  +  G A_G^-1 G^T r  +  Pi A_Pi^-1 Pi^T r
//
//   S    Punktglaetter im Kantenraum (Jacobi)
//   G    diskreter Gradient, A_G   = G^T A G     (skalarer Hilfsraum)
//   Pi   Knoten-Vektor-Interpolation, A_Pi = Pi^T A Pi  (vektorieller Hilfsraum)
//
// -----------------------------------------------------------------------------
// Warum beide Hilfsraeume
// -----------------------------------------------------------------------------
// Der Operator hat einen riesigen NAHnullraum aus Gradientenfeldern: fuer
// A = grad phi ist curl A = 0, es bleibt nur der Massenterm, der in
// nichtleitenden Gebieten verschwindet. Punktglaetter daempfen diese Moden
// nicht -- sie liegen im Nahnullraum. Genau daher stammt die schlechte
// Konvergenz des reinen FEM-Systems dieses Loesers (329 Iterationen im
// nobem-Fall, wo 20 bis 50 zu erwarten waeren).
//
// Der skalare Hilfsraum faengt die Gradientenmoden. Er genuegt aber NICHT:
// wegen curl(grad) = 0 faellt der curl-curl-Anteil aus A_G = G^T A G
// vollstaendig heraus, es bleibt A_G = G^T (sigma/dt M) G. Bei sigma = 0 --
// also magnetostatisch und in den Spulen -- ist das exakt null (am Testnetz
// nachgemessen: max|A_G|/max|A| = 6.5e-16). Der vektorielle Hilfsraum schliesst
// die Luecke: dort ist max|A_Pi|/max|A| = 0.50.
//
// -----------------------------------------------------------------------------
// Was AMS NICHT abdeckt
// -----------------------------------------------------------------------------
// Auf einem Gebiet mit b1 != 0 gibt es harmonische Felder: rotationsfrei, aber
// keine Gradienten einer eindeutigen Funktion. Sie liegen im Nullraum von
// curl-curl und werden weder von G noch von Pi erfasst. Bei TEAM 28 sind das
// DREI Volltori (Platte mit zentraler Bohrung, zwei Ringspulen), also ein
// dreidimensionaler harmonischer Nullraum. Die Kohomologie-Projektion muss
// daher erhalten bleiben, auch wenn die Baum-Kobaum-Eichung entfaellt.
//
// -----------------------------------------------------------------------------
// Singulaere Hilfsprobleme
// -----------------------------------------------------------------------------
// Beide Hilfsmatrizen koennen singulaer sein:
//   A_G  ist bei sigma = 0 identisch null.
//   A_Pi hat mindestens den dreidimensionalen Nullraum der konstanten
//        Vektorfelder (ein konstantes Feld hat verschwindende Rotation, was
//        S * Pi * const = 0 am Testnetz bestaetigt).
// Deshalb loest ein AGGREGATIONS-AMG die Hilfsprobleme, mit Diagonalshift auf
// der groebsten Ebene. Als Vorkonditionierer ist das zulaessig -- er muss die
// Matrix nicht exakt invertieren, nur gut annaehern und darf dabei nicht
// entarten.
//
// Warum AMG und nicht ILU(0): eine erste Fassung benutzte ILU(0) mit Shift und
// erreichte gegenueber Jacobi nur rund Faktor 2 statt der netzunabhaengigen
// Konvergenz, die das Verfahren liefern soll. Die Hilfsprobleme SIND
// Knoten-Laplace-Probleme, und dort skaliert ILU(0) bekannt schlecht, waehrend
// AMG die Iterationszahl von der Problemgroesse entkoppelt (am
// 3D-Laplace nachgemessen: 12 bis 14 Iterationen ueber achtfache DOF-Zahl,
// gegenueber 38 bis 92 mit Jacobi).
// =============================================================================

#include <vector>

#include "AlgebraicMultigrid.cuh"
#include "DiscreteGradient.cuh"
#include "Mesh.cuh"

/// Blendet in einer Knotenmatrix die Zeilen aus, deren Diagonale relativ zum
/// groessten Diagonaleintrag kleiner als 'ratio' ist: Zeile und Spalte werden
/// geleert, die Diagonale auf 1 gesetzt. Rueckgabe ist die Maske.
///
/// Gedacht fuer A_G = G^T A G, wo ausserhalb der Leiter die ganze Zeile
/// rundungsklein ist (A_cc * G = 0 exakt in der Analysis). Ausfuehrliche
/// Begruendung an der Definition in src/AmsPreconditioner.cpp.
std::vector<char> ams_mask_meaningless_rows(HostCsr& M, double ratio);

class AmsPreconditioner {
public:
    /// Baut G, Pi, A_G, A_Pi und die beiden ILU(0)-Zerlegungen auf.
    /// 'A_edge' ist die assemblierte Kantenmatrix (num_edges x num_edges),
    /// Spaltenindizes je Zeile aufsteigend.
    ///
    /// 'shift' ist der relative Diagonalshift auf der groebsten AMG-Ebene.
    void setup(const HostCsr& A_edge, const Mesh& mesh, double shift = 1e-10);

    /// z = B r
    void apply(const double* r, double* z) const;

    // ---- Diagnose, damit der Aufbau nicht stillschweigend entartet
    int    num_edges()            const { return m_num_edges; }
    int    scalar_dofs()          const { return static_cast<int>(m_rg.size()); }
    int    vector_dofs()          const { return static_cast<int>(m_rp.size()); }
    int    scalar_levels()        const { return m_amg_G.num_levels(); }
    int    vector_levels()        const { return m_amg_Pi.num_levels(); }
    double scalar_complexity()    const { return m_amg_G.operator_complexity(); }
    double vector_complexity()    const { return m_amg_Pi.operator_complexity(); }
    double auxiliary_scale_ratio() const { return m_ratio_Pi_over_G; }

    /// Zeilen mit verschwindender Diagonale in den Hilfsmatrizen. Das ist die
    /// Kennzahl, an der eine kollabierte Vergroeberung auffaellt: A_G ist bei
    /// sigma = 0 identisch null, und wenn -- wie bei TEAM 28 -- zwei von drei
    /// Koerpern nichtleitend sind, besteht A_G ueberwiegend aus leeren Zeilen.
    /// Auf so einer Matrix erzeugt die Aggregation lauter Einzelaggregate, die
    /// Grobgitter bleiben fast leer (Operatorkomplexitaet nahe 1), und der
    /// Hilfsraum traegt nichts.
    /// Diagonalen von A_G, die rundungsklein sind (|d| <= 1e-10 * max|d|).
    /// Aussagekraeftiger als scalar_empty_rows(): exakt null werden nur die
    /// Knoten, an denen sich alles ausloescht, rundungsklein dagegen alle
    /// Knoten ausserhalb der Leiter -- dort ist A_G = G^T A_cc G bis auf
    /// Rundung null, weil A_cc * G = 0 gilt.
    int scalar_tiny_rows() const { return m_tiny_G; }

    /// Knoten, an denen der skalare Hilfsraum ausgeblendet wurde, weil A_G
    /// dort keine Information traegt (kein sigma). Siehe
    /// mask_meaningless_rows in src/AmsPreconditioner.cpp.
    int scalar_masked_rows() const { return m_masked_G; }

    /// Konvergenzfaktor je V-Zyklus, gemessen auf einer Zufalls-rechten-Seite.
    /// Unter etwa 0.5 taugt der Hilfsraum als Vorkonditionierer, nahe 1.0
    /// traegt er nichts bei. Das ist die Kennzahl, die ueber den Hilfsraum
    /// entscheidet -- Level-Groessen und Operatorkomplexitaet sagen darueber
    /// nichts (siehe measure_amg_convergence in src/AmsPreconditioner.cpp).
    double scalar_convergence() const { return m_conv_G; }

    /// Derselbe Wert OHNE die Maske gemessen. Die Differenz zeigt, wie viel
    /// eines schlechten Werts blosses Messartefakt der ausgeblendeten Zeilen
    /// ist -- apply() regt die naemlich gar nicht an.
    double scalar_convergence_unmasked() const { return m_conv_G_unmasked; }
    double vector_convergence() const { return m_conv_Pi; }

    /// Grundlage der AMG-Hierarchie im vektoriellen Hilfsraum: knotenweiser
    /// Vektor-Poisson-Operator (true, Vorgabe) oder das alte Pi^T A Pi
    /// (false). Muss VOR setup() gesetzt werden. Der Vergleich laeuft ueber
    /// vector_convergence().
    void set_use_alpha_poisson(bool on) { m_use_alpha_poisson = on; }

    /// Materialkoeffizienten je Hexaeder fuer den Alpha-Poisson-Operator:
    /// nu = 1/mu (Steifigkeit) und beta = sigma/dt (gelumpte Masse). Muss VOR
    /// setup() gesetzt werden. Ohne beide ist der Operator nicht spektral
    /// aequivalent zu Pi^T A Pi -- siehe die Begruendung in
    /// src/DiscreteGradient.cpp.
    void set_alpha_coefficients(const double* nu_per_hex, const double* beta_per_hex) {
        m_nu_per_hex = nu_per_hex;
        m_beta_per_hex = beta_per_hex;
    }

    /// Diagonalen, die das AMG ueber alle Ebenen als nicht vorhanden
    /// eingestuft hat (skalarer bzw. vektorieller Hilfsraum).
    int scalar_amg_tiny() const { return m_amg_G.num_tiny_diagonals(); }
    int vector_amg_tiny() const { return m_amg_Pi.num_tiny_diagonals(); }

    int scalar_empty_rows() const { return m_empty_G; }
    int vector_empty_rows() const { return m_empty_Pi; }
    /// Groessen der AMG-Ebenen, zur Kontrolle der Vergroeberungsrate.
    const std::vector<int>& scalar_level_sizes() const { return m_lv_G; }
    const std::vector<int>& vector_level_sizes() const { return m_lv_Pi; }

private:
    int      m_num_edges = 0;
    HostCsr  m_G, m_Pi;
    AlgebraicMultigrid m_amg_G, m_amg_Pi;
    std::vector<double> m_inv_diag_edge;   // Jacobi-Glaetter im Kantenraum
    double   m_ratio_Pi_over_G = 0.0;
    int      m_empty_G = 0, m_empty_Pi = 0;
    int m_tiny_G = 0;
    int m_masked_G = 0;
    double m_conv_G = -1.0;
    double m_conv_G_unmasked = -1.0;
    double m_conv_Pi = -1.0;
    // VORGABE AN, seit die Materialkoeffizienten durchgereicht werden.
    // Gemessen am End-to-End-Test (4x4x4, halb Leiter halb Luft):
    //   vektorieller Hilfsraum   0.93 -> 0.23 je V-Zyklus
    //   aeusserer GMRES          keine Konvergenz -> 28 Iterationen
    // Ohne set_alpha_coefficients() faellt der Operator auf nu = 1 und
    // beta = 0 zurueck und wird SCHAEDLICH; setup() warnt in dem Fall.
    bool m_use_alpha_poisson = true;
    const double* m_nu_per_hex = nullptr;
    const double* m_beta_per_hex = nullptr;
    std::vector<char> m_mask_G;
    std::vector<int> m_lv_G, m_lv_Pi;

    // Arbeitsspeicher, damit apply() nicht allokiert
    mutable std::vector<double> m_rg, m_eg, m_rp, m_ep, m_edge_tmp;
};

#endif // AMS_PRECONDITIONER_CUH
