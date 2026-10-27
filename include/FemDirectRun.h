#ifndef FEM_DIRECT_RUN_H
#define FEM_DIRECT_RUN_H

#include <string>
#include <vector>

#include "Mesh.cuh"
#include "MaterialTypes.cuh"

// =============================================================================
// FEM-direkt: Vorbedingungen prüfen, bevor gerechnet wird
// =============================================================================
//
// AP8 aus docs/design/fem-direct-plan.md.
//
// WOZU EIN EIGENES MODUL
//
// Der FEM-direkt-Pfad hat mehrere Vorbedingungen, die bei Verletzung KEINEN
// Fehler erzeugen, sondern ein falsches Ergebnis. Die wichtigste:
//
//   Ohne mitvernetztes Luftgebiet ist mesh.boundary_faces die Oberfläche der
//   LEITER, nicht die Außenwand eines Luftkastens. A x n = 0 dort bedeutet
//   dann "das Feld endet an der Aluminiumplatte" -- physikalischer Unsinn, der
//   sich als saubere, konvergierte, falsche Lösung zeigt.
//
// Solche Bedingungen gehören geprüft, nicht kommentiert. Und die Prüfung
// gehört in ein Modul ohne CUDA-Bibliotheken, damit sie im Test wirklich
// ausgeführt werden kann.

namespace fem_direct {

struct Box {
    double lo[3] = {0, 0, 0};
    double hi[3] = {0, 0, 0};
    bool valid = false;

    double extent(int axis) const { return hi[axis] - lo[axis]; }
};

struct PreflightReport {
    // --- Netz ---------------------------------------------------------------
    size_t num_elements = 0;
    size_t num_air_elements = 0;
    size_t num_conductor_elements = 0;
    size_t num_edges = 0;
    size_t num_boundary_faces = 0;
    size_t num_dirichlet_edges = 0;
    size_t num_free_dofs = 0;
    size_t block_system_rows = 0;      ///< 2 * num_free... bzw. 2 * num_edges

    Box conductor_box;
    Box air_box;

    /// Abschneidefaktor je Achse: Ausdehnung des Luftgebiets geteilt durch die
    /// der Leiter. Der Plan verlangt eine Studie über 2x, 3x, 5x.
    double truncation_factor[3] = {0, 0, 0};
    double truncation_factor_min = 0.0;

    // --- Bewertung ----------------------------------------------------------
    /// Rechnen ist nicht sinnvoll möglich.
    std::vector<std::string> problems;
    /// Rechnen ist möglich, das Ergebnis aber mit Vorbehalt.
    std::vector<std::string> warnings;

    bool ok() const { return problems.empty(); }

    /// Mehrzeilige Zusammenfassung für die Konsole.
    std::string to_string() const;
};

/// Prüft Netz und Materialien gegen die Vorbedingungen des FEM-direkt-Pfades.
///
/// materials darf nullptr sein; dann entfallen die materialabhängigen
/// Prüfungen (Regularisierung), und das wird als Warnung vermerkt.
///
/// air_component_id ist die component_id des Luftgebiets, per Vorgabe 0 --
/// so schreibt tools/import_bdf.py es.
/// `nullspace_handled` sagt, dass der Curl-Curl-Nullraum ANDERS als durch
/// Regularisierung beseitigt wird -- konkret durch die Baum-Kobaum-Eichung.
/// Dann ist kappa = 0 in der Luft kein Fehler, sondern der Normalfall, und die
/// entsprechende Beanstandung entfaellt.
///
/// Ohne dieses Kennzeichen beanstandet der Preflight jedes nichtleitende
/// Element -- richtig fuer den regularisierten Weg, falsch fuer den geeichten.
PreflightReport preflight_fem_direct(
    const Mesh& mesh,
    const MaterialProperties* materials,
    int air_component_id = 0,
    double min_truncation_factor = 2.0,
    bool nullspace_handled = false);

}  // namespace fem_direct

#endif  // FEM_DIRECT_RUN_H
