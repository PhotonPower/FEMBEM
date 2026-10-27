#include "FemDirectRun.h"
#include "FemDirectBC.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace fem_direct {

namespace {

void grow(Box& b, const double3& p) {
    if (!b.valid) {
        b.lo[0] = b.hi[0] = p.x;
        b.lo[1] = b.hi[1] = p.y;
        b.lo[2] = b.hi[2] = p.z;
        b.valid = true;
        return;
    }
    b.lo[0] = std::min(b.lo[0], p.x); b.hi[0] = std::max(b.hi[0], p.x);
    b.lo[1] = std::min(b.lo[1], p.y); b.hi[1] = std::max(b.hi[1], p.y);
    b.lo[2] = std::min(b.lo[2], p.z); b.hi[2] = std::max(b.hi[2], p.z);
}

std::string fmt(double v) {
    std::ostringstream s;
    s.precision(4);
    s << v;
    return s.str();
}

}  // namespace

std::string PreflightReport::to_string() const {
    std::ostringstream s;
    s.precision(4);
    s << "[FEM-direkt] Netz: " << num_elements << " Elemente ("
      << num_conductor_elements << " Leiter, " << num_air_elements << " Luft), "
      << num_edges << " Kanten\n";
    s << "[FEM-direkt] Rand: " << num_boundary_faces << " Flaechen, "
      << num_dirichlet_edges << " festgehaltene Kanten -> "
      << num_free_dofs << " freie Freiheitsgrade\n";
    s << "[FEM-direkt] Blocksystem: " << block_system_rows << " Zeilen (2n)\n";

    if (conductor_box.valid && air_box.valid) {
        s << "[FEM-direkt] Abschneidefaktor je Achse: "
          << fmt(truncation_factor[0]) << " / "
          << fmt(truncation_factor[1]) << " / "
          << fmt(truncation_factor[2])
          << "  (kleinster: " << fmt(truncation_factor_min) << ")\n";
    }

    for (const auto& w : warnings) s << "[FEM-direkt] WARNUNG: " << w << "\n";
    for (const auto& p : problems) s << "[FEM-direkt] ABBRUCH: " << p << "\n";
    return s.str();
}

PreflightReport preflight_fem_direct(
    const Mesh& mesh,
    const MaterialProperties* materials,
    int air_component_id,
    double min_truncation_factor,
    bool nullspace_handled)
{
    PreflightReport rep;
    rep.num_elements = mesh.num_hexes;
    rep.num_edges = mesh.num_edges;
    rep.num_boundary_faces = mesh.num_boundary_faces;

    if (mesh.num_hexes == 0) {
        rep.problems.push_back("Das Netz hat keine Elemente.");
        return rep;
    }
    if (mesh.num_edges == 0) {
        rep.problems.push_back(
            "Das Netz hat keine Kanten. TopologyExtractor::extract_edges() "
            "wurde nicht aufgerufen.");
        return rep;
    }
    if (mesh.num_boundary_faces == 0) {
        rep.problems.push_back(
            "Das Netz hat keine Randflaechen. "
            "TopologyExtractor::extract_boundary_faces() wurde nicht aufgerufen. "
            "Ohne Rand gibt es keinen Ort fuer A x n = 0.");
        return rep;
    }

    // -------------------------------------------------------------------------
    // Luftgebiet und Ausdehnungen
    // -------------------------------------------------------------------------
    mesh.hex_component_ids.prefetch_to_host();
    mesh.hex_nodes.prefetch_to_host();
    mesh.nodes_initial.prefetch_to_host();

    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        const int cid = mesh.hex_component_ids[h];
        const bool is_air = (cid == air_component_id);
        if (is_air) ++rep.num_air_elements; else ++rep.num_conductor_elements;

        const int nn = element_num_nodes(mesh.elem_type(h));
        const Hexahedron e = mesh.hex_nodes[h];
        for (int i = 0; i < nn; ++i) {
            if (e.v[i] < 0) continue;
            const double3 p = mesh.nodes_initial[e.v[i]];
            grow(is_air ? rep.air_box : rep.conductor_box, p);
        }
    }

    // DIE zentrale Pruefung. Ohne Luftgebiet ist boundary_faces die Oberflaeche
    // der Leiter, und A x n = 0 dort heisst "das Feld endet an der
    // Aluminiumplatte". Das Ergebnis sieht sauber aus und ist falsch.
    if (rep.num_air_elements == 0) {
        rep.problems.push_back(
            "Kein Luftgebiet im Netz (kein Element mit component_id "
            + std::to_string(air_component_id) + "). Damit ist der aeussere Rand "
            "die LEITEROBERFLAECHE, und A x n = 0 dort bedeutet, dass das Feld "
            "an der Platte endet. Das ergibt eine konvergierte, plausibel "
            "aussehende, falsche Loesung. Der FEM-direkt-Pfad braucht ein "
            "mitvernetztes Luftgebiet mit abgeschnittenem Aussenrand; fuer ein "
            "Netz ohne Luft ist der BEM-FEM-Pfad der richtige.");
        return rep;
    }
    if (rep.num_conductor_elements == 0) {
        rep.problems.push_back(
            "Alle Elemente sind als Luft eingetragen. Dann gibt es keinen "
            "Leiter, keine Quelle und nichts zu rechnen. Pruefe die "
            "PID-Zuordnung im Importer (tools/import_bdf.py --map).");
        return rep;
    }

    // -------------------------------------------------------------------------
    // Abschneidefaktor
    // -------------------------------------------------------------------------
    if (rep.conductor_box.valid && rep.air_box.valid) {
        rep.truncation_factor_min = 1e300;
        for (int a = 0; a < 3; ++a) {
            const double ce = rep.conductor_box.extent(a);
            const double ae = rep.air_box.extent(a);
            rep.truncation_factor[a] = (ce > 0.0) ? ae / ce : 0.0;
            if (ce > 0.0)
                rep.truncation_factor_min =
                    std::min(rep.truncation_factor_min, rep.truncation_factor[a]);
        }
        if (rep.truncation_factor_min > 1e299) rep.truncation_factor_min = 0.0;

        if (rep.truncation_factor_min < min_truncation_factor) {
            std::ostringstream w;
            w.precision(3);
            w << "Der Luftkasten ist knapp: kleinster Abschneidefaktor "
              << rep.truncation_factor_min << ", verlangt sind mindestens "
              << min_truncation_factor << ". A x n = 0 so nah am Leiter "
              << "verformt das Feld messbar. Das Ergebnis ist brauchbar zur "
              << "Inbetriebnahme, fuer eine belastbare Kraft braucht es die "
              << "Abschneidestudie ueber mehrere Kastengroessen.";
            rep.warnings.push_back(w.str());
        }
    }

    // -------------------------------------------------------------------------
    // Dirichlet-Kanten
    // -------------------------------------------------------------------------
    try {
        const auto fixed = collect_dirichlet_edges(mesh);
        for (char c : fixed) if (c) ++rep.num_dirichlet_edges;
        rep.num_free_dofs = mesh.num_edges - rep.num_dirichlet_edges;
    } catch (const std::exception& e) {
        rep.problems.push_back(std::string("Randkanten nicht bestimmbar: ") + e.what());
        return rep;
    }

    rep.block_system_rows = 2 * mesh.num_edges;

    if (rep.num_free_dofs == 0) {
        rep.problems.push_back(
            "Alle Kanten liegen auf dem Rand, es bleibt kein freier "
            "Freiheitsgrad. Das Netz hat kein Inneres -- typischerweise eine "
            "einzige Elementschicht.");
        return rep;
    }
    if (rep.num_free_dofs * 4 < mesh.num_edges) {
        rep.warnings.push_back(
            "Mehr als drei Viertel aller Kanten liegen auf dem Rand ("
            + std::to_string(rep.num_dirichlet_edges) + " von "
            + std::to_string(mesh.num_edges) + "). Das Netz ist sehr grob; die "
            "Randbedingung dominiert dann die Loesung.");
    }

    // -------------------------------------------------------------------------
    // Regularisierung
    // -------------------------------------------------------------------------
    if (!materials) {
        rep.warnings.push_back(
            "Materialien nicht uebergeben, die Regularisierung des "
            "Curl-Curl-Nullraums konnte nicht geprueft werden.");
    } else {
        size_t zero_kappa = 0;
        double max_kappa = 0.0;
        for (size_t h = 0; h < mesh.num_hexes; ++h) {
            if (!(materials[h].kappa > 0.0)) ++zero_kappa;
            max_kappa = std::max(max_kappa, materials[h].kappa);
        }
        if (zero_kappa > 0 && !nullspace_handled) {
            rep.problems.push_back(
                std::to_string(zero_kappa) + " Elemente haben kappa = 0. K ist "
                "dort singulaer (alle Gradienten liegen im Nullraum), und eine "
                "Zerlegung bricht daran ab oder liefert Datenmuell. Vorher "
                "fem_direct::regularize_nonconducting_regions() aufrufen -- oder "
                "den Nullraum eichen und nullspace_handled setzen.");
        }
        if (zero_kappa > 0 && nullspace_handled) {
            rep.warnings.push_back(
                std::to_string(zero_kappa) + " Elemente mit kappa = 0, nicht "
                "regularisiert. Der Nullraum wird geeicht -- die Eichmaske muss "
                "jeden Knoten dieser Gebiete erreichen, sonst bleibt K dort "
                "singulaer.");
        }
        if (max_kappa <= 0.0) {
            rep.problems.push_back(
                "Kein Element hat eine positive Leitfaehigkeit. Ohne Leiter "
                "gibt es keinen Wirbelstrom.");
        }
    }

    return rep;
}

}  // namespace fem_direct
