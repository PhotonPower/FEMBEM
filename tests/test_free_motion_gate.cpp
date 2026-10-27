// =============================================================================
// Regressionstest: kraftgetriebene Objekte müssen is_moving() == true melden.
//
//   g++ -O2 -std=c++17 -o test_free_motion_gate tests/test_free_motion_gate.cpp
//   ./test_free_motion_gate
//
// HINTERGRUND
// -----------
// apply_kinematics() (src/Kinematics.cu) prüft is_moving() als Gate, bevor es
// die von apply_newtons_law() integrierte Translation überhaupt auf die
// Netzknoten anwendet:
//
//     if (!obj.is_moving() || obj.get_num_unique_nodes() == 0) return;
//
// is_moving() wurde bislang ausschließlich von set_velocity() gesetzt, gedacht
// für kinematisch VORGEGEBENE Geschwindigkeit. Ein kraftgetriebenes Objekt wie
// die Aluminiumplatte in TEAM 28 -- deren Bewegung über apply_newtons_law()
// integriert wird -- setzte is_moving() nirgends und blieb daher für immer
// false. Folge: apply_newtons_law() integrierte Translation und
// Geschwindigkeit bei jedem Zeitschritt korrekt, mesh.nodes_current wurde
// aber nie aktualisiert. Der Maxwell-Tensor hätte damit in jedem Schritt am
// ursprünglichen Spalt ausgewertet -- die für TEAM 28 zentrale
// Schwebehöhen-Zeitreihe hätte so nie entstehen können.
//
// Fix: PhysicalObject::enable_free_motion() setzt is_moving() explizit, ohne
// eine Geschwindigkeit vorzugeben. mainloop() ruft sie für die Platte nach
// set_mechanics() auf.
//
// Dieser Test prüft die Randbedingung ohne CUDA-Kontext: reiner Zustand von
// PhysicalObject, kein Kernel-Aufruf.
// =============================================================================

#include "PhysicalObject.h"
#include <cstdio>

int main()
{
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        printf("  [%s] %s\n", ok ? "OK  " : "FEHL", what);
        if (!ok) ++failures;
    };

    // ---- 1: frisch konstruiertes Objekt bewegt sich nicht ----
    {
        LinearMetal plate("Test_Plate", 3.40e7, 1.0);
        check(!plate.is_moving(), "frisch konstruiertes Objekt: is_moving() == false");
    }

    // ---- 2: set_mechanics allein setzt is_moving nicht ----
    // (Regressionsschutz: enable_free_motion() darf nicht in set_mechanics
    // versteckt werden -- ein Objekt, das nur Masse/Trägheit für spätere
    // Zwecke trägt, soll nicht stillschweigend anfangen, sich zu bewegen.)
    {
        LinearMetal plate("Test_Plate", 3.40e7, 1.0);
        plate.set_mechanics(0.107, 1.1306e-4, 1.1306e-4, 2.2609e-4);
        check(!plate.is_moving(),
              "set_mechanics() allein setzt is_moving() nicht (kein versteckter Nebeneffekt)");
    }

    // ---- 3: der eigentliche Fix ----
    {
        LinearMetal plate("Test_Plate", 3.40e7, 1.0);
        plate.set_mechanics(0.107, 1.1306e-4, 1.1306e-4, 2.2609e-4);
        plate.enable_free_motion();
        check(plate.is_moving(),
              "enable_free_motion() setzt is_moving() -- apply_kinematics() wird nicht mehr uebersprungen");
    }

    // ---- 4: set_velocity setzt weiterhin is_moving (bestehendes Verhalten) ----
    {
        LinearMetal coil("Test_Coil", 0.0, 1.0);
        coil.set_velocity(0.0, 0.0, 1.0);
        check(coil.is_moving(), "set_velocity() setzt weiterhin is_moving() (unveraendertes Verhalten)");
    }

    printf("\n=============== %s ===============\n\n",
           failures == 0 ? "BESTANDEN: is_moving()-Gate korrekt"
                         : "FEHLGESCHLAGEN");
    return failures == 0 ? 0 : 1;
}
