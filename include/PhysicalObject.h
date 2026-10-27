#ifndef PHYSICAL_OBJECT_H
#define PHYSICAL_OBJECT_H

#include <vector>
#include <string>
#include "MaterialTypes.cuh"
#include "Mesh.cuh"
#include <cmath>

// Hilfsstruktur für Quaternionen (w + xi + yj + zk)
struct Quaternion {
    double w, x, y, z;

    // Identitäts-Quaternion (keine Rotation)
    Quaternion() : w(1.0), x(0.0), y(0.0), z(0.0) {}

    // Normalisierung, um numerischen Drift (Akkumulation von Rundungsfehlern) zu verhindern
    void normalize() {
        double mag = std::sqrt(w * w + x * x + y * y + z * z);
        if (mag > 0.0) {
            w /= mag; x /= mag; y /= mag; z /= mag;
        }
    }
};

class PhysicalObject {
protected:
    std::string name;
    MaterialProperties props;
    std::vector<int> hex_indices;

    // --- Rigorose Mechanik ---
    double mass;
    double inertia[3]; // Hauptträgheitsmomente I_xx, I_yy, I_zz im körperfesten System

    double3 velocity;         // v_x, v_y, v_z (im weltfesten System)
    double3 angular_velocity; // omega_x, omega_y, omega_z (im KÖRPERFESTEN System!)

    Quaternion orientation;   // Aktuelle Orientierung des Körpers im Raum

    // Hilfsfunktion: Wandelt das aktuelle Quaternion in die 3x3 Rotationsmatrix für CUDA um
    void update_rotation_matrix();

    // Cache für die eindeutigen Knoten-IDs dieses Objekts
    DeviceArray<int> cached_unique_nodes;

public:
    PhysicalObject(const std::string& obj_name);
    virtual ~PhysicalObject() = default;

    std::string get_name() const { return name; }

    // Setup der Mechanik
    void set_mechanics(double m, double I_xx, double I_yy, double I_zz) {
        mass = m;
        inertia[0] = I_xx;
        inertia[1] = I_yy;
        inertia[2] = I_zz;
    }

    // Baut die Liste der eindeutigen Knoten auf (wird 1x beim Setup aufgerufen)
    void build_node_cache(const Mesh& mesh);

    // Getter für den Kernel
    int get_num_unique_nodes() const { return cached_unique_nodes.size(); }
    const int* get_unique_nodes_ptr() const { return cached_unique_nodes.data(); }

    // Aufgabe 7.3: Rigorose Integration der Bewegungsgleichungen
    void apply_newtons_law(double3 force, double3 torque, double dt);

    virtual void update_state(double t, double dt) = 0;
//};

// Abstrakte Basisklasse für alle physikalischen Gebiete
//class PhysicalObject {

protected:
    // Mechanische Eigenschaften
    //double mass;
    double inertia_tensor[3]; // Diagonalelemente I_xx, I_yy, I_zz (für vereinfachte Rotation)

    // Kinematische Zustandsgrößen (Geschwindigkeit)
    //double3 velocity;
    //double3 angular_velocity;

    //std::string name;
    //MaterialProperties props;
    
    // Liste aller globalen Hexaeder-IDs, die zu diesem Objekt gehören
    //std::vector<int> hex_indices;

public:
    /*
    PhysicalObject(const std::string& obj_name) : name(obj_name) {
        props.is_moving = false;
        props.translation = make_double3(0.0, 0.0, 0.0);
        // Identitätsmatrix für Rotation
        for(int i=0; i<3; ++i) 
            for(int j=0; j<3; ++j) 
                props.rot_matrix[i][j] = (i == j) ? 1.0 : 0.0;
    }

    virtual ~PhysicalObject() = default;
    */
    // Weist dem Objekt ein Hexaeder zu
    void add_hex(int hex_id) { hex_indices.push_back(hex_id); }
    
    const std::vector<int>& get_hex_indices() const { return hex_indices; }

    bool   has_lower_stop = false;
    double lower_stop_z   = 0.0;
    MaterialProperties get_properties() const { return props; }
    bool is_moving() const { return props.is_moving; }

    // Berechnet den Schwerpunkt (Center of Mass) basierend auf den zugeordneten Knoten
    void compute_center_of_mass(const Mesh& mesh);

    // Virtuelle Funktion, die in jedem Zeitschritt aufgerufen wird, 
    // um die Kinematik (Translation/Rotation) oder den Quellenstrom anzupassen.
    //virtual void update_state(double t, double dt) = 0;

    void set_mass(double m) { mass = m; }
    /// Lesezugriff auf die Masse. Es gab bisher nur set_mass() -- die Masse
    /// wurde ausserhalb der Klasse nirgends gebraucht, weil apply_newtons_law()
    /// die Bewegungsgleichung selbst loest. Der harmonische Pfad braucht sie
    /// aber: die Schwebehoehe ist die Hoehe, bei der F_z = m*g gilt, und
    /// dieser Vergleich findet ausserhalb statt.
    double get_mass() const { return mass; }
    void set_inertia(double Ixx, double Iyy, double Izz) {
        inertia_tensor[0] = Ixx; inertia_tensor[1] = Iyy; inertia_tensor[2] = Izz;
    }

    // FIX: is_moving() gated bislang ausschliesslich ueber set_velocity(), das
    // fuer kinematisch VORGEGEBENE Geschwindigkeit gedacht ist. Ein kraftgetrieb-
    // enes Objekt -- wie die Platte, deren Bewegung ueber apply_newtons_law()
    // integriert wird -- setzte is_moving() nirgends, blieb also fuer immer
    // false. apply_kinematics() (Kinematics.cu) prueft is_moving() als Gate,
    // bevor es die integrierte Translation ueberhaupt auf die Netzknoten
    // anwendet.
    //
    // Folge ohne diesen Aufruf: apply_newtons_law() integriert
    // props.translation und velocity bei jedem Zeitschritt korrekt, aber
    // mesh.nodes_current wird nie aktualisiert. Der Maxwell-Tensor wertet
    // damit in jedem Schritt am urspruenglichen Spalt (3.8 mm) aus. Die fuer
    // TEAM 28 zentrale Schwebehoehen-Zeitreihe koennte so nie entstehen --
    // die Platte bliebe geometrisch bei z = 3.8 mm, unabhaengig davon, was
    // die Kraftintegration berechnet.
    //
    // Absichtlich als eigene Methode und nicht als Nebenwirkung von
    // set_mechanics(), damit ein Objekt, das rein informativ Masse und
    // Traegheit traegt (z.B. fuer eine spaetere Erweiterung), nicht
    // stillschweigend anfaengt, sich zu bewegen.
    void enable_free_motion() { props.is_moving = true; }

    // -------------------------------------------------------------------------
    // Untere Anschlagbedingung (Auflage)
    // -------------------------------------------------------------------------
    // Bei TEAM 28 ruht die Platte fuer t <= 0 auf der Wickelform, also auf
    // z = 3.8 mm ueber der Spulenoberkante -- das ist genau die Startlage des
    // Netzes. Sie kann NICHT tiefer sinken. Ohne diese Bedingung faellt die
    // Platte im Modell durch die Wickelform, sobald die Kraft einmal kleiner
    // als das Gewicht ist. Genau das passierte im ersten Referenzlauf: die
    // Verschiebung erreichte -1.2852 mm, also eine Schwebehoehe von 2.5 mm,
    // rund 1.3 mm INNERHALB der Wickelform.
    //
    // Bewusst als Bedingung auf translation.z >= stop und nicht auf eine
    // absolute Hoehe: dann braucht das Objekt keine Kenntnis der Geometrie, und
    // fuer einen Koerper, der auf einer Auflage startet, ist stop = 0 richtig.
    // Standardmaessig AUS, damit sich fuer Objekte ohne Auflage nichts aendert.
    /// Platte auf eine vorgegebene Verschiebung setzen. Fuer --plate-z: damit
    /// laesst sich der Loeser an der GEMESSENEN Ruhelage pruefen, wo das
    /// Zyklusmittel der Vertikalkraft exakt m*g sein MUSS. Das ist der einzige
    /// streng exakte Referenzwert, den die Messung liefert -- alles andere
    /// (Anfangsbeschleunigung, Scheitelhoehe) ist eine Ableitung mit Fehlerbalken.
    void set_translation_z(double dz) { props.translation.z = dz; }

    void set_lower_stop(double translation_z_min) {
        lower_stop_z = translation_z_min;
        has_lower_stop = true;
    }
    bool at_lower_stop() const {
        return has_lower_stop && props.translation.z <= lower_stop_z;
    }
};

// -----------------------------------------------------------------------------
// Abgeleitete Klassen
// -----------------------------------------------------------------------------

class LinearMetal : public PhysicalObject {
public:
    LinearMetal(const std::string& name, double kappa, double mu_r) : PhysicalObject(name) {
        props.type = RegionType::LINEAR_METAL;
        props.kappa = kappa;
        props.mu_relative = mu_r;
    }

    void update_state(double t, double dt) override {
        (void)t; (void)dt;   // noch ungenutzt, siehe unten
        // Hier könnte später die Logik für Newtons Bewegungsgleichungen aufgerufen werden
        // z.B. props.translation.z = 0.5 * 9.81 * t * t; (freier Fall)
    }
    
    // Setup für erzwungene Kinematik (z.B. konstante Geschwindigkeit)
    void set_velocity(double vx, double vy, double vz) {
        props.is_moving = true;
        // Speichere temporär in translation für update_state
        props.translation = make_double3(vx, vy, vz); 
    }
};

class CoilCurrentDriven : public PhysicalObject {
private:
    double amplitude;
    double frequency;
    double3 current_direction;

public:
    CoilCurrentDriven(const std::string& name, double amp, double freq, double3 dir) 
        : PhysicalObject(name), amplitude(amp), frequency(freq), current_direction(dir) {
        props.type = RegionType::COIL_CURRENT;
        props.kappa = 0.0; // Spulen werden in der FEM oft als strömungsfreie B-Feld-Quellen modelliert
        props.mu_relative = 1.0;
        props.current_is_azimuthal = false;
        props.current_axis_origin = make_double3(0.0, 0.0, 0.0);
    }

    // Ringspule: der Strom laeuft azimutal um die z-Achse durch axis_origin.
    // Fuer TEAM 28 ist das die physikalisch richtige Wahl, siehe die Erklaerung
    // an current_is_azimuthal in MaterialTypes.cuh. Eine konstante kartesische
    // Richtung ist im Ringgebiet nicht divergenzfrei.
    void set_azimuthal(double3 axis_origin) {
        props.current_is_azimuthal = true;
        props.current_axis_origin = axis_origin;
    }

    void set_cartesian(double3 dir) {
        props.current_is_azimuthal = false;
        current_direction = dir;
    }

    void update_state(double t, double dt) override {
        (void)dt;
        // Sinusförmige Anregung i(t) = I_max * sin(2*pi*f*t) (vgl. Kurz et al.)
        double current = amplitude * sin(2.0 * 3.14159265359 * frequency * t);
        if (props.current_is_azimuthal) {
            // Nur der Betrag; die Richtung berechnet der Kernel pro Gausspunkt.
            props.current_density_x = current;
            props.current_density_y = 0.0;
            props.current_density_z = 0.0;
        } else {
            props.current_density_x = current_direction.x * current;
            props.current_density_y = current_direction.y * current;
            props.current_density_z = current_direction.z * current;
        }
    }
};


void assign_materials_by_bounding_box(Mesh& mesh, PhysicalObject* obj,
    double x_min, double x_max,
    double y_min, double y_max,
    double z_min, double z_max);

void assign_materials_by_component_id(Mesh& mesh, PhysicalObject* obj, int target_component_id);

#endif // PHYSICAL_OBJECT_H