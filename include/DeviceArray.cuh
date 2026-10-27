#ifndef DEVICE_ARRAY_CUH
#define DEVICE_ARRAY_CUH

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

// Backend-Umschaltung. DeviceArray ist die einzige Stelle, an der der
// grosse, langlebige Speicher dieses Loesers angelegt wird -- entsprechend
// laeuft die gesamte Allokation ueber bf::malloc_managed_raw(). Im GPU-Modus
// ist das unveraendert cudaMallocManaged, im CPU-Modus schlichtes std::malloc.
// Dadurch braucht der CPU-Modus weder Treiber noch Geraet, und der restliche
// Code (der ohnehin ueber operator[] und data() auf dieselben Zeiger
// zugreift) bleibt unangetastet.
#include "Backend.cuh"

// =================================================================================
// Rigoroses Fehlerbehandlungs-Makro für CUDA-Aufrufe.
// Bricht nicht das Programm mit exit() ab, sondern wirft eine saubere C++ Exception.
// =================================================================================
#define CUDA_CHECK_ERROR(call)                                                              \
    do {                                                                                    \
        cudaError_t cuda_status = (call);                                                   \
        if (cuda_status != cudaSuccess) {                                                   \
            std::string error_message = std::string("CUDA Error: ") +                       \
                                        cudaGetErrorString(cuda_status) +                   \
                                        " at " + __FILE__ + ":" + std::to_string(__LINE__); \
            throw std::runtime_error(error_message);                                        \
        }                                                                                   \
    } while (0)


// =================================================================================
// Template-Klasse für die Verwaltung von Unified Memory (Host & Device)
// =================================================================================
template <typename T>
class DeviceArray {
private:
    T* m_data;          // Zeiger auf den Unified Memory Block
    size_t m_size;      // Anzahl der Elemente vom Typ T im Array

    // Statische Überprüfung zur Kompilierzeit:
    // cudaMallocManaged reserviert nur rohen Speicher und ruft KEINE Konstruktoren auf.
    // Daher dürfen nur triviale Datentypen (int, double, double3 etc.) verwendet werden.
    static_assert(std::is_trivially_copyable<T>::value, 
                  "DeviceArray requires trivially copyable types (PODs). Complex C++ objects are not allowed.");

public:
    // -----------------------------------------------------------------------------
    // Konstruktoren & Destruktor
    // -----------------------------------------------------------------------------
    
    // Standard-Konstruktor: Erstellt ein leeres Array
    DeviceArray() : m_data(nullptr), m_size(0) {}

    // Initialisierungs-Konstruktor: Alloziert direkt Speicher
    explicit DeviceArray(size_t number_of_elements) : m_data(nullptr), m_size(0) {
        allocate(number_of_elements);
    }

    // Destruktor: Gibt den Speicher auf der GPU sicher frei
    ~DeviceArray() {
        free();
    }

    // -----------------------------------------------------------------------------
    // Rule of Five: Kopier- und Verschiebe-Semantik (Move Semantics)
    // -----------------------------------------------------------------------------

    // Kopier-Konstruktor DEAKTIVIERT: Verhindert versehentliche "Double Free" Fehler
    // und versteckte, teure Speicherkopien.
    DeviceArray(const DeviceArray& other) = delete;
    
    // Kopier-Zuweisung DEAKTIVIERT
    DeviceArray& operator=(const DeviceArray& other) = delete;

    // Verschiebe-Konstruktor (Move Constructor): Übernimmt den Speicher eines anderen Arrays
    DeviceArray(DeviceArray&& other) noexcept : m_data(other.m_data), m_size(other.m_size) {
        other.m_data = nullptr;
        other.m_size = 0;
    }

    // Verschiebe-Zuweisung (Move Assignment)
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            free(); // Alten eigenen Speicher freigeben
            m_data = other.m_data;
            m_size = other.m_size;
            
            other.m_data = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    // -----------------------------------------------------------------------------
    // Speichermanagement
    // -----------------------------------------------------------------------------

    // Alloziert neuen Speicher. Wenn bereits Speicher alloziert war, wird dieser gelöscht.
    void allocate(size_t number_of_elements) {
        if (m_size == number_of_elements) {
            return; // Nichts tun, wenn Größe identisch
        }
        
        free();
        
        if (number_of_elements > 0) {
            m_size = number_of_elements;
            void* raw = nullptr;
            bf::malloc_managed_raw(&raw, m_size * sizeof(T));
            m_data = static_cast<T*>(raw);

            // Frisch alloziierter Speicher wird IMMER genullt.
            //
            // Frueher blieb er uninitialisiert, und das hat einen realen Fehler
            // ermoeglicht: apply_physical() benutzt d_temp_z als Zwischenpuffer
            // mit beta = 0, und bf::dgemv las y auch bei beta == 0 (inzwischen
            // behoben). Ein NaN-Bitmuster im rohen malloc-Speicher ergab dann
            // 0 * NaN = NaN und verwandelte ganze BEM-Bloecke in NaN.
            //
            // Der Fehler lag streng genommen in dgemv. Aber uninitialisierter
            // Speicher ist eine Falle, die jederzeit die naechste Stelle
            // findet: im Repo gibt es 62 allocate()-Aufrufe und keinen
            // einzigen, dem ein fill_zero() folgt. Ein memset bei der
            // Allokation kostet gegenueber dem, was danach hineingerechnet
            // wird, nichts -- und nimmt eine ganze Fehlerklasse aus dem Spiel.
            //
            // ACHTUNG: ein Aufruf mit UNVERAENDERTER Groesse kehrt oben frueh
            // zurueck und nullt NICHT. Wer eine bestehende Allokation leeren
            // will, ruft fill_zero() auf.
            bf::memset_d(m_data, 0, m_size * sizeof(T));
        }
    }

    // Gibt den Speicher explizit frei und setzt Parameter zurück
    void free() {
        if (m_data != nullptr) {
            bf::free_managed(m_data);
            m_data = nullptr;
            m_size = 0;
        }
    }

    // Setzt alle Bytes des Arrays auf Null (besonders wichtig für initiale Matrizen)
    void fill_zero() {
        if (m_data != nullptr && m_size > 0) {
            bf::memset_d(m_data, 0, m_size * sizeof(T));
        }
    }

    // -----------------------------------------------------------------------------
    // Performance-Optimierung für Unified Memory (Prefetching)
    // -----------------------------------------------------------------------------
    
    // Zieht die Daten asynchron in den L2 Cache der GPU
    // Im CPU-Modus ein No-Op: es gibt nur einen Adressraum, es ist nichts zu
    // migrieren. Die zahlreichen prefetch-Aufrufe im Restcode bleiben dadurch
    // unveraendert stehen und kosten dort nichts.
    void prefetch_to_device(int device_id = 0, cudaStream_t stream = 0) const {
        bf::prefetch_to_device_hint(m_data, m_size * sizeof(T), device_id, stream);
    }

    // Zieht die Daten asynchron in den RAM der CPU
    void prefetch_to_host(cudaStream_t stream = 0) const {
        bf::prefetch_to_host_hint(m_data, m_size * sizeof(T), stream);
    }

    // -----------------------------------------------------------------------------
    // Datenzugriff
    // -----------------------------------------------------------------------------

    // Gibt die Anzahl der Elemente zurück
    size_t size() const {
        return m_size;
    }

    // Gibt den rohen Pointer zurück (wird benötigt, um ihn an CUDA-Kernel zu übergeben)
    T* data() {
        return m_data;
    }

    const T* data() const {
        return m_data;
    }

    // Array-Operator für Lese-/Schreibzugriff auf der CPU
    // ACHTUNG: Dieser Operator darf NUR vom Host (CPU) aufgerufen werden!
    T& operator[](size_t index) {
        // Optional: Assert für Bounds-Checking im Debug-Modus
        // assert(index < m_size);
        return m_data[index];
    }

    // Array-Operator für reinen Lesezugriff auf der CPU
    const T& operator[](size_t index) const {
        return m_data[index];
    }
};

#endif // DEVICE_ARRAY_CUH