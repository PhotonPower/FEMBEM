// main.cpp
/*
Copyright (c) 2026 PhotonPower
Source code and documentation were created using AI systems
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and /or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


#include "fem.cuh"
#include "Backend.cuh"

#include <cstdlib>

// =================================================================================
// Das Hauptprogramm
// =================================================================================
int main(int argc, char* argv[]) {
    // Wahl des Rechenwerks. MUSS als allererstes passieren -- ab hier
    // legt jede DeviceArray-Instanz Speicher an, und ob das cudaMallocManaged
    // oder std::malloc wird, entscheidet genau dieser Aufruf. Ein spaeterer
    // Wechsel wuerde Puffer des einen Modus im anderen freigeben.
    //
    // Ausgewertet werden --cpu, --gpu, --threads=N und --help sowie die
    // Umgebungsvariable FEMBEM_BACKEND. Alle uebrigen Argumente bleiben
    // unangetastet und werden wie bisher in mainloop() gelesen (Schrittzahl,
    // "nobem", "noilu").
    if (!bf::select_backend_from_args(argc, argv)) {
        return EXIT_SUCCESS;   // --help wurde angefordert
    }

    // Optionales Argument: Anzahl Zeitschritte (Kurztest), z. B. "fembem 3"
    return mainloop(argc, argv);
}
