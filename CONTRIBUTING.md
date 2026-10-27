# Mitwirken

Beiträge sind willkommen: Fehlerberichte, Messungen, Tests, Korrekturen und
Dokumentation. Der Quelltext und die Dokumentation wurden mit Hilfe von
KI-Systemen erzeugt; Änderungen von Hand oder mit KI-Unterstützung werden
gleich behandelt. Entscheidend ist, dass sie nachvollziehbar und geprüft sind.

## Bauen und testen

```sh
cmake -B build -DFEMBEM_CUDA_ARCH=75       # Compute Capability ohne Punkt
cmake --build build --target host_tests -j
ctest --test-dir build -L host --output-on-failure
```

Das ist der Lauf, den die CI macht. Er braucht das CUDA-Toolkit zum Bauen,
aber keine GPU. Weitere Hinweise (Windows, GCC-Versionen, cuDSS) in
[BUILD.md](BUILD.md).

## Vor einem Pull Request

- Die Tests mit der Marke `host` müssen bestehen.
- Eine Änderung am Verhalten bekommt einen Test, der **ohne** die Änderung
  fehlschlägt. Das Muster der vorhandenen Tests: ein Kontrollfall mit dem
  fehlerhaften Original, der durchfallen muss
  (siehe [tests/README.md](tests/README.md)).
- Ein neuer Test, der ohne GPU und ohne Dateien läuft, setzt als erste Zeile
  in `main()` `bf::g_backend = bf::Backend::CPU;` und bekommt die Marke
  `host` samt Eintrag in `host_tests`.
- Keine Build-Verzeichnisse, Binärdateien, `.obj`/`.exe`, Logdateien oder
  erzeugte PDFs einchecken (siehe `.gitignore`).
- Dateinamen nur mit ASCII-Zeichen, Zeilenenden LF.

## Stil

- Kommentare erklären das **Warum**: Annahmen, Fallstricke und was gemessen
  wurde. Die vorhandenen Kopfkommentare der Header sind das Vorbild.
- Zahlen in Kommentaren und Dokumenten nennen Netz, Parameter und Stand, damit
  sie überprüfbar sind.
- Ein Vergleich mit `NaN` ist immer falsch: `a < b` und `a > b` sind beide
  unwahr. Siehe [docs/design/nan-and-thresholds.md](docs/design/nan-and-thresholds.md).

## Fehler melden

Bitte mit Betriebssystem, Compiler- und CUDA-Version, GPU oder `--cpu`, dem
Aufruf und der Ausgabe. Die Vorlage unter *Issues* fragt das ab.

## Lizenz

Mit einem Beitrag stimmen Sie zu, dass er unter der [MIT-Lizenz](LICENSE)
steht.
