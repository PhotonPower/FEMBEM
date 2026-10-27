#!/usr/bin/env bash
# apt-Pakete in der CI installieren, robust gegen haengende Runner.
#
#   .github/scripts/apt-install.sh paket1 paket2 ...
#
# Auf GitHub-Runnern hing "apt-get install" mehrfach ueber zehn Minuten,
# waehrend derselbe Schritt in einem Nachbarjob in zwei Minuten durchlief.
# Deshalb: jeder Versuch mit Zeitgrenze, bis zu drei Versuche, vor einem
# neuen Versuch einen halb fertigen Paketstand aufraeumen.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
APT=(sudo DEBIAN_FRONTEND=noninteractive apt-get
     -o Acquire::Retries=3 -o Acquire::http::Timeout=60 -o Dpkg::Use-Pty=0)

# Beobachtet (Annotationen der CI-Laeufe): die Haenger sitzen in
# "apt-get install", nicht im Update; ein zweiter Versuch geht meist durch.
# Vermutete Ursache, ohne Logs nicht belegt: der man-db-Trigger, der nach der
# Installation den Handbuchindex neu aufbaut und auf Ubuntu-24.04-Runnern sehr
# lange brauchen kann (CUDA-Toolkit und TeX bringen viele Handbuchseiten mit).
# Das Abschalten ist unschaedlich: die CI braucht keinen Handbuchindex.
if [ -e /var/lib/man-db/auto-update ]; then
    sudo rm -f /var/lib/man-db/auto-update
    echo "man-db: automatische Aktualisierung abgeschaltet"
fi

# Zeitgrenzen in Sekunden; ueberschreibbar, um das Skript selbst zu testen.
# Ein normaler Durchlauf braucht unter einer Minute (CUDA-Toolkit, TeX).
UPDATE_TIMEOUT="${APT_UPDATE_TIMEOUT:-300}"
INSTALL_TIMEOUT="${APT_INSTALL_TIMEOUT:-300}"

# Zustand festhalten, wenn ein Versuch in die Zeitgrenze laeuft: welche
# Prozesse laufen (und wie lange schon), wie viel ist heruntergeladen, was
# meldeten dpkg und apt zuletzt. Als Annotation (::warning::), damit es ohne
# das Log ueber die API lesbar ist.
diagnose() {   # $1: Beschreibung des Zeitpunkts
    local procs cache partial last_dpkg last_term
    # Verdaechtige einer langen Installation: Download (http), dpkg, Trigger,
    # mandb, ldconfig, Python-Nachbearbeitung, DKMS (Kernelmodul uebersetzen).
    procs=$(ps -eo etimes=,args= 2>/dev/null \
        | grep -E 'apt|dpkg|http|man-db|mandb|trigger|ldconfig|update-|depmod|dkms|cc1|gcc|make|python3|sh -c' \
        | grep -v -E 'grep|apt-install\.sh|^ *[0-9]+ sleep [0-9]+$' \
        | awk '{t=$1; $1=""; printf "%ss %s; ", t, substr($0, 2, 110)}' \
        | cut -c1-900)
    cache=$(du -sm /var/cache/apt/archives 2>/dev/null | cut -f1)
    partial=$(du -sm /var/cache/apt/archives/partial 2>/dev/null | cut -f1)
    last_dpkg=$(tail -n 2 /var/log/dpkg.log 2>/dev/null | tr '\n' ' ' | cut -c1-220)
    last_term=$(tr '\r' '\n' < /var/log/apt/term.log 2>/dev/null | grep -v '^ *$' \
        | tail -n 3 | tr '\n' ' ' | tr -s ' ' | cut -c1-300)
    echo "::warning::apt-Diagnose Versuch $attempt, $1: Cache ${cache:-?} MB, davon partial ${partial:-?} MB | dpkg.log: ${last_dpkg:-leer} | term.log: ${last_term:-leer} | Prozesse: ${procs:-keine}"
}

for attempt in 1 2 3; do
    echo "::group::apt, Versuch $attempt: $*"
    snap="/tmp/apt-diagnose-$attempt.txt"; rm -f "$snap"
    timeout "$UPDATE_TIMEOUT" "${APT[@]}" update -q
    rc=$?                     # 124 = Zeitgrenze erreicht (haengender apt)
    if [ "$rc" -eq 0 ]; then
        # Waechter: Schnappschuss 10 s vor der Zeitgrenze der Installation,
        # solange apt noch laeuft -- danach sind die Prozesse schon beendet.
        ( sleep $((INSTALL_TIMEOUT > 20 ? INSTALL_TIMEOUT - 10 : INSTALL_TIMEOUT / 2)) \
          && diagnose "Installation, kurz vor der Zeitgrenze" > "$snap" ) &
        watcher=$!
        timeout "$INSTALL_TIMEOUT" "${APT[@]}" install -y -q --no-install-recommends "$@"
        rc=$?
        kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
    fi
    echo "::endgroup::"
    if [ "$rc" -eq 0 ]; then
        exit 0
    fi
    if [ -s "$snap" ]; then cat "$snap"; else diagnose "nach Abbruch (Exit $rc)"; fi
    echo "::warning::apt-Versuch $attempt fehlgeschlagen oder abgebrochen (Exit $rc), neuer Versuch"
    sudo pkill -9 -x apt-get || true
    sudo pkill -9 -x dpkg || true
    sudo DEBIAN_FRONTEND=noninteractive dpkg --configure -a || true
    sleep 20
done
echo "::error::apt nach drei Versuchen nicht erfolgreich"
exit 1
