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

for attempt in 1 2 3; do
    echo "::group::apt, Versuch $attempt: $*"
    timeout "$UPDATE_TIMEOUT" "${APT[@]}" update -q \
        && timeout "$INSTALL_TIMEOUT" "${APT[@]}" install -y -q --no-install-recommends "$@"
    rc=$?                     # 124 = Zeitgrenze erreicht (haengender apt)
    echo "::endgroup::"
    if [ "$rc" -eq 0 ]; then
        exit 0
    fi
    echo "::warning::apt-Versuch $attempt fehlgeschlagen oder abgebrochen (Exit $rc), neuer Versuch"
    sudo pkill -9 -x apt-get || true
    sudo pkill -9 -x dpkg || true
    sudo DEBIAN_FRONTEND=noninteractive dpkg --configure -a || true
    sleep 20
done
echo "::error::apt nach drei Versuchen nicht erfolgreich"
exit 1
