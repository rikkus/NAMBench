#!/usr/bin/env bash
# Read the fpi USB CDC console.
#
#   ./monitor.sh            follow the console until Ctrl-C
#   ./monitor.sh --raw      same, but via cat (no terminal control at all)
#
# Uses screen by default because it handles the serial line properly, and falls
# back to cat. Either way this is more reliable than typing the device node by
# hand — a mistyped baud rate is a favourite way to get an apparently dead
# terminal, since the board only ever writes when it has something to say.

set -uo pipefail

find_port() {
    local p
    for p in /dev/cu.usbmodem*; do
        [[ -e "$p" ]] && { echo "$p"; return 0; }
    done
    return 1
}

PORT="$(find_port)" || {
    cat >&2 <<'EOF'
No /dev/cu.usbmodem* node found.

The board is either not attached, or attached but running firmware that does not
expose a CDC console (the RP2350 boot ROM shows up as a mass-storage volume and
no serial port at all).

  ./status.sh              # see what the laptop currently detects
  ls /Volumes/RP2350       # if this exists, flash a firmware first
EOF
    exit 1
}

echo "console: $PORT   (screen: quit with Ctrl-A then k then y)" >&2
echo "if you flashed just now, expect the banner immediately, then a beat every 250 ms" >&2
echo >&2

if [[ "${1:-}" == "--raw" ]]; then
    exec cat "$PORT"
fi

if command -v screen >/dev/null; then
    exec screen "$PORT" 115200
fi

exec cat "$PORT"
