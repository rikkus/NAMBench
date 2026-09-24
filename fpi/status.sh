#!/usr/bin/env bash
# Is fpi attached, and in which state?
#
#   ./status.sh
#
# Three states matter (see docs/fpi/README.md):
#   BOOTSEL  — a volume named RP2350 is mounted, ready for a .uf2
#   CDC      — firmware is running and exposing a serial console
#   AUDIO    — firmware is running as a UAC2 sound card, so there is no /dev node

set -uo pipefail

echo "== USB devices (Raspberry Pi / RP2350) =="
# ioreg, not system_profiler: SPUSBDataType returns nothing at all on this
# macOS build (verified), which made an earlier version of this section report an
# empty device list even with the board attached.
if command -v ioreg >/dev/null; then
    usb="$(ioreg -p IOUSB -w0 -l 2>/dev/null \
           | grep -iE '"USB Product Name"|"USB Vendor Name"|"idVendor"|"idProduct"')"
    if [[ -n "$usb" ]]; then
        echo "$usb" | sed 's/^ *| *//; s/^/  /'
    else
        echo "  no USB devices reported by ioreg"
    fi
    # 11914 = 0x2e8a, Raspberry Pi. Product 9 = the SDK's CDC descriptor;
    # 15 = 0x000f, the RP2350 boot ROM.
    if grep -q '"idVendor" = 11914' <<<"$usb"; then
        echo "  -> Raspberry Pi device present"
    fi
else
    echo "  (ioreg unavailable)"
fi

echo
echo "== BOOTSEL volume =="
if [[ -d /Volumes/RP2350 ]]; then
    echo "  /Volumes/RP2350 is mounted  <- board is in the bootloader, ready to flash"
    ls -la /Volumes/RP2350 2>/dev/null | sed 's/^/    /'
else
    echo "  no /Volumes/RP2350  (board is not in BOOTSEL, or not plugged in)"
fi

echo
echo "== Serial console nodes =="
found=0
for n in /dev/cu.usbmodem* /dev/tty.usbmodem*; do
    [[ -e "$n" ]] || continue
    echo "  $n"
    found=1
done
[[ $found -eq 1 ]] || echo "  none  (no CDC firmware running — expected on a blank board)"

echo
echo "== Audio devices =="
if command -v system_profiler >/dev/null; then
    system_profiler SPAudioDataType 2>/dev/null \
        | grep -iE "pico|nam|rp2350" | sed 's/^/  /' \
        || true
fi

cat <<'EOF'

Next steps depend on the above:
  RP2350 mounted  -> cp fpi/build/src/hello.uf2 /Volumes/RP2350
  usbmodem node   -> ./serial-probe.py 20     (NOT screen/cat/dd — see
                     docs/fpi/TOOLCHAIN.md, "the silent-console trap")
  nothing at all  -> hold BOOT, plug in, release BOOT; and check the cable
                     carries data, not just power.
EOF
