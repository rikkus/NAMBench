#!/usr/bin/env python3
"""Read the fpi USB CDC console, surviving re-enumeration.

Why this exists, and why `cat`/`dd`/`screen` are not substitutes:

  * The Pico SDK gates USB stdio output on stdio_usb_connected(), which by
    default "actually checks DTR". macOS does not raise DTR when a program opens
    /dev/cu.usbmodem*, so a firmware built without
    PICO_STDIO_USB_CONNECTION_WITHOUT_DTR can run perfectly while emitting
    nothing. This script raises DTR/RTS anyway, so it works either way.
  * Rebooting or reflashing the board invalidates the open file descriptor
    ("Device not configured"), which kills a naive reader at exactly the moment
    the interesting output starts. This reopens and keeps going.

    ./serial-probe.py [seconds] [port]
"""

import fcntl
import glob
import os
import select
import struct
import sys
import termios
import time

DEFAULT_SECONDS = 15


def find_port() -> str | None:
    ports = sorted(glob.glob("/dev/cu.usbmodem*"))
    return ports[0] if ports else None


def open_port(port: str) -> int | None:
    try:
        fd = os.open(port, os.O_RDWR | os.O_NONBLOCK | os.O_NOCTTY)
    except OSError:
        return None

    attrs = termios.tcgetattr(fd)
    # Fully raw: no canonical processing, no echo, no signals, no flow control.
    attrs[0] = attrs[1] = attrs[2] = 0
    attrs[3] = 0
    attrs[4] = attrs[5] = termios.B115200
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)

    bits = getattr(termios, "TIOCM_DTR", 0x002) | getattr(termios, "TIOCM_RTS", 0x004)
    try:
        fcntl.ioctl(fd, termios.TIOCMBIS, struct.pack("I", bits))
    except (OSError, AttributeError):
        pass  # the firmware should not need it; not fatal
    return fd


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SECONDS
    port = sys.argv[2] if len(sys.argv) > 2 else find_port()
    if not port:
        port = "/dev/cu.usbmodem1101"  # it may reappear after a reboot

    print(f"[probe] watching {port} for {seconds:g}s (survives re-enumeration)",
          file=sys.stderr)

    deadline = time.time() + seconds
    total = 0
    fd = None
    reopens = 0
    last_open_attempt = 0.0

    try:
        while time.time() < deadline:
            if fd is None:
                now = time.time()
                if now - last_open_attempt < 0.25:
                    time.sleep(0.05)
                    continue
                last_open_attempt = now
                path = find_port() or port
                newfd = open_port(path)
                if newfd is None:
                    continue
                fd = newfd
                reopens += 1
                if reopens > 1:
                    print(f"[probe] reopened {path} (device re-enumerated)",
                          file=sys.stderr)
                continue

            try:
                ready, _, _ = select.select([fd], [], [], 0.25)
                if not ready:
                    continue
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                continue
            except OSError as exc:
                # Device went away (reboot/reflash). Drop it and wait for it back.
                print(f"[probe] port lost ({exc.strerror}) — waiting for it to return",
                      file=sys.stderr)
                try:
                    os.close(fd)
                except OSError:
                    pass
                fd = None
                continue

            if chunk:
                total += len(chunk)
                sys.stdout.write(chunk.decode("utf-8", "replace"))
                sys.stdout.flush()
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    if total == 0:
        print(
            "\n[probe] NOTHING received on the CDC console.\n"
            "        The firmware writes a beat every 250 ms, so either it is not\n"
            "        running, or it is faulting before its first write. A build with\n"
            "        the fault handlers will say so; reflash and re-run.",
            file=sys.stderr,
        )
        return 1

    print(f"\n[probe] {total} bytes received", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
