#!/usr/bin/env python3
"""Survey a folder of impulse-response WAVs: what the plugin would convolve,
and how much of each IR a shorter cut would throw away.

    python3 ir-study/ir_survey.py ~/Documents/Audio/IRs [sample-size]

Every file by default; pass a sample size for a quicker, seeded random subset.
A subset can miss the extremes: in the library this was written against, the
worst cases are a handful of rear, side and room mic positions.

1. Inventory of every WAV: sample rate, bit depth, channels, and the tap count
   after resampling to 48 kHz and AudioDSPTools' 8192-tap truncation.
2. For every file (or a seeded random sample), in 48 kHz taps:
   - the last sample within 60 dB of the peak (the per-sample, RT60-style view)
   - the point by which 99.99% of the energy has arrived
   - the energy a cut at 1024/2048/4096/8192 taps discards, relative to the
     whole IR. For broadband input that is roughly the level of the error the
     cut adds to the output, which is why it, not the per-sample view, says
     whether a cut is audible.
   - the level of the last 10% of the file relative to the peak (50 ms for the
     usual 500 ms file): the recording's noise floor, for comparison.

Stdlib only; slow-ish because 24-bit samples are decoded in Python.
"""
import collections, math, os, random, struct, sys


def read(path):
    d = open(path, 'rb').read()
    i, fmt, ch, sr, bits, raw = 12, None, 1, 0, 0, b''
    while i + 8 <= len(d):
        cid, sz = d[i:i + 4], struct.unpack('<I', d[i + 4:i + 8])[0]
        if cid == b'fmt ':
            fmt, ch, sr, _, _, bits = struct.unpack('<HHIIHH', d[i + 8:i + 24])
            if fmt == 0xFFFE:
                fmt = struct.unpack('<H', d[i + 32:i + 34])[0]
        elif cid == b'data':
            raw = d[i + 8:i + 8 + sz]
        i += 8 + sz + (sz & 1)
    return fmt, ch, sr, bits, raw


def samples(fmt, ch, bits, raw):
    step = bits // 8 * ch
    n = len(raw) // step
    if fmt == 3 and bits == 32:
        return [struct.unpack_from('<f', raw, j * step)[0] for j in range(n)]
    if bits == 16:
        return [struct.unpack_from('<h', raw, j * step)[0] / 32768 for j in range(n)]
    if bits == 24:
        return [int.from_bytes(raw[j * step:j * step + 3], 'little', signed=True) / 8388608 for j in range(n)]
    if bits == 32:
        return [struct.unpack_from('<i', raw, j * step)[0] / 2147483648 for j in range(n)]
    raise ValueError(f'unsupported format {fmt}/{bits}')


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs if f.lower().endswith('.wav'))

    inv = collections.Counter()
    taps = collections.Counter()
    for p in paths:
        fmt, ch, sr, bits, raw = read(p)
        n = len(raw) // (bits // 8 * ch)
        inv[(sr, bits, ch)] += 1
        taps[min(8192, round(n * 48000 / sr))] += 1
    print(f'{len(paths)} WAV files')
    for (sr, bits, ch), c in sorted(inv.items()):
        print(f'  {c:5d}  {sr} Hz  {bits}-bit  {ch} ch')
    print('taps after resampling to 48 kHz and truncating at 8192:')
    for t, c in sorted(taps.items()):
        print(f'  {c:5d}  {t} taps')

    random.seed(1)
    sample = random.sample(paths, k) if 0 < k < len(paths) else paths
    last60, e9999, floors = [], [], []
    cuts = [1024, 2048, 4096, 8192]
    lost = {c: [] for c in cuts}
    for p in sample:
        fmt, ch, sr, bits, raw = read(p)
        x = samples(fmt, ch, bits, raw)
        pk = max(abs(v) for v in x) or 1.0
        scale = 48000 / sr
        last60.append(max((i for i, v in enumerate(x) if abs(v) > pk * 1e-3), default=0) * scale)
        e = [v * v for v in x]
        tot = sum(e)
        acc = 0.0
        for i, v in enumerate(e):
            acc += v
            if acc >= tot * 0.9999:
                e9999.append(i * scale)
                break
        for c in cuts:
            n = round(c / scale)
            lost[c].append(10 * math.log10(max(sum(e[n:]), 1e-30) / tot))
        tail = x[-len(x) // 10:]
        floors.append(20 * math.log10(math.sqrt(sum(v * v for v in tail) / len(tail)) / pk + 1e-30))

    def spread(v):
        v = sorted(v)
        n = len(v)
        return f'min {v[0]:.0f}  median {v[n // 2]:.0f}  p90 {v[int(n * .9)]:.0f}  max {v[-1]:.0f}'

    print(f'\n{len(sample)} files, in 48 kHz taps:')
    print('  last sample within 60 dB of peak:', spread(last60))
    print('  99.99% of energy arrived by:     ', spread(e9999))
    print('energy discarded by a cut, relative to the whole IR:')
    for c in cuts:
        v = sorted(lost[c])
        print(f'  at {c:5d} taps: median {v[len(v) // 2]:6.1f} dB, p90 {v[int(len(v) * .9)]:6.1f} dB, '
              f'worst {v[-1]:6.1f} dB')
    f = sorted(floors)
    print(f'last 10% of each file relative to peak: median {f[len(f) // 2]:.1f} dB, loudest {f[-1]:.1f} dB')


if __name__ == '__main__':
    main()
