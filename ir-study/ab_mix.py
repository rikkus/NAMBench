#!/usr/bin/env python3
"""Turn ab_render output into level-matched A/B WAVs and a boosted difference.

    python3 ir-study/ab_mix.py DIR NAME [NAME ...]

For each NAME, reads DIR/NAME_up.f64 (AudioDSPTools main) and DIR/NAME_fft.f64
(the partitioned-ir branch), both raw float64 at 48 kHz, and writes to DIR/out/:

    N-NAME-A-upstream.wav                 float32, shared peak at -1 dBFS
    N-NAME-B-fft.wav                      the same gain as A
    N-NAME-difference-boosted-XdB.wav     (B - A) in double, raised X dB so its
                                          peak lands just under -6 dBFS

The same gain goes on A and B, so they can be switched between without
re-levelling. Stdlib only.
"""
import array, math, os, struct, sys


def load(path):
    a = array.array('d')
    a.frombytes(open(path, 'rb').read())
    return a


def write_f32(path, samples, rate=48000):
    data = array.array('f', samples).tobytes()
    hdr = b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVE'
    hdr += b'fmt ' + struct.pack('<IHHIIHH', 16, 3, 1, rate, rate * 4, 4, 32)
    open(path, 'wb').write(hdr + b'data' + struct.pack('<I', len(data)) + data)


def db(x):
    return 20 * math.log10(x) if x > 0 else float('-inf')


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, names = sys.argv[1], sys.argv[2:]
    out = os.path.join(src, 'out')
    os.makedirs(out, exist_ok=True)
    for i, name in enumerate(names, 1):
        up, ff = load(f'{src}/{name}_up.f64'), load(f'{src}/{name}_fft.f64')
        peak = max(max(abs(v) for v in up), max(abs(v) for v in ff))
        g = 10 ** (-1 / 20) / peak
        up = [v * g for v in up]
        ff = [v * g for v in ff]
        diff = [b - a for a, b in zip(up, ff)]
        rs = math.sqrt(sum(v * v for v in up) / len(up))
        rd = math.sqrt(sum(v * v for v in diff) / len(diff))
        pd = max(abs(v) for v in diff)
        boost = math.floor(db(0.5 / pd)) if pd > 0 else 0
        gb = 10 ** (boost / 20)
        write_f32(f'{out}/{i}-{name}-A-upstream.wav', up)
        write_f32(f'{out}/{i}-{name}-B-fft.wav', ff)
        write_f32(f'{out}/{i}-{name}-difference-boosted-{boost}dB.wav', [v * gb for v in diff])
        print(f'{name:12} difference RMS {db(rd / rs):7.1f} dB re signal | '
              f'difference peak {db(pd):7.1f} dBFS | boost +{boost} dB')


if __name__ == '__main__':
    main()
