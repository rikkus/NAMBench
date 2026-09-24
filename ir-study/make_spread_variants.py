#!/usr/bin/env python3
"""Generate the convolvers spread_ab.cpp compares.

    python3 ir-study/make_spread_variants.py <adt tree> OUTDIR <rev>...

  PC0, PC1, ...  dsp/PartitionedConvolution.{h,cpp} at each <rev>, in order.
                 PC0 is the reference every other variant's output is compared
                 with, so pass the baseline first.
  PCt            e2dc6bc with a clock read between the phases of _RunFftBlock:
                 forward FFT, multiplies, inverse FFT, overlap-add. spread_ab
                 reports them as the attribution table.

Each is renamed so all of them link into one binary; the
ConvolutionImplementation enum lives once, in common_enum.h, as in
make_storage_variants.py. variants.h includes them all and lists them for
spread_ab.cpp.

Then, from the repository root:
  c++ -std=c++17 -O3 -DNDEBUG -IOUTDIR -isystem vendor/eigen-adt \\
    ir-study/spread_ab.cpp OUTDIR/PC*.cpp -o spread_ab
  ./spread_ab [cpu] [rounds] [frames]
"""
import os, re, subprocess, sys

ATTRIBUTION_REV = 'e2dc6bc'


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    tree, out, revs = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out, exist_ok=True)

    def git_show(rev, path):
        return subprocess.check_output(['git', '-C', tree, 'show', f'{rev}:{path}'], text=True)

    enum_re = re.compile(r'/// Selects the convolution engine\.\nenum class ConvolutionImplementation\n\{.*?\};\n', re.S)
    enum = re.search(r'enum class ConvolutionImplementation\n\{.*?\};\n',
                     git_show(revs[0], 'dsp/PartitionedConvolution.h'), re.S).group(0)
    open(os.path.join(out, 'common_enum.h'), 'w').write('#pragma once\nnamespace dsp {\n' + enum + '}\n')

    def write(name, cpp, h):
        open(os.path.join(out, f'{name}.cpp'), 'w').write(cpp.replace('PartitionedConvolution', name))
        h = enum_re.sub('', h.replace('PartitionedConvolution', name))
        assert 'enum class ConvolutionImplementation' not in h, f'{name}: enum not stripped'
        open(os.path.join(out, f'{name}.h'), 'w').write(h.replace('#pragma once', '#pragma once\n#include "common_enum.h"'))

    def instrument(s):
        def sub(old, new):
            assert s.count(old) == 1, old
            return s.replace(old, new)
        s = sub('namespace dsp\n{\n',
                'namespace pct { extern unsigned long long phase_ns[4]; unsigned long long now_ns(); }\n\nnamespace dsp\n{\n')
        fwd = '  s.fft.fwd(current_spectrum.data(), s.input_time.data(), s.fft_size);\n'
        s = sub(fwd, '  const unsigned long long t0 = pct::now_ns();\n' + fwd + '  const unsigned long long t1 = pct::now_ns();\n')
        inv = '  s.fft.inv(s.ifft_time.data(), s.accumulator.data(), s.fft_size);\n'
        s = sub(inv, '  const unsigned long long t2 = pct::now_ns();\n' + inv + '  const unsigned long long t3 = pct::now_ns();\n')
        tail = '  std::fill(s.input_time.begin(), s.input_time.begin() + s.block_size, 0.0f);\n'
        return sub(tail, '  const unsigned long long t4 = pct::now_ns();\n'
                         '  pct::phase_ns[0] += t1 - t0;\n  pct::phase_ns[1] += t2 - t1;\n'
                         '  pct::phase_ns[2] += t3 - t2;\n  pct::phase_ns[3] += t4 - t3;\n' + tail)

    names = []
    for n, rev in enumerate(revs):
        write(f'PC{n}', git_show(rev, 'dsp/PartitionedConvolution.cpp'), git_show(rev, 'dsp/PartitionedConvolution.h'))
        names.append((f'PC{n}', subprocess.check_output(['git', '-C', tree, 'rev-parse', '--short=7', rev], text=True).strip()))
    write('PCt', instrument(git_show(ATTRIBUTION_REV, 'dsp/PartitionedConvolution.cpp')),
          git_show(ATTRIBUTION_REV, 'dsp/PartitionedConvolution.h'))

    with open(os.path.join(out, 'variants.h'), 'w') as f:
        f.write('#pragma once\n')
        for name, _ in names + [('PCt', '')]:
            f.write(f'#include "{name}.h"\n')
        f.write('#define SPREAD_VARIANTS(X) ' + ' '.join(f'X({n}, "{r}")' for n, r in names) + '\n')
    print(f'wrote {", ".join(n for n, _ in names)}, PCt, common_enum.h and variants.h to {out}')


if __name__ == '__main__':
    main()
