#!/usr/bin/env python3
"""Generate the three convolvers spectrum_storage_ab.cpp compares.

    python3 ir-study/make_storage_variants.py vendor/adt-partitioned OUTDIR

  PC0  ba646f4: multiplies all N bins of every spectrum, stores all N
  PC1  the current source with every spectrum buffer allocated at N anyway
       (the loop still stops at N/2): the half-spectrum change without its
       memory saving
  PC2  the current source, as committed: N/2 + 1 bins stored and multiplied

PC1 and PC2 differ only in how much is allocated, which is what isolates the
storage half of a09e360 from its arithmetic half. Each is renamed so all three
link into one binary; the ConvolutionImplementation enum lives once, in
common_enum.h.

Then, from OUTDIR:
  c++ -std=c++17 -O3 -I. -isystem ../vendor/eigen-adt \\
    ../ir-study/spectrum_storage_ab.cpp PC0.cpp PC1.cpp PC2.cpp -o storage_ab
  ./storage_ab [cpu] [rounds]
"""
import os, re, subprocess, sys


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    tree, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)

    def git_show(rev, path):
        return subprocess.check_output(['git', '-C', tree, 'show', f'{rev}:{path}'], text=True)

    cur_cpp = open(os.path.join(tree, 'dsp/PartitionedConvolution.cpp')).read()
    cur_h = open(os.path.join(tree, 'dsp/PartitionedConvolution.h')).read()
    old_cpp = git_show('ba646f4', 'dsp/PartitionedConvolution.cpp')
    old_h = git_show('ba646f4', 'dsp/PartitionedConvolution.h')

    enum_re = re.compile(r'/// Selects the convolution engine\.\nenum class ConvolutionImplementation\n\{.*?\};\n', re.S)
    enum = re.search(r'enum class ConvolutionImplementation\n\{.*?\};\n', cur_h, re.S).group(0)
    open(os.path.join(out, 'common_enum.h'), 'w').write('#pragma once\nnamespace dsp {\n' + enum + '}\n')

    def write(n, cpp, h):
        open(os.path.join(out, f'PC{n}.cpp'), 'w').write(cpp.replace('PartitionedConvolution', f'PC{n}'))
        h = enum_re.sub('', h.replace('PartitionedConvolution', f'PC{n}'))
        open(os.path.join(out, f'PC{n}.h'), 'w').write(h.replace('#pragma once', '#pragma once\n#include "common_enum.h"'))

    def full_storage(s):
        for a in ['std::vector<FftState::Complex>(static_cast<size_t>(s.num_bins)));',
                  'std::vector<FftState::Complex>(static_cast<size_t>(s.num_bins), FftState::Complex{}));',
                  's.accumulator.assign(static_cast<size_t>(s.num_bins), FftState::Complex{});']:
            assert s.count(a) == 1, a
            s = s.replace(a, a.replace('s.num_bins', 's.fft_size'))
        a = 'std::fill(s.accumulator.begin(), s.accumulator.end(), FftState::Complex{});'
        assert s.count(a) == 1, a
        return s.replace(a, 'std::fill(s.accumulator.begin(), s.accumulator.begin() + s.num_bins, FftState::Complex{});')

    write(0, old_cpp, old_h)
    write(1, full_storage(cur_cpp), cur_h)
    write(2, cur_cpp, cur_h)
    print(f'wrote PC0, PC1, PC2 and common_enum.h to {out}')


if __name__ == '__main__':
    main()
