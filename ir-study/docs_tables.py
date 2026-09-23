#!/usr/bin/env python3
"""Print README/BENCHMARKING-style IR tables from nam_ir_benchmark reports.

    python3 ir-study/docs_tables.py benchmark-results/*-ir-block64.json

Bolds whichever of shipping and FFT has the lower core%, and the 8192-tap
ratio. Also prints each length's worst single block and the branch's direct
path against shipping, for the prose around the tables.
"""
import json, sys
MS = {256:'5.3',512:'10.7',1024:'21.3',2048:'42.7',4096:'85.3',8192:'170.7'}
for path in sys.argv[1:]:
    d = json.load(open(path))
    rows = {}
    spreads = []
    for r in d['results']:
        key = 'ship' if r['variant']=='adt_upstream' else ('fft' if r['requested']=='fft' else 'pdir')
        rows.setdefault(r['taps'], {})[key] = r
        spreads.append(r['spread']*100)
    print(f"### {path}\n(spread {min(spreads):.2f}-{max(spreads):.2f}%)\n")
    print("| taps | ms of IR | shipping | | partitioned FFT | | |\n|---:|---:|---:|---:|---:|---:|---:|\n| | | core% | p99 | core% | p99 | |")
    for t in sorted(rows):
        s, f = rows[t]['ship'], rows[t]['fft']
        sc, fc = s['corePercent'], f['corePercent']
        ratio = sc/fc
        scs, fcs = f"{sc:.3f}%", f"{fc:.3f}%"
        if fc < sc: fcs = f"**{fcs}**"
        else: scs = f"**{scs}**"
        rs = f"{ratio:.2f}x"
        if t == 8192: rs = f"**{rs}**"
        print(f"| {t} | {MS[t]} | {scs} | {s['blockP99Percent']:.2f}% | {fcs} | {f['blockP99Percent']:.2f}% | {rs} |")
    print()
    for t in sorted(rows):
        s, f, p = rows[t]['ship'], rows[t]['fft'], rows[t]['pdir']
        print(f"  {t}: max ship {s['blockMaxPercent']:.1f}% fft {f['blockMaxPercent']:.1f}%  pdir/ship time {p['corePercent']/s['corePercent']:.3f}  parts {f['fftPartitions']}")
    for p in d['parities']: print('  parity', {k:v for k,v in p.items() if k in ('taps','reference','candidate','dbBelowSignal','maxAbsDiff','relativeRmsDb')})
    print()
