#!/usr/bin/env python3
"""Reduce ir_trace's per-block CSV to the JSON the burstiness page charts.

    python3 ir-study/trace_reduce.py trace.csv out.json

Times become percentages of each block's real-time deadline. FFT subjects are
split into quiet blocks and transform blocks by position: the transform lands
on the block that completes each FFT block (256 samples up to 2048 taps, 512
above), so with 64-frame blocks that is every 4th or every 8th block.
"""
import sys, csv, json, collections, statistics as st
FFTB = {512:256,1024:256,2048:256,4096:512,8192:512}
data = collections.defaultdict(lambda: collections.defaultdict(list))  # key -> pass -> list
for r in csv.DictReader(open(sys.argv[1])):
    k = (r['exp'], r['subject'], int(r['taps']), int(r['block']), r['input'], r['ir'])
    data[k][int(r['pass'])].append(int(r['ns']))
def budget(block): return block/48000*1e9
def pct(v, block): return v/budget(block)*100
def q(v, p):
    s = sorted(v); return s[min(len(s)-1, int(p*len(s)))]
def split(k):
    """quiet vs heavy blocks for fft subjects"""
    exp, subj, taps, block, *_ = k
    allv = [x for p in data[k].values() for x in p]
    if subj == 'upstream' or subj == 'direct':
        return allv, []
    period = max(1, FFTB[taps] // block)
    if period == 1: return [], allv
    quiet, heavy = [], []
    for p in data[k].values():
        for i, x in enumerate(p):
            (heavy if i % period == period-1 else quiet).append(x)
    return quiet, heavy
out = {}
# 1. timeline: median across passes per block index, blocks 400..463 (steady state)
def key(exp, subj, taps=8192, block=64, inp='guitar', ir='synthetic'): return (exp, subj, taps, block, inp, ir)
tl = {}
for subj in ('upstream', 'fft'):
    passes = data[key('headline', subj)]
    tl[subj] = [round(pct(st.median(passes[p][i] for p in passes), 64), 3) for i in range(400, 464)]
    tl[subj+'_raw'] = [round(pct(passes[3][i], 64), 3) for i in range(400, 464)]
out['timeline'] = tl
# 2. histogram of all blocks (percent of budget), 0.1% bins up to 8%
hist = {}
for subj in ('upstream', 'fft'):
    allv = [pct(x, 64) for p in data[key('headline', subj)].values() for x in p]
    bins = [0]*81
    for v in allv: bins[min(80, int(v/0.1))] += 1
    hist[subj] = {'bins': bins, 'n': len(allv), 'mean': round(st.mean(allv),3),
                  'p50': round(q(allv,.5),3), 'p99': round(q(allv,.99),3), 'max': round(max(allv),3)}
out['hist'] = hist
# 3. ladder
lad = []
for taps in (512, 1024, 2048, 4096, 8192):
    up = [x for p in data[key('ladder','upstream',taps)].values() for x in p]
    quiet, heavy = split(key('ladder','fft',taps))
    allf = quiet + heavy
    lad.append(dict(taps=taps, fftBlock=FFTB[taps], period=FFTB[taps]//64,
        partitions={512:1,1024:3,2048:7,4096:7,8192:15}[taps],
        up=round(pct(st.median(up),64),3), up_mean=round(pct(st.mean(up),64),3),
        quiet=round(pct(st.median(quiet),64),3), heavy=round(pct(st.median(heavy),64),3),
        fft_mean=round(pct(st.mean(allf),64),3)))
out['ladder'] = lad
# 4. input + IR content rows
rows = []
for inp in ('guitar','white noise','silence','110 Hz sine'):
    up = [x for p in data[key('input','upstream',inp=inp)].values() for x in p]
    quiet, heavy = split(key('input','fft',inp=inp))
    rows.append(dict(group='input', label=inp, ship=round(pct(st.median(up),64),3),
                     quiet=round(pct(st.median(quiet),64),3), heavy=round(pct(st.median(heavy),64),3)))
irs = sorted({k[5] for k in data if k[0]=='ircontent'}, key=lambda s: (s!='synthetic', s!='single click', s))
for ir in irs:
    d = [x for p in data[('ircontent','direct',8192,64,'guitar',ir)].values() for x in p]
    quiet, heavy = split(('ircontent','fft',8192,64,'guitar',ir))
    rows.append(dict(group='ir', label=ir, ship=round(pct(st.median(d),64),3),
                     quiet=round(pct(st.median(quiet),64),3), heavy=round(pct(st.median(heavy),64),3)))
out['content'] = rows
# 5. block sizes
bs = []
for block in (32, 64, 128, 256, 512, 1024):
    up = [pct(x,block) for p in data[key('blocksize','upstream',block=block)].values() for x in p]
    ff = [pct(x,block) for p in data[key('blocksize','fft',block=block)].values() for x in p]
    passes = data[key('blocksize','fft',block=block)]
    period = max(1, 512//block)
    prof = [round(st.median(pct(passes[p][i],block) for p in passes for i in range(200+j, len(passes[p]), period) ),3) for j in range(period)] if period>1 else None
    bs.append(dict(block=block, ms=round(block/48,3), period=period,
        up_p50=round(q(up,.5),3), up_p99=round(q(up,.99),3), up_mean=round(st.mean(up),3),
        fft_p50=round(q(ff,.5),3), fft_p99=round(q(ff,.99),3), fft_mean=round(st.mean(ff),3)))
out['blocksize'] = bs
json.dump(out, open(sys.argv[2],'w'), separators=(',',':'))
print(json.dumps(out['ladder'], indent=0)); print(json.dumps(out['content'])); print(json.dumps(out['blocksize']))
print(out['timeline']['fft'][:16]); print(out['timeline']['upstream'][:16]); print(out['hist']['fft'])
