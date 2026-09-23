Upstream vs partitioned-FFT impulse-response convolution — A/B samples

Input:  audio-input/input.wav (10.9 s clean guitar DI, 48 kHz) + 0.5 s silence for the tail
Chain:  dsp::ImpulseResponse loaded from the IR file, exactly as the plugin does
        (resample to 48 kHz, truncate to 8192 taps), 64-frame blocks
A:      AudioDSPTools main (direct convolution)
B:      partitioned-ir branch, Auto (picks FFT for every IR here)
Level:  the same gain on A and B, shared peak at -1 dBFS. Float32 WAV.

difference-boosted-NdB = (B - A), computed in double precision, then raised by N dB
so it is audible at all. Unboosted it sits around -140 dB below the signal.

IR                                  difference RMS re signal   difference peak
1 Orange 4x12 (York 421v-CN)        -142.7 dB                  -136.1 dBFS
2 Ampeg 4x10 (York 57-CNT)          -143.0 dB                  -137.9 dBFS
3 Mesa OS 4x12 57                   -143.6 dB                  -139.0 dBFS
4 Deftones White Pony (2040 taps)   -141.5 dB                  -139.3 dBFS
