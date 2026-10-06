# NAMBench

Compares NAM A2 WaveNet processing code and IR convolution, on macOS, iOS, Linux and
Android: the reference implementation against the planar-NEON and slimmed/fused kernels,
all measured in one process against one in-RAM model and one in-RAM audio file. It
reaches the pedal only through `namchain`, as a vendor patch or a port.

## Build and run

`README.md` owns this. First supply an A2 capture in `nam-files/` — the `.nam` files are
other people's work and are not committed; `nam-files/README.md` names the one every
published number was measured on, with its checksum. Then:

```sh
./Scripts/fetch-vendor.sh && xcodegen generate
xcodebuild -project NAMBench.xcodeproj -scheme nambench-cli -configuration Release build
```

`nambench --help` lists the options. `BENCHMARKING.md` is the measurement protocol for
hardware you own; `CONFORMANCE.md` is the half that runs in the cloud and reads no timing.
Reports land in `benchmark-results/`.

## Constraints

- `vendor/` and `fpi/sdk` are fetched by `Scripts/fetch-vendor.sh` at pinned commits and are
  gitignored. They are other people's repositories inside this one, and they are fetched
  trees rather than checkouts: `.gitignore` and the fetch script own them, and nothing adds
  them to the index. A fetched tree that is not ignored is refused by the guard in
  `pedal.project/tools/git-hooks/`, because committing it would record a pointer that
  clones as an empty directory.
- Nothing here runs on the Core host: it arrives at the pedal through `namchain`.
- Every published number is pinned to a commit and a build configuration. A number you
  cannot rebuild is a number you cannot defend.

System context (the pedal, how the repos fit together, the plan):
`~/code.local/pedal/pedal.project`, starting at its `AGENTS.md` and
`docs/working-across-repos.md`.
