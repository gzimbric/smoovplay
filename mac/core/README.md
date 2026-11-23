# SmoovPlay's macOS core

`smoovplay` is a Rust program that does on a Mac what `run.py` and the `python -m framegen.*` helpers do. It takes
the same flags, prints the same `[run]` lines and speaks the same `--control` protocol, so the Mac app in `mac/app`
can run either core. The Python pipeline in `framegen/` is the reference it was checked against.

## Crates

| Crate | What it does |
|---|---|
| `sp-core` | Shared types. Frames are Metal textures, and every crate uses one GPU device and queue. Also ticks, sources, the clock, block means, and the traits the other crates implement. |
| `sp-tap` | The frame tap, the shared memory that chiaki-ng writes frames and sound into. Byte-compatible with `chiaki_tap/frametap.c` and `framegen/chiaki_tap.py`. |
| `sp-media` | VideoToolbox decoding, conversion of NV12 and P010 into Metal frames with HDR tone mapping, and the sources for Remote Play, the frame tap, video files, the simulated PS5 and test mode. |
| `sp-rife` | RIFE on Apple's GPU. MPSGraph runs the convolutions and Metal kernels do the rest. |
| `sp-sched` | The pipeline itself. Repeat and cut detection, the scheduler for 60, 90 and 120 Hz, even timing, dip smoothing, scoring, stats, the results report, the stream check and the delay test. |
| `sp-present` | The video window. FSR 1, fill modes, looks, overlays and the menu, all drawn in Metal, plus HDR output through EDR. |
| `sp-settings` | Settings in the same JSON file as the Python, the `--control` protocol, looks and their command line, and Auto-detect's recommendations. |
| `sp-replay` | Instant replay. VideoToolbox encodes the last 30 seconds to 5 minutes into memory and saves it as MP4. Also the before and after video. |
| `sp-capture` | Capture cards through AVFoundation, with their sound played in step with the picture. Experimental. |
| `smoovplay` | The program. It ties the crates together the way `run.py` does and adds the helpers as subcommands. |

## Building

```
./build_release.sh                       # builds target/release/smoovplay and signs it
cargo test -p smoovplay -p sp-settings -p sp-present
```

`build_release.sh` signs with the keychain's Apple Development identity if there is one, otherwise ad hoc. RIFE's
weights come from the repo's `models/` folder and are downloaded there on first use. The looks preview takes its
sample pictures from `framegen/samples`. Set `SMOOVPLAY_MODELS` or `SMOOVPLAY_SAMPLES` to use other folders.

## Running

`smoovplay --help` lists every flag. Some common runs:

```
smoovplay testdata/clip_1080p60.mp4 --display --realtime          # a video file played as a 30 fps PS5 game
smoovplay testdata/clip_1080p60.mp4 --score --settings none        # no window, scored against the left-out frames
smoovplay --chiaki --display                                       # PS5 Remote Play through chiaki-ng
smoovplay --chiaki --test-mode --display --results out/results.txt # test mode with the written report
smoovplay --source tap --tap smoovplay-window --display            # an app window copied by the Mac app
smoovplay --source chiaki --display --control --results <file>     # what the Mac app runs
```

With `--display`, the window opens on the main thread, which then runs AppKit's event loop, and the pipeline moves
to a thread of its own at user-interactive priority. The Mac stays awake while a session plays. Without a window,
everything runs on the main thread.

Instant replay works as it does in `run.py`. Turn it on under Settings, Instant replay, then press R or B, or send
`key R` or `key B` from the app. Looks can be drawn into clips. The before and after video is made by
`smoovplay replay-compose` in a separate, lower priority process.

`--capture` takes a capture card by number or name, or a video file standing in for one. The card's sound is held
back to meet the picture. `--max-ticks` works for every source, where the Python honours it only for video files.

The helpers are subcommands with the same arguments and output as the Python ones:

```
smoovplay autodetect [--apply] [--if-new] [--settings P] [--timing F,H] [--present-ms MS]   # framegen.autodetect
smoovplay looks schema|encode|decode|import|save|delete|add-lut|preview ...              # framegen.looks
smoovplay looks-preview                                                                  # framegen.looks_preview
smoovplay prepare [--model M] [--no-detect]                                              # framegen.prepare
smoovplay settings                                                                       # framegen.settings
smoovplay sources --json                                                                 # framegen.sources
smoovplay capture --list [--json]                                                        # framegen.capture
smoovplay replay-compose --after A --before B --out OUT ...                              # framegen.replay compose
```

## The control link

`--control` takes the commands `framegen/remote.py` documents and prints a `[settings-state]` line at the start of
each session and after every change. The Mac app needs a few more, and only this core has them.

`key <name>` acts as if the key were pressed in the video window. It accepts M, F, I, S, U, Z, K, J, L, R, B and Q
in either case, the arrow keys, Enter, Esc, Backspace and Tab. `Shift+J` goes to the previous look. A bare `key`
does nothing, which lets the app check whether the core takes keys.

`overlay <x> <y> <w> <h>` puts the window over that rectangle, given in points from the top left of the main
display. The window becomes borderless and floating, lets clicks through and never takes the keys. F then fills that
monitor. `overlay none` turns it back into a normal window.

The state gains a `menu_open` field after `restart`, published again whenever the menu opens or closes. The core
quits when its standard input closes, so it never outlives the app. And `minimize_chiaki` stays available on a Mac,
because the app hides chiaki-ng's window itself.

## Measured

All on an Apple M5 Max with a 6016x3384 display at 160 Hz. Each comparison ran the same input through both cores,
one after the other.

On a real PS5 over Ethernet, playing a 30 fps game for about four minutes at 90 fps and full motion detail:

| | Rust core | Python core |
|---|---|---|
| Extra frames shown on time | 100% | 67% |
| Stream to screen, median | 29.8 ms | 42.5 ms |
| Stream to screen, 95th percentile | 30.9 ms | 49.7 ms |
| Interpolation tick, median | 9.4 ms | 21.4 ms |

With the fake PS5 (`scripts/frametap_fake.py --native`, HEVC, 12 seconds, test mode, a 3200x1800 window with FSR 1):

| Output | Detail | On time, Rust | On time, Python | To screen p50, Rust | To screen p50, Python | Tick p50, Rust | Tick p50, Python |
|---|---|---|---|---|---|---|---|
| 60 fps | full | | | 19.0 ms | 27.1 ms | 5.2 ms | 10.2 ms |
| 60 fps | half | | | 19.0 ms | 25.2 ms | 4.4 ms | 8.3 ms |
| 90 fps | full | 99% | 73% | 28.8 ms | 42.7 ms | 8.8 ms | 21.3 ms |
| 90 fps | half | 99% | 87% | 27.8 ms | 41.2 ms | 6.6 ms | 17.0 ms |
| 120 fps | full | 98% | 36% | 31.7 ms | 43.2 ms | 8.8 ms | 21.3 ms |
| 120 fps | half | 98% | 50% | 30.2 ms | 34.8 ms | 5.9 ms | 15.8 ms |

At 90 and 120 fps a tick draws two or three in-between frames and paces them. RIFE scored the same against the real
frames in both cores, 29.3 dB at full detail and 27.9 dB at half (Python 29.1 and 27.8).

Scoring the test clip without a window (`--score --settings none --max-ticks 600`) gives exactly the Python's
quality line, 28.41 dB for RIFE, 23.33 for a blend and 20.85 for the repeat, and the same repeat detection. An
interpolation tick takes 4.8 ms there, against 9.2 ms in Python.

Auto-detect times one new frame plus one in-between frame at 1080p. That takes 4.4 ms at full detail and 2.4 ms at
half, against 7.7 and 5.1 in Python on the same Mac, so it suggests 120 fps at full detail. It also adds the cost of
showing a frame, which is 1.0 ms here (`run::PRESENT_MS`). The Python adds 3.0 ms, because it copies each frame
through the CPU into OpenGL. Measured, the window takes 0.08 ms of CPU per frame, and FSR 1 takes 0.45 ms of GPU in
the 3200x1800 window and 1.17 ms fullscreen. `--present-ms 3.0` makes the output match the Python's byte for byte.

## Checked against the Python

These give byte-identical output for the same input: `settings`, `sources --json`, `--list-sources --json`,
`looks schema`, `decode`, `encode` and `import` with the file it saves, the error for a bad share code, the reply to
`looks preview`, and `autodetect --timing 9.3,6.8` with and without `--apply` and `--if-new` at the Python's 3.0 ms
present cost.

So do the `[settings-state]` lines for `set hz 90`, `state`, `restart`, `quit` and `set motion "half"`, apart from
`menu_open` and three reasons reworded for this core. The engine row says RIFE runs on Apple's GPU through Metal
instead of PyTorch, the presentation row says Metal instead of OpenGL, and RTX VSR's row says it needs an NVIDIA
card.

The looks preview gives the same replies, errors included. Its pictures are within 0.2 levels of the OpenGL ones on
average, and 0.6 for looks that sharpen. The results report has the Python's lines and layout. Two things differ:
the usage text for a bad `looks` subcommand, which only goes to standard error, and `capture --list --json`, which
gives AVFoundation's device ID as `alt` and prints compact JSON.

## Not in this core yet

- `--out` for writing the result to a video file, and `--display-cv`.
- `--fp32`. The Metal engine runs in half precision only.
- `--source window`. On a Mac the app copies the window and sends it through the frame tap, so use `--source tap`.
- NVIDIA's pieces (TensorRT, VFG and RTX VSR) and DXGI. They are greyed out with the reason, as in the Python.
- HDR10 output on a display with EDR headroom. It has only been checked on an SDR display, where it falls back to
  tone-mapped SDR.
- The watchdog prints macOS's `sample` of the process when the loop stalls, because there are no Python stacks.
