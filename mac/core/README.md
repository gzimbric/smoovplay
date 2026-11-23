# SmoovPlay's macOS core (Rust)

`smoovplay` is the Rust binary that replaces `run.py` (and the `python -m framegen.*` helpers the launcher calls) on
a Mac. The Python pipeline (`run.py`, `framegen/`) stays the reference: the same flags, the same `[run] ...` lines and
the same `--control` protocol, so the Mac app (`mac/app`) switches cores by changing only the command it starts
(`SMOOVPLAY_CORE=rust`, or the Help page).

## Crates

| Crate | What it is |
|---|---|
| `sp-core` | Frames on the GPU (`Frame`, Metal textures), `Gpu`, ticks and sources (`Tick`, `Event`, `Waiting`, `SourceInfo`), the clock, block means, and the traits the pieces plug into (`Source`, `Interpolator`, `Presenter`). |
| `sp-tap` | The frame tap: chiaki-ng's shared-memory frame and sound rings, byte-compatible with `chiaki_tap/frametap.c` and `framegen/chiaki_tap.py`. |
| `sp-media` | VideoToolbox decoding, NV12/P010 to Metal frames (HDR tone mapping), and the sources: `ChiakiSource` (PS5 Remote Play), `TapSource`/`open_tap` (an app's frames), `FileSource`, `Ps5SimSource`, test mode, `Clock60`. |
| `sp-rife` | RIFE on Apple's GPU: MPSGraph executables for the convolutions, Metal kernels for the rest. |
| `sp-sched` | The pipeline: repeat and cut detection, the cadence scheduler, 60/90/120 Hz, even timing, dip smoothing, scoring, the stats, the results report, the stream check, the delay test. |
| `sp-present` | The video window: Metal drawing (FSR 1, fill modes, looks, overlays, the menu), presentation, HDR through EDR. |
| `sp-settings` | Settings (the same JSON file as the Python), the `--control` protocol, looks and their CLI, Auto-detect's recommendations. |
| `sp-replay` | Instant replay: the last 30 s to 5 min encoded by VideoToolbox in memory, saved as MP4; the before & after video. |
| `sp-capture` | Capture cards through AVFoundation (Experimental), their sound played in step with the picture, the device list. |
| `smoovplay` | The binary: run.py's `main()` and the helpers' subcommands, wiring the crates together. |

## Building

```
cargo build --release -p smoovplay      # -> target/release/smoovplay (what the Mac app starts)
cargo test -p smoovplay -p sp-settings -p sp-present
```

The RIFE weights come from the repo's `models/` folder (downloaded there on first use), the looks preview's sample
pictures from `framegen/samples` (`SMOOVPLAY_MODELS` and `SMOOVPLAY_SAMPLES` point elsewhere).

## Running

Play (run.py's flags; `smoovplay --help` lists them):

```
smoovplay testdata/clip_1080p60.mp4 --display --realtime          # a video file as a simulated PS5 30 fps game
smoovplay testdata/clip_1080p60.mp4 --score --settings none        # headless, scored against the left-out frames
smoovplay --chiaki --display                                       # PS5 Remote Play through chiaki-ng's frame tap
smoovplay --chiaki --test-mode --display --results out/results.txt # test mode, with the plain-language report
smoovplay --source tap --tap smoovplay-window --display            # the Mac app's App window (or any frame-tap app)
smoovplay --source chiaki --display --control --results <file>     # what the Mac app starts
```

With `--display` the source is opened and RIFE built on the main thread, the window opens there, and the pipeline
moves to its own thread (QoS user-interactive) while the main thread runs AppKit's event loop. The display is kept
awake while playing (an `IOPMAssertion`). Headless runs stay on the main thread. With the window, instant replay
records as run.py's does (Settings > Instant replay; R and B save, `key R` / `key B` from the app), with
"Looks in clips" drawn by a renderer of its own; the before & after video is made by `smoovplay replay-compose` in a
process of its own. `--capture N|NAME|FILE` reads a capture card (or a recording standing in for one) through
sp-capture, its sound held back to meet the picture. `--max-ticks` works for any source
(the Python honours it for video files only).

Subcommands (the Python helpers, same arguments and output):

```
smoovplay autodetect [--apply] [--if-new] [--settings P] [--timing F,H] [--present-ms MS]   # python -m framegen.autodetect
smoovplay looks schema|encode|decode|import|save|delete|add-lut|preview ...              # python -m framegen.looks
smoovplay looks-preview                                                                  # python -m framegen.looks_preview
smoovplay prepare [--model M] [--no-detect]                                              # python -m framegen.prepare
smoovplay settings                                                                       # python -m framegen.settings
smoovplay sources --json      (also smoovplay --list-sources --json)                     # python -m framegen.sources
smoovplay capture --list|--formats NAME [--json]                                         # python -m framegen.capture
smoovplay replay-compose --after A --before B --out OUT ...                              # python -m framegen.replay compose
```

## The launcher link (`--control`)

As `framegen/remote.py`: `set <key> <json>`, `restart`, `quit`, `state`, `look-show <json>`, `look-hide`,
`looks-reload`, and a `[settings-state] {json}` line at the start of every session and after every change. The Mac
app's additions:

- `key <name>`: a key as if pressed in the video window. M, F, I, S, U, Z, K, J, L, R, B, Q (either case), Up,
  Down, Left, Right, Enter (Return), Esc (Escape), Backspace (Delete), Tab; `Shift+J` is the previous look. `key`
  alone does nothing (the app's probe).
- `overlay <x> <y> <w> <h>` (points from the main display's top left, as ScreenCaptureKit gives them) puts the window
  over that rectangle: borderless, floating, click-through, never taking the keys; F then covers that monitor.
  `overlay none` makes it a normal window again.
- `"menu_open": true|false` after `"restart"` in the state, published again whenever the menu opens or closes.
- When stdin closes under `--control`, the core quits as `quit` does, so it never outlives the app.
- `minimize_chiaki` isn't greyed out on a Mac (the app hides chiaki-ng's window itself).

## Measured (Apple M5 Max, 6016x3384 160 Hz display, same inputs, one run after the other)

Headless, `smoovplay testdata/clip_1080p60.mp4 --score --settings none --max-ticks 600` vs run.py: the PSNR line is
identical (RIFE 28.41 | blend 23.33 | repeat 20.85 dB, n=299, worst 25.47), and so is the repeat check
(tp 300, tn 300). Interp ticks: 4.78 ms p50 (Python 9.21).

PS5 Remote Play: `scripts/frametap_fake.py testdata/clip_1080p60.mp4 --seconds 12 --codec hevc --native --chiaki-like`
into `--chiaki --test-mode --score --display --hz H --scale S --settings none` (720 ticks; the window 3200x1800,
FSR 1):

| fps | motion detail | extra frames on time, Rust / Python | arrival to screen p50 / p95 ms, Rust / Python | interp tick p50 ms, Rust / Python | RIFE dB, Rust / Python |
|---|---|---|---|---|---|
| 60 | full | n/a | 19.0 / 19.7 vs 27.1 / 29.6 | 5.23 vs 10.24 | 29.28 vs 29.14 |
| 60 | half | n/a | 19.0 / 19.5 vs 25.2 / 27.5 | 4.43 vs 8.25 | 27.89 vs 27.82 |
| 90 | full | 99% vs 73% | 28.8 / 40.4 vs 42.7 / 47.8 | 8.79 vs 21.32 | (scored at 60 and 120) |
| 90 | half | 99% vs 87% | 27.8 / 40.1 vs 41.2 / 46.1 | 6.58 vs 16.98 | (scored at 60 and 120) |
| 120 | full | 98% vs 36% | 31.7 / 39.7 vs 43.2 / 48.3 | 8.81 vs 21.34 | 29.28 vs 29.14 |
| 120 | half | 98% vs 50% | 30.2 / 37.6 vs 34.8 / 46.2 | 5.90 vs 15.79 | 27.89 vs 27.74 |

(At 90 and 120 an interp tick holds two or three generated frames and the pacing between them.)

Auto-detect's test (one new frame plus one in-between frame at 1080p): 4.41 ms at full motion detail, 2.41 at half
(Python on the same Mac: 7.74 and 5.10), so it suggests 120 fps at full detail where the Python suggests 90 at half.
The window's cost per frame shown, which Auto-detect and Auto motion detail add (`run::PRESENT_MS`, 1.0 ms; the
Python's `MAC_PRESENT_MS` is 3.0 for its CPU copy into OpenGL): `write()` 0.08 ms of CPU (p50), FSR 1 0.45 ms of GPU
into the 3200x1800 window and 1.17 ms fullscreen at 6016x3384, less the 0.24 ms of FSR `recommend()` counts itself.
`smoovplay autodetect --present-ms 3.0` reproduces the Python's JSON and settings file byte for byte.

## Parity with the Python helpers

Byte for byte (same inputs, the user's folders swapped for temporary ones): `settings`, `sources --json`,
`--list-sources --json`, `looks schema|decode|encode|import` (and the saved file), a bad share code's error,
`looks preview`'s reply, `autodetect --timing 9.3,6.8 [--apply] [--if-new]` at the Python's 3.0 ms present cost, and
the `[settings-state]` lines for `set hz 90`, `state`, `restart`, `quit` and `set motion "half"` (restart list
["Motion detail"]), apart from `"menu_open"` and three reasons worded for this core:

- engine: "On a Mac, RIFE runs on Apple's GPU (Metal)." (Python: "...through PyTorch on...")
- present: "On a Mac, the window always presents through Metal." (Python: "...through OpenGL.")
- vsr_quality: "RTX VSR isn't available: it needs an NVIDIA RTX card." (Python: "...it needs frames decoded on the GPU...")

`looks-preview` gives the same replies (errors included); its pictures are within 0.2 levels on average of the
OpenGL ones (0.6 for looks that sharpen). The results report has the Python's lines and layout ("AI engine: Metal").
Not the same: argparse's usage text for a bad `looks` subcommand (stderr only), and `capture --list --json`, whose
"alt" is AVFoundation's unique ID (sp-capture's choice) and whose JSON is compact.

## Not in this core yet

- `--out` (writing the result to a video file) and `--display-cv`; `--fp32` (the Metal engine is half precision).
- `--source window`: on a Mac the app copies the window and sends it through the frame tap (`--source tap`).
- NVIDIA's pieces (TensorRT, VFG, RTX VSR), DXGI presentation: greyed out with the reason, as in the Python on a Mac.
- HDR10 output needs a display with EDR headroom; it was checked only as far as "shown in SDR" on an SDR display.
- The watchdog prints macOS's `sample` of the process for "where each thread is" (no Python stacks to show).
