# SmoovPlay on a Mac: the native app and core

SmoovPlay's Mac version, written for the Mac: a SwiftUI app ([`app/`](app/)) and a Rust core ([`core/`](core/)) that
runs the video on Apple's GPU through Metal. It does what the Python pipeline (`framegen/`, `run.py`) does on a Mac,
faster, with the same command line, settings file, looks and launcher protocol, so either core works with the app.
Windows keeps the Python pipeline and the WPF launcher.

| Piece | What it is |
|---|---|
| [`core/`](core/README.md) | The Rust core, `smoovplay`: the frame tap and VideoToolbox decoding, RIFE on Metal, the scheduler, the Metal video window (FSR 1, looks, HDR output), settings, instant replay, capture cards |
| [`app/`](app/README.md) | `SmoovPlay.app`: the launcher's pages, window capture with ScreenCaptureKit, keys from any app, hiding chiaki-ng's window |
| [`../chiaki_tap/`](../chiaki_tap/README.md#build-on-macos) | The patched chiaki-ng for PS5 Remote Play, built for macOS by `build_macos.sh` |

## Building it all

Needs an Apple silicon Mac, Xcode, [Homebrew](https://brew.sh) and Rust (`brew install rustup`, then
`rustup default stable`). From the repo root:

```sh
git clone --branch v1.10.1 --recurse-submodules https://github.com/streetpea/chiaki-ng.git build/chiaki-ng-src
bash chiaki_tap/build_macos.sh build/chiaki-ng-src build/chiaki-ng-frametap   # the patched chiaki-ng
mac/core/build_release.sh                                                      # the core
mac/app/bundle.sh --open                                                       # the app, opened
```

Each script signs what it builds with the keychain's Apple Development identity if there is one (else ad hoc), so
macOS keeps the app's permissions (Screen Recording for window capture, Local Network for chiaki-ng finding the PS5)
from one build to the next. `SIGN_IDENTITY` picks another identity, `-` signs ad hoc.

## How the pieces talk

The app starts the core as a child process with `--control` and the source's flags, reads its `[settings-state]`
lines and sends it commands on stdin, as the Windows launcher does with `run.py` (`framegen/remote.py`). PS5 frames
reach the core from chiaki-ng through the frame tap, POSIX shared memory (`docs/frame-tap.md`); an app's window
reaches it the same way, copied by the app with ScreenCaptureKit.

## Measured (Apple M5 Max, 1080p, the fake PS5 stream `scripts/frametap_fake.py --native`)

| | Rust core | Python core |
|---|---|---|
| One in-between frame, full motion detail | 3.7 ms | 6.9 ms |
| 90 fps, full detail: extra frames shown on time | 99% | 73% |
| 120 fps, full detail: extra frames shown on time | 98% | 36% |
| 60 fps: a frame from the tap to the screen (p50) | 19 ms | 27 ms |

[`core/README.md`](core/README.md) has the full comparison and how each piece was checked against the Python
(identical tick sequences from the same stream, identical scheduler decisions, settings files and share codes byte
for byte). Not yet tried with a real PS5 or a real capture card.
