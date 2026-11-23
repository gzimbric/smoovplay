# SmoovPlay on a Mac

This folder is a native Mac version of SmoovPlay. It has two parts. [`core/`](core/README.md) is a Rust program that
runs the whole pipeline on Apple's GPU through Metal. [`app/`](app/README.md) is a SwiftUI app that replaces the
Windows launcher. Together with the patched chiaki-ng from [`chiaki_tap/`](../chiaki_tap/README.md#build-on-macos),
they play a PS5 over Remote Play with frame generation, as the Windows version does.

The core does what the Python pipeline in `framegen/` does, with the same command line, settings file, looks and
launcher protocol, and it was checked against it piece by piece. The app can run either one. Windows keeps the
Python pipeline and the WPF launcher.

## Building

You need an Apple silicon Mac with Xcode, [Homebrew](https://brew.sh) and Rust (`brew install rustup`, then
`rustup default stable`). From the repo root:

```sh
git clone --branch v1.10.1 --recurse-submodules https://github.com/streetpea/chiaki-ng.git build/chiaki-ng-src
bash chiaki_tap/build_macos.sh build/chiaki-ng-src build/chiaki-ng-frametap   # the patched chiaki-ng
mac/core/build_release.sh                                                      # the core
mac/app/bundle.sh --open                                                       # the app
```

Each script signs what it builds with your Apple Development identity if the keychain has one, and ad hoc
otherwise. A real identity matters because macOS ties permissions to the signature. With it, the app keeps its
Screen Recording permission and chiaki-ng keeps its Local Network permission when you rebuild. Set `SIGN_IDENTITY`
to use another identity, or `-` for ad hoc.

## How the pieces fit

The app starts the core with `--control` and the source's flags. It reads the core's `[settings-state]` lines and
sends commands on its standard input, the way the Windows launcher drives `run.py`. Frames from the PS5 reach the
core from chiaki-ng through the frame tap, a block of POSIX shared memory described in `docs/frame-tap.md`. An app
window reaches it the same way, copied by the app with ScreenCaptureKit.

## How fast it is

On an Apple M5 Max with a PS5 over Ethernet, playing a 30 fps game at 90 fps and full motion detail:

| | Rust core | Python core |
|---|---|---|
| Extra frames shown on time | 100% | 67% |
| Stream to screen, median | 29.8 ms | 42.5 ms |
| One in-between frame at 1080p | 3.7 ms | 6.9 ms |

With the fake PS5 stream, the Rust core also holds 120 fps at full detail, showing 98% of the extra frames on time
where the Python core shows 36%. [`core/README.md`](core/README.md) has the full figures and how each piece was
checked against the Python.

Capture cards haven't been tried with a real card yet.
