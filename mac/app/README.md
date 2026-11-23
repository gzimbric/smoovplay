# SmoovPlay for Mac: the app

The SwiftUI app that replaces the Windows launcher (`installer/launcher`) on a Mac, plus what only a Mac app can do:
copying another app's window with ScreenCaptureKit, keys that work from any app, and hiding chiaki-ng's window while
the video plays. The video itself comes from the core, a child process: the Rust `smoovplay` (`mac/core`, built by
`mac/core/build_release.sh`) when it's there, else `run.py`, with the same command line and `--control` commands.
With the Rust core, the helpers (Auto-detect, looks, the looks preview) are its subcommands too, so no Python runs.

```
./bundle.sh --open          # builds build/SmoovPlay.app (release, signed with your Apple Development identity
                            # if the keychain has one, else ad hoc) and opens it
swift test                  # the checks below (about 6 s; the core's are skipped without the repo's .venv)
```

It runs against this checkout: the Rust core (or the repo's `.venv` Python) runs the video and the helpers, and
`build/chiaki-ng-frametap/chiaki-ng.app` is the patched chiaki-ng (`chiaki_tap/README.md`, Build on macOS).
`bundle.sh` writes the repo's path into the app's Info.plist, so the app finds it from anywhere. It uses Xcode's
toolchain when `/Applications/Xcode.app` is there, else the command line tools, and needs macOS 14 or newer.

## Pages

The Windows launcher's, in a rail on the left:

| Page | What it does |
|---|---|
| Play | The Play button (Stop while it plays), what's happening while it starts (starting, waiting for the stream, stream found, restarting for a setting, playing, ended, stopped, stopped with an error), where the video comes from (PS5 Remote Play, Capture card, App window, Frame tap; the Video file too with `SMOOVPLAY_SOURCES=all`), the capture card or app window to use, App window mode's keys, the settings in use, notes from the core, Test with a 60 fps game, the test results card, and the session's log under Show details. While the video plays the window goes to the Dock, with a badge (Live, …, !), and comes back when the session ends. |
| Connect your PS5 | The eight steps to tick off (remembered), Open chiaki-ng (the patched one, with `CHIAKI_FRAMETAP=1`), chiaki-ng's guide. Other sources get their own setup guide here, as on Windows. The app opens on this page until the PS5 guide is done. |
| Mac check | The Windows PC check's cards that apply to a Mac: the chip and memory, the display's refresh rate (with Open Displays when it's set lower than it can go), disk space, the pipeline, chiaki-ng (built or not), the network, and Screen Recording for App window mode. |
| Settings | Built from the pipeline's own list (`python -m framegen.settings`): the big choices first (frame rate with Off, model, upscaler), Auto-detect, then folding sections. With nothing playing it reads and writes the settings file (`~/Library/Application Support/framegen/settings.json`, in the core's own layout); while playing it sends each change to the core (`set <key> <json>`) and follows its `[settings-state]` lines, with Restart now for restart-only changes. Choices this Mac can't use (RTX VSR, NVIDIA VFG, TensorRT, DXGI) are greyed out with the reason. Fill the screen has its drawing. |
| Looks | The looks (SmoovPlay's, then yours), the picked one's effects to edit, the preview on sample pictures (`python -m framegen.looks_preview`, hold the mouse on it for the original), Use this look, Save as mine, Reset, New look, Copy code, Paste a code, Add a .cube LUT. While playing, the video shows the look as you edit it (`look-show`). |
| Help | Copy diagnostics (the user folder becomes `~`, device owners' names are left out, NVIDIA VFG results stay out), the logs folder, the video window's keys, which core to use (Python, Rust or another binary), links. |

Logs go to `~/Library/Logs/SmoovPlay`: `session-*.log` (the core's output, the commands sent, and the app's own
`[launcher]` and `[capture]` lines), `results-*.txt` (test mode), `autodetect.txt`. The launcher's own choices (the
guide's ticks, the source, the window keys...) are in `~/Library/Application Support/framegen/smoovplay-launcher.txt`,
in the Windows launcher's format.

## What the Mac app does itself

**App window** (window capture). On Windows the core copies the window; on a Mac the app does, with
ScreenCaptureKit, and hands each picture to the core through the frame tap (`docs/frame-tap.md`, version 2, BGRA,
POSIX shared memory `/smoovplay-window`), which the core reads with `--source tap --tap smoovplay-window`:

- The list puts known streaming apps first (PS Remote Play, Moonlight, Steam Link, GeForce NOW, Parsec), then a
  browser whose tab looks like a cloud game (Xbox, GeForce NOW, Luna, Boosteroid...), then other browsers, then the
  rest. The core's own video window is left out.
- The window is copied on its own (`SCContentFilter(desktopIndependentWindow:)`), so windows over it, SmoovPlay's
  video among them, don't get in the way. Its standard title bar is left out (32 points on macOS 26 and later; a
  switch on the Play page keeps it, for apps that draw under it).
- The picture is the window's shape at the start, fitted into 1920x1080, at up to 60 a second. A picture is sent
  only when the window changed (`FEATURE_VARIABLE`): ScreenCaptureKit's "idle" frames send nothing, and the core's
  60 Hz beat counts those as repeats for certain. A "complete" frame with nothing marked changed goes as a repeat
  (`FLAG_REPEAT`); otherwise the core compares pictures, as a streaming app's repeats are decoded again and never
  quite identical.
- The capture's figures go in the session log (`[capture] stopped: 796 pictures sent ...`).

**Screen Recording.** App window mode needs it. The Play page shows a card while it's off: Allow asks macOS (its
prompt comes once), and Open System Settings goes to Privacy & Security > Screen & System Audio Recording, where the
user turns SmoovPlay on and opens it again. macOS ties the permission to the app's signature: an ad hoc signature
changes with every build, so each new build has to be allowed again. `SIGN_IDENTITY="Apple Development: ..."
./bundle.sh` signs with a real identity, which keeps it. Started from a terminal
(`build/SmoovPlay.app/Contents/MacOS/SmoovPlay`), the app has the terminal's permission instead.

**Keys from any app**, with Carbon's `RegisterEventHotKey` (no Accessibility permission, and the app only hears its
own shortcuts):

- Control-Option-R (save the instant replay) and Control-Option-B (a before & after video), while a session plays
  with instant replay on. On Windows the core holds Ctrl+Alt+R and B itself; its key code is Windows-only.
- In App window mode, the video window's keys, as WindowKeys.cs and `framegen/hotkeys.py` have them, with
  Control-Option: M menu (else Space), F fullscreen, Esc quit (else Q), S stats, I smooth on/off. A shortcut macOS
  or another app has goes to the next one in its list, and people can pick their own on the Play page.
- Each press becomes a command for the core (`key R`...). Until the core takes `key`, fullscreen, stats and quit
  use today's commands (`set fullscreen`, `set overlay`/`overlay_style`, `quit`); the others wait for it.

**chiaki-ng.** Play starts the core first; when it says it's waiting for chiaki-ng, the app opens the patched
chiaki-ng with `CHIAKI_FRAMETAP=1` (and any other `CHIAKI_*` variable, as `run_chiaki_frametap.command` does), unless
the one it opened is still running. A chiaki-ng the app didn't open gets the Windows launcher's question (close it,
play anyway, cancel). While the video plays chiaki-ng is hidden (it keeps streaming; the setting "Hide chiaki-ng" is
followed), hidden again if it shows itself in the first 10 seconds, and shown again when the session ends.

**Awake.** While the video plays the app holds a `PreventUserIdleDisplaySleep` power assertion (a controller isn't
input to macOS).

## Commands beyond the Windows launcher's

The Rust core (`mac/core`) takes all of these; `run.py` answers `[remote] unknown command: ...` to them, and the app
then stops sending them for that session (`CoreProtocol.swift` has the protocol in full):

| Command | For | What the core does |
|---|---|---|
| `key <name>` | Instant replay's keys, App window mode's keys | Acts as if that key were pressed in the video window: M, F, Q, S, I, U, Z, K, J, L, R, B, and Up, Down, Left, Right, Enter, Esc, Backspace for the menu. `key` alone does nothing: the app sends it once the video is up, to learn whether the core takes keys. |
| `overlay <x> <y> <w> <h>`, `overlay none` | App window mode | Sits over the copied window, as `overlay_of` does on Windows: borderless, floating, click-through and never focused, at that rectangle in points from the main display's top left. `none` returns it to a normal window. The app sends the rectangle when the window moves, twice a second at most. |
| `menu_open` in the state | App window mode | Says when the video window's menu is open, so the app takes the menu's own keys (arrows, Return, Esc, Delete) only then. |

The Rust core also quits when its stdin closes under `--control`, so it never outlives the app, and doesn't grey out
`minimize_chiaki` on a Mac (the app hides chiaki-ng's window itself).

## Switches

Environment, as the Windows launcher's dev mode has them: `SMOOVPLAY_APP` (the repo), `SMOOVPLAY_PYTHON`,
`SMOOVPLAY_CORE` (`python`, `rust`, or a path to a core binary; overrides the Help page), `SMOOVPLAY_RUN_ARGS` (more
core arguments), `SMOOVPLAY_SETTINGS` (another settings file, passed to the core and Auto-detect as `--settings`),
`SMOOVPLAY_PREFS`, `SMOOVPLAY_LOGS`, `SMOOVPLAY_CHIAKI` (another chiaki-ng.app), `SMOOVPLAY_SOURCES=all`,
`SMOOVPLAY_SOURCE_FILE`, `SMOOVPLAY_TAP`, `SMOOVPLAY_DETECT_ARGS=--timing 9.3,6.8`, `SMOOVPLAY_CAPTURE_HZ` (the
capture's top rate, default 60), `SMOOVPLAY_SHOTS=<folder>` (a PNG of the window on every page or state change,
drawn by the app itself, so nothing but its own window is in it), `SMOOVPLAY_TRACE=<file>`.

Command line: `--page play|guide|check|settings|looks|help`, `--source <kind>`, `--window <app, bundle id or title
words>`, `--play`, `--test` (Test with a 60 fps game), `--stop-after <seconds>`, `--quit-after-stop`,
`--keep-window`, `--detect`, `--show-results <file>`, `--diagnostics <file>`.

An end-to-end run without a console, the way these were tried: a window playing `testdata/clip_1080p60.mp4` in a
loop, then

```
build/SmoovPlay.app/Contents/MacOS/SmoovPlay --source window --window <its app> --play --stop-after 20 --quit-after-stop
```

or Remote Play with the fake PS5 on a tap of its own (chiaki-ng still opens, and is hidden and shown again):

```
.venv/bin/python scripts/frametap_fake.py testdata/clip_1080p60.mp4 --delay 10 --native --tap 'Local\sp-e2e' &
SMOOVPLAY_RUN_ARGS='--chiaki-tap Local\sp-e2e' build/SmoovPlay.app/Contents/MacOS/SmoovPlay --test --stop-after 10
```

## Checks

`swift test` (Swift Testing), with fixtures captured from the real core (`Tests/SmoovPlayKitTests/Fixtures`, the
home folder taken out):

- The protocol: every line the launcher reads, a real `--control` session's lines, the state from `run.py
  --control` (including a pending restart and frame generation off), the commands, the core's arguments per source,
  the core choice.
- The Settings page model from `python -m framegen.settings` and those states: the frame-rate pick with Off, the
  sections, rows hidden by `applies`, the Mac's greyed-out choices with the core's reasons, restart-only rows, picks
  (Off, back on at a rate), the file when nothing plays, locked while starting, a file write that keeps the rest, and
  the pipeline reading back what the app wrote.
- The frame tap writer: the layout as the docs give it, frames (padded rows too) and their new/repeat flags read back
  by `framegen/chiaki_tap.py`'s `FrameTap` on the reader's clock, and `run.py --source tap` playing what the writer
  sends until it ends the stream.
- Auto-detect's result and changes, the test report's card, looks (editing, Save as mine, share codes decoded by the
  pipeline), shortcuts and App window keys, diagnostics leaving out the user.
- `SMOOVPLAY_TEST_DISPLAY=1 swift test --filter liveSession` drives a real session through `CoreSession` (it opens
  the video window): a setting changed live, confirmed by the state, then quit.

## Files

| File | Contents |
|---|---|
| `Package.swift` | Three targets and the tests. |
| `bundle.sh` | Builds `build/SmoovPlay.app`: the binary, Info.plist, the icon (from `docs/brand`), the signature. |
| `Sources/CFrameTap` | POSIX shared memory and barriers for the frame tap (`shm_open` is variadic, which Swift can't call). |
| `Sources/SmoovPlayKit/CoreProtocol.swift` | The protocol: line parsing, the state, the commands. |
| `Sources/SmoovPlayKit/CoreProcess.swift` | `LineProcess` (a child's lines in, commands out), `CoreSession` (a play session: arguments, log, Stop, exit), `PyTool` (the one-shot helpers). |
| `Sources/SmoovPlayKit/Paths.swift` | Where everything is, the core choice, the switches. |
| `Sources/SmoovPlayKit/Settings.swift` | The Settings page model, the settings file. |
| `Sources/SmoovPlayKit/FrameTap.swift`, `WindowCapture.swift` | The tap writer; window capture and Screen Recording. |
| `Sources/SmoovPlayKit/Hotkeys.swift` | Carbon shortcuts, App window mode's keys. |
| `Sources/SmoovPlayKit/Chiaki.swift` | Opening, hiding and showing chiaki-ng; keeping the Mac awake. |
| `Sources/SmoovPlayKit/AutoDetect.swift`, `Looks.swift`, `Sources.swift`, `Checks.swift`, `Diagnostics.swift`, `Prefs.swift`, `JSON.swift` | The rest of the model. |
| `Sources/SmoovPlay` | The app: `AppModel.swift` (state and actions), one file per page, `Theme.swift`. |

## Not done yet

- The core's side of `key`, `overlay` and `menu_open` (above). Until then, in App window mode the smooth video is in
  its own window rather than over the app's, and the menu, smooth on/off and instant replay keys from other apps do
  nothing (a note says so).
- macOS's own window picker (`SCContentSharingPicker`), which needs no Screen Recording permission, as a second way
  to choose the window.
- Capture cards: listed and played through the core (by number, OpenCV's AVFoundation input, as `run.py` does on a
  Mac), untried with a real card, and without the card's sound on a Mac yet.
- Setup, updates and uninstall: the app runs from a checkout. Packaging the core, Python or the Rust binary, and
  chiaki-ng inside the app is for later.
