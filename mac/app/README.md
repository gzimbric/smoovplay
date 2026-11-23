# SmoovPlay for Mac

A SwiftUI app that takes the place of the Windows launcher on a Mac. It also does three things only a Mac app can:
it copies another app's window with ScreenCaptureKit, it registers keys that work from any app, and it hides
chiaki-ng's window while the video plays.

The video itself comes from the core, which the app runs as a child process. That's the Rust `smoovplay` in
`mac/core` once it's built, and `run.py` otherwise. Both take the same command line and `--control` commands. With
the Rust core, Auto-detect, the looks commands and the looks preview run as its subcommands, so no Python is needed.

```
./bundle.sh --open          # builds and opens build/SmoovPlay.app
swift test                  # runs the checks below in about 6 seconds
```

`bundle.sh` signs the app with the keychain's Apple Development identity if there is one, otherwise ad hoc. With a
real identity macOS keeps the app's Screen Recording permission from one build to the next. Set `SIGN_IDENTITY` to
use another identity, or `-` for ad hoc. The app needs macOS 14 or newer.

The app runs against this checkout. It finds the repo through a path `bundle.sh` writes into its Info.plist, and the
patched chiaki-ng at `build/chiaki-ng-frametap/chiaki-ng.app` (see `chiaki_tap/README.md`).

## Pages

| Page | What it does |
|---|---|
| Play | Starts and stops a session and shows what's happening. Picks the source (PS5 Remote Play, capture card, app window or frame tap) and the card or window to use. Shows the settings in use, notes from the core, Test with a 60 fps game and its results card, and the session log. While the video plays the window goes to the Dock with a badge. |
| Connect your PS5 | The eight setup steps, which it remembers, and a button that opens the patched chiaki-ng with the frame tap on. Other sources have their own guide here. The app opens on this page until the PS5 guide is done. |
| Mac check | The chip and memory, the display's refresh rate, disk space, the core, chiaki-ng, the network, and Screen Recording for app window mode. |
| Settings | Built from the core's own list of settings. With nothing playing it reads and writes the settings file. While playing it sends each change to the core and follows its state, with Restart now for settings that need one. Choices a Mac can't use, such as RTX VSR, are greyed out with the reason. |
| Looks | SmoovPlay's looks and yours. Edit a look's effects with a live preview on sample pictures, save it, copy or paste a share code, or add a .cube LUT. While playing, the video shows the look as you edit it. |
| Help | Copy diagnostics (with your user folder and device names left out), the logs folder, the video window's keys, and which core to use. |

Logs go to `~/Library/Logs/SmoovPlay`. Each session gets `session-*.log` with the core's output, the commands sent
and the app's own `[launcher]` and `[capture]` lines. Test mode adds `results-*.txt`. The app's own choices, such as
the guide's ticks and the source, are in `~/Library/Application Support/framegen/smoovplay-launcher.txt`.

## What the app does itself

### App window

On Windows the core copies the window. On a Mac the app does it with ScreenCaptureKit and hands each picture to the
core through the frame tap (`docs/frame-tap.md`) as POSIX shared memory named `/smoovplay-window`. The core reads it
with `--source tap --tap smoovplay-window`.

The window list puts known streaming apps first: PS Remote Play, Moonlight, Steam Link, GeForce NOW and Parsec. Then
come browsers with a tab that looks like a cloud game, then other browsers, then everything else. The core's own
video window is left out.

The window is copied on its own, so windows on top of it don't get in the way. Its title bar is left out unless you
turn that off on the Play page, for apps that draw under it. Pictures are fitted into 1920x1080 at up to 60 a second
and sent only when the window changed. When ScreenCaptureKit says nothing changed, the frame goes as a repeat.
Otherwise the core compares pictures, because a streaming app decodes its repeats again and they are never quite
identical.

With the Rust core, the app asks for the video window to be put over the copied one through the `overlay` command,
and sends the new position whenever the window moves.

### Screen Recording

App window mode needs Screen Recording permission. While it's off, the Play page shows a card with two buttons.
Allow brings up macOS's prompt, which it only shows once. Open System Settings goes to Privacy & Security, Screen &
System Audio Recording, where you turn SmoovPlay on and then reopen it. Started from a terminal, the app uses the
terminal's permission instead.

### Keys from any app

These use Carbon's `RegisterEventHotKey`, so they need no Accessibility permission and the app only hears its own
shortcuts.

While a session plays with instant replay on, Control-Option-R saves the replay and Control-Option-B makes a before
and after video. In app window mode, Control-Option also gives the video window's keys: M for the menu, F for
fullscreen, Esc to quit, S for stats and I to switch frame generation off and on. If macOS or another app already
uses a shortcut, the next one on the list is taken, and you can pick your own on the Play page. Each press is sent
to the core as a `key` command.

### chiaki-ng

Play starts the core first. When the core says it's waiting for chiaki-ng, the app opens the patched chiaki-ng with
`CHIAKI_FRAMETAP=1`, unless the copy it opened earlier is still running. If a chiaki-ng it didn't open is running,
it asks what to do, as the Windows launcher does. While the video plays, chiaki-ng's window is hidden (it keeps
streaming) and is hidden again if it shows itself in the first 10 seconds. It comes back when the session ends.

While the video plays the app also keeps the display awake, since a controller doesn't count as input to macOS.

## Commands beyond the Windows launcher's

The Rust core takes these. `run.py` answers `[remote] unknown command` to them, and the app then stops sending them
for that session. `CoreProtocol.swift` has the whole protocol.

| Command | Used for | What the core does |
|---|---|---|
| `key <name>` | Instant replay keys and app window keys | Acts as if the key were pressed in the video window. A bare `key` does nothing, which tells the app whether the core takes keys. |
| `overlay <x> <y> <w> <h>` and `overlay none` | App window mode | Puts the video window over that rectangle, borderless and floating, letting clicks through and never taking the keys. The app sends the rectangle when the window moves, at most twice a second. `none` makes it a normal window again. |
| `menu_open` in the state | App window mode | Says when the video window's menu is open, so the app passes on the menu's own keys only then. |

## Switches

Environment variables:

| Variable | Meaning |
|---|---|
| `SMOOVPLAY_APP` | The repo |
| `SMOOVPLAY_CORE` | `python`, `rust`, or the path to a core binary. Overrides the Help page. |
| `SMOOVPLAY_PYTHON` | The Python to use with `run.py` |
| `SMOOVPLAY_RUN_ARGS` | More arguments for the core |
| `SMOOVPLAY_SETTINGS` | Another settings file, passed to the core and Auto-detect |
| `SMOOVPLAY_PREFS`, `SMOOVPLAY_LOGS` | Other places for the app's choices and logs |
| `SMOOVPLAY_CHIAKI` | Another chiaki-ng.app |
| `SMOOVPLAY_SOURCES=all` | Also offer the Video file source, with `SMOOVPLAY_SOURCE_FILE` for the file |
| `SMOOVPLAY_TAP` | Another tap name for app window mode |
| `SMOOVPLAY_DETECT_ARGS` | Extra Auto-detect arguments, for example `--timing 9.3,6.8` to skip timing the GPU |
| `SMOOVPLAY_CAPTURE_HZ` | The window capture's top rate, 60 by default |
| `SMOOVPLAY_SHOTS` | A folder that gets a picture of the app's own window on every page or state change |
| `SMOOVPLAY_TRACE` | A file for a trace of what the app does |

Command line: `--page play|guide|check|settings|looks|help`, `--source <kind>`, `--window <app or title words>`,
`--play`, `--test`, `--stop-after <seconds>`, `--quit-after-stop`, `--keep-window`, `--detect`,
`--show-results <file>` and `--diagnostics <file>`.

To try it end to end without a console, play `testdata/clip_1080p60.mp4` in a window in a loop, then run

```
build/SmoovPlay.app/Contents/MacOS/SmoovPlay --source window --window <its app> --play --stop-after 20 --quit-after-stop
```

or use the fake PS5 on a tap of its own. chiaki-ng still opens, and is hidden and shown again.

```
.venv/bin/python scripts/frametap_fake.py testdata/clip_1080p60.mp4 --delay 10 --native --tap 'Local\sp-e2e' &
SMOOVPLAY_RUN_ARGS='--chiaki-tap Local\sp-e2e' build/SmoovPlay.app/Contents/MacOS/SmoovPlay --test --stop-after 10
```

## Checks

`swift test` uses fixtures captured from the real core, with the home folder taken out. It covers:

- the protocol, including a real `--control` session and states with a pending restart and with frame generation off
- the Settings page model built from the core's list and those states, including writes to the settings file that
  the core then reads back
- the frame tap writer, read back by `framegen/chiaki_tap.py` and played by `run.py --source tap`
- Auto-detect's result, the test report's card, looks and share codes, shortcuts, and diagnostics leaving out the
  user

`SMOOVPLAY_TEST_DISPLAY=1 swift test --filter liveSession` runs a real session through the app's `CoreSession`. It
opens the video window, changes a setting live, checks the state and quits.

## Files

| File | Contents |
|---|---|
| `Package.swift` | Three targets and the tests |
| `bundle.sh` | Builds and signs `build/SmoovPlay.app` |
| `Sources/CFrameTap` | POSIX shared memory for the frame tap, in C because Swift can't call the variadic `shm_open` |
| `Sources/SmoovPlayKit/CoreProtocol.swift` | Parsing the core's lines, the state and the commands |
| `Sources/SmoovPlayKit/CoreProcess.swift` | Running the core and the one-shot helpers |
| `Sources/SmoovPlayKit/Paths.swift` | Where everything is, the core choice and the switches |
| `Sources/SmoovPlayKit/Settings.swift` | The Settings page model and the settings file |
| `Sources/SmoovPlayKit/FrameTap.swift`, `WindowCapture.swift` | The tap writer, window capture and Screen Recording |
| `Sources/SmoovPlayKit/Hotkeys.swift` | Shortcuts and app window keys |
| `Sources/SmoovPlayKit/Chiaki.swift` | Opening, hiding and showing chiaki-ng, and keeping the Mac awake |
| `Sources/SmoovPlay` | The app's state and one file per page |

## Not done yet

- macOS's own window picker, `SCContentSharingPicker`, which needs no Screen Recording permission.
- Capture cards with a real card. The core reads them through AVFoundation with their sound, but only a recording
  standing in for a card has been tried.
- Packaging. The app runs from a checkout, and bundling the core and chiaki-ng inside it is for later.
