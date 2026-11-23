#!/bin/bash
# Start chiaki-ng with the frame tap on, so SmoovPlay's "run.py --chiaki" receives every video frame.
# Without CHIAKI_FRAMETAP=1 this build behaves exactly like stock chiaki-ng.
#
# build_macos.sh copies this next to chiaki-ng.app. Double-click it in Finder or run it from a terminal. Arguments
# go to chiaki-ng. CHIAKI_APP=/path/to/chiaki-ng.app picks another app.
#
# The app is started through LaunchServices (open), so it doesn't depend on this terminal. LaunchServices doesn't
# pass the shell's environment on, so every CHIAKI_* variable set here (CHIAKI_FRAMETAP_NAME,
# CHIAKI_FRAMETAP_RENDERER and the rest) is handed over explicitly.
set -euo pipefail

APP="${CHIAKI_APP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/chiaki-ng.app}"
BIN="$APP/Contents/MacOS/chiaki"
if [[ ! -x "$BIN" ]]; then
    echo "no chiaki-ng at $APP (build it with chiaki_tap/build_macos.sh, or set CHIAKI_APP)" >&2
    exit 1
fi
# A chiaki-ng that's already running would only come to the front, without the tap.
if pgrep -qf "^$BIN"; then
    echo "chiaki-ng is already running. Quit it first, so it starts again with the tap on" >&2
    exit 1
fi

export CHIAKI_FRAMETAP=1
ENV_ARGS=()
while IFS= read -r var; do
    ENV_ARGS+=(--env "$var=${!var}")
done < <(compgen -e | /usr/bin/grep '^CHIAKI_')

open -n "${ENV_ARGS[@]}" "$APP" --args "$@"
echo "chiaki-ng started with the frame tap on${CHIAKI_FRAMETAP_NAME:+ (tap name: $CHIAKI_FRAMETAP_NAME)}"
