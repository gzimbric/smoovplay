#!/bin/bash
# Builds the Rust core (target/release/smoovplay) and signs it, so the SmoovPlay app can start it.
#
#   ./build_release.sh
#
# Signing: SIGN_IDENTITY picks the code-signing identity (a name or SHA-1 from `security find-identity -v -p
# codesigning`); unset or "auto", the keychain's first Apple Development identity, else ad hoc; "-" forces ad hoc.
# The same rule as mac/app/bundle.sh and chiaki_tap/build_macos.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
command -v cargo >/dev/null || export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
cargo build --release -p smoovplay

IDENTITY="${SIGN_IDENTITY:-auto}"
if [[ "$IDENTITY" == auto ]]; then
    IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null | awk '/"Apple Development: /{print $2; exit}')"
    IDENTITY="${IDENTITY:--}"
fi
BIN="$HERE/target/release/smoovplay"
codesign --force --sign "$IDENTITY" --timestamp=none --identifier com.smoovplay.core "$BIN"
codesign --verify --strict "$BIN"
echo "built $BIN (signed $([[ "$IDENTITY" == "-" ]] && echo "ad hoc" || echo "with identity $IDENTITY"))"
