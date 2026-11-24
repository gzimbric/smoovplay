#!/bin/bash
# Packages SmoovPlay for a Mac as one app that needs nothing else: no checkout, no Python and no Homebrew.
#
#   mac/package.sh                  # build/package/SmoovPlay.app and build/SmoovPlay-macOS.dmg
#   mac/package.sh --no-dmg         # just the app
#
# The app carries the Rust core as Contents/Helpers/smoovplay, the patched chiaki-ng as
# Contents/Helpers/chiaki-ng.app, and RIFE's weights and the looks preview's pictures in Contents/Resources.
# Build chiaki-ng first with chiaki_tap/build_macos.sh. CHIAKI_APP points to another build.
#
# Signing follows the other scripts: SIGN_IDENTITY by name or SHA-1, unset or "auto" for the keychain's first Apple
# Development identity (ad hoc if it has none), "-" for ad hoc. This is for running on your own Macs. Sharing it
# with other people would need a Developer ID certificate, the hardened runtime and notarisation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="$REPO/build/package"
DMG="$REPO/build/SmoovPlay-macOS.dmg"
CHIAKI="${CHIAKI_APP:-$REPO/build/chiaki-ng-frametap/chiaki-ng.app}"
MODEL="4.22.lite"
MAKE_DMG=1
for a in "$@"; do
    case "$a" in
    --no-dmg) MAKE_DMG=0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
    esac
done

die() { echo "package: $*" >&2; exit 1; }
step() { echo; echo "=== $*"; }

[[ "$(uname)" == Darwin ]] || die "this packages the Mac app, so it runs on a Mac"
[[ -d "$CHIAKI" ]] || die "no chiaki-ng at $CHIAKI. Build it first:
  bash chiaki_tap/build_macos.sh <chiaki-ng source> $REPO/build/chiaki-ng-frametap"
WEIGHTS="$REPO/models/flownet_v$MODEL.pkl"

IDENTITY="${SIGN_IDENTITY:-auto}"
if [[ "$IDENTITY" == auto ]]; then
    IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null | awk '/"Apple Development: /{print $2; exit}')"
    IDENTITY="${IDENTITY:--}"
fi
export SIGN_IDENTITY="$IDENTITY"  # the two build scripts below sign the same way

step "the Rust core"
"$REPO/mac/core/build_release.sh"
CORE="$REPO/mac/core/target/release/smoovplay"
if [[ ! -f "$WEIGHTS" ]]; then
    step "RIFE $MODEL weights"
    "$CORE" prepare --model "$MODEL" --no-detect
fi
[[ -f "$WEIGHTS" ]] || die "RIFE's weights aren't at $WEIGHTS"

step "the app"
"$REPO/mac/app/bundle.sh"

step "assembling"
STAGE="$OUT/SmoovPlay.new.app"
APP="$OUT/SmoovPlay.app"
mkdir -p "$OUT"
rm -rf "$STAGE"
ditto "$REPO/mac/app/build/SmoovPlay.app" "$STAGE"
# A packaged app finds everything inside itself, not in this checkout
plutil -remove SmoovPlayRepo "$STAGE/Contents/Info.plist" 2>/dev/null || true
# Not in Contents/MacOS: the Mac's filesystem ignores case, so "smoovplay" there would replace "SmoovPlay", the app
mkdir -p "$STAGE/Contents/Helpers"
install -m 755 "$CORE" "$STAGE/Contents/Helpers/smoovplay"
ditto "$CHIAKI" "$STAGE/Contents/Helpers/chiaki-ng.app"
RES="$STAGE/Contents/Resources"
mkdir -p "$RES/models" "$RES/samples" "$RES/installer"
cp "$WEIGHTS" "$RES/models/"
cp "$REPO"/framegen/samples/* "$RES/samples/"
cp "$REPO/LICENSE" "$REPO/NOTICE" "$RES/"
cp "$REPO/installer/THIRD_PARTY_NOTICES.txt" "$RES/installer/"

step "signing"
echo "signing $([[ "$IDENTITY" == "-" ]] && echo "ad hoc" || echo "with identity $IDENTITY")"
# Inside out. chiaki-ng was signed whole by its own build, so it's checked rather than signed again.
codesign --verify --deep --strict "$STAGE/Contents/Helpers/chiaki-ng.app" || die "chiaki-ng's signature doesn't verify"
codesign --force --sign "$IDENTITY" --timestamp=none --identifier com.smoovplay.core "$STAGE/Contents/Helpers/smoovplay"
codesign --force --sign "$IDENTITY" --timestamp=none "$STAGE"
codesign --verify --deep --strict "$STAGE" || die "the app's signature doesn't verify"

# Swap it in only once it's whole and signed
if pgrep -f "$APP/Contents/MacOS/SmoovPlay" >/dev/null; then
    die "SmoovPlay is running from $APP. Quit it, then run this again (the new app is in $STAGE)"
fi
rm -rf "$APP"
mv "$STAGE" "$APP"
echo "built $APP ($(du -sh "$APP" | cut -f1))"

if [[ $MAKE_DMG == 1 ]]; then
    step "disk image"
    DIR="$(mktemp -d)"
    ditto "$APP" "$DIR/SmoovPlay.app"
    ln -s /Applications "$DIR/Applications"
    rm -f "$DMG"
    hdiutil create -quiet -volname SmoovPlay -srcfolder "$DIR" -fs HFS+ -format UDZO "$DMG"
    rm -rf "$DIR"
    [[ "$IDENTITY" == "-" ]] || codesign --force --sign "$IDENTITY" --timestamp=none "$DMG"
    echo "built $DMG ($(du -h "$DMG" | cut -f1))"
fi
