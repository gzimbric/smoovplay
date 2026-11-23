#!/bin/bash
# Builds build/SmoovPlay.app from this package: the release binary, an Info.plist, the icon (docs/brand's SVG, or
# installer/icon's PNGs without rsvg-convert), signed ad hoc from the inside out, then checked.
#
#   ./bundle.sh            # build/SmoovPlay.app
#   ./bundle.sh --open     # ...and open it
#   ./bundle.sh --debug    # a debug build (faster to build)
#
# Signing: SIGN_IDENTITY picks the code-signing identity (a name or SHA-1 from `security find-identity -v -p
# codesigning`); unset or "auto", the keychain's first Apple Development identity, else ad hoc; "-" forces ad hoc.
# A real identity matters while testing window capture: macOS ties the Screen Recording permission to the app's
# signature, and an ad hoc signature changes with every build, so each new build would have to be allowed again.
# The first signing with a keychain identity asks to use its key: Always Allow makes that once.
# Uses Xcode's toolchain when it's there (DEVELOPER_DIR), else the command line tools.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CONFIG=release
OPEN=0
for a in "$@"; do
    case "$a" in
        --open) OPEN=1 ;;
        --debug) CONFIG=debug ;;
        *) echo "unknown option $a" >&2; exit 2 ;;
    esac
done
if [[ -z "${DEVELOPER_DIR:-}" && -d /Applications/Xcode.app/Contents/Developer ]]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi
VERSION="${SMOOVPLAY_VERSION:-0.2.0}"
BUILD="$(git -C "$REPO" rev-list --count HEAD 2>/dev/null || echo 1)"

cd "$HERE"
swift build -c "$CONFIG" --product SmoovPlay
BIN="$(swift build -c "$CONFIG" --product SmoovPlay --show-bin-path)/SmoovPlay"

OUT="$HERE/build"
APP="$OUT/SmoovPlay.app"
NEW="$OUT/SmoovPlay.new.app"
rm -rf "$NEW"
mkdir -p "$NEW/Contents/MacOS" "$NEW/Contents/Resources"
cp "$BIN" "$NEW/Contents/MacOS/SmoovPlay"

# ---- the icon: a Mac icon's body is 824 of its 1024 pixels, so the brand's square goes in that much smaller
ICONSET="$OUT/AppIcon.iconset"
rm -rf "$ICONSET"
mkdir -p "$ICONSET"
SVG="$REPO/docs/brand/smoovplay-icon.svg"
if command -v rsvg-convert >/dev/null && [[ -f "$SVG" ]]; then
    PADDED="$OUT/icon-padded.svg"
    INNER="$(sed -e 's/^<svg[^>]*>//' -e 's/<\/svg>[[:space:]]*$//' "$SVG")"
    printf '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024"><g transform="translate(100 100) scale(0.8046875)">%s</g></svg>' "$INNER" > "$PADDED"
    for s in 16 32 128 256 512; do
        rsvg-convert -w $s -h $s "$PADDED" -o "$ICONSET/icon_${s}x${s}.png"
        rsvg-convert -w $((s * 2)) -h $((s * 2)) "$PADDED" -o "$ICONSET/icon_${s}x${s}@2x.png"
    done
    rm -f "$PADDED"
else
    PNG="$REPO/installer/icon"
    for s in 16 32 128 256; do
        src="$PNG/smoovplay-$([[ $s -gt 256 ]] && echo 256 || echo $s).png"
        [[ -f "$src" ]] || src="$PNG/smoovplay-256.png"
        sips -z $s $s "$src" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
        sips -z $((s * 2)) $((s * 2)) "$PNG/smoovplay-256.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
    done
fi
iconutil -c icns "$ICONSET" -o "$NEW/Contents/Resources/AppIcon.icns"
rm -rf "$ICONSET"

cat > "$NEW/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleExecutable</key><string>SmoovPlay</string>
    <key>CFBundleIdentifier</key><string>io.github.smoovplay.SmoovPlay</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>SmoovPlay</string>
    <key>CFBundleDisplayName</key><string>SmoovPlay</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$BUILD</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.games</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
    <key>NSHumanReadableCopyright</key><string>MIT licence. Not affiliated with Sony, NVIDIA or AMD.</string>
    <key>NSCameraUsageDescription</key><string>SmoovPlay reads your capture card's picture, to smooth a console's HDMI output. It's only used while you play with a capture card.</string>
    <key>NSMicrophoneUsageDescription</key><string>SmoovPlay plays your capture card's sound in step with the smoothed picture. It's only used while you play with a capture card.</string>
    <key>NSScreenCaptureUsageDescription</key><string>SmoovPlay reads the window of the app you stream with (PS Remote Play, Moonlight, a cloud game in a browser), to smooth its picture. Nothing is recorded or sent anywhere.</string>
    <key>SmoovPlayRepo</key><string>$REPO</string>
</dict>
</plist>
EOF
plutil -lint "$NEW/Contents/Info.plist" >/dev/null

# ---- sign from the inside out (there's one binary), then the app; check it
IDENTITY="${SIGN_IDENTITY:-auto}"
if [[ "$IDENTITY" == auto ]]; then
    IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null | awk '/"Apple Development: /{print $2; exit}')"
    IDENTITY="${IDENTITY:--}"
fi
codesign --force --sign "$IDENTITY" --timestamp=none "$NEW/Contents/MacOS/SmoovPlay"
codesign --force --sign "$IDENTITY" --timestamp=none "$NEW"
codesign --verify --deep --strict "$NEW"

# Swap it in only once it's whole (a running copy keeps its own files until it quits)
rm -rf "$APP"
mv "$NEW" "$APP"
echo "built $APP ($CONFIG, signed $([[ "$IDENTITY" == "-" ]] && echo "ad hoc" || echo "as $IDENTITY"))"
if [[ $OPEN == 1 ]]; then open "$APP"; fi
