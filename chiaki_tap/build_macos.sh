#!/bin/bash
# Build chiaki-ng with the frame tap on macOS (Apple Silicon, Homebrew):
#
#   bash chiaki_tap/build_macos.sh <chiaki-ng source> <build folder>
#
# The source is a chiaki-ng v1.10.1 checkout with its submodules (git clone --branch v1.10.1 --recurse-submodules
# https://github.com/streetpea/chiaki-ng.git). The build folder takes the CMake build tree and the finished app,
# <build folder>/chiaki-ng.app, with run_chiaki_frametap.command beside it.
#
# This follows chiaki-ng's own macOS CI (the macOS jobs in .github/workflows/build-release.yaml and
# build-macos-arm.yml). It uses the same Homebrew packages, configure flags and deploy steps (macdeployqt, MoltenVK in
# the bundle, SDL next to the binary), and runs apply_patch.py before compiling. Where it differs, it's so that it
# needs no sudo and writes nothing outside the two folders:
#   - SDL3 and sdl2-compat come from Homebrew, at the versions the CI builds from source into /usr/local
#     (scripts/build-sdl2-compat.sh), and this script warns if Homebrew's differ
#   - MoltenVK comes from Homebrew's molten-vk, not the release tarball
#   - nanopb's generator gets protobuf from a venv in the build folder, not pip --user
#   - openssl@4 stays linked, and the build puts openssl@3's headers first instead (the CI unlinks openssl@4)
#   - no DMG
# It also adds three things. The app is assembled in <build folder>/chiaki-ng.new and swapped in only once it's
# signed, verified and free of Homebrew paths. It's signed inside out, in the order of the CI's scripts/macos-sign.sh,
# with an Apple Development identity if the keychain has one and ad hoc otherwise. And the WebEngine helper's links
# into Homebrew's Qt are pointed at the app's own.
# Re-running only rebuilds what changed and deploys the app again.
set -euo pipefail

usage="usage: build_macos.sh <chiaki-ng source> <build folder>"
SRC="$(cd "${1:?$usage}" && pwd)"
OUT="${2:?$usage}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$OUT/chiaki-ng.app"

step() { echo; echo "=== $*"; }
die() { echo "build_macos.sh: $*" >&2; exit 1; }

[[ "$(uname -s)" == Darwin ]] || die "this is the macOS build. On Windows use build_windows.sh"
command -v brew >/dev/null || die "needs Homebrew (https://brew.sh)"
[[ -f "$SRC/gui/src/streamsession.cpp" ]] || die "$SRC is not a chiaki-ng checkout"
[[ -f "$SRC/third-party/nanopb/CMakeLists.txt" ]] || die "$SRC has no submodules: git -C $SRC submodule update --init --recursive"
[[ "$(uname -m)" == arm64 ]] || echo "warning: chiaki-ng's macOS CI builds arm64 only, and this Mac is $(uname -m)" >&2
BREW="$(brew --prefix)"

step "Homebrew packages"
# chiaki-ng's CI list (qt@6 and pkgconfig are aliases of qt and pkgconf), plus sdl3, sdl2-compat and molten-vk,
# which the CI builds or downloads itself. wget isn't needed.
PKGS=(qt ffmpeg pkgconf opus openssl@3 cmake ninja nasm protobuf@29 speexdsp libplacebo python-setuptools json-c
      miniupnpc libevent sdl3 sdl2-compat molten-vk)
missing=()
for p in "${PKGS[@]}"; do
    brew list --versions "$p" >/dev/null 2>&1 || missing+=("$p")
done
if ((${#missing[@]})); then
    # Homebrew's Qt 6 links into the Homebrew prefix, where a force-linked qt@5 is in the way. Unlinking it is the
    # user's call (qt@5 stays installed, under $BREW/opt/qt@5), so stop rather than do it here.
    if [[ " ${missing[*]} " == *" qt "* ]] && [[ "$(readlink "$BREW/bin/qmake" 2>/dev/null)" == *qt@5* ]]; then
        die "Homebrew's Qt 6 can't be installed while qt@5 is linked. Run brew unlink qt@5, then run this again
(qt@5 stays installed, with its tools in $BREW/opt/qt@5/bin)."
    fi
    echo "installing: ${missing[*]}"
    brew install "${missing[@]}"
else
    echo "all installed"
fi

# The SDL versions chiaki-ng's CI builds.
sdl_ver="$(sed -n 's/^SDL_VER=//p' "$SRC/scripts/build-sdl2-compat.sh")"
compat_ver="$(sed -n 's/^SDL_COMPAT_VER=//p' "$SRC/scripts/build-sdl2-compat.sh")"
have_sdl="$(brew list --versions sdl3 | awk '{print $2}')"
have_compat="$(brew list --versions sdl2-compat | awk '{print $2}')"
if [[ "$have_sdl" != "$sdl_ver"* || "$have_compat" != "$compat_ver"* ]]; then
    echo "warning: chiaki-ng's CI builds SDL3 $sdl_ver and sdl2-compat $compat_ver, but Homebrew has $have_sdl and $have_compat" >&2
fi

step "Python for nanopb's generator (protobuf 5, as the CI)"
if [[ -x "$OUT/pyenv/bin/python" ]] && "$OUT/pyenv/bin/python" -c 'import google.protobuf' 2>/dev/null; then
    echo "already set up"
else
    "$BREW/bin/python3" -m venv "$OUT/pyenv"
    "$OUT/pyenv/bin/python" -m pip install --quiet --disable-pip-version-check 'protobuf>=5,<6' setuptools
fi
PY="$OUT/pyenv/bin/python"

step "frame tap patch"
"$PY" -I "$HERE/apply_patch.py" "$SRC"

step "configure chiaki-ng"
# A linked openssl@4 puts its headers in $BREW/include, which chiaki-lib has on its include path (for opus) ahead of
# openssl@3's. The build would then compile against OpenSSL 4 and fail to link against 3. The CI unlinks openssl@4.
# Here a folder holding only an "openssl" link to openssl@3's headers goes first on every target's include path. It
# has to be a folder of its own, because clang ignores an -I folder that is also an -isystem one, and
# openssl@3/include is one through the OpenSSL target.
mkdir -p "$OUT/openssl3-include"
ln -sfn "$BREW/opt/openssl@3/include/openssl" "$OUT/openssl3-include/openssl"
cat > "$OUT/openssl3_first.cmake" <<EOF
# Written by SmoovPlay's chiaki_tap/build_macos.sh. Puts openssl@3's headers before any in $BREW/include.
include_directories(BEFORE "$OUT/openssl3-include")
EOF
cmake -S "$SRC" -B "$OUT" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCHIAKI_ENABLE_CLI=OFF \
    -DCHIAKI_ENABLE_STEAMDECK_NATIVE=OFF \
    -DOPENSSL_ROOT_DIR="$BREW/opt/openssl@3" \
    -DOPENSSL_INCLUDE_DIR="$BREW/opt/openssl@3/include" \
    -DOPENSSL_CRYPTO_LIBRARY="$BREW/opt/openssl@3/lib/libcrypto.dylib" \
    -DOPENSSL_SSL_LIBRARY="$BREW/opt/openssl@3/lib/libssl.dylib" \
    -DCMAKE_PREFIX_PATH="$BREW/opt/openssl@3;$BREW/opt/qt@6;$BREW/opt/protobuf@29" \
    -DCMAKE_PROJECT_chiaki_INCLUDE="$OUT/openssl3_first.cmake" \
    -DPYTHON_EXECUTABLE="$PY" -DPython_EXECUTABLE="$PY"

step "build chiaki-ng"
export CPATH="$BREW/opt/ffmpeg/include"
cmake --build "$OUT" --config Release --target chiaki

step "deploy $APP"
# Assembled in a staging bundle and moved into place only once it's complete and verified, so a chiaki-ng that's
# running, or launched meanwhile, never sees a half-built app. Every file in it is new. macOS caches a file's
# signature once it has run, and editing a signed file in place after it ran gets the process that loads it killed.
STAGE_DIR="$OUT/chiaki-ng.new"
STAGE="$STAGE_DIR/chiaki-ng.app"  # still named .app, because macdeployqt and codesign want that
DEPLOYQT="$BREW/opt/qt@6/bin/macdeployqt"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
ditto "$OUT/gui/chiaki.app" "$STAGE"
# Makes macdeployqt bundle QtWebEngine's QML (chiaki-ng's PSN sign-in), as the CI does.
cp "$SRC/scripts/qtwebengine_import.qml" "$SRC/gui/src/qml/"
"$DEPLOYQT" "$STAGE" -qmldir="$SRC/gui/src/qml" -libpath="$BREW/lib"
# MoltenVK goes in the app, as in the CI. The Vulkan loader looks in an app's Contents/Resources/vulkan/icd.d first
# and, finding a driver there, doesn't load others, so not Homebrew's on a Mac that has it. The release tarball the CI
# uses has the driver with a manifest pointing next to it. Homebrew's manifest points into Homebrew, so write that one.
# The CI runs macdeployqt a second time after this. MoltenVK needs only system libraries, so that's skipped here.
ICD="$STAGE/Contents/Resources/vulkan/icd.d"
mkdir -p "$ICD"
cp -L "$BREW/opt/molten-vk/lib/libMoltenVK.dylib" "$ICD/"
api="$(sed -n 's/.*"api_version" *: *"\([^"]*\)".*/\1/p' "$BREW/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json")"
cat > "$ICD/MoltenVK_icd.json" <<EOF
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "./libMoltenVK.dylib",
        "api_version" : "${api:-1.4.0}",
        "is_portability_driver" : true
    }
}
EOF
# QtWebEngine's helper (chiaki-ng's PSN sign-in page) is an app inside QtWebEngineCore.framework, which macdeployqt
# leaves alone. The CI links its Frameworks folder to the app's. This goes further. The helper also links some Qt
# frameworks straight from Homebrew, which would load a second copy of Qt beside the app's, so point those at the app's.
HELPER="$STAGE/Contents/Frameworks/QtWebEngineCore.framework/Versions/A/Helpers/QtWebEngineProcess.app"
if [[ -d "$HELPER" ]]; then
    [[ -e "$HELPER/Contents/Frameworks" ]] || ln -s ../../../../../../../Frameworks "$HELPER/Contents/Frameworks"
    HELPER_BIN="$HELPER/Contents/MacOS/QtWebEngineProcess"
    otool -L "$HELPER_BIN" | awk -v p="$BREW/" 'NR > 1 && index($1, p) == 1 {print $1}' | while read -r dep; do
        fw="${dep#*/lib/}"  # QtCore.framework/Versions/A/QtCore
        [[ -e "$STAGE/Contents/Frameworks/$fw" ]] || die "the WebEngine helper needs $fw, which isn't in the app"
        install_name_tool -change "$dep" "@executable_path/../Frameworks/$fw" "$HELPER_BIN" 2>/dev/null
    done
fi
# SDL goes next to the binary, as in the CI, because sdl2-compat loads SDL3 at run time and looks beside itself first.
MACOS="$STAGE/Contents/MacOS"
cp -L "$BREW/opt/sdl3/lib/libSDL3.dylib" "$MACOS/libSDL3.dylib"
cp -L "$BREW/opt/sdl2-compat/lib/libSDL2-2.0.0.dylib" "$MACOS/libSDL2-2.0.0.dylib"
chmod u+w "$MACOS/libSDL3.dylib" "$MACOS/libSDL2-2.0.0.dylib"
install_name_tool -id @executable_path/libSDL3.dylib "$MACOS/libSDL3.dylib" 2>/dev/null
install_name_tool -id @executable_path/libSDL2-2.0.0.dylib "$MACOS/libSDL2-2.0.0.dylib" 2>/dev/null
# Wherever macdeployqt pointed chiaki-ng's SDL2 (usually its own copy in Frameworks), point it at this one.
old_sdl2="$(otool -L "$MACOS/chiaki" | awk '/libSDL2-2\.0\.0\.dylib/ && !n++ {print $1}')"
if [[ -n "$old_sdl2" && "$old_sdl2" != @executable_path/libSDL2-2.0.0.dylib ]]; then
    install_name_tool -change "$old_sdl2" @executable_path/libSDL2-2.0.0.dylib "$MACOS/chiaki" 2>/dev/null
fi
rm -f "$STAGE/Contents/Frameworks/libSDL2-2.0.0.dylib" "$STAGE/Contents/Frameworks/libSDL3.dylib" \
      "$STAGE/Contents/Frameworks/libSDL3.0.dylib"
[[ -e "$STAGE/Contents/Frameworks/vulkan" ]] || ln -s libvulkan.1.dylib "$STAGE/Contents/Frameworks/vulkan"

step "sign"
# Inside out, in the order of chiaki-ng's scripts/macos-sign.sh (which signs releases with a Developer ID). First
# every loose library (Qt and QML plugins, FFmpeg and the other dylibs, SDL, MoltenVK in Resources), then the
# WebEngine helper, each framework, and last the app with chiaki-ng's entitlements. Apple Silicon runs nothing
# unsigned, and the edits above broke the libraries' own signatures. There's no hardened runtime, since a build for
# this Mac has no use for it.
# SIGN_IDENTITY picks the identity, a name or SHA-1 from `security find-identity -v -p codesigning`. Unset or "auto"
# means the keychain's first Apple Development identity, else ad hoc, and "-" forces ad hoc. A real identity keeps
# the app the same to macOS from build to build, so its Local Network permission (finding the PS5) isn't asked for
# again.
IDENTITY="${SIGN_IDENTITY:-auto}"
if [[ "$IDENTITY" == auto ]]; then
    IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null | awk '/"Apple Development: /{print $2; exit}')"
    IDENTITY="${IDENTITY:--}"
fi
echo "signing $([[ "$IDENTITY" == "-" ]] && echo "ad hoc" || echo "with identity $IDENTITY")"
sign_code() { codesign --force --sign "$IDENTITY" --timestamp=none "$@"; }
find "$STAGE/Contents" -type f \( -name '*.dylib' -o -name '*.so' \) -print0 |
    xargs -0 codesign --force --sign "$IDENTITY" --timestamp=none
find "$STAGE/Contents/Frameworks" -type d -name '*.app' -path '*/Helpers/*' -print0 | while IFS= read -r -d '' h; do
    sign_code --entitlements "$SRC/scripts/macos/QtWebEngineProcess.entitlements" "$h"
done
find "$STAGE/Contents/Frameworks" -maxdepth 1 -type d -name '*.framework' -print0 |
    xargs -0 codesign --force --sign "$IDENTITY" --timestamp=none
sign_code --entitlements "$SRC/gui/entitlements.xml" "$STAGE"
codesign --verify --deep --strict --verbose=2 "$STAGE" || die "the app's signature doesn't verify"

# Check every library and program in the app. None may still load from Homebrew, since it would run here and break
# on a Mac without it. Each must carry a valid signature of its own, because codesign --deep doesn't check code in
# Resources. Not --strict here, because that rejects the WebEngine helper's Frameworks link, which leads out of the
# helper (but not out of the app).
links_brew() {  # whether the libraries a Mach-O file loads (not its own install name) include one in Homebrew
    local id
    id="$(otool -D "$1" | tail -n +2)"
    otool -L "$1" | awk -v p="$BREW/" -v id="$id" 'NR > 1 && $1 != id && index($1, p) == 1 {f = 1} END {exit !f}'
}
bad="$(find "$STAGE/Contents" -type f -print0 | xargs -0 file | awk -F': ' '/Mach-O/{print $1}' |
    sed 's/ (for architecture .*//' | sort -u | while read -r f; do
        links_brew "$f" && echo "links Homebrew: $f"
        codesign --verify "$f" 2>/dev/null || echo "bad signature: $f"
    done)" || true
[[ -z "$bad" ]] || die "the app isn't self-contained:
$bad"

step "replace $APP"
if pgrep -qf "^$APP/Contents/MacOS/chiaki"; then
    die "chiaki-ng is running from $APP. Quit it and run this again (the new app is ready at $STAGE)"
fi
rm -rf "$OUT/chiaki-ng.app.old"
[[ -e "$APP" ]] && mv "$APP" "$OUT/chiaki-ng.app.old"
mv "$STAGE" "$APP"
rm -rf "$OUT/chiaki-ng.app.old" "$STAGE_DIR"

# The launcher beside the app, replaced rather than written over.
rm -f "$OUT/run_chiaki_frametap.command"
cp "$HERE/run_chiaki_frametap.command" "$OUT/"
chmod +x "$OUT/run_chiaki_frametap.command"
echo
echo "done: $APP"
echo "start it with the tap on: $OUT/run_chiaki_frametap.command"
