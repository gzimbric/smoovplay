// swift-tools-version: 6.0
// SmoovPlay for macOS: the SwiftUI launcher (the Windows WPF launcher's pages, installer/launcher), plus what only
// the Mac app does: window capture with ScreenCaptureKit into the frame tap, keys that work from any app, and
// hiding chiaki-ng's window while the video plays. The core (run.py today, the Rust `smoovplay` later) is a child
// process driven through its --control protocol (framegen/remote.py).
//
//   swift build                     # debug build of the app and its library
//   swift test                      # protocol, settings page model, frame tap writer (read back with Python)
//   ./bundle.sh                     # build/SmoovPlay.app, signed ad hoc (bundle.sh --open also opens it)
import PackageDescription

let package = Package(
    name: "SmoovPlay",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "SmoovPlay", targets: ["SmoovPlay"]),
    ],
    targets: [
        // POSIX shared memory and barriers for the frame tap (shm_open is variadic, which Swift can't call).
        .target(name: "CFrameTap", path: "Sources/CFrameTap"),
        // Everything that isn't a view, so tests can reach it.
        .target(
            name: "SmoovPlayKit",
            dependencies: ["CFrameTap"],
            path: "Sources/SmoovPlayKit",
            linkerSettings: [
                .linkedFramework("ScreenCaptureKit"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("Carbon"),
                .linkedFramework("IOKit"),
                .linkedFramework("AppKit"),
            ]
        ),
        .executableTarget(
            name: "SmoovPlay",
            dependencies: ["SmoovPlayKit"],
            path: "Sources/SmoovPlay"
        ),
        .testTarget(
            name: "SmoovPlayKitTests",
            dependencies: ["SmoovPlayKit", "CFrameTap"],
            path: "Tests/SmoovPlayKitTests",
            resources: [.copy("Fixtures")]
        ),
    ],
    swiftLanguageModes: [.v5]
)
