// Child processes: the core (a play session, run.py --control or the Rust core) and the one-shot Python helpers.
import Foundation

/// A child process whose stdout and stderr come in as lines, and whose stdin takes lines. Callbacks come on the
/// queue given (the main queue for the UI). Writing to a child that has gone never raises SIGPIPE: the app ignores
/// it (SmoovPlayKit.ignoreSigpipe), and a failed write just returns false.
public final class LineProcess: @unchecked Sendable {
    public let process = Process()
    private let input = Pipe(), output = Pipe(), errors = Pipe()
    private let lock = NSLock()
    private var buffers: [ObjectIdentifier: Data] = [:]
    private let group = DispatchGroup()
    private let queue: DispatchQueue
    public var onLine: (String) -> Void = { _ in }
    public var onExit: (Int32) -> Void = { _ in }
    private var stdinOpen = true

    public init(executable: URL, arguments: [String], cwd: URL?, env: [String: String], queue: DispatchQueue = .main) {
        self.queue = queue
        process.executableURL = executable
        process.arguments = arguments
        if let cwd { process.currentDirectoryURL = cwd }
        process.environment = env
        process.standardInput = input
        process.standardOutput = output
        process.standardError = errors
    }

    public func start() throws {
        SmoovPlayKit.ignoreSigpipe()
        for pipe in [output, errors] {
            group.enter()
            let id = ObjectIdentifier(pipe)
            pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
                let d = h.availableData
                guard let self else { return }
                if d.isEmpty {
                    h.readabilityHandler = nil
                    self.flush(id, final: true)
                    self.group.leave()
                } else {
                    self.lock.lock()
                    self.buffers[id, default: Data()].append(d)
                    self.lock.unlock()
                    self.flush(id, final: false)
                }
            }
        }
        group.enter()
        process.terminationHandler = { [weak self] _ in self?.group.leave() }
        try process.run()
        group.notify(queue: queue) { [weak self] in
            guard let self else { return }
            self.onExit(self.process.terminationStatus)
        }
    }

    private func flush(_ id: ObjectIdentifier, final: Bool) {
        var lines: [String] = []
        lock.lock()
        var buf = buffers[id] ?? Data()
        while let nl = buf.firstIndex(of: 0x0A) {
            var lineData = buf[buf.startIndex..<nl]
            if lineData.last == 0x0D { lineData = lineData.dropLast() }
            lines.append(String(decoding: lineData, as: UTF8.self))
            buf = Data(buf[(nl + 1)...])
        }
        if final, !buf.isEmpty {
            lines.append(String(decoding: buf, as: UTF8.self))
            buf = Data()
        }
        buffers[id] = buf
        lock.unlock()
        guard !lines.isEmpty else { return }
        queue.async { [weak self] in
            guard let self else { return }
            for l in lines { self.onLine(l) }
        }
    }

    /// One line on the child's stdin. False if it has gone.
    @discardableResult
    public func send(_ line: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard stdinOpen, process.isRunning else { return false }
        do {
            try input.fileHandleForWriting.write(contentsOf: Data((line + "\n").utf8))
            return true
        } catch {
            return false
        }
    }

    /// Closes stdin: a child reading commands ends by itself (looks_preview, the core's command thread).
    public func closeInput() {
        lock.lock()
        defer { lock.unlock() }
        guard stdinOpen else { return }
        stdinOpen = false
        try? input.fileHandleForWriting.close()
    }

    public var running: Bool { process.isRunning }
    public var pid: Int32 { process.processIdentifier }

    public func terminate() { if process.isRunning { process.terminate() } }
    public func kill() { if process.isRunning { Darwin.kill(process.processIdentifier, SIGKILL) } }
}

public enum SmoovPlayKit {
    private static let once: Void = { signal(SIGPIPE, SIG_IGN) }()
    public static func ignoreSigpipe() { _ = once }
}

/// One play session: the core started with --display --control, its log, its state, and Stop.
public final class CoreSession: @unchecked Sendable {
    public struct Options: Sendable {
        public var source: String  // a SourceDef kind
        public var sourceArgs: [String]
        public var testMode = false
        public var extraArgs: [String] = []
        public init(source: String, sourceArgs: [String]) { self.source = source; self.sourceArgs = sourceArgs }
    }

    public let paths: AppPaths
    public let options: Options
    public private(set) var logURL: URL
    public private(set) var resultsURL: URL
    public private(set) var lastLine = ""
    public private(set) var live = false  // a state line came: the video window reads commands
    public private(set) var streamEnded = false
    public private(set) var exitCode: Int32?
    public private(set) var stopping = false
    public private(set) var arguments: [String] = []
    /// Commands the core answered "unknown command" to, this session (key, overlay...).
    public private(set) var unsupported: Set<String> = []
    private var proc: LineProcess?
    private var log: FileHandle?
    private let logLock = NSLock()

    public var onLine: (String) -> Void = { _ in }
    public var onEvent: (CoreEvent) -> Void = { _ in }
    public var onState: (PlayState) -> Void = { _ in }
    /// Where the callbacks come (the main queue for the UI).
    public var queue: DispatchQueue = .main

    public init(paths: AppPaths, options: Options, now: Date = Date()) {
        self.paths = paths
        self.options = options
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd-HHmmss"
        f.locale = Locale(identifier: "en_US_POSIX")
        let stamp = f.string(from: now)
        logURL = paths.logs.appendingPathComponent("session-\(stamp).log")
        resultsURL = paths.logs.appendingPathComponent("results-\(stamp).txt")
    }

    /// The core's arguments (after the executable's own, e.g. run.py): the Windows launcher's, from Session.Start.
    public static func arguments(options: Options, paths: AppPaths, results: URL) -> [String] {
        var a = options.sourceArgs
        a += ["--display", "--control"]
        if options.testMode { a.append("--test-mode") }
        a += ["--results", results.path]
        a += paths.settingsArgs
        a += options.extraArgs
        a += paths.runArgs
        return a
    }

    public var running: Bool { proc?.running ?? false }

    /// Ends the core at once (SIGKILL), for quitting when it didn't stop.
    public func kill() { proc?.kill() }
    public var pid: Int32? { proc?.pid }

    public func start(header: String) throws {
        try FileManager.default.createDirectory(at: paths.logs, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        log = try? FileHandle(forWritingTo: logURL)
        write(header)
        let (exe, pre) = paths.coreCommand
        arguments = CoreSession.arguments(options: options, paths: paths, results: resultsURL)
        write("[launcher] core: \(exe.path) \(pre.map { ($0 as NSString).lastPathComponent }.joined(separator: " ")) \(arguments.joined(separator: " "))")
        let p = LineProcess(executable: exe, arguments: pre + arguments, cwd: paths.repo, env: paths.env, queue: queue)
        p.onLine = { [weak self] l in self?.line(l) }
        p.onExit = { [weak self] rc in self?.exited(rc) }
        proc = p
        onState(.starting)
        do {
            try p.start()
        } catch {
            write("[launcher] couldn't start the core: \(error.localizedDescription)")
            closeLog()
            throw error
        }
    }

    /// One command for the running core, e.g. "set hz 90". False if it has gone.
    @discardableResult
    public func send(_ command: String) -> Bool {
        let what = command == "quit" ? "[launcher] Stop: asked the video window to close"
            : command.hasPrefix("look-show ") ? "[launcher] settings: look-show (\(command.count) characters)"
            : "[launcher] settings: " + command
        write(what)
        return proc?.send(command) ?? false
    }

    /// True once the core said it doesn't know this command (its first word).
    public func supports(_ command: String) -> Bool { !unsupported.contains(command) }

    /// Stops the session. Once the video is up it's asked to quit, as Esc does (clips being saved are finished
    /// first); it's ended outright if it hasn't gone 20 s later, if it isn't up yet, or on a second Stop.
    public func stop() {
        let again = stopping
        stopping = true
        guard let p = proc, p.running else { return }
        if !again, live, send("quit") {
            DispatchQueue.global().asyncAfter(deadline: .now() + 20) { [weak p] in
                guard let p, p.running else { return }
                p.terminate()
                DispatchQueue.global().asyncAfter(deadline: .now() + 3) { p.kill() }
            }
            return
        }
        p.terminate()
        DispatchQueue.global().asyncAfter(deadline: .now() + 3) { [weak p] in p?.kill() }
    }

    private func line(_ line: String) {
        let ev = CoreLine.parse(line)
        if case .state = ev {
            live = true  // (not for the log)
            onEvent(ev)
            return
        }
        write(line)
        let t = line.trimmingCharacters(in: .whitespaces)
        if !t.isEmpty { lastLine = t }
        onLine(line)
        switch ev {
        case .waitingForChiaki, .waitingForApp: onState(.waiting)
        case let .found(d): onState(.found(d))
        case let .building(d): onState(.building(d))
        case .playing: onState(.playing)
        case .streamEnded: streamEnded = true
        case let .unknownCommand(c): unsupported.insert(c)
        default: break
        }
        onEvent(ev)
    }

    private func exited(_ rc: Int32) {
        exitCode = rc
        write("[launcher] the core exited with code \(rc)")
        closeLog()
        if stopping { onState(.stopped) }
        else if rc == 0 && streamEnded { onState(.ended) }
        else if rc == 0 { onState(.stopped) }
        else { onState(.error(lastLine.isEmpty ? "It stopped unexpectedly (exit code \(rc))." : lastLine)) }
    }

    /// A line in the session's log (after the core exited too: the window capture's last word comes then).
    public func write(_ s: String) {
        logLock.lock()
        defer { logLock.unlock() }
        let d = Data((s + "\n").utf8)
        if let log { try? log.write(contentsOf: d); return }
        guard let h = try? FileHandle(forWritingTo: logURL) else { return }
        _ = try? h.seekToEnd()
        try? h.write(contentsOf: d)
        try? h.close()
    }

    private func closeLog() {
        logLock.lock()
        defer { logLock.unlock() }
        try? log?.close()
        log = nil
    }
}

/// Runs a Python module of the pipeline once and collects its lines (python -m framegen.settings, .looks ...).
public enum PyTool {
    public struct Result: Sendable {
        public var code: Int32
        public var lines: [String]
        /// The last line that looks like JSON (the helpers answer with one).
        public var json: String? { lines.last(where: { $0.hasPrefix("{") }) }
        public var lastText: String? { lines.last(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }) }
    }

    /// The Rust core's subcommand for a Python helper module (smoovplay --help lists them), or nil.
    public static let subcommands: [String: String] = [
        "framegen.autodetect": "autodetect", "framegen.looks": "looks", "framegen.looks_preview": "looks-preview",
        "framegen.prepare": "prepare", "framegen.settings": "settings", "framegen.sources": "sources",
        "framegen.capture": "capture",
    ]

    /// How to run a helper: the Rust core's subcommand when the core is a Rust binary that has it, else Python's
    /// `-m module`.
    public static func command(_ paths: AppPaths, module: String) -> (URL, [String]) {
        if paths.coreKind != .python, let bin = paths.coreBinary, let sub = subcommands[module],
           FileManager.default.isExecutableFile(atPath: bin.path) {
            return (bin, [sub])
        }
        return (paths.python, ["-m", module])
    }

    /// Blocking: call it off the main thread. onLine sees each line as it comes (on a background queue).
    public static func run(_ paths: AppPaths, module: String, _ args: [String], stdin: String? = nil,
                           timeout: TimeInterval = 300, onLine: ((String) -> Void)? = nil) -> Result {
        let q = DispatchQueue(label: "pytool")
        let (exe, pre) = command(paths, module: module)
        let p = LineProcess(executable: exe, arguments: pre + args, cwd: paths.repo, env: paths.env, queue: q)
        var lines: [String] = []
        let done = DispatchSemaphore(value: 0)
        var code: Int32 = -1
        p.onLine = { l in lines.append(l); onLine?(l) }
        p.onExit = { rc in code = rc; done.signal() }
        do { try p.start() } catch { return Result(code: -1, lines: ["couldn't start Python: \(error.localizedDescription)"]) }
        if let s = stdin { p.send(s) }
        p.closeInput()
        if done.wait(timeout: .now() + timeout) == .timedOut {
            p.kill()
            _ = done.wait(timeout: .now() + 5)
            return q.sync { Result(code: -1, lines: lines + ["it took longer than \(Int(timeout)) s"]) }
        }
        return q.sync { Result(code: code, lines: lines) }
    }

    public static func json(_ r: Result) -> JSON? { r.json.flatMap { try? JSON.parse($0) } }
}
