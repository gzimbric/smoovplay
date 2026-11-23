//! run.py's command line (argparse's rules: `--flag value` or `--flag=value`, unique prefixes of long options,
//! `--no-x` for the on/off ones, errors as "usage: ...\nsmoovplay: error: ..." with exit code 2).

use std::path::PathBuf;

pub const PROG: &str = "smoovplay";

pub const DESCRIPTION: &str = "\
SmoovPlay's pipeline: real-time frame generation (a 30 fps game shown at 60, 90 or 120 fps) and upscaling, for
PS5 Remote Play, an app's frames through the frame tap or a video file. The macOS core (run.py's flags).

Examples
--------
  # Simulate PS5 output from a native-60fps recording, watch it in a window at real speed:
  smoovplay testdata/clip_1080p60.mp4 --display --realtime

  # Same, as fast as possible, scored against the real dropped frames:
  smoovplay testdata/clip_1080p60.mp4 --score

  # PS5 Remote Play through chiaki-ng's frame tap (start chiaki-ng with CHIAKI_FRAMETAP=1, see chiaki_tap/):
  smoovplay --chiaki --display --chiaki-log out/session.csv

  # Any app that sends its frames through the frame tap (docs/frame-tap.md), compressed or decoded; the SmoovPlay
  # app sends a copied window this way (App window):
  smoovplay --source tap --tap smoovplay-window --display

  # Test mode: a 60 fps PS5 game played as if it were 30 fps, each generated frame scored against the real one:
  smoovplay --chiaki --test-mode --display --results out/results.txt

Subcommands (the Python helpers the launcher runs):
  smoovplay autodetect [--apply] [--if-new] [--settings P] [--timing F,H]   python -m framegen.autodetect
  smoovplay looks schema|encode|decode|import|save|delete|add-lut|preview  python -m framegen.looks
  smoovplay looks-preview                                                  python -m framegen.looks_preview
  smoovplay prepare [--model M] [--no-detect]                              python -m framegen.prepare
  smoovplay settings                                                       python -m framegen.settings
  smoovplay sources [--json] [--all]                                       python -m framegen.sources
  smoovplay capture --list --json                                          python -m framegen.capture
  smoovplay replay-compose ...                                             python -m framegen.replay compose

Settings (frame rate, upscaler, ...) are chosen in the window's menu (key M) and saved to
~/Library/Application Support/framegen/settings.json. Command-line flags override them for one run without saving.";

#[derive(Clone, Debug, Default)]
pub struct Args {
    pub video: Option<String>,
    pub source: Option<String>,
    pub list_sources: bool,
    pub json: bool,
    pub window: Option<String>,
    pub window_keys: Option<String>,
    pub capture: Option<String>,
    pub capture_size: Option<String>,
    pub capture_fps: f64,
    pub capture_format: String,
    pub capture_colour: String,
    pub capture_audio: String,
    pub capture_backend: String,
    pub capture_fourcc: String,
    pub list_devices: bool,
    pub chiaki: bool,
    pub chiaki_decoder: String,
    pub chiaki_log: Option<String>,
    pub minimize_chiaki: Option<bool>,
    pub test_mode: bool,
    pub chiaki_tap: Option<String>,
    pub tap: Option<String>,
    pub raw: bool,
    pub realtime: bool,
    pub max_ticks: Option<u64>,
    pub hitch: f64,
    pub noise: f64,
    pub model: Option<String>,
    pub vfg_quality: Option<String>,
    pub scale: Option<f64>,
    pub fp32: bool,
    pub engine: Option<String>,
    pub trt: bool,
    pub no_graph: bool,
    pub dup_threshold: f64,
    pub off: bool,
    pub display: bool,
    pub settings: String,
    pub vsync: Option<bool>,
    pub fullscreen: Option<bool>,
    pub fill: Option<String>,
    pub hdr: Option<String>,
    pub present: Option<String>,
    pub steady_hud: Option<bool>,
    pub scaler: Option<String>,
    pub vsr_quality: Option<i64>,
    pub fsr_sharpness: Option<f64>,
    pub hz: Option<i64>,
    pub display_cv: bool,
    pub out: Option<String>,
    pub score: bool,
    pub save_worst: Option<String>,
    pub control: bool,
    pub results: Option<String>,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Flag,
    /// BooleanOptionalAction: --x / --no-x
    OnOff,
    Str,
    Int,
    Float,
}

struct Opt {
    names: &'static [&'static str],
    kind: Kind,
    metavar: &'static str,
    choices: &'static [&'static str],
    help: &'static str,
}

const fn o(names: &'static [&'static str], kind: Kind, metavar: &'static str, choices: &'static [&'static str],
           help: &'static str) -> Opt {
    Opt { names, kind, metavar, choices, help }
}

use Kind::*;

/// run.py's options, in its order. Hidden ones (argparse.SUPPRESS) have help "".
const OPTS: &[Opt] = &[
    o(&["--source"], Str, "", &["chiaki", "capture", "window", "tap", "file"],
      "where frames come from (default: from the flags below; a video file if one is given)"),
    o(&["--list-sources"], Flag, "", &[], "list the sources and exit; with --json, as JSON for the launcher"),
    o(&["--json"], Flag, "", &[], ""),
    o(&["--window"], Str, "APP", &[],
      "--source window: on a Mac the SmoovPlay app copies the window and sends it through the frame tap (--source tap)"),
    o(&["--window-keys"], Str, "KEYS", &[], "--source window: SmoovPlay's shortcuts (on a Mac, the app's)"),
    o(&["--capture"], Str, "NAME|N|FILE", &[], "a capture card by name or number (from --list-devices), or a video file standing in for one"),
    o(&["--capture-size"], Str, "WxH", &[], "the size to ask the card for (default: Capture card resolution in the settings, 1080p)"),
    o(&["--capture-fps"], Float, "", &[], ""),
    o(&["--capture-format"], Str, "", &[], "auto (NV12, then YUY2, then MJPEG) or one the card lists"),
    o(&["--capture-colour"], Str, "", &["auto", "709-limited", "709-full", "601-limited", "601-full"],
      "how to read the card's colours (auto: limited range unless the card says full; BT.709 from 720p up)"),
    o(&["--capture-audio"], Str, "auto|off|NAME|N", &[], "the card's sound, played on this Mac in step with the picture"),
    o(&["--capture-backend"], Str, "", &["pyav", "opencv"], "(Windows' choice; on a Mac the card is read through AVFoundation)"),
    o(&["--capture-fourcc"], Str, "", &[], "[opencv backend] MJPG, YUY2 or NV12"),
    o(&["--list-devices"], Flag, "", &[], "list capture devices and exit"),
    o(&["--chiaki"], Flag, "", &[], "PS5 Remote Play frames from chiaki-ng's frame tap"),
    o(&["--chiaki-decoder"], Str, "", &["cuda", "sw", "d3d11va", "videotoolbox"],
      "how to decode the Remote Play stream (on a Mac always VideoToolbox, which the others also mean)"),
    o(&["--chiaki-log"], Str, "CSV", &[], "write one row per tick: stream facts and pipeline decisions"),
    o(&["--minimize-chiaki"], OnOff, "", &[], "minimise chiaki-ng's window while ours is up (on a Mac the SmoovPlay app does it)"),
    o(&["--test-mode", "--chiaki-as-30"], Flag, "", &[],
      "test mode: play a 60 fps game as if it were 30 fps by keeping every other frame, and rate each generated frame \
       against the real one that was left out (also in the window's menu: Simulate a 30 fps game). Remote Play, the \
       frame tap, or a video file with --raw"),
    o(&["--chiaki-tap"], Str, "", &[], ""),
    o(&["--tap"], Str, "NAME", &[], "--source tap: the frame tap's shared-memory name (default Local\\smoovplay-tap; docs/frame-tap.md)"),
    o(&["--raw"], Flag, "", &[], "play the file's frames as-is (a real capture recording) instead of simulating PS5 30fps"),
    o(&["--realtime"], Flag, "", &[], "pace file input at 60 Hz like a capture card"),
    o(&["--max-ticks"], Int, "", &[], "stop after this many ticks (any source)"),
    o(&["--hitch"], Float, "", &[], "[sim] probability a game frame is held 3 ticks"),
    o(&["--noise"], Float, "", &[], "[sim] gaussian noise sigma added per tick"),
    o(&["--model"], Str, "", &["4.22.lite", "4.25.lite", "4.25", "4.26", "vfg"],
      "RIFE version (default: the saved setting, 4.22.lite). The other RIFE versions are for testing; vfg is NVIDIA's"),
    o(&["--vfg-quality"], Str, "", &["low", "medium", "high"], "NVIDIA VFG's quality mode (NVIDIA only)"),
    o(&["--scale"], Float, "", &[], "flow resolution scale (default 1.0, 0.5 above 1440p)"),
    o(&["--fp32"], Flag, "", &[], ""),
    o(&["--engine"], Str, "", &["auto", "trt", "torch"], "(NVIDIA's choice; on a Mac RIFE always runs on Metal)"),
    o(&["--trt"], Flag, "", &[], "same as --engine trt"),
    o(&["--no-graph"], Flag, "", &[], "disable CUDA graphs (NVIDIA only)"),
    o(&["--dup-threshold"], Float, "", &[], ""),
    o(&["--off"], Flag, "", &[], "start with interpolation disabled (toggle with 'i')"),
    o(&["--display"], Flag, "", &[], "show output in the Metal window"),
    o(&["--settings"], Str, "JSON", &[], "settings file the menu loads and saves (default ~/Library/Application Support/framegen/settings.json); 'none' = defaults, unsaved"),
    o(&["--vsync"], OnOff, "", &[], "vsync the window (use with variable-refresh displays)"),
    o(&["--fullscreen"], OnOff, "", &[], "start fullscreen"),
    o(&["--fill"], Str, "", &["fit", "zoom", "stretch", "panorama"],
      "on a screen that isn't 16:9: fit (bars), zoom (crops), stretch or panorama; all of them the same 16:9 picture"),
    o(&["--hdr"], Str, "", &["off", "sdr", "hdr10"],
      "an HDR stream from PS5 Remote Play: off (as before), sdr (tone-mapped to SDR) or hdr10 (HDR output through EDR, \
       Experimental). Default: the saved setting"),
    o(&["--present"], Str, "", &["gl", "dxgi"], "(Windows' choice; on a Mac the window always presents through Metal)"),
    o(&["--steady-hud"], OnOff, "", &[], "keep unchanged parts of the picture (HUD, map) as the game drew them. Default: the saved setting"),
    o(&["--scaler"], Str, "", &["fsr", "vsr", "bilinear"], "how the window upscales when it is bigger than the frame (cycle live with 'u'); vsr is NVIDIA's"),
    o(&["--vsr-quality"], Int, "", &["1", "2", "3", "4"], "RTX VSR quality (NVIDIA only)"),
    o(&["--fsr-sharpness"], Float, "", &["0.0", "0.25", "0.5", "1.0", "2.0"],
      "FSR 1 sharpening, in stops below maximum: 0 = sharpest, 1 = half, 2 = quarter"),
    o(&["--hz"], Int, "", &["60", "90", "120"],
      "output rate for a 30 fps game: 120 = 3 generated frames per real one (half-res flow by default), 90 = 2. Needs \
       --display; falls back to 60 Hz output whenever it can't keep up"),
    o(&["--display-cv"], Flag, "", &[], ""),
    o(&["--out"], Str, "", &[], "write output video (not in the Mac core yet)"),
    o(&["--score"], Flag, "", &[], "[sim] PSNR of interpolated frames vs real dropped frames"),
    o(&["--save-worst"], Str, "JPG", &[], "[with --score] save prev | RIFE | real | next for the worst frame"),
    o(&["--control"], Flag, "", &[],
      "take settings commands on stdin and print [settings-state] lines (the launcher's Settings page; see framegen/remote.py)"),
    o(&["--results"], Str, "TXT", &[],
      "when test mode (a simulated 30 fps game) or --score was used, write a plain-language report of performance and \
       quality here: every 10 s while playing, and at the end"),
];

pub fn usage() -> String {
    format!("usage: {PROG} [-h] [options] [video]")
}

/// argparse's error: usage, the message, exit code 2.
pub fn error(msg: &str) -> ! {
    eprintln!("{}\n{PROG}: error: {msg}", usage());
    std::process::exit(2);
}

pub fn help() -> String {
    let mut s = format!("{}\n\n{DESCRIPTION}\n\npositional arguments:\n  video                 input video file\n\noptions:\n  -h, --help            show this help message and exit\n", usage());
    for opt in OPTS {
        if opt.help.is_empty() {
            continue;
        }
        let dest = opt.names[0].trim_start_matches("--").replace('-', "_").to_uppercase();
        let meta = if !opt.choices.is_empty() {
            format!("{{{}}}", opt.choices.join(","))
        } else if !opt.metavar.is_empty() {
            opt.metavar.to_string()
        } else {
            dest
        };
        let head = match opt.kind {
            Flag => opt.names.join(", "),
            OnOff => format!("{0}, --no-{1}", opt.names[0], &opt.names[0][2..]),
            _ => opt.names.iter().map(|n| format!("{n} {meta}")).collect::<Vec<_>>().join(", "),
        };
        if head.len() <= 20 {
            s += &format!("  {head:<22}{}\n", opt.help);
        } else {
            s += &format!("  {head}\n                        {}\n", opt.help);
        }
    }
    s
}

fn py_repr(s: &str) -> String {
    if s.contains('\'') && !s.contains('"') { format!("\"{s}\"") } else { format!("'{}'", s.replace('\'', "\\'")) }
}

/// The option a word names (exactly, or as a unique prefix), and whether it was the --no- form.
fn find(word: &str) -> Result<(&'static Opt, bool), String> {
    for opt in OPTS {
        if opt.names.contains(&word) {
            return Ok((opt, false));
        }
        if opt.kind == OnOff && word == format!("--no-{}", &opt.names[0][2..]) {
            return Ok((opt, true));
        }
    }
    let mut hits: Vec<(&'static Opt, bool, String)> = Vec::new();
    for opt in OPTS {
        for n in opt.names {
            if n.starts_with(word) {
                hits.push((opt, false, n.to_string()));
            }
        }
        if opt.kind == OnOff {
            let no = format!("--no-{}", &opt.names[0][2..]);
            if no.starts_with(word) {
                hits.push((opt, true, no));
            }
        }
    }
    match hits.len() {
        1 => Ok((hits[0].0, hits[0].1)),
        0 => Err(format!("unrecognized arguments: {word}")),
        _ => Err(format!(
            "ambiguous option: {word} could match {}",
            hits.iter().map(|h| h.2.as_str()).collect::<Vec<_>>().join(", ")
        )),
    }
}

pub fn parse(argv: &[String]) -> Args {
    let mut a = Args {
        capture_fps: 60.0,
        capture_format: "auto".into(),
        capture_colour: "auto".into(),
        capture_audio: "auto".into(),
        capture_backend: "pyav".into(),
        capture_fourcc: "MJPG".into(),
        chiaki_decoder: "cuda".into(),
        dup_threshold: 2.0,
        settings: sp_settings::settings::default_path().to_string_lossy().into_owned(),
        ..Default::default()
    };
    let mut extra: Vec<String> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let word = &argv[i];
        i += 1;
        if word == "-h" || word == "--help" {
            print!("{}", help());
            std::process::exit(0);
        }
        if !word.starts_with("--") || word == "--" {
            if a.video.is_none() && !word.starts_with('-') {
                a.video = Some(word.clone());
            } else {
                extra.push(word.clone());
            }
            continue;
        }
        let (name, inline) = match word.split_once('=') {
            Some((n, v)) => (n.to_string(), Some(v.to_string())),
            None => (word.clone(), None),
        };
        let (opt, negated) = find(&name).unwrap_or_else(|e| error(&e));
        let shown = opt.names.join("/");
        let value = match opt.kind {
            Flag | OnOff => {
                if inline.is_some() {
                    error(&format!("argument {shown}: ignored explicit argument {}", py_repr(inline.as_deref().unwrap())));
                }
                None
            }
            _ => Some(match inline {
                Some(v) => v,
                None => {
                    if i >= argv.len() || (argv[i].starts_with("--") && argv[i].len() > 2) {
                        error(&format!("argument {shown}: expected one argument"));
                    }
                    i += 1;
                    argv[i - 1].clone()
                }
            }),
        };
        let int = |v: &str| -> i64 {
            v.trim().parse::<i64>().unwrap_or_else(|_| error(&format!("argument {shown}: invalid int value: {}", py_repr(v))))
        };
        let float = |v: &str| -> f64 {
            v.trim().parse::<f64>().unwrap_or_else(|_| error(&format!("argument {shown}: invalid float value: {}", py_repr(v))))
        };
        if let Some(v) = &value
            && !opt.choices.is_empty()
        {
            let ok = match opt.kind {
                Int => opt.choices.iter().any(|c| c.parse::<i64>().ok() == Some(int(v))),
                Float => opt.choices.iter().any(|c| c.parse::<f64>().ok() == Some(float(v))),
                _ => opt.choices.contains(&v.as_str()),
            };
            if !ok {
                let shown_v = if opt.kind == Str { py_repr(v) } else { v.clone() };
                let list = if opt.kind == Str {
                    opt.choices.iter().map(|c| py_repr(c)).collect::<Vec<_>>().join(", ")
                } else {
                    opt.choices.join(", ")
                };
                error(&format!("argument {shown}: invalid choice: {shown_v} (choose from {list})"));
            }
        }
        let s = || value.clone().unwrap();
        match opt.names[0] {
            "--source" => a.source = Some(s()),
            "--list-sources" => a.list_sources = true,
            "--json" => a.json = true,
            "--window" => a.window = Some(s()),
            "--window-keys" => a.window_keys = Some(s()),
            "--capture" => a.capture = Some(s()),
            "--capture-size" => a.capture_size = Some(s()),
            "--capture-fps" => a.capture_fps = float(&s()),
            "--capture-format" => a.capture_format = s(),
            "--capture-colour" => a.capture_colour = s(),
            "--capture-audio" => a.capture_audio = s(),
            "--capture-backend" => a.capture_backend = s(),
            "--capture-fourcc" => a.capture_fourcc = s(),
            "--list-devices" => a.list_devices = true,
            "--chiaki" => a.chiaki = true,
            "--chiaki-decoder" => a.chiaki_decoder = s(),
            "--chiaki-log" => a.chiaki_log = Some(s()),
            "--minimize-chiaki" => a.minimize_chiaki = Some(!negated),
            "--test-mode" => a.test_mode = true,
            "--chiaki-tap" => a.chiaki_tap = Some(s()),
            "--tap" => a.tap = Some(s()),
            "--raw" => a.raw = true,
            "--realtime" => a.realtime = true,
            "--max-ticks" => a.max_ticks = Some(int(&s()).max(0) as u64),
            "--hitch" => a.hitch = float(&s()),
            "--noise" => a.noise = float(&s()),
            "--model" => a.model = Some(s()),
            "--vfg-quality" => a.vfg_quality = Some(s()),
            "--scale" => a.scale = Some(float(&s())),
            "--fp32" => a.fp32 = true,
            "--engine" => a.engine = Some(s()),
            "--trt" => a.trt = true,
            "--no-graph" => a.no_graph = true,
            "--dup-threshold" => a.dup_threshold = float(&s()),
            "--off" => a.off = true,
            "--display" => a.display = true,
            "--settings" => a.settings = s(),
            "--vsync" => a.vsync = Some(!negated),
            "--fullscreen" => a.fullscreen = Some(!negated),
            "--fill" => a.fill = Some(s()),
            "--hdr" => a.hdr = Some(s()),
            "--present" => a.present = Some(s()),
            "--steady-hud" => a.steady_hud = Some(!negated),
            "--scaler" => a.scaler = Some(s()),
            "--vsr-quality" => a.vsr_quality = Some(int(&s())),
            "--fsr-sharpness" => a.fsr_sharpness = Some(float(&s())),
            "--hz" => a.hz = Some(int(&s())),
            "--display-cv" => a.display_cv = true,
            "--out" => a.out = Some(s()),
            "--score" => a.score = true,
            "--save-worst" => a.save_worst = Some(s()),
            "--control" => a.control = true,
            "--results" => a.results = Some(s()),
            other => unreachable!("{other}"),
        }
    }
    if !extra.is_empty() {
        error(&format!("unrecognized arguments: {}", extra.join(" ")));
    }
    a
}

impl Args {
    /// --settings none: the defaults, never saved.
    pub fn settings_path(&self) -> Option<PathBuf> {
        if self.settings.eq_ignore_ascii_case("none") { None } else { Some(PathBuf::from(&self.settings)) }
    }
}
