//! `smoovplay` is SmoovPlay's macOS core. It puts run.py and the framegen helpers the launcher calls into one binary,
//! built on the Rust crates (sp-core, sp-tap, sp-media, sp-rife, sp-sched, sp-present, sp-settings). The Python
//! pipeline is the reference. The flags, the `[run]` lines and the --control protocol match the Python's, so the Mac
//! app switches cores by changing only the command it starts.
//!
//!     smoovplay [run.py's flags]          play (see --help)
//!     smoovplay autodetect ...            python -m framegen.autodetect
//!     smoovplay looks ...                 python -m framegen.looks
//!     smoovplay looks-preview             python -m framegen.looks_preview
//!     smoovplay prepare ...               python -m framegen.prepare
//!     smoovplay settings                  python -m framegen.settings
//!     smoovplay sources [--json]          python -m framegen.sources (run.py --list-sources)
//!     smoovplay capture --list --json     python -m framegen.capture
//!     smoovplay replay-compose ...        python -m framegen.replay compose

mod adapt;
mod args;
mod img;
mod run;
mod sources;
mod system;
mod tools;
mod win;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let rest = || argv[1..].to_vec();
    let rc = match argv.first().map(String::as_str) {
        Some("autodetect") => tools::autodetect_main(&rest()),
        Some("looks") => tools::looks_main(&rest()),
        Some("looks-preview") => tools::looks_preview_main(),
        Some("prepare") => tools::prepare_main(&rest()),
        Some("settings") => tools::settings_main(),
        Some("sources") => {
            let r = rest();
            tools::sources_main(r.iter().any(|a| a == "--json"), r.iter().any(|a| a == "--all"))
        }
        Some("capture") => tools::capture_main(&rest()),
        Some("replay-compose") => tools::replay_compose_main(&rest()),
        _ => run::main(&argv),
    };
    use std::io::Write;
    let _ = std::io::stdout().flush();
    std::process::exit(rc);
}
