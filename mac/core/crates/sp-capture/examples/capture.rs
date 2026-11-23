//! `python -m framegen.capture` for the Mac: the capture devices macOS has, and what one offers.
//!
//! ```text
//! cargo run -p sp-capture --example capture -- --list [--json] [--raw]
//! cargo run -p sp-capture --example capture -- --formats NAME [--json]
//! ```

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(sp_capture::cli::main(&args));
}
