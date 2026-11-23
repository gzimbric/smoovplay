//! The before & after maker on its own, as `sp-replay-compose --after A --before B --out OUT ...`. The smoovplay binary
//! runs the same code as `smoovplay replay-compose ...`. An optional leading `replay-compose` or `compose` is skipped.

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "replay-compose" || a == "compose") {
        args.remove(0);
    }
    std::process::exit(sp_replay::compose_main(&args));
}
