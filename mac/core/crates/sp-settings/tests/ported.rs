//! The Python tests' assertions for this logic, ported: tests/test_settings.py (not the in-video menu's drawing),
//! test_looks.py (not the renderers), test_fill.py (the arithmetic) and test_autodetect.py. Each test names the
//! Python test it comes from.

mod common;

use std::sync::{Arc, Mutex};

use common::*;
use sp_settings::autodetect::{self, Display, Machine, Options, Timing, VfgTiming, auto_scale, recommend};
use sp_settings::fill;
use sp_settings::json::{self, Map, Value};
use sp_settings::looks::{
    self, Library, LookState, builtin, builtin_look, decode, encode, encode_look, node, normalize,
};
use sp_settings::remote::{FnRules, OpenRules, Remote, Request};
use sp_settings::settings::{self, LookChoices, Settings, spec, specs};

fn v<T: Into<Value>>(x: T) -> Value {
    x.into()
}

fn settings_at(path: &std::path::Path) -> Settings {
    Settings::load(path, LookChoices::builtin())
}

fn saved(path: &std::path::Path) -> Value {
    json::loads(&std::fs::read_to_string(path).unwrap()).unwrap()
}

// ================================================================================================ test_settings.py
#[test]
fn test_defaults_are_valid_choices() {
    let looks = LookChoices::builtin();
    for s in specs() {
        assert!(s.is_choice(&s.default_value(), &looks), "{}", s.key);
        assert!(!s.help.is_empty(), "{}", s.key);
    }
    let mut keys: Vec<&str> = specs().iter().map(|s| s.key).collect();
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), specs().len()); // unique keys
}

#[test]
fn test_load_save_roundtrip_and_bad_values_ignored() {
    let path = tmp_dir("roundtrip").join("settings.json");
    std::fs::write(&path, r#"{"hz": 90, "scaler": "nope", "unknown": 1, "fsr_sharpness": 0.25}"#).unwrap();
    let mut st = settings_at(&path);
    assert_eq!((st.get_i64("hz"), st.get_str("scaler"), st.get_f64("fsr_sharpness")), (90, "fsr", 0.25));
    st.set("scaler", "bilinear").unwrap();
    assert_eq!(saved(&path).get("scaler"), Some(&v("bilinear")));
    assert_eq!(settings_at(&path).get_str("scaler"), "bilinear");
}

#[test]
fn test_missing_or_corrupt_file_gives_defaults() {
    let d = tmp_dir("corrupt");
    assert_eq!(settings_at(&d.join("absent.json")).get("hz"), Some(&spec("hz").unwrap().default_value()));
    std::fs::write(d.join("bad.json"), "{not json").unwrap();
    assert_eq!(settings_at(&d.join("bad.json")).get("interp"), Some(&v(true)));
}

#[test]
fn test_override_is_not_saved() {
    let path = tmp_dir("override").join("s.json");
    let mut st = settings_at(&path);
    st.override_value("hz", 120).unwrap();
    st.set("overlay", false).unwrap(); // saving something else must not persist the override
    let f = saved(&path);
    assert_eq!(st.get_i64("hz"), 120);
    assert_eq!(f.get("hz"), Some(&spec("hz").unwrap().default_value()));
    assert_eq!(f.get("overlay"), Some(&v(false)));
}

#[test]
fn test_test_mode_is_this_session_only() {
    let path = tmp_dir("testmode").join("s.json");
    std::fs::write(&path, r#"{"simulate_30": true}"#).unwrap(); // e.g. hand-edited: still starts off
    let mut st = settings_at(&path);
    assert!(!st.get_bool("simulate_30"));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s2 = Arc::clone(&seen);
    st.watch(move |_, k| s2.lock().unwrap().push(k.to_string()));
    st.set("simulate_30", true).unwrap(); // from the menu: applies (and is heard) but isn't written
    assert!(st.get_bool("simulate_30") && *seen.lock().unwrap() == ["simulate_30"]);
    st.set("overlay", false).unwrap();
    let f = saved(&path);
    assert_eq!((f.get("simulate_30"), f.get("overlay")), (Some(&v(false)), Some(&v(false))));
    assert!(!settings_at(&path).get_bool("simulate_30"));
}

#[test]
fn test_invalid_value_rejected() {
    let e = Settings::defaults().set("hz", 75).unwrap_err();
    assert_eq!(e.to_string(), "hz: 75 is not one of (60, 90, 120)");
}

#[test]
fn test_step_wraps_and_skips_disallowed() {
    let mut st = Settings::defaults();
    assert_eq!(st.step("hz", 1, None).unwrap(), v(90));
    assert_eq!(st.step("hz", 1, None).unwrap(), v(120));
    assert_eq!(st.step("hz", 1, None).unwrap(), v(60)); // wraps
    let no_vsr = |_: &str, x: &Value| x != &v("vsr");
    assert_eq!(st.step("scaler", 1, Some(&no_vsr)).unwrap(), v("bilinear"));
    assert_eq!(st.step("scaler", 1, Some(&no_vsr)).unwrap(), v("fsr"));
}

#[test]
fn test_watchers_fire_on_change_only() {
    let mut st = Settings::defaults();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s2 = Arc::clone(&seen);
    let stop = st.watch(move |_, k| s2.lock().unwrap().push(k.to_string()));
    st.set("hz", 90).unwrap();
    st.set("hz", 90).unwrap(); // unchanged: no call
    st.override_value("interp", false).unwrap();
    st.unwatch(stop);
    st.set("hz", 60).unwrap();
    assert_eq!(*seen.lock().unwrap(), ["hz", "interp"]);
}

#[test]
fn test_every_setting_applies_where_the_menu_shows_it() {
    // (test_every_setting_is_on_a_page and test_rows_that_dont_apply_are_hidden: the menu shows the settings that
    // apply to the values; here, which those are)
    let applies = |vals: &[(&str, Value)]| -> Vec<&'static str> {
        let m: Map = vals.iter().cloned().collect();
        let mut all = Settings::defaults().values();
        for (k, x) in m.iter() {
            all.insert(k, x.clone());
        }
        specs().iter().filter(|s| s.applies_to(&all)).map(|s| s.key).collect()
    };
    let d = applies(&[]);
    assert!(!d.contains(&"vfg_quality") && d.contains(&"fsr_sharpness") && !d.contains(&"vsr_quality"));
    let vsr = applies(&[("scaler", v("vsr"))]);
    assert!(!vsr.contains(&"fsr_sharpness") && vsr.contains(&"vsr_quality"));
    let vfg = applies(&[("model", v("vfg"))]);
    assert!(vfg.contains(&"vfg_quality") && !vfg.contains(&"engine") && !vfg.contains(&"motion"));
    assert!(vfg.contains(&"steady_hud")); // Steady HUD works with NVIDIA VFG too
}

#[test]
fn test_specs_json_is_what_the_launcher_needs() {
    let data = settings::specs_json(&settings::default_path(), &LookChoices::builtin());
    let data = json::loads(&json::dumps(&data)).unwrap(); // JSON-safe
    assert!(s(&data, "path").ends_with("settings.json"));
    let rows = list(&data, "specs");
    let keys: Vec<&str> = rows.iter().map(|r| s(r, "key")).collect();
    assert_eq!(keys, specs().iter().map(|s| s.key).collect::<Vec<_>>());
    for r in rows {
        assert!(!s(r, "group").is_empty() && !s(r, "label").is_empty() && !s(r, "help").is_empty(), "{}", s(r, "key"));
        let values: Vec<&Value> = list(r, "choices").iter().map(|c| c.get("value").unwrap()).collect();
        assert!(values.iter().any(|x| x.py_eq(r.get("default").unwrap())), "{}", s(r, "key"));
    }
    let row = |k: &str| rows.iter().find(|r| s(r, "key") == k).unwrap();
    assert_eq!(row("simulate_30").get("saved"), Some(&v(false)));
    let sharp = list(row("fsr_sharpness"), "choices");
    assert_eq!((s(&sharp[0], "name"), s(&sharp[1], "name")), ("Max", "High"));
    let model = row("model");
    assert_eq!(model.get("live"), Some(&v(false)));
    assert!(!s(model, "later").is_empty() && s(&list(model, "choices")[0], "name").starts_with("RIFE"));
    let scaler = row("scaler");
    assert!(!s(scaler, "later").is_empty() && list(scaler, "choices").iter().all(|c| !s(c, "hint").is_empty()));
    assert_eq!(s(sharp.iter().find(|c| s(c, "name") == "High").unwrap(), "hint"), ""); // not every choice needs one
}

#[test]
fn test_hints_are_for_real_choices_and_short() {
    let looks = LookChoices::builtin();
    let data = settings::specs_json(&settings::default_path(), &looks);
    for (sp, r) in specs().iter().zip(list(&data, "specs")) {
        let hints: Vec<&str> = list(r, "choices").iter().map(|c| s(c, "hint")).filter(|h| !h.is_empty()).collect();
        for h in hints {
            assert!(h.chars().count() <= 140, "{}: {h}", sp.key); // two or three lines in the dropdown
        }
    }
}

#[test]
fn test_impact_lines_are_plain_short_and_never_about_vfg() {
    for s in specs() {
        for (k, text) in s.impact {
            assert!(["picture", "speed"].contains(k), "{}", s.key);
            assert!(text.is_ascii(), "{} {k}", s.key); // the in-video menu's font draws ASCII only
        }
    }
    for key in ["model", "vfg_quality"] {
        assert!(spec(key).unwrap().impact.is_empty(), "{key}"); // NVIDIA's licence: no published figures
    }
    let data = settings::specs_json(&settings::default_path(), &LookChoices::builtin());
    assert!(!s(&data, "impact_pc").is_empty());
    let steady = list(&data, "specs").iter().find(|r| s(r, "key") == "steady_hud").unwrap();
    assert!(!s(steady.get("impact").unwrap(), "speed").is_empty());
}

fn remote_with(
    dir: &std::path::Path,
    commands: &[&str],
    rules: impl sp_settings::remote::Rules + 'static,
) -> (Settings, Remote, Buf, Arc<Mutex<Vec<String>>>) {
    let mut st = settings_at(&dir.join("settings.json"));
    let out = Buf::default();
    let mut r = Remote::new(&mut st, rules, out.clone());
    let log = Arc::new(Mutex::new(Vec::new()));
    let l2 = Arc::clone(&log);
    r.set_log(move |m| l2.lock().unwrap().push(m.to_string()));
    let input = commands.iter().map(|c| format!("{c}\n")).collect::<String>();
    r.read_from(std::io::Cursor::new(input.into_bytes())).join().unwrap(); // (the reader thread, to the end)
    (st, r, out, log)
}

fn states(out: &Buf) -> Vec<Value> {
    out.text()
        .lines()
        .filter_map(|l| l.strip_prefix(sp_settings::remote::PREFIX))
        .map(|j| json::loads(j).unwrap())
        .collect()
}

#[test]
fn test_remote_set_applies_saves_and_reports() {
    let d = tmp_dir("remote1");
    let (mut st, mut r, out, _) =
        remote_with(&d, &["set hz 90", "set fsr_sharpness 1", "set overlay false"], OpenRules);
    assert_eq!(r.poll(&mut st), None);
    assert_eq!((st.get_i64("hz"), st.get_f64("fsr_sharpness"), st.get_bool("overlay")), (90, 1.0, false));
    assert_eq!(st.get("fsr_sharpness"), Some(&v(1.0))); // the spec's own type, so the menu still finds it
    let f = saved(&d.join("settings.json"));
    assert_eq!((f.get("hz"), f.get("overlay")), (Some(&v(90)), Some(&v(false))));
    let all = states(&out);
    let last = all.last().unwrap();
    assert_eq!(last.get("values").unwrap().get("hz"), Some(&v(90)));
    assert_eq!(list(last, "restart").len(), 0);
    assert_eq!(all.len(), 3); // one per change
}

#[test]
fn test_remote_refuses_bad_and_unavailable_values() {
    let d = tmp_dir("remote2");
    let rules = FnRules {
        allowed: |_: &Settings, k: &str, x: &Value| !(k == "scaler" && x == &v("vsr")),
        disabled: |_: &Settings, k: &str| (k == "vsr_quality").then(|| "no RTX VSR here".to_string()),
        restart_reasons: |_: &Settings| Vec::new(),
    };
    let cmds = ["set hz 75", "set nope 1", "set scaler \"vsr\"", "set vsr_quality 4", "set hz {broken", "bogus"];
    let (mut st, mut r, out, log) = remote_with(&d, &cmds, rules);
    r.poll(&mut st);
    assert_eq!((st.get_i64("hz"), st.get_str("scaler"), st.get_i64("vsr_quality")), (60, "fsr", 2));
    assert!(!d.join("settings.json").exists());
    let all = states(&out);
    assert_eq!(all.len(), 5); // each refused set re-sends the real values
    let last = all.last().unwrap();
    assert!(!list(last.get("choices").unwrap(), "scaler").contains(&v("vsr")));
    assert_eq!(json::dumps(last.get("disabled").unwrap()), r#"{"vsr_quality": "no RTX VSR here"}"#);
    let log = log.lock().unwrap().join("\n");
    assert!(log.contains("can't set vsr_quality: no RTX VSR here") && log.contains("unknown command: bogus"));
}

#[test]
fn test_remote_restart_and_state() {
    let d = tmp_dir("remote3");
    let rules = FnRules {
        allowed: |_: &Settings, _: &str, _: &Value| true,
        disabled: |_: &Settings, _: &str| None,
        restart_reasons: |_: &Settings| vec!["AI engine".to_string()],
    };
    let (mut st, mut r, out, _) = remote_with(&d, &["set engine \"torch\"", "restart", "state"], rules);
    assert_eq!(r.poll(&mut st), Some(Request::Restart));
    assert_eq!(st.get_str("engine"), "torch");
    let restarts: Vec<String> = states(&out).iter().map(|x| json::dumps(x.get("restart").unwrap())).collect();
    assert_eq!(restarts, [r#"["AI engine"]"#, r#"["AI engine"]"#]);
}

#[test]
fn test_the_launchers_stop_quits_as_esc_does() {
    let d = tmp_dir("remote4");
    let (mut st, mut r, _, _) = remote_with(&d, &["restart", "quit", "restart"], OpenRules);
    assert_eq!(r.poll(&mut st), Some(Request::Quit)); // a Stop wins over a restart that came with it
}

#[test]
fn test_specs_json_says_what_applies_when() {
    let data = settings::specs_json(&settings::default_path(), &LookChoices::builtin());
    let rows = list(&data, "specs");
    let row = |k: &str| rows.iter().find(|r| s(r, "key") == k).unwrap();
    assert_eq!(json::dumps(row("vfg_quality").get("applies").unwrap()), r#"{"key": "model", "values": ["vfg"]}"#);
    assert_eq!(row("fullscreen").get("applies"), Some(&Value::Null));
    let quick: Vec<&str> = rows.iter().filter(|r| r.get("quick") == Some(&v(true))).map(|r| s(r, "key")).collect();
    assert_eq!(quick, ["interp", "hz", "model", "scaler"]);
    assert!(rows[..3].iter().all(|r| s(r, "group") == "Smoothness"));
}

#[test]
fn test_capture_card_resolution_says_what_it_costs() {
    let sp = spec("capture_size").unwrap();
    let looks = LookChoices::builtin();
    assert_eq!(sp.default_value(), v("1080p"));
    assert_eq!(settings::capture_size("1080p"), Some((1920, 1080)));
    assert!(!sp.live && sp.experimental);
    assert_eq!(sp.name(&v("2160p"), &looks), "4K");
    for c in sp.choices(&looks) {
        assert!(settings::capture_size(c.as_str().unwrap()).is_some());
    }
    assert!(sp.hint(&v("1440p")).contains("twice") && sp.hint(&v("2160p")).contains("four times"));
    assert!(sp.hint(&v("1080p")).contains("Recommended"));
    let speed = sp.impact.iter().find(|(k, _)| *k == "speed").unwrap().1;
    for t in [sp.hint(&v("1440p")), sp.hint(&v("2160p")), sp.help, speed] {
        assert!(t.contains("sluggish"), "{t}");
    }
    assert!(sp.help.contains("gains nothing") && sp.help.contains("Capture cards only"));
}

#[test]
fn test_settings_for_one_source_say_why_not_with_the_others() {
    let (size, mini) = (spec("capture_size").unwrap(), spec("minimize_chiaki").unwrap());
    assert!(size.unused_with("capture").is_none() && mini.unused_with("chiaki").is_none());
    assert!(size.unused_with("chiaki").unwrap().contains("1080p at most"));
    assert!(
        size.unused_with("window").unwrap().contains("1080p") && size.unused_with("tap").unwrap().contains("own size")
    );
    assert_eq!(size.unused_with("file").unwrap(), "Only used with a capture card.");
    assert_eq!(mini.unused_with("tap"), mini.unused_with("capture"));
    assert_eq!(mini.unused_with("tap").unwrap(), "Only used with PS5 Remote Play.");
    assert!(spec("fill").unwrap().unused_with("tap").is_none()); // most settings: every source
    let data = settings::specs_json(&settings::default_path(), &LookChoices::builtin());
    let row = |k: &str| list(&data, "specs").iter().find(|r| s(r, "key") == k).unwrap().clone();
    assert_eq!(json::dumps(row("capture_size").get("sources").unwrap()), r#"["capture"]"#);
    assert!(!s(row("capture_size").get("elsewhere").unwrap(), "").is_empty());
    assert_eq!((row("fill").get("sources"), row("fill").get("elsewhere")), (Some(&Value::Null), Some(&Value::Null)));
}

// ================================================================================================ test_looks.py
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn random_look(rng: &mut Rng, lut: &str) -> Value {
    let mut nodes = Vec::new();
    for _ in 0..(1 + rng.below(6)) {
        let t = &looks::types()[rng.below(looks::types().len())];
        if t.key == "lut" && lut.is_empty() {
            continue;
        }
        let mut params = Vec::new();
        for f in t.params {
            params.push((f.key, if f.text { v(lut) } else { v(f.lo + rng.unit() * (f.hi - f.lo)) }));
        }
        let on = rng.unit() > 0.1;
        nodes.push(node(t.key, on, rng.unit(), &params));
    }
    let look = json::obj([("name", "Random".into()), ("nodes", Value::List(nodes))]);
    normalize(&look).unwrap().to_value()
}

#[test]
fn test_every_builtin_look_survives_its_code_and_codes_are_short() {
    for (key, look) in builtin() {
        let code = encode_look(look);
        assert!(code.starts_with("SPL1-") && code.len() <= 64, "{key} {code}");
        assert_eq!(&decode(&code).unwrap(), look, "{key}");
    }
}

#[test]
fn test_random_looks_survive_their_codes() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..200 {
        let look = random_look(&mut rng, "My grade.cube");
        assert_eq!(decode(&encode(&look).unwrap()).unwrap().to_value(), look);
    }
}

#[test]
fn test_codes_survive_chat_apps_and_say_what_is_wrong() {
    let film = builtin_look("film").unwrap();
    let code = encode_look(film);
    assert_eq!(&decode(&format!("  {}\n{} \n", &code[..20], &code[20..])).unwrap(), film); // wrapped and padded
    let lower = code.to_lowercase().replacen("spl1-", "SPL1-", 1);
    assert_eq!(&decode(&format!("{}{}", &lower[..5], &code[5..])).unwrap(), film);
    let c12 = code.as_bytes()[12] as char;
    let swapped = format!("{}{}{}", &code[..12], if c12 != 'A' { 'A' } else { 'B' }, &code[13..]);
    for (bad, says) in [
        ("hello".to_string(), "start with SPL1-"),
        ("SPL2-AAAA".to_string(), "newer version"),
        (code[..code.len() - 3].to_string(), "incomplete or has a typo"),
        (swapped, "incomplete or has a typo"),
    ] {
        let e = decode(&bad).unwrap_err();
        assert!(e.0.contains(says), "{bad}: {}", e.0);
    }
}

fn make_code(body: &[u8]) -> String {
    let mut data = vec![0u8];
    data.extend_from_slice(body);
    let crc = (sp_settings::zlib::crc32(&data) & 0xFFFF) as u16;
    data.extend(crc.to_be_bytes());
    // URL-safe base64 without padding
    const B: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::from("SPL1-");
    for ch in data.chunks(3) {
        let b = [ch[0], *ch.get(1).unwrap_or(&0), *ch.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for i in 0..=ch.len() {
            out.push(B[((n >> (18 - 6 * i)) & 63) as usize] as char);
        }
    }
    out
}

#[test]
fn test_codes_from_a_newer_version() {
    // a vignette with a setting this version doesn't have (bit 6, past roundness): skipped
    let got = decode(&make_code(&[b"\x01X\x01".as_slice(), &[13 << 1], &[1 << 6], b"\x04"].concat())).unwrap();
    let want = normalize(&json::obj([("nodes", Value::List(vec![node("vignette", true, 1.0, &[])]))])).unwrap();
    assert_eq!(got.nodes, want.nodes);
    let e = decode(&make_code(&[b"\x01X\x01".as_slice(), &[99 << 1], b"\x00"].concat())).unwrap_err();
    assert!(e.0.contains("doesn't have")); // an effect it doesn't have
}

#[test]
fn test_normalize_cleans_up() {
    let mut nodes = vec![
        json::loads(r#"{"type": "color", "contrast": 7, "saturation": "x", "exposure": 0.123}"#).unwrap(),
        json::loads(r#"{"type": "warp-drive"}"#).unwrap(),
        v("junk"),
        json::loads(r#"{"type": "lut", "file": "..\\..\\secret\\grade.cube"}"#).unwrap(),
        json::loads(r#"{"type": "lut", "file": "notalut.png"}"#).unwrap(),
    ];
    nodes.extend(std::iter::repeat_n(json::loads(r#"{"type": "grain"}"#).unwrap(), 30));
    let look = normalize(&json::obj([("name", "  A   look ".into()), ("nodes", Value::List(nodes))])).unwrap();
    let color = &look.nodes[0];
    assert_eq!(look.name, "A look");
    assert_eq!((color.get("contrast"), color.get("saturation")), (Some(1.0), Some(1.0)));
    assert!((color.get("exposure").unwrap() - 0.1).abs() < 1e-9 && color.on && color.mix() == 1.0);
    assert_eq!((look.nodes[1].file(), look.nodes[2].file()), ("grade.cube", "")); // no folders, .cube only
    assert_eq!(look.nodes.len(), looks::MAX_NODES);
    assert!(normalize(&v("not a look")).is_err());
}

#[test]
fn test_types_are_numbered_once_and_their_settings_are_sane() {
    let mut codes: Vec<u32> = looks::types().iter().map(|t| t.code).collect();
    codes.sort();
    codes.dedup();
    assert_eq!(codes.len(), looks::types().len());
    for t in looks::types() {
        assert!(["point", "glow", "blur", "clarity", "ca", "sharpen"].contains(&t.kind) && !t.help.is_empty());
        assert!(t.field_count() < 32); // the share code's mask
        let mut keys: Vec<&str> = t.fields().map(|f| f.key).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), t.field_count(), "{}", t.key);
        assert!(!keys.contains(&"type") && !keys.contains(&"on"), "{}", t.key);
        for f in t.fields() {
            if let looks::PDefault::Num(d) = f.default {
                assert!(f.text || (f.lo <= d && d <= f.hi), "{} {}", t.key, f.key);
            }
        }
    }
}

fn look_json(name: &str, nodes: Vec<Value>) -> Value {
    json::obj([("name", name.into()), ("nodes", Value::List(nodes))])
}

#[test]
fn test_saving_loading_and_deleting_looks() {
    let d = tmp_dir("saving");
    let mut lib = Library::new(&d);
    let id =
        lib.save(&look_json("Moody blues", vec![node("color", true, 1.0, &[("temperature", v(-0.4))])]), None).unwrap();
    assert!(id == "my:moody-blues" && d.join("moody-blues.json").exists());
    assert_eq!(lib.save(&look_json("Moody blues", vec![]), None).unwrap(), "my:moody-blues-2");
    let ids = lib.ids();
    let n = builtin().len();
    assert_eq!(ids[..n], builtin().iter().map(|(k, _)| k.to_string()).collect::<Vec<_>>()[..]);
    assert_eq!(ids[n..], ["my:moody-blues", "my:moody-blues-2"]);
    assert_eq!(lib.get(&id).unwrap().nodes[0].get("temperature"), Some(-0.4));
    assert_eq!(lib.choices().name(&id), Some("Moody blues"));
    assert_eq!(lib.save(&look_json("Renamed", vec![]), Some(&id)).unwrap(), id); // the same file
    assert_eq!(lib.get(&id).unwrap().name, "Renamed");
    std::fs::write(d.join("broken.json"), "{nope").unwrap();
    lib.reload(); // a broken file is skipped
    assert!(!lib.ids().contains(&"my:broken".to_string()));
    assert!(lib.delete(&id).unwrap() && !lib.ids().contains(&id) && !lib.delete(&id).unwrap());
    assert!(lib.path("my:..\\..\\x").is_none() && !lib.delete("cinematic").unwrap());
}

#[test]
fn test_the_look_settings() {
    let sp = spec("look").unwrap();
    assert_eq!(sp.default_value(), v(looks::DEFAULT_LOOK));
    assert_eq!(
        json::dumps(&sp.applies().map(|(k, vs)| json::obj([(k, Value::List(vs))])).unwrap()),
        r#"{"look_on": [true]}"#
    );
    assert_eq!(spec("look_on").unwrap().default_value(), v(false));
    assert_eq!(spec("look_clips").unwrap().default_value(), v(false));
    assert_eq!(spec("look_clips").unwrap().applies().unwrap().0, "replay");
    assert_eq!(spec("look_clips").unwrap().group, "Instant replay");
    let mut lib = Library::new(tmp_dir("looksettings"));
    let my = lib.save(&look_json("Mine", vec![node("grain", true, 1.0, &[])]), None).unwrap();
    let vals: Map = [("look", v(my.as_str())), ("look_on", v(true))].into_iter().collect();
    let st = Settings::new(&vals, None, lib.choices());
    assert_eq!(st.get_str("look"), my); // a saved look of the user's is a valid choice
}

#[test]
fn test_k_and_j_and_the_launchers_preview() {
    let mut lib = Library::new(tmp_dir("kandj"));
    let mut st = Settings::new(&Map::new(), None, lib.choices());
    let mut ls = LookState::new(&mut st);
    assert!(ls.current(&st, &lib).is_none() && st.get_str("look") == "cinematic");
    let rev = ls.rev();
    ls.toggle(&mut st); // K
    assert!(st.get_bool("look_on") && ls.current(&st, &lib) == builtin_look("cinematic") && ls.rev() > rev);
    ls.next(&mut st, 1); // J
    assert_eq!((st.get_str("look"), ls.name(&st, &lib).as_str()), ("film", "Film"));
    ls.next(&mut st, -1);
    assert_eq!(st.get_str("look"), "cinematic");
    ls.toggle(&mut st);
    assert!(ls.current(&st, &lib).is_none());
    ls.next(&mut st, 1); // J while off: turns it on, as it was
    assert!(st.get_bool("look_on") && st.get_str("look") == "cinematic");
    ls.show(Some(&look_json("Editing", vec![node("vignette", true, 1.0, &[])]))).unwrap(); // the Looks page, live
    assert_eq!(ls.name(&st, &lib), "Editing");
    ls.show(None).unwrap();
    assert_eq!(ls.name(&st, &lib), "Cinematic");
    let my = lib.save(&look_json("Gone soon", vec![]), None).unwrap();
    st.set_look_choices(lib.choices());
    st.set("look", my.as_str()).unwrap();
    lib.delete(&my).unwrap();
    ls.reload(&mut st, &mut lib); // its file was deleted: back to the default
    assert_eq!(st.get_str("look"), looks::DEFAULT_LOOK);
    ls.close(&mut st);
}

fn cube(path: &std::path::Path, size: usize, f: impl Fn(f64, f64, f64) -> (f64, f64, f64)) {
    let g = |i: usize| i as f64 / (size - 1) as f64;
    let mut text = format!("TITLE \"test\"\nLUT_3D_SIZE {size}\n# a comment\n");
    let mut rows = Vec::new();
    for b in 0..size {
        for gi in 0..size {
            for r in 0..size {
                let (x, y, z) = f(g(r), g(gi), g(b));
                rows.push(format!("{x:.6} {y:.6} {z:.6}"));
            }
        }
    }
    text.push_str(&rows.join("\n"));
    std::fs::write(path, text).unwrap();
}

#[test]
fn test_cube_files() {
    let d = tmp_dir("cubes");
    let mut lib = Library::new(d.join("lib"));
    let src = d.join("src").join("Warm.cube");
    std::fs::create_dir_all(src.parent().unwrap()).unwrap();
    cube(&src, 9, |r, g, b| (r.sqrt(), g, b * 0.5));
    assert_eq!(lib.add_lut(&src).unwrap(), "Warm.cube");
    assert_eq!(lib.luts(), ["Warm.cube"]);
    let lut = lib.lut("Warm.cube").unwrap();
    assert!(lut.size == 9 && lut.title == "test" && lut.table.len() == 9 * 9 * 9 * 3);
    assert_eq!(lut.at(8, 0, 0), [1.0, 0.0, 0.0]);
    assert!((lut.at(0, 0, 8)[2] - 0.5).abs() < 1e-6);
    let bad = d.join("bad.cube");
    for (text, says) in [
        ("LUT_1D_SIZE 4\n0 0 0\n", "1D LUT"),
        ("LUT_3D_SIZE 2\n0 0 0\n", "should have 8 rows"),
        ("hello\n", "no LUT_3D_SIZE"),
    ] {
        std::fs::write(&bad, text).unwrap();
        assert!(looks::load_cube(&bad).unwrap_err().contains(says), "{text}");
    }
    assert!(lib.add_lut(&d.join("x.png")).unwrap_err().to_string().contains(".cube"));
    assert!(lib.lut("Missing.cube").is_none());
}

#[test]
fn test_a_missing_lut_draws_nothing() {
    // (test_a_lut_node: the stages part; the drawing is the renderer's)
    let lib = Library::new(tmp_dir("gone"));
    let gone = normalize(&look_json("", vec![node("lut", true, 1.0, &[("file", v("Gone.cube"))])])).unwrap();
    assert!(looks::stages(&gone, &|n| lib.lut(n).is_some()).is_empty());
}

#[test]
fn test_passes() {
    let st = looks::stages(builtin_look("dreamy").unwrap(), &|_| true); // color, split | blur | bloom
    let starts: Vec<Option<&str>> = st.iter().map(|s| s.start.map(|n| n.type_key())).collect();
    assert_eq!(starts, [None, Some("blur"), Some("bloom")]);
    assert_eq!(st[0].nodes.iter().map(|n| n.type_key()).collect::<Vec<_>>(), ["color", "split"]);
    let sharpen = normalize(&look_json("", vec![node("sharpen", true, 1.0, &[])])).unwrap();
    assert!(looks::stages(&sharpen, &|_| true)[0].start.is_none()); // the first pass: never a start
    let off = normalize(&look_json("", vec![node("color", false, 1.0, &[]), node("grain", true, 0.0, &[])])).unwrap();
    assert!(looks::stages(&off, &|_| true).is_empty());
}

#[test]
fn test_the_command_line() {
    let d = tmp_dir("cli");
    let samples = samples_dir();
    let run = |args: &[&str], stdin: &str| -> (i32, Value) {
        let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        let (line, rc) = looks::cli::main(&args, &d, &samples, &mut std::io::Cursor::new(stdin.as_bytes().to_vec()));
        (rc, json::loads(&line).unwrap())
    };
    let (rc, out) = run(&["schema"], "");
    let keys: Vec<&str> = list(&out, "types").iter().map(|t| s(t, "key")).collect();
    assert_eq!(rc, 0);
    assert_eq!(keys, looks::types().iter().map(|t| t.key).collect::<Vec<_>>());
    assert_eq!(s(&out, "folder"), d.to_string_lossy());
    assert_eq!(s(&list(&list(&out, "types")[0], "params")[0], "key"), "mix");
    assert_eq!(list(&out, "looks").len(), builtin().len());
    let noir = builtin_look("noir").unwrap();
    let (rc, out) = run(&["encode"], &json::dumps(&noir.to_value()));
    assert!(rc == 0 && s(&out, "code") == encode_look(noir));
    let code = s(&out, "code").to_string();
    let (rc, out) = run(&["decode", &code], "");
    assert!(rc == 0 && out.get("look") == Some(&noir.to_value()));
    let (rc, out) = run(&["decode", "SPL1-garbage"], "");
    assert!(rc == 1 && s(&out, "error").contains("typo"));
    let (rc, out) = run(&["save"], r#"{"name": "From the launcher", "nodes": []}"#);
    assert!(rc == 0 && s(&out, "id") == "my:from-the-launcher");
    let (_, out) = run(&["schema"], "");
    let last = list(&out, "looks").last().unwrap().clone();
    assert!(s(&last, "id") == "my:from-the-launcher" && s(&last, "path").ends_with(".json"));
    let (rc, out) = run(&["encode", s(&last, "path")], ""); // a saved look's file
    assert!(rc == 0 && decode(s(&out, "code")).unwrap().name == "From the launcher");
    let vhs = builtin_look("vhs").unwrap();
    let (rc, out) = run(&["import", &encode_look(vhs)], ""); // a pasted code becomes one of the user's looks
    assert!(rc == 0 && s(&out, "id") == "my:retro-vhs" && out.get("look") == Some(&vhs.to_value()));
    let (_, out) = run(&["delete", "my:from-the-launcher"], "");
    assert_eq!(json::dumps(&out), r#"{"deleted": true}"#);
}

#[test]
fn test_the_remote_link_takes_look_commands() {
    let mut st = Settings::defaults();
    let mut r = Remote::new(&mut st, OpenRules, Buf::default());
    r.set_log(|_| {});
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let s1 = Arc::clone(&seen);
    r.add_handler("look-show", move |_, t| {
        let x = json::loads(t).map_err(|e| e.to_string())?;
        s1.lock().unwrap().push(format!("show {}", json::dumps(&x)));
        Ok(())
    });
    let s2 = Arc::clone(&seen);
    r.add_handler("look-hide", move |_, _| {
        s2.lock().unwrap().push("hide".into());
        Ok(())
    });
    let s3 = Arc::clone(&seen);
    r.add_handler("looks-reload", move |_, _| {
        s3.lock().unwrap().push("reload".into());
        Ok(())
    });
    let lines = "look-show {\"name\": \"x\", \"nodes\": []}\nlook-hide\nlooks-reload\nlook-show oops\n";
    r.read_from(std::io::Cursor::new(lines.as_bytes().to_vec())).join().unwrap();
    r.poll(&mut st); // the bad JSON is reported, not raised
    assert_eq!(*seen.lock().unwrap(), [r#"show {"name": "x", "nodes": []}"#, "hide", "reload"]);
}

// ================================================================================================ test_fill.py
const W: i64 = 1920;
const H: i64 = 1080;

#[test]
fn test_fit_keeps_all_of_the_picture_with_bars() {
    assert_eq!(fill::layout("fit", 3440, 1440, W, H), (2560, 1440, 440, 0)); // 21:9: bars at the sides
    assert_eq!(fill::layout("fit", 1920, 1200, W, H), (1920, 1080, 0, 60)); // 16:10: bars top and bottom
    assert_eq!(fill::layout("fit", 2560, 1440, W, H), (2560, 1440, 0, 0)); // 16:9: the whole screen
    assert_eq!(fill::layout("panorama", 3440, 1440, W, H), fill::layout("fit", 3440, 1440, W, H));
}

#[test]
fn test_zoom_fills_by_cropping_and_stretch_by_widening() {
    let (dw, dh, x0, y0) = fill::layout("zoom", 3440, 1440, W, H);
    assert!((dw, x0) == (3440, 0) && (dw as f64 / dh as f64 - W as f64 / H as f64).abs() < 0.002);
    assert!(y0 < 0 && y0 + dh > 1440);
    let cut = 1.0 - 1440.0 / dh as f64;
    assert!(0.24 < cut && cut < 0.27);
    let (_, dh, x0, y0) = fill::layout("zoom", 1920, 1200, W, H);
    assert!((dh, y0) == (1200, 0) && x0 < 0); // 16:10: the sides cut instead
    assert_eq!(fill::layout("stretch", 3440, 1440, W, H), (3440, 1440, 0, 0));
    for f in fill::FILLS {
        assert_eq!(fill::layout(f, 2560, 1440, W, H), (2560, 1440, 0, 0), "{f}");
    }
}

#[test]
fn test_messages_and_the_menu_go_on_the_part_of_the_picture_on_screen() {
    let vis = |f: &str| {
        let (a, b, c, d) = fill::layout(f, 3440, 1440, W, H);
        fill::visible(a, b, c, d, 3440, 1440)
    };
    assert_eq!(vis("zoom"), (0, 0, 3440, 1440));
    assert_eq!(vis("fit"), (440, 0, 2560, 1440));
}

#[test]
fn test_panorama_keeps_the_middle_close_to_true_and_never_folds() {
    assert!(!fill::wider(2560, 1440, W, H) && !fill::wider(1920, 1200, W, H) && fill::wider(3440, 1440, W, H));
    assert!(fill::panorama_t(2560, 1440, W, H) == 0.0 && fill::panorama_u(0.3, 0.0) == 0.3);
    for (fw, fh, middle, edge) in [(3440, 1440, 1.06, 2.5), (2560, 1080, 1.06, 2.5), (5120, 1440, 1.16, 5.5)] {
        let t = fill::panorama_t(fw, fh, W, H);
        let k = (fw as f64 / fh as f64) / (W as f64 / H as f64);
        let xs: Vec<f64> = (0..2001).map(|i| -1.0 + 2.0 * i as f64 / 2000.0).collect();
        let us: Vec<f64> = xs.iter().map(|x| fill::panorama_u(*x, t)).collect();
        assert!((us[0] + 1.0).abs() < 1e-9 && (us[2000] - 1.0).abs() < 1e-9 && us[1000].abs() < 1e-9);
        assert!(us.windows(2).all(|p| p[1] > p[0])); // always moving on across the picture
        let widen: Vec<f64> = us.windows(2).map(|p| k * (xs[1] - xs[0]) / (p[1] - p[0])).collect();
        assert!(1.0 < widen[1000] && widen[1000] < middle && widen[0] < edge, "{fw}x{fh}");
        assert!(widen[1000] < widen[750] && widen[750] < widen[500] && widen[500] < widen[0]);
        assert!((widen[1000] - k * t.atan() / t).abs() < 0.01);
    }
}

#[test]
fn test_the_setting_says_it_is_the_same_16_9_picture() {
    let sp = spec("fill").unwrap();
    let looks = LookChoices::builtin();
    assert_eq!(sp.choices(&looks), fill::FILLS.iter().map(|f| v(*f)).collect::<Vec<_>>());
    assert!(sp.default_value() == v("fit") && sp.live);
    assert!(
        sp.help.contains("16:9") && sp.help.contains("none of these is real ultrawide") && sp.help.contains("Key: Z")
    );
    assert!(sp.impact.iter().any(|(k, t)| *k == "picture" && t.contains("16:9")));
}

// ================================================================================================ test_autodetect.py
fn timing(full: f64, half: f64) -> Timing {
    Timing::for_testing(full, half, true)
}

fn this_pc() -> Timing {
    timing(9.3, 6.8) // RTX 4070 Ti SUPER, what the test measures there
}

const UW120: Display = Display { width: 3440, height: 1440, hz: 120, max_hz: 120 };

fn disp(w: i64, h: i64, hz: i64, max_hz: i64) -> Display {
    Display { width: w, height: h, hz, max_hz }
}

struct Pick {
    t: Timing,
    display: Option<Display>,
    current: Vec<(&'static str, Value)>,
    opts: Options,
}

impl Default for Pick {
    fn default() -> Self {
        Pick { t: this_pc(), display: Some(UW120), current: vec![], opts: Options { trt: true, ..Options::default() } }
    }
}

/// test_autodetect.py pick(): every suggestion is a valid choice, with a reason.
fn pick(p: Pick) -> autodetect::Recommendation {
    let mut current = Settings::defaults().values();
    for (k, x) in p.current {
        current.insert(k, x);
    }
    let r = recommend(&current, &p.t, p.display.as_ref(), &p.opts);
    for (k, x) in r.settings.iter() {
        assert!(spec(k).unwrap().is_choice(x, &LookChoices::builtin()), "{k}");
        assert!(r.reasons.get(k).and_then(Value::as_str).is_some_and(|t| !t.is_empty()), "{k}");
    }
    r
}

fn hz(r: &autodetect::Recommendation) -> i64 {
    r.settings.get("hz").unwrap().as_i64().unwrap()
}

fn setting<'a>(r: &'a autodetect::Recommendation, k: &str) -> &'a str {
    r.settings.get(k).and_then(Value::as_str).unwrap_or("")
}

#[test]
fn test_tested_card_gets_the_mode_it_was_tuned_for() {
    let r = pick(Pick::default());
    assert_eq!(
        json::dumps(&Value::Dict(r.settings.clone())),
        r#"{"hz": 90, "motion": "auto", "engine": "auto", "scaler": "fsr"}"#
    );
    assert!(r.reasons.get("hz").unwrap().as_str().unwrap().contains("120 fps would only fit with half motion detail"));
}

#[test]
fn test_display_caps_the_frame_rate() {
    assert_eq!(hz(&pick(Pick { display: Some(disp(2560, 1440, 60, 60)), ..Pick::default() })), 60);
    assert_eq!(hz(&pick(Pick { display: Some(disp(2560, 1440, 100, 100)), ..Pick::default() })), 90);
    assert_eq!(
        hz(&pick(Pick { t: timing(5.0, 3.5), display: Some(disp(2560, 1440, 119, 119)), ..Pick::default() })),
        120
    );
}

#[test]
fn test_display_set_below_what_it_can_do_gets_a_note() {
    let r = pick(Pick { display: Some(disp(2560, 1440, 60, 144)), ..Pick::default() });
    assert_eq!(hz(&r), 60);
    assert!(r.notes.iter().any(|n| n.contains("can run at 144 Hz")));
}

#[test]
fn test_fast_card_keeps_full_detail_at_120() {
    let r = pick(Pick { t: timing(5.0, 3.5), display: Some(disp(2560, 1440, 144, 144)), ..Pick::default() });
    assert!(hz(&r) == 120 && setting(&r, "motion") == "full");
    assert!(r.notes.iter().any(|n| n.contains("G-Sync"))); // 120 into 144 Hz
}

#[test]
fn test_slower_cards_step_down() {
    let d144 = Some(disp(2560, 1440, 144, 144));
    let mid = pick(Pick { t: timing(12.0, 8.5), display: d144, ..Pick::default() });
    assert_eq!((hz(&mid), setting(&mid, "motion")), (90, "half"));
    let slow = pick(Pick { t: timing(20.0, 13.0), display: d144, ..Pick::default() });
    assert_eq!((hz(&slow), setting(&slow, "motion")), (60, "half"));
    let too_slow = pick(Pick { t: timing(40.0, 30.0), ..Pick::default() });
    assert!(
        hz(&too_slow) == 60 && too_slow.reasons.get("hz").unwrap().as_str().unwrap().contains("Even 60 fps is tight")
    );
}

#[test]
fn test_laptop_leaves_more_room() {
    let t = timing(8.9, 6.6); // fits 90 fps on a desktop card, not with a laptop's margin
    assert_eq!(hz(&pick(Pick { t: t.clone(), ..Pick::default() })), 90);
    let opts = Options { trt: true, gpu: "NVIDIA GeForce RTX 4080 Laptop GPU".into(), ..Options::default() };
    let r = pick(Pick { t, opts, ..Pick::default() });
    assert_eq!((hz(&r), setting(&r, "motion")), (90, "half"));
    assert!(r.notes.iter().any(|n| n.contains("Laptop")));
}

#[test]
fn test_rtx_vsr_kept_only_where_it_fits() {
    let vsr = || vec![("scaler", v("vsr"))];
    let r = pick(Pick { current: vsr(), ..Pick::default() }); // 90 fps leaves ~0.7 ms: not enough
    assert!(setting(&r, "scaler") == "fsr" && !r.settings.contains_key("vsr_quality"));
    let d60 = Some(disp(2560, 1440, 60, 60));
    let r = pick(Pick { current: vsr(), display: d60, ..Pick::default() });
    assert!(setting(&r, "scaler") == "vsr" && r.settings.get("vsr_quality") == Some(&v(4)));
    let opts = Options { trt: true, vsr: Some("RTX Video SDK not set up".into()), ..Options::default() };
    let r = pick(Pick { current: vsr(), display: d60, opts, ..Pick::default() });
    assert!(
        setting(&r, "scaler") == "fsr" && r.reasons.get("scaler").unwrap().as_str().unwrap().contains("isn't set up")
    );
}

#[test]
fn test_fsr_is_never_swapped_for_vsr() {
    let r = pick(Pick { t: timing(3.0, 2.0), display: Some(disp(2560, 1440, 60, 60)), ..Pick::default() });
    assert_eq!(setting(&r, "scaler"), "fsr");
}

#[test]
fn test_without_tensorrt_or_display() {
    let t = Timing::for_testing(10.4, 6.8, false);
    let r = recommend(&Settings::defaults().values(), &t, None, &Options::default());
    assert!(hz(&r) == 60 && setting(&r, "engine") == "auto");
    assert!(r.reasons.get("engine").unwrap().as_str().unwrap().contains("TensorRT isn't installed"));
    assert!(r.notes.iter().any(|n| n.contains("assumes 60 Hz")));
    let mut t = this_pc();
    t.full.estimated = true;
    assert!(pick(Pick { t, ..Pick::default() }).notes.iter().any(|n| n.contains("estimated")));
}

#[test]
fn test_busy_card_note() {
    let busy = |b| Pick { opts: Options { trt: true, busy: Some(b), ..Options::default() }, ..Pick::default() };
    assert!(pick(busy(85)).notes.iter().any(|n| n.contains("busy")));
    assert!(!pick(busy(3)).notes.iter().any(|n| n.contains("busy")));
}

#[test]
fn test_rtx_40_and_50_cards_without_vfg_are_offered_it() {
    let text = autodetect::vfg_offer(true, "NVIDIA GeForce RTX 5070 Ti", Some("621.24"));
    assert!(text.starts_with("Your RTX 5070 Ti can also run NVIDIA VFG"));
    assert!(text.contains("NVIDIA's licence") && text.contains("Auto-detect times it"));
    assert!(!text.contains("driver")); // new enough
    let digit_ms = text.as_bytes().windows(3).any(|w| w[0].is_ascii_digit() && (w[1..] == *b"ms" || &w[1..] == b" m"));
    assert!(!digit_ms && !text.contains("faster")); // NVIDIA's licence: no speed or quality claims
}

#[test]
fn test_vfg_offer_asks_for_a_newer_driver_first() {
    let gpu = "NVIDIA GeForce RTX 4070";
    assert!(autodetect::vfg_offer(true, gpu, Some("566.36")).contains("570.65 or newer, and this PC has 566.36"));
    assert!(!autodetect::vfg_offer(true, gpu, Some("570.65")).contains("driver"));
    assert!(!autodetect::vfg_offer(true, gpu, None).contains("driver")); // unknown: VFG itself will say
    assert!(!autodetect::vfg_offer(true, gpu, Some("r570")).contains("driver"));
}

#[test]
fn test_no_vfg_offer_when_set_up_or_the_card_cant_run_it() {
    assert_eq!(autodetect::vfg_offer(false, "NVIDIA GeForce RTX 3080", Some("621.24")), "");
    assert_eq!(autodetect::vfg_offer(false, "NVIDIA GeForce RTX 5070 Ti", Some("621.24")), "");
}

fn detect_on(vfg_why: &str, driver: &str) -> Value {
    let machine = Machine {
        display: Some(UW120),
        driver: Some(driver.into()),
        busy: Some(0),
        gpu: "NVIDIA GeForce RTX 4070".into(),
        vram_gb: 12.0,
        mac_mode: false,
        dml_mode: false,
        trt: true,
        vsr: None,
        vfg_why: vfg_why.into(),
    };
    autodetect::detect(&Settings::defaults().values(), &machine, &this_pc(), None)
}

#[test]
fn test_detect_offers_vfg_only_where_it_can_be_set_up() {
    let r = detect_on(autodetect::VFG_NOT_SET_UP, "621.24");
    assert!(
        s(&r, "vfg_offer").starts_with("Your RTX 4070 can also run NVIDIA VFG")
            && r.get("vfg_offer_ready") == Some(&v(true))
    );
    let row = list(&r, "found").iter().find(|f| s(f, "label") == "NVIDIA VFG").unwrap().clone();
    assert_eq!(s(&row, "detail"), "Optional: see below.");
    let r = detect_on(autodetect::VFG_NOT_SET_UP, "566.36"); // too old: offered, but the button waits for the driver
    assert!(s(&r, "vfg_offer").contains("566.36") && r.get("vfg_offer_ready") == Some(&v(false)));
    for why in [
        "NVIDIA VFG needs an RTX 40 or 50 series card.",
        "No NVIDIA graphics card was found.",
        "NVIDIA VFG couldn't be checked: boom",
    ] {
        let r = detect_on(why, "621.24");
        assert!(r.get("vfg_offer").is_none() && r.get("vfg_offer_ready").is_none(), "{why}");
    }
}

#[test]
fn test_auto_motion_is_half_where_full_doesnt_fit_this_card() {
    let sh = autodetect::SHARE;
    assert_eq!(auto_scale(1920, 1080, 60, None, sh), 1.0); // no times saved: the plain rule
    assert_eq!(auto_scale(1920, 1080, 60, Some(19.7), sh), 0.5); // an RTX 2080 Ti
    assert_eq!(auto_scale(1920, 1080, 60, Some(14.4), sh), 1.0);
    assert_eq!(auto_scale(1920, 1080, 90, Some(9.3), sh), 1.0); // the card the rule was tuned on keeps full at 90
    assert_eq!(auto_scale(1920, 1080, 90, Some(10.5), sh), 0.5);
    assert_eq!(auto_scale(1920, 1080, 60, Some(14.4), 0.80), 0.5); // a laptop's smaller share
    assert_eq!(auto_scale(1920, 1080, 120, Some(5.0), sh), 0.5); // 120 fps is half anyway
    assert!(auto_scale(2560, 1440, 60, Some(19.7), sh) == 1.0 && auto_scale(2560, 1440, 90, None, sh) == 0.5);
    assert_eq!(auto_scale(3840, 2160, 60, Some(5.0), sh), 0.25);
}

#[test]
fn test_the_tests_times_are_kept_for_that_card_only() {
    let settings = tmp_dir("timing").join("framegen").join("settings.json"); // its folder isn't there yet
    assert_eq!(autodetect::saved_full_ms(&settings, "NVIDIA GeForce RTX 2080 Ti"), None);
    autodetect::save_timing(&settings, "NVIDIA GeForce RTX 2080 Ti", &timing(19.7, 14.5).to_value());
    assert_eq!(autodetect::saved_full_ms(&settings, "NVIDIA GeForce RTX 2080 Ti"), Some(19.7));
    assert_eq!(autodetect::saved_full_ms(&settings, "NVIDIA GeForce RTX 4070"), None); // another card since
    std::fs::write(settings.with_file_name("card-timing.json"), "{not json").unwrap();
    assert_eq!(autodetect::saved_full_ms(&settings, "NVIDIA GeForce RTX 2080 Ti"), None);
}

#[test]
fn test_detect_saves_its_times_only_when_it_measured_them() {
    let path = tmp_dir("measured").join("settings.json");
    let mut st = settings_at(&path);
    let made_up = detect_on("", "621.24");
    autodetect::finish(&mut st, &path, &made_up, false, false, false); // made-up times (--timing): not kept
    assert_eq!(autodetect::saved_full_ms(&path, "NVIDIA GeForce RTX 4070"), None);
    let machine = Machine { gpu: "NVIDIA GeForce RTX 4070".into(), ..Machine::mac("x", 0.0, None) };
    let measured = autodetect::detect(&st.values(), &Machine { mac_mode: false, ..machine }, &timing(19.7, 14.5), None);
    let lines = autodetect::finish(&mut st, &path, &measured, true, false, false);
    assert!(lines.is_empty());
    assert_eq!(autodetect::saved_full_ms(&path, "NVIDIA GeForce RTX 4070"), Some(19.7));
}

#[test]
fn test_detect_apply_saves_the_suggestions_unless_setup_finds_the_card_busy() {
    // (autodetect.main --apply and --if-new)
    let path = tmp_dir("apply").join("settings.json");
    let mut st = settings_at(&path);
    let mut busy_machine = Machine::mac("Apple M5 Max", 64.0, Some(disp(6016, 3384, 160, 160)));
    busy_machine.busy = Some(80);
    let busy = autodetect::detect(&st.values(), &busy_machine, &Timing::measured(7.6, 5.0, "torch"), None);
    let lines = autodetect::finish(&mut st, &path, &busy, true, true, true);
    assert_eq!(lines, ["[detect] the graphics card was 80% busy, so nothing was saved"]);
    assert!(!path.exists());
    let mac = Machine::mac("Apple M5 Max", 64.0, Some(disp(6016, 3384, 160, 160)));
    let r = autodetect::detect(&st.values(), &mac, &Timing::measured(7.6, 5.0, "torch"), None);
    let lines = autodetect::finish(&mut st, &path, &r, true, true, false);
    assert_eq!(lines, [format!("[detect] saved to {}", path.display())]);
    let f = saved(&path);
    assert_eq!((f.get("hz"), f.get("motion")), (Some(&v(90)), Some(&v("half")))); // the M5 Max on a 160 Hz 6K
}

#[test]
fn test_vfg_is_suggested_when_it_keeps_up_at_least_as_well() {
    let vfg = VfgTiming { low: 5.0, medium: 6.0, high: 7.4 };
    let r = pick(Pick { opts: Options { trt: true, vfg: Some(vfg), ..Options::default() }, ..Pick::default() });
    assert!(
        setting(&r, "model") == "vfg" && r.settings.contains_key("vfg_quality") && !r.settings.contains_key("engine")
    );
}

#[test]
fn test_this_mac_reads_its_display_and_chip() {
    // (primary_display() and gpu.name() on a Mac; on another machine they just say they can't)
    if cfg!(target_os = "macos") {
        let d = autodetect::primary_display().expect("the main display");
        assert!(d.width >= 640 && d.height >= 480 && d.hz >= 24 && d.max_hz >= d.hz, "{d:?}");
        assert!(autodetect::mac_memory_gb() > 1.0);
        assert!(autodetect::mac_gpu_name().starts_with("Apple"), "{}", autodetect::mac_gpu_name());
    }
}
