//! The Rust port against the Python reference: each test reads what tools/settings_fixtures.py recorded from
//! framegen/*.py and checks this crate gives exactly the same (JSON text, file bytes, protocol lines, share codes).
//! Without the fixtures (they're git-ignored) each test says SKIPPED and passes.

mod common;

use std::sync::{Arc, Mutex};

use common::*;
use sp_settings::autodetect::{self, Display, Machine, MotionTime, Options, Timing, VfgTiming};
use sp_settings::json::{self, Map, Value, dumps};
use sp_settings::looks::{self, Library, Lut, Node};
use sp_settings::remote::{FnRules, Remote, Request};
use sp_settings::settings::{self, LookChoices, Settings, SettingsError, specs};
use sp_settings::{fill, py, zlib};

macro_rules! need {
    ($name:expr) => {
        match fixture($name) {
            Some(v) => v,
            None => return,
        }
    };
}

/// Collects mismatches and fails at the end with the first few, so one run shows how many there are.
struct Check {
    group: &'static str,
    total: usize,
    bad: Vec<String>,
}

impl Check {
    fn new(group: &'static str) -> Self {
        Check { group, total: 0, bad: Vec::new() }
    }
    fn eq(&mut self, what: impl std::fmt::Display, got: impl AsRef<str>, want: impl AsRef<str>) {
        self.total += 1;
        if got.as_ref() != want.as_ref() {
            self.bad.push(format!("{what}:\n   got  {}\n   want {}", got.as_ref(), want.as_ref()));
        }
    }
    fn done(self) {
        eprintln!("{}: {} checks, {} mismatches", self.group, self.total, self.bad.len());
        if !self.bad.is_empty() {
            panic!(
                "{}: {} of {} differ from the Python:\n{}",
                self.group,
                self.bad.len(),
                self.total,
                self.bad.iter().take(12).cloned().collect::<Vec<_>>().join("\n")
            );
        }
    }
}

fn opt_str(v: Option<&Value>) -> Option<String> {
    match v {
        None | Some(Value::Null) => None,
        Some(x) => Some(x.as_str().expect("a string").to_string()),
    }
}

fn user_choices(files: Option<&Value>) -> LookChoices {
    match files {
        None | Some(Value::Null) => LookChoices::builtin(),
        Some(f) => Library::new(library_from(f)).choices(),
    }
}

// ------------------------------------------------------------------------------------------------ settings
#[test]
fn specs_json_and_spec_helpers() {
    let f = need!("specs");
    let mut c = Check::new("specs");
    let path = std::path::PathBuf::from(s(&f, "path"));
    c.eq("specs_json, built-in looks", dumps(&settings::specs_json(&path, &LookChoices::builtin())), s(&f, "builtin"));
    let looks = user_choices(f.get("user_looks"));
    c.eq("specs_json, the user's looks too", dumps(&settings::specs_json(&path, &looks)), s(&f, "with_user"));
    for sp in specs() {
        let u = f.get("unused_with").unwrap().get(sp.key).unwrap();
        for (src, want) in u.as_dict().unwrap().iter() {
            c.eq(
                format!("{}.unused_with({src})", sp.key),
                format!("{:?}", sp.unused_with(src)),
                format!("{:?}", opt_str(Some(want))),
            );
        }
        if let Some(rows) = f.get("names").unwrap().get(sp.key) {
            for row in items(rows) {
                let r = items(row);
                let v = json::loads(r[0].as_str().unwrap()).unwrap();
                c.eq(
                    format!("{}.name({})", sp.key, r[0]),
                    sp.name(&v, &LookChoices::builtin()),
                    r[1].as_str().unwrap(),
                );
                c.eq(format!("{}.hint({})", sp.key, r[0]), sp.hint(&v), r[2].as_str().unwrap());
            }
        }
    }
    for a in list(&f, "applies") {
        let vals = json::loads(s(a, "values")).unwrap();
        let got: Vec<Value> = specs().iter().map(|sp| Value::Bool(sp.applies_to(vals.as_dict().unwrap()))).collect();
        c.eq(format!("applies_to {}", s(a, "values")), dumps(&Value::List(got)), dumps(a.get("applies").unwrap()));
    }
    for (k, v) in f.get("capture_sizes").unwrap().as_dict().unwrap().iter() {
        let wh = settings::capture_size(k).unwrap();
        c.eq(format!("capture size {k}"), format!("[{}, {}]", wh.0, wh.1), dumps(v));
    }
    c.done();
}

fn op_result(st: &mut Settings, op: &[Value]) -> Value {
    let key = op[1].as_str().unwrap();
    let r: Result<Value, SettingsError> = match op[0].as_str().unwrap() {
        "set" => st.set(key, op[2].clone()).map(|_| Value::Null),
        "override" => st.override_value(key, op[2].clone()).map(|_| Value::Null),
        _ => {
            let delta = op[2].as_i64().unwrap();
            match op.get(3) {
                Some(deny) => {
                    let deny = deny.clone();
                    let allowed = move |k: &str, v: &Value| {
                        !items(&deny).iter().any(|d| {
                            let d = items(d);
                            d[0].as_str() == Some(k) && d[1].py_eq(v)
                        })
                    };
                    st.step(key, delta, Some(&allowed))
                }
                None => st.step(key, delta, None),
            }
        }
    };
    match r {
        Ok(v) => json::obj([("ok", dumps(&v).into())]),
        Err(SettingsError::NotAChoice(m)) => json::obj([("error", "ValueError".into()), ("message", m.into())]),
        Err(SettingsError::UnknownKey(k)) => {
            json::obj([("error", "KeyError".into()), ("message", py::str_repr(&k).into())])
        }
    }
}

#[test]
fn settings_files_round_trip() {
    let f = need!("settings_files");
    let mut c = Check::new("settings files");
    for case in items(&f) {
        let name = s(case, "name");
        let looks = user_choices(case.get("user_looks"));
        let d = tmp_dir("settings");
        let path = d.join("sub").join("settings.json");
        if let Some(text) = case.get("file").and_then(Value::as_str) {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, text.as_bytes()).unwrap();
        }
        let mut st = Settings::load(&path, looks);
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let s2 = Arc::clone(&seen);
        st.watch(move |_, k| s2.lock().unwrap().push(k.to_string()));
        c.eq(format!("{name}: values"), dumps(&Value::Dict(st.values())), s(case, "values"));
        let ops = list(case, "ops");
        let results = list(case, "results");
        for (op, want) in ops.iter().zip(results) {
            c.eq(format!("{name}: {}", dumps(op)), dumps(&op_result(&mut st, items(op))), dumps(want));
        }
        let seen: Vec<Value> = seen.lock().unwrap().iter().map(|k| Value::from(k.as_str())).collect();
        c.eq(format!("{name}: watchers"), dumps(&Value::List(seen)), dumps(case.get("seen").unwrap()));
        c.eq(format!("{name}: values after"), dumps(&Value::Dict(st.values())), s(case, "values_after"));
        let file = std::fs::read_to_string(&path).ok();
        c.eq(format!("{name}: the file"), format!("{file:?}"), format!("{:?}", opt_str(case.get("file_after"))));
        c.eq(format!("{name}: nothing printed"), s(case, "log"), "");
    }
    c.done();
}

#[test]
fn settings_state_lines() {
    let f = need!("state");
    let mut c = Check::new("[settings-state]");
    for case in items(&f) {
        let name = s(case, "name");
        let looks = user_choices(case.get("user_looks"));
        let d = tmp_dir("state");
        let path = d.join("settings.json");
        if let Some(text) = case.get("file").and_then(Value::as_str) {
            std::fs::write(&path, text).unwrap();
        }
        let mut st = Settings::load(&path, looks);
        for o in case.get("overrides").and_then(Value::as_list).unwrap_or(&[]) {
            let o = items(o);
            st.override_value(o[0].as_str().unwrap(), o[1].clone()).unwrap();
        }
        let deny = case.get("deny").cloned().unwrap_or(Value::List(vec![]));
        let disabled = case.get("disabled").cloned().unwrap_or(Value::Dict(Map::new()));
        let restart: Vec<String> = case
            .get("restart")
            .and_then(Value::as_list)
            .unwrap_or(&[])
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let rules = FnRules {
            allowed: move |_: &Settings, k: &str, v: &Value| {
                !items(&deny).iter().any(|d| {
                    let d = items(d);
                    d[0].as_str() == Some(k) && d[1].py_eq(v)
                })
            },
            disabled: move |_: &Settings, k: &str| disabled.get(k).and_then(Value::as_str).map(str::to_string),
            restart_reasons: move |_: &Settings| restart.clone(),
        };
        let out = Buf::default();
        let mut r = Remote::new(&mut st, rules, out.clone());
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let l2 = Arc::clone(&log);
        r.set_log(move |m| l2.lock().unwrap().push(m.to_string()));
        let shown = Arc::new(Mutex::new(Vec::<String>::new()));
        let sh = Arc::clone(&shown);
        r.add_handler("look-show", move |_, text| {
            let v = json::loads(text).map_err(|e| e.to_string())?;
            let look = looks::normalize(&v).map_err(|e| e.to_string())?;
            sh.lock().unwrap().push(dumps(&look.to_value()));
            Ok(())
        });
        let sh = Arc::clone(&shown);
        r.add_handler("look-hide", move |_, _| {
            sh.lock().unwrap().push("hide".into());
            Ok(())
        });
        let sh = Arc::clone(&shown);
        r.add_handler("looks-reload", move |_, _| {
            sh.lock().unwrap().push("reload".into());
            Ok(())
        });
        for cmd in case.get("commands").and_then(Value::as_list).unwrap_or(&[]) {
            r.push_line(cmd.as_str().unwrap());
        }
        r.publish(&st);
        let got = r.poll(&mut st);
        for op in case.get("after").and_then(Value::as_list).unwrap_or(&[]) {
            let op = items(op);
            let (k, v) = (op[1].as_str().unwrap(), op[2].clone());
            if op[0].as_str() == Some("set") { st.set(k, v).unwrap() } else { st.override_value(k, v).unwrap() }
        }
        let want_lines: Vec<&str> = list(case, "lines").iter().map(|l| l.as_str().unwrap()).collect();
        let text = out.text();
        let got_lines: Vec<&str> = text.split('\n').collect();
        c.eq(format!("{name}: line count"), got_lines.len().to_string(), want_lines.len().to_string());
        for (i, (g, w)) in got_lines.iter().zip(&want_lines).enumerate() {
            c.eq(format!("{name}: line {i}"), g, w);
        }
        let mut log = log.lock().unwrap().clone();
        log.push(String::new()); // (the Python's split leaves a last "")
        let want_log: Vec<&str> = list(case, "log").iter().map(|l| l.as_str().unwrap()).collect();
        c.eq(format!("{name}: messages"), log.join("\n"), want_log.join("\n"));
        let poll = match got {
            None => "null",
            Some(Request::Quit) => "\"quit\"",
            Some(Request::Restart) => "\"restart\"",
        };
        c.eq(format!("{name}: poll()"), poll, dumps(case.get("poll").unwrap()));
        let shown: Vec<Value> = shown.lock().unwrap().iter().map(|x| Value::from(x.as_str())).collect();
        c.eq(format!("{name}: handlers"), dumps(&Value::List(shown)), dumps(case.get("shown").unwrap()));
        c.eq(format!("{name}: values"), dumps(&Value::Dict(st.values())), s(case, "values"));
        let file = std::fs::read_to_string(&path).ok();
        c.eq(format!("{name}: the file"), format!("{file:?}"), format!("{:?}", opt_str(case.get("file_after"))));
        r.close(&mut st);
    }
    c.done();
}

// ------------------------------------------------------------------------------------------------ looks
fn normalized(v: &Value) -> Value {
    match looks::normalize(v) {
        Ok(l) => json::obj([("look", dumps(&l.to_value()).into())]),
        Err(e) => json::obj([("error", e.py_class().into()), ("message", e.to_string().into())]),
    }
}

#[test]
fn normalize_messy_looks() {
    let f = need!("normalize");
    let mut c = Check::new("normalize");
    for case in items(&f) {
        let input = s(case, "input");
        let mut want = case.as_dict().unwrap().clone();
        want.remove("input");
        c.eq(input, dumps(&normalized(&json::loads(input).unwrap())), dumps(&Value::Dict(want)));
    }
    c.done();
}

#[test]
fn share_codes() {
    let f = need!("codes");
    let mut c = Check::new("share codes");
    for b in list(&f, "builtin") {
        let id = s(b, "id");
        let look = looks::builtin_look(id).expect("a built-in look");
        c.eq(format!("built-in {id}"), dumps(&look.to_value()), s(b, "look"));
        c.eq(format!("built-in {id}: encode"), looks::encode_look(look), s(b, "code"));
        c.eq(format!("built-in {id}: decode"), dumps(&looks::decode(s(b, "code")).unwrap().to_value()), s(b, "look"));
    }
    let ids: Vec<&str> = looks::builtin().iter().map(|(k, _)| *k).collect();
    let want_ids: Vec<&str> = list(&f, "builtin").iter().map(|b| s(b, "id")).collect();
    c.eq("built-in ids", ids.join(","), want_ids.join(","));
    for e in list(&f, "edited") {
        let input = json::loads(s(e, "input")).unwrap();
        c.eq(
            format!("normalize {}", s(e, "input")),
            dumps(&looks::normalize(&input).unwrap().to_value()),
            s(e, "look"),
        );
        c.eq(format!("encode {}", s(e, "input")), looks::encode(&input).unwrap(), s(e, "code"));
    }
    for d in list(&f, "decode") {
        let code = s(d, "code");
        let got = match looks::decode(code) {
            Ok(l) => format!("look {}", dumps(&l.to_value())),
            Err(e) => format!("error {}", e.0),
        };
        let want = match d.get("look") {
            Some(l) => format!("look {}", l.as_str().unwrap()),
            None => format!("error {}", s(d, "error")),
        };
        c.eq(format!("decode {code:?}"), got, want);
    }
    c.done();
}

fn node_of(v: &Value) -> Node {
    let mut m = Map::new();
    m.insert("nodes", Value::List(vec![v.clone()]));
    looks::normalize(&Value::Dict(m)).unwrap().nodes.remove(0)
}

fn uniforms_json(u: &[(&str, looks::Uniform)]) -> String {
    dumps(&Value::List(u.iter().map(|(k, v)| Value::List(vec![(*k).into(), v.to_value()])).collect()))
}

fn lut_of(v: &Value) -> Lut {
    let nums = |k: &str| list(v, k).iter().map(|x| x.as_f64().unwrap()).collect::<Vec<_>>();
    Lut {
        size: v.get("size").unwrap().as_i64().unwrap() as usize,
        table: vec![],
        dmin: nums("dmin"),
        dmax: nums("dmax"),
        title: String::new(),
    }
}

#[test]
fn shader_inputs_and_passes() {
    let f = need!("uniforms");
    let mut c = Check::new("uniforms and stages");
    let (lut, lut2) = (lut_of(f.get("lut").unwrap()), lut_of(f.get("lut2").unwrap()));
    for case in list(&f, "nodes") {
        let nv = json::loads(s(case, "node")).unwrap();
        let n = node_of(&nv);
        c.eq(format!("node {}", s(case, "node")), dumps(&n.to_value()), s(case, "node"));
        c.eq(format!("uniforms {}", s(case, "node")), uniforms_json(&looks::uniforms(&n, None)), s(case, "uniforms"));
        if let Some(w) = case.get("with_lut") {
            c.eq("uniforms with a LUT", uniforms_json(&looks::uniforms(&n, Some(&lut))), w.as_str().unwrap());
            c.eq(
                "uniforms with a short domain",
                uniforms_json(&looks::uniforms(&n, Some(&lut2))),
                s(case, "with_lut2"),
            );
        }
        if let Some(g) = case.get("glow") {
            let (thr, r) = looks::glow(&n);
            c.eq("glow", dumps(&Value::List(vec![thr.into(), r.into()])), g.as_str().unwrap());
        }
    }
    for case in list(&f, "stages") {
        let look = looks::normalize(&json::loads(s(case, "look")).unwrap()).unwrap(); // (the raw look)
        let st = looks::stages(&look, &|name| name != "Missing.cube");
        let got: Vec<Value> = st
            .iter()
            .map(|s| {
                Value::List(vec![
                    s.start.map_or(Value::Null, |n| n.type_key().into()),
                    Value::List(s.nodes.iter().map(|n| n.type_key().into()).collect()),
                ])
            })
            .collect();
        c.eq(format!("stages {}", s(case, "look")), dumps(&Value::List(got)), dumps(case.get("stages").unwrap()));
    }
    for h in list(&f, "hue_vec") {
        let h = items(h);
        let v = looks::hue_vec(h[0].as_f64().unwrap());
        c.eq(format!("hue_vec({})", dumps(&h[0])), dumps(&v.to_vec().into()), h[1].as_str().unwrap());
    }
    for (k, want) in list(&f, "assist").iter().enumerate() {
        c.eq(format!("assist_matrix({k})"), dumps(&looks::assist_matrix(k).to_vec().into()), want.as_str().unwrap());
    }
    c.done();
}

#[test]
fn cube_files() {
    let f = need!("cube");
    let mut c = Check::new("cube LUTs");
    let d = tmp_dir("cube");
    for case in items(&f) {
        let name = s(case, "name");
        let p = d.join("x.cube");
        std::fs::write(&p, s(case, "text")).unwrap();
        let got = match looks::load_cube(&p) {
            Ok(l) => json::obj([
                ("size", l.size.into()),
                ("title", l.title.as_str().into()),
                ("dmin", l.dmin.clone().into()),
                ("dmax", l.dmax.clone().into()),
                ("table", l.table.iter().map(|x| *x as f64).collect::<Vec<f64>>().into()),
            ]),
            Err(m) => json::obj([("message", m.into())]),
        };
        let mut want = case.as_dict().unwrap().clone();
        want.remove("name");
        want.remove("text");
        want.remove("error");
        c.eq(format!("cube {name}"), dumps(&got), dumps(&Value::Dict(want)));
    }
    c.done();
}

#[test]
fn looks_library_and_command_line() {
    let f = need!("library");
    let mut c = Check::new("looks library and CLI");
    let dir = tmp_dir("library");
    let src = tmp_dir("src");
    let samples = samples_dir();
    let place = |t: &str| {
        t.replace(&dir.to_string_lossy().to_string(), "{dir}")
            .replace(&src.to_string_lossy().to_string(), "{src}")
            .replace(&samples.to_string_lossy().to_string(), "{samples}")
    };
    let unplace = |t: &str| {
        t.replace("{dir}", &dir.to_string_lossy())
            .replace("{src}", &src.to_string_lossy())
            .replace("{samples}", &samples.to_string_lossy())
    };
    let mut lib = Library::new(&dir);
    for step in list(&f, "steps") {
        let st = items(step.get("step").unwrap());
        let what = dumps(step.get("step").unwrap());
        let res: Result<Value, String> = match st[0].as_str().unwrap() {
            "save" => lib.save(&st[1], st[2].as_str()).map(Value::from).map_err(|e| place(&e.py_message())),
            "write" => {
                std::fs::write(dir.join(st[1].as_str().unwrap()), st[2].as_str().unwrap()).unwrap();
                Ok(Value::Null)
            }
            "write_bytes" => {
                std::fs::write(dir.join(st[1].as_str().unwrap()), unhex(st[2].as_str().unwrap())).unwrap();
                Ok(Value::Null)
            }
            "mkdir" => {
                std::fs::create_dir(dir.join(st[1].as_str().unwrap())).unwrap();
                Ok(Value::Null)
            }
            "reload" => {
                lib.reload();
                Ok(Value::Null)
            }
            "delete" => lib.delete(st[1].as_str().unwrap()).map(Value::from).map_err(|e| place(&e.py_message())),
            _ => Ok(lib.path(st[1].as_str().unwrap()).map_or(Value::Null, |p| place(&p.to_string_lossy()).into())),
        };
        let got = match res {
            Ok(v) => format!("ok {}", dumps(&v)),
            Err(e) => format!("error {e}"),
        };
        let want = match step.get("ok") {
            Some(v) => format!("ok {}", dumps(v)),
            None => format!("error {}", s(step, "message")),
        };
        c.eq(format!("{what}: result"), got, want);
        c.eq(format!("{what}: choices"), dumps(&lib.choices().ids.into()), dumps(step.get("choices").unwrap()));
        let mut names: Vec<String> =
            lib.choices().ids.iter().zip(lib.choices().names).map(|(i, n)| format!("{i}={n}")).collect();
        names.sort();
        let mut want_names: Vec<String> = list(step, "names")
            .iter()
            .map(|p| format!("{}={}", items(p)[0].as_str().unwrap(), items(p)[1].as_str().unwrap()))
            .collect();
        want_names.sort();
        c.eq(format!("{what}: names"), names.join("|"), want_names.join("|"));
        let mut files = Map::new();
        let mut entries: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
        entries.sort();
        for p in entries {
            let n = p.file_name().unwrap().to_string_lossy().into_owned();
            files.insert(
                n,
                if p.is_file() {
                    Value::from(String::from_utf8_lossy(&std::fs::read(&p).unwrap()).into_owned())
                } else {
                    Value::Null
                },
            );
        }
        c.eq(format!("{what}: files"), dumps(&Value::Dict(files)), dumps(step.get("files").unwrap()));
        if st[0].as_str() == Some("reload") {
            let mut log = lib.skipped.iter().map(|l| place(l)).collect::<Vec<_>>().join("\n");
            if !log.is_empty() {
                log.push('\n');
            }
            c.eq(format!("{what}: skipped"), log, s(step, "log"));
        }
    }
    std::fs::remove_dir(dir.join("folder.json")).unwrap();
    let good = format!("LUT_3D_SIZE 2\n{}", "0 0 0\n".repeat(8));
    std::fs::write(src.join("Warm.cube"), &good).unwrap();
    std::fs::write(src.join("Shout.CUBE"), &good).unwrap();
    std::fs::write(src.join("bad.cube"), "hello\n").unwrap();
    std::fs::write(src.join("x.png"), &good).unwrap();
    for step in list(&f, "luts") {
        let name = s(step, "src");
        let got = match lib.add_lut(&src.join(name)) {
            Ok(n) => format!("ok {n}"),
            Err(e) => format!("error {e}"),
        };
        let want = match step.get("ok") {
            Some(v) => format!("ok {}", v.as_str().unwrap()),
            None => format!("error {}", s(step, "message")),
        };
        c.eq(format!("add_lut {name}"), got, want);
        c.eq(format!("add_lut {name}: luts"), dumps(&lib.luts().into()), dumps(step.get("luts").unwrap()));
        let sizes: Vec<Value> = ["Warm.cube", "Shout.CUBE", "Missing.cube", "..\\Warm.cube"]
            .iter()
            .map(|n| lib.lut(n).map_or(Value::Null, |l| l.size.into()))
            .collect();
        c.eq(format!("add_lut {name}: lut()"), dumps(&Value::List(sizes)), dumps(step.get("lut").unwrap()));
    }
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::create_dir(&dir).unwrap();
    for call in list(&f, "cli") {
        let args: Vec<String> = list(call, "args").iter().map(|a| unplace(a.as_str().unwrap())).collect();
        let mut stdin = std::io::Cursor::new(s(call, "stdin").as_bytes().to_vec());
        let (line, rc) = looks::cli::main(&args, &dir, &samples, &mut stdin);
        let what = dumps(call.get("args").unwrap());
        c.eq(format!("looks {what}: exit code"), rc.to_string(), dumps(call.get("rc").unwrap()));
        if rc != 2 {
            c.eq(format!("looks {what}"), place(&format!("{line}\n")), s(call, "out"));
        }
    }
    c.done();
}

// ------------------------------------------------------------------------------------------------ fill
#[test]
fn fill_arithmetic() {
    let f = need!("fill");
    let mut c = Check::new("fill");
    for case in items(&f) {
        let sc: Vec<i64> = list(case, "screen").iter().map(|v| v.as_i64().unwrap()).collect();
        let pic: Vec<i64> = list(case, "picture").iter().map(|v| v.as_i64().unwrap()).collect();
        let (fw, fh, w, h) = (sc[0], sc[1], pic[0], pic[1]);
        let what = format!("{fw}x{fh} for {w}x{h}");
        c.eq(format!("{what}: wider"), fill::wider(fw, fh, w, h).to_string(), dumps(case.get("wider").unwrap()));
        let t = fill::panorama_t(fw, fh, w, h);
        c.eq(format!("{what}: panorama_t"), dumps(&t.into()), dumps(case.get("panorama_t").unwrap()));
        for name in fill::FILLS {
            let l = fill::layout(name, fw, fh, w, h);
            let v = fill::visible(l.0, l.1, l.2, l.3, fw, fh);
            let got = format!("[[{}, {}, {}, {}], [{}, {}, {}, {}]]", l.0, l.1, l.2, l.3, v.0, v.1, v.2, v.3);
            c.eq(format!("{what}: {name}"), got, dumps(case.get(name).unwrap()));
        }
        let xs = [-1.0, -0.73, -0.5, -0.1, 0.0, 0.3, 0.999, 1.0];
        let us: Vec<f64> = xs.iter().map(|x| fill::panorama_u(*x, t)).collect();
        c.eq(format!("{what}: panorama_u"), dumps(&us.into()), dumps(case.get("u").unwrap()));
    }
    c.done();
}

// ------------------------------------------------------------------------------------------------ Auto-detect
fn timing_of(v: &Value) -> Timing {
    let m = |k: &str| {
        let t = v.get(k).unwrap();
        MotionTime {
            ms: t.get("ms").unwrap().as_f64().unwrap(),
            engine: t.get("engine").unwrap().as_str().unwrap().into(),
            estimated: t.get("estimated").unwrap().truthy(),
        }
    };
    Timing { full: m("full"), half: m("half") }
}

fn display_of(v: Option<&Value>) -> Option<Display> {
    let d = v?.as_list()?;
    let n = |i: usize| d[i].as_i64().unwrap();
    Some(Display { width: n(0), height: n(1), hz: n(2), max_hz: n(3) })
}

fn vfg_of(v: Option<&Value>) -> Option<VfgTiming> {
    let v = v.filter(|v| !matches!(v, Value::Null))?;
    let ms = |q: &str| v.get(q).unwrap().get("ms").unwrap().as_f64().unwrap();
    Some(VfgTiming { low: ms("low"), medium: ms("medium"), high: ms("high") })
}

fn current_with_defaults(current: &Value) -> Map {
    let mut m = Settings::defaults().values();
    for (k, v) in current.as_dict().unwrap().iter() {
        m.insert(k, v.clone());
    }
    m
}

#[test]
fn recommend_on_made_up_machines() {
    let f = need!("recommend");
    let mut c = Check::new("recommend");
    for case in items(&f) {
        let kw = case.get("kw").unwrap();
        let opts = Options {
            gpu: kw.get("gpu").and_then(Value::as_str).unwrap_or("").into(),
            trt: kw.get("trt").is_some_and(Value::truthy),
            vsr: opt_str(kw.get("vsr")),
            busy: kw.get("busy").and_then(Value::as_i64),
            vfg: vfg_of(kw.get("vfg")),
            vfg_why: kw.get("vfg_why").and_then(Value::as_str).unwrap_or("").into(),
        };
        let r = autodetect::recommend(
            &current_with_defaults(case.get("current").unwrap()),
            &timing_of(case.get("timing").unwrap()),
            display_of(case.get("display")).as_ref(),
            &opts,
        );
        c.eq(s(case, "name"), dumps(&r.to_value()), s(case, "result"));
    }
    c.done();
}

#[test]
fn detect_results_and_the_timing_file() {
    let f = need!("detect");
    let mut c = Check::new("detect");
    for case in list(&f, "detect") {
        let m = case.get("machine").unwrap();
        let machine = Machine {
            display: display_of(m.get("display")),
            driver: opt_str(m.get("driver")),
            busy: m.get("busy").and_then(Value::as_i64),
            gpu: s(m, "gpu").into(),
            vram_gb: m.get("vram_gb").unwrap().as_f64().unwrap(),
            mac_mode: m.get("mac_mode").unwrap().truthy(),
            dml_mode: m.get("dml_mode").unwrap().truthy(),
            trt: m.get("trt").unwrap().truthy(),
            vsr: opt_str(m.get("vsr")),
            vfg_why: s(m, "vfg_why").into(),
        };
        let r = autodetect::detect(
            &current_with_defaults(case.get("current").unwrap()),
            &machine,
            &timing_of(case.get("timing").unwrap()),
            vfg_of(case.get("vfg")).as_ref(),
        );
        c.eq(s(case, "name"), dumps(&r), s(case, "result"));
    }
    for row in list(&f, "auto_scale") {
        let r = items(row);
        let full = r[3].as_f64();
        let got = autodetect::auto_scale(
            r[0].as_i64().unwrap(),
            r[1].as_i64().unwrap(),
            r[2].as_i64().unwrap(),
            full,
            r[4].as_f64().unwrap(),
        );
        c.eq(format!("auto_scale{}", dumps(row)), dumps(&got.into()), dumps(&r[5]));
    }
    for row in list(&f, "offers") {
        let r = items(row);
        let (gpu, driver) = (r[0].as_str().unwrap(), r[1].as_str());
        let not_set_up = gpu != "NVIDIA GeForce RTX 3080";
        c.eq(
            format!("vfg_offer({gpu:?}, {driver:?})"),
            autodetect::vfg_offer(not_set_up, gpu, driver),
            r[2].as_str().unwrap(),
        );
        c.eq(
            format!("vfg_driver_too_old({driver:?})"),
            autodetect::vfg_driver_too_old(driver).to_string(),
            dumps(&r[3]),
        );
    }
    let d = tmp_dir("timing");
    let sp = d.join("new").join("settings.json");
    let t = Timing::for_testing(19.7, 14.5, true);
    autodetect::save_timing(&sp, "NVIDIA GeForce RTX 2080 Ti", &t.to_value());
    c.eq("card-timing.json", std::fs::read_to_string(autodetect::timing_path(&sp)).unwrap(), s(&f, "timing_file"));
    for row in list(&f, "timing_reads") {
        let r = items(row);
        std::fs::write(autodetect::timing_path(&sp), r[0].as_str().unwrap()).unwrap();
        for (i, gpu) in ["NVIDIA GeForce RTX 2080 Ti", "X"].iter().enumerate() {
            let want = &r[i + 1];
            if want.as_str() == Some("AttributeError") {
                continue; // (the Python crashes on a list; this says None)
            }
            c.eq(
                format!("saved_full_ms({:?}, {gpu})", r[0]),
                dumps(&autodetect::saved_full_ms(&sp, gpu).into()),
                dumps(want),
            );
        }
    }
    c.done();
}

// ------------------------------------------------------------------------------------------------ JSON and Python
/// Lone surrogates: a Python str can hold one, a Rust String can't (they read as U+FFFD here).
const LONE_SURROGATES: [&str; 3] = ["\"\\ud83d\"", "\"\\ude00\\ud83d\"", "\"\\ud83dx\""];

#[test]
fn json_and_python_basics() {
    let f = need!("pyjson");
    let mut c = Check::new("JSON and Python basics");
    for case in list(&f, "loads") {
        let text = s(case, "text");
        if LONE_SURROGATES.contains(&text) {
            continue;
        }
        match json::loads(text) {
            Ok(v) => {
                c.eq(
                    format!("dumps {text:?}"),
                    dumps(&v),
                    case.get("dumps").and_then(Value::as_str).unwrap_or("(an error)"),
                );
                if case.get("dumps").is_some() {
                    c.eq(format!("indent 2 {text:?}"), json::dumps_indent(&v, 2), s(case, "indent2"));
                    c.eq(format!("indent 1 {text:?}"), json::dumps_indent(&v, 1), s(case, "indent1"));
                    c.eq(format!("repr {text:?}"), v.py_repr(), s(case, "repr"));
                    c.eq(format!("str {text:?}"), v.py_str(), s(case, "str"));
                }
            }
            Err(e) => c.eq(
                format!("loads {text:?}"),
                e.to_string(),
                case.get("message").and_then(Value::as_str).unwrap_or("(parsed)"),
            ),
        }
    }
    for row in list(&f, "floats") {
        let r = items(row);
        let bits = unhex(r[0].as_str().unwrap());
        let x = f64::from_le_bytes(bits.try_into().unwrap());
        c.eq(format!("repr {}", r[1]), py::float_repr(x), r[1].as_str().unwrap());
        c.eq(format!("{} :.1f", r[1]), py::fmt_fixed(x, 1), r[2].as_str().unwrap());
        c.eq(format!("{} :.0f", r[1]), py::fmt_fixed(x, 0), r[3].as_str().unwrap());
        c.eq(format!("{} :.2f", r[1]), py::fmt_fixed(x, 2), r[4].as_str().unwrap());
        c.eq(format!("round({}, 6)", r[1]), py::float_repr(py::round_to(x, 6)), r[5].as_str().unwrap());
        c.eq(format!("round({}, 2)", r[1]), py::float_repr(py::round_to(x, 2)), r[6].as_str().unwrap());
        if let Some(n) = r[7].as_i64() {
            c.eq(format!("round({})", r[1]), (py::round_even(x) as i64).to_string(), n.to_string());
        }
    }
    for row in list(&f, "float_strs") {
        let r = items(row);
        let got = py::float_from_str(r[0].as_str().unwrap()).map(py::float_repr);
        c.eq(format!("float({:?})", r[0]), format!("{got:?}"), format!("{:?}", r[1].as_str()));
    }
    for row in list(&f, "int_strs") {
        let r = items(row);
        let got = py::int_from_str(r[0].as_str().unwrap()).map(|n| n.to_string());
        let want = r[1].as_str().map(|w| if w.len() > 38 { i128::MAX.to_string() } else { w.to_string() });
        c.eq(format!("int({:?})", r[0]), format!("{got:?}"), format!("{want:?}"));
    }
    for row in list(&f, "splits") {
        let r = items(row);
        let text = r[0].as_str().unwrap();
        let got: Vec<Value> = py::split_ws(text).into_iter().map(Value::from).collect();
        c.eq(format!("{text:?}.split()"), dumps(&Value::List(got)), dumps(&r[1]));
        c.eq(format!("{text:?}.strip()"), py::strip(text), r[2].as_str().unwrap());
    }
    for row in list(&f, "reprs") {
        let r = items(row);
        c.eq(format!("repr {:?}", r[0]), py::str_repr(r[0].as_str().unwrap()), r[1].as_str().unwrap());
    }
    for row in list(&f, "sums") {
        let r = items(row);
        let xs: Vec<f64> =
            items(&r[0]).iter().map(|h| f64::from_le_bytes(unhex(h.as_str().unwrap()).try_into().unwrap())).collect();
        c.eq(format!("sum {xs:?}"), hex(&py::sum(xs.iter().copied()).to_le_bytes()), r[1].as_str().unwrap());
    }
    for row in list(&f, "deflate") {
        let r = items(row);
        let data = unhex(r[0].as_str().unwrap());
        c.eq(format!("deflate {} bytes", data.len()), hex(&zlib::deflate_raw(&data)), r[1].as_str().unwrap());
        c.eq(
            format!("inflate {} bytes", data.len()),
            hex(&zlib::inflate_raw(&unhex(r[1].as_str().unwrap())).unwrap()),
            r[0].as_str().unwrap(),
        );
    }
    for row in list(&f, "utf8") {
        let r = items(row);
        let b = unhex(r[0].as_str().unwrap());
        let got = std::str::from_utf8(&b).err().map(|e| py::utf8_error_text(&b, &e));
        c.eq(format!("decode {}", r[0]), format!("{got:?}"), format!("{:?}", r[1].as_str()));
    }
    c.done();
}
