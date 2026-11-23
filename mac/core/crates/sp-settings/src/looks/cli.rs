//! `python -m framegen.looks schema|encode|decode|import|save|delete|add-lut|preview`, the launcher's way in, as
//! functions the binary exposes with the same arguments and the same one line of JSON (ASCII, \u escapes). Errors are
//! {"error": "..."} with exit code 1.
//!
//! `preview` draws a look, which needs the renderer: [`parse`] returns it as [`Command::Preview`] for the binary to
//! run; [`run`] answers it with an error.

use std::io::Read;
use std::path::{Path, PathBuf};

use super::library::LibraryError;
use super::{DEFAULT_LOOK, Library, Look, MAX_NODES, builtin, decode, encode, encode_look, samples, types};
use crate::json::{self, Map, Value};
use crate::py;

#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    Schema,
    /// A look's JSON file, or stdin.
    Encode(Option<PathBuf>),
    Decode(String),
    /// Saves a pasted code as one of the user's looks.
    Import(String),
    /// Look JSON on stdin; an id to overwrite one of the user's.
    Save(Option<String>),
    Delete(String),
    AddLut(PathBuf),
    /// LOOK: an id or a code.
    Preview {
        look: String,
        src: PathBuf,
        out: PathBuf,
        frame: i64,
    },
}

const USAGE: &str =
    "usage: python -m framegen.looks [-h] {schema,encode,decode,import,save,delete,add-lut,preview} ...";

/// The arguments after `-m framegen.looks`, as argparse reads them. Err: argparse's message (it exits with code 2).
pub fn parse(args: &[String]) -> Result<Command, String> {
    let Some(cmd) = args.first() else {
        return Err(format!("{USAGE}\npython -m framegen.looks: error: the following arguments are required: cmd"));
    };
    let rest = &args[1..];
    let one = |name: &str| -> Result<String, String> {
        match rest {
            [a] => Ok(a.clone()),
            [] => Err(format!(
                "{USAGE}\npython -m framegen.looks {cmd}: error: the following arguments are required: {name}"
            )),
            _ => Err(format!(
                "{USAGE}\npython -m framegen.looks: error: unrecognized arguments: {}",
                rest[1..].join(" ")
            )),
        }
    };
    let optional = || -> Result<Option<String>, String> {
        match rest {
            [] => Ok(None),
            [a] => Ok(Some(a.clone())),
            _ => Err(format!(
                "{USAGE}\npython -m framegen.looks: error: unrecognized arguments: {}",
                rest[1..].join(" ")
            )),
        }
    };
    Ok(match cmd.as_str() {
        "schema" if rest.is_empty() => Command::Schema,
        "schema" => {
            return Err(format!("{USAGE}\npython -m framegen.looks: error: unrecognized arguments: {}", rest.join(" ")));
        }
        "encode" => Command::Encode(optional()?.map(PathBuf::from)),
        "decode" => Command::Decode(one("code")?),
        "import" => Command::Import(one("code")?),
        "save" => Command::Save(optional()?),
        "delete" => Command::Delete(one("id")?),
        "add-lut" => Command::AddLut(PathBuf::from(one("file")?)),
        "preview" => {
            let mut pos = Vec::new();
            let mut frame = 0i64;
            let mut i = 0;
            while i < rest.len() {
                let a = &rest[i];
                if a == "--frame" {
                    let v = rest.get(i + 1).ok_or_else(|| {
                        format!(
                            "{USAGE}\npython -m framegen.looks preview: error: argument --frame: expected one argument"
                        )
                    })?;
                    frame = v.parse().map_err(|_| {
                        format!(
                            "{USAGE}\npython -m framegen.looks preview: error: argument --frame: invalid int value: {}",
                            py::str_repr(v)
                        )
                    })?;
                    i += 2;
                } else if let Some(v) = a.strip_prefix("--frame=") {
                    frame = v.parse().map_err(|_| {
                        format!(
                            "{USAGE}\npython -m framegen.looks preview: error: argument --frame: invalid int value: {}",
                            py::str_repr(v)
                        )
                    })?;
                    i += 1;
                } else {
                    pos.push(a.clone());
                    i += 1;
                }
            }
            if pos.len() < 3 {
                let names = ["look", "src", "out"];
                return Err(format!(
                    "{USAGE}\npython -m framegen.looks preview: error: the following arguments are required: {}",
                    names[pos.len()..].join(", ")
                ));
            }
            if pos.len() > 3 {
                return Err(format!(
                    "{USAGE}\npython -m framegen.looks: error: unrecognized arguments: {}",
                    pos[3..].join(" ")
                ));
            }
            Command::Preview { look: pos[0].clone(), src: PathBuf::from(&pos[1]), out: PathBuf::from(&pos[2]), frame }
        }
        other => {
            return Err(format!(
                "{USAGE}\npython -m framegen.looks: error: argument cmd: invalid choice: {} (choose from 'schema', \
                 'encode', 'decode', 'import', 'save', 'delete', 'add-lut', 'preview')",
                py::str_repr(other)
            ));
        }
    })
}

/// What `schema` says: the node types and their settings, the looks (built-in, then the user's) with their codes,
/// the LUTs, the folders and the preview's sample pictures (`samples_dir`: framegen/samples, as shipped).
pub fn schema(lib: &Library, samples_dir: &Path) -> Value {
    let entry = |id: &str, look: &Look, builtin: bool| -> Value {
        let mut m = Map::new();
        m.insert("id", id);
        m.insert("name", look.name.as_str());
        m.insert("builtin", builtin);
        m.insert("look", look.to_value());
        m.insert("code", encode_look(look));
        if !builtin {
            m.insert("path", lib.path(id).map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| "None".into()));
        }
        Value::Dict(m)
    };
    let num = |x: f64, int: bool| if int { Value::Int(x as i64) } else { Value::Float(x) };
    let mut looks: Vec<Value> = builtin().iter().map(|(id, l)| entry(id, l, true)).collect();
    for id in lib.choices().ids {
        if let Some(l) = lib.user_look(&id) {
            looks.push(entry(&id, l, false));
        }
    }
    json::obj([
        ("folder", lib.folder.to_string_lossy().as_ref().into()),
        ("lut_folder", lib.lut_folder().to_string_lossy().as_ref().into()),
        ("luts", lib.luts().into()),
        ("max_nodes", MAX_NODES.into()),
        ("default", DEFAULT_LOOK.into()),
        (
            "types",
            Value::List(
                types()
                    .iter()
                    .map(|t| {
                        json::obj([
                            ("key", t.key.into()),
                            ("label", t.label.into()),
                            ("kind", t.kind.into()),
                            ("help", t.help.into()),
                            (
                                "params",
                                Value::List(
                                    t.fields()
                                        .map(|f| {
                                            json::obj([
                                                ("key", f.key.into()),
                                                ("label", f.label.into()),
                                                ("min", num(f.lo, f.int_range)),
                                                ("max", num(f.hi, f.int_range)),
                                                ("default", f.default_value()),
                                                ("step", f.step.into()),
                                                ("unit", f.unit.into()),
                                                ("names", f.names.to_vec().into()),
                                                ("text", f.text.into()),
                                            ])
                                        })
                                        .collect(),
                                ),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("looks", Value::List(looks)),
        ("samples", Value::List(samples(samples_dir))),
    ])
}

fn error(text: impl Into<String>) -> (Value, i32) {
    (json::obj([("error", Value::Str(text.into()))]), 1)
}

fn lib_error(e: LibraryError) -> (Value, i32) {
    match e {
        LibraryError::Io(e, _) => error(py::strerror(&e)),
        other => error(other.to_string()),
    }
}

fn read_json(text: Result<String, String>) -> Result<Value, (Value, i32)> {
    let text = text.map_err(error)?;
    json::loads(&text).map_err(|e| error(e.to_string()))
}

fn read_file(p: &Path) -> Result<String, String> {
    match py::read_text(p) {
        Ok(t) => Ok(t),
        Err(py::ReadError::Os(e)) => Err(py::strerror(&e)),
        Err(py::ReadError::Decode(t)) => Err(t),
    }
}

fn read_stdin(stdin: &mut dyn Read) -> Result<String, String> {
    let mut buf = Vec::new();
    stdin.read_to_end(&mut buf).map_err(|e| py::strerror(&e))?;
    match std::str::from_utf8(&buf) {
        Ok(s) => Ok(py::universal_newlines(s)),
        Err(e) => Err(py::utf8_error_text(&buf, &e)),
    }
}

/// Runs one command: (the JSON to print, the exit code). `library`: where the user's looks are (looks_dir());
/// `samples_dir`: the preview's pictures, for schema.
pub fn run(cmd: &Command, library: &Path, samples_dir: &Path, stdin: &mut dyn Read) -> (Value, i32) {
    let ok = |v: Value| (v, 0);
    match cmd {
        Command::Schema => ok(schema(&Library::new(library), samples_dir)),
        Command::Encode(file) => {
            let text = match file {
                Some(p) => read_file(p),
                None => read_stdin(stdin),
            };
            match read_json(text) {
                Err(e) => e,
                Ok(v) => match encode(&v) {
                    Ok(code) => ok(json::obj([("code", code.into())])),
                    Err(e) => error(e.to_string()),
                },
            }
        }
        Command::Decode(code) => match decode(code) {
            Ok(look) => ok(json::obj([("look", look.to_value())])),
            Err(e) => error(e.0),
        },
        Command::Import(code) => {
            let mut lib = Library::new(library);
            match decode(code) {
                Err(e) => error(e.0),
                Ok(look) => match lib.save(&look.to_value(), None) {
                    Ok(id) => ok(json::obj([("id", id.into()), ("look", look.to_value())])),
                    Err(e) => lib_error(e),
                },
            }
        }
        Command::Save(id) => {
            let mut lib = Library::new(library);
            match read_json(read_stdin(stdin)) {
                Err(e) => e,
                Ok(v) => match lib.save(&v, id.as_deref()) {
                    Ok(id) => ok(json::obj([("id", id.into())])),
                    Err(e) => lib_error(e),
                },
            }
        }
        Command::Delete(id) => {
            let mut lib = Library::new(library);
            match lib.delete(id) {
                Ok(d) => ok(json::obj([("deleted", d.into())])),
                Err(e) => lib_error(e),
            }
        }
        Command::AddLut(file) => {
            let mut lib = Library::new(library);
            match lib.add_lut(file) {
                Ok(name) => ok(json::obj([("name", name.into())])),
                Err(e) => lib_error(e),
            }
        }
        Command::Preview { .. } => error("preview needs the renderer: the binary runs it"),
    }
}

/// parse() and run() together: the line to print (JSON, or argparse's message) and the exit code.
pub fn main(args: &[String], library: &Path, samples_dir: &Path, stdin: &mut dyn Read) -> (String, i32) {
    match parse(args) {
        Err(usage) => (usage, 2),
        Ok(cmd) => {
            let (v, rc) = run(&cmd, library, samples_dir, stdin);
            (json::dumps(&v), rc)
        }
    }
}
