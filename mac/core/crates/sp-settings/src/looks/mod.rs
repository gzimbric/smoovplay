//! Looks (framegen/looks.py): optional effects on the final picture, stacked like the nodes of a color grading app.
//!
//! A look is a name and a list of nodes, applied in order. Each node is one effect with an on-off switch, an amount
//! ("mix") and its own settings. As JSON (the saved files, and what the launcher sends):
//!
//! ```json
//! {"name": "Cinematic", "nodes": [{"type": "color", "on": true, "mix": 1.0, "contrast": 0.25}, ...]}
//! ```
//!
//! - [`types`]: the node types and their settings. A type's `code` and its settings' order are part of the share-code
//!   format, so never change or reuse them, only add.
//! - [`builtin`]: the looks SmoovPlay comes with. [`Library`]: those plus the user's own, and their .cube LUTs.
//! - [`encode`] and [`decode`]: SPL1 share codes, bit for bit as the Python writes them.
//! - [`stages`] and [`uniforms`]: the passes and shader inputs a renderer draws (the Metal one must match
//!   framegen/looks_gl.py).
//! - [`cli`]: `python -m framegen.looks schema|encode|decode|import|save|delete|add-lut` as functions.

pub mod cli;
mod code;
mod library;

use std::sync::OnceLock;

pub use code::{CodeError, PREFIX, decode, encode, encode_look};
pub use library::{Library, LookState, Lut, load_cube, looks_dir, samples};

use crate::generated;
use crate::json::{Map, Value};
use crate::py;

pub const MAX_NODES: usize = 16;
/// LUT nodes a look can use at once (texture units).
pub const MAX_LUTS: usize = 4;
/// Bytes of UTF-8 of the name in a share code.
pub const MAX_NAME: usize = 48;
/// Luma weights.
pub const W709: [f64; 3] = [0.2126, 0.7152, 0.0722];
pub const DEFAULT_LOOK: &str = generated::DEFAULT_LOOK;

/// A setting's default, a number or, for the LUT's file, text.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PDefault {
    Num(f64),
    Text(&'static str),
}

/// One setting of a node type.
#[derive(Debug, PartialEq)]
pub struct Param {
    pub key: &'static str,
    pub label: &'static str,
    pub lo: f64,
    pub hi: f64,
    /// lo and hi are ints in the Python (the LUT's file), so the schema shows them as 0, not 0.0.
    pub(crate) int_range: bool,
    pub default: PDefault,
    pub step: f64,
    pub unit: &'static str,
    /// A choice, with a name for each value from lo up.
    pub names: &'static [&'static str],
    /// A file name (the LUT's), not a number.
    pub text: bool,
}

/// A node's value for one setting.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldValue {
    Num(f64),
    Text(String),
}

impl FieldValue {
    pub fn num(&self) -> f64 {
        match self {
            FieldValue::Num(v) => *v,
            FieldValue::Text(_) => 0.0,
        }
    }
    pub fn text(&self) -> &str {
        match self {
            FieldValue::Text(s) => s,
            FieldValue::Num(_) => "",
        }
    }
    fn to_value(&self) -> Value {
        match self {
            FieldValue::Num(v) => Value::Float(*v),
            FieldValue::Text(s) => Value::Str(s.clone()),
        }
    }
}

/// Why normalize() can't make a look of something.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum LookError {
    /// ValueError("not a look").
    #[error("not a look")]
    NotALook,
    /// A TypeError in the Python (a node's "type" that's a list or an object). It isn't caught there either.
    #[error("{0}")]
    Type(String),
    /// An OverflowError in the Python (a number too big for a float). It isn't caught there either.
    #[error("int too large to convert to float")]
    Overflow,
}

impl LookError {
    /// The Python exception's class name.
    pub fn py_class(&self) -> &'static str {
        match self {
            LookError::NotALook => "ValueError",
            LookError::Type(_) => "TypeError",
            LookError::Overflow => "OverflowError",
        }
    }
}

impl Param {
    pub(crate) fn default_value(&self) -> Value {
        match self.default {
            PDefault::Num(v) => Value::Float(v),
            PDefault::Text(s) => Value::Str(s.to_string()),
        }
    }

    pub(crate) fn default_num(&self) -> f64 {
        match self.default {
            PDefault::Num(v) => v,
            PDefault::Text(_) => 0.0,
        }
    }

    /// In range and on its step (Param.clean).
    pub fn clean(&self, v: &Value) -> Result<FieldValue, LookError> {
        if self.text {
            return Ok(FieldValue::Text(lut_name(v)));
        }
        let x = match v {
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::Int(n) => Some(*n as f64),
            Value::BigInt(s) => {
                let f: f64 = s.parse().unwrap_or(f64::INFINITY);
                if f.is_infinite() {
                    return Err(LookError::Overflow);
                }
                Some(f)
            }
            Value::Float(f) => Some(*f),
            Value::Str(s) => py::float_from_str(s),
            _ => None,
        };
        let Some(x) = x.filter(|x| x.is_finite()) else {
            return Ok(FieldValue::Num(self.default_num()));
        };
        let x = if self.lo > x { self.lo } else { x };
        let x = if self.hi < x { self.hi } else { x };
        let q = py::round_even((x - self.lo) / self.step);
        Ok(FieldValue::Num(py::round_to(self.lo + q * self.step, 6)))
    }

    /// The setting as a whole number of steps from lo (Param.q).
    pub fn q(&self, v: f64) -> i64 {
        py::round_even((v - self.lo) / self.step) as i64
    }
}

/// A kind of node.
#[derive(Debug, PartialEq)]
pub struct NodeType {
    pub key: &'static str,
    /// Its number in share codes.
    pub code: u32,
    pub label: &'static str,
    /// What it reads. A "point" node reads only its own pixel, "glow", "blur" and "clarity" read a blurred copy, and
    /// "ca" and "sharpen" read neighbours.
    pub kind: &'static str,
    pub help: &'static str,
    pub params: &'static [Param],
}

impl NodeType {
    /// mix, then the type's own settings.
    pub fn fields(&self) -> impl Iterator<Item = &'static Param> + '_ {
        std::iter::once(&generated::MIX).chain(self.params.iter())
    }
    pub fn field_count(&self) -> usize {
        1 + self.params.len()
    }
    pub fn field(&self, i: usize) -> Option<&'static Param> {
        if i == 0 { Some(&generated::MIX) } else { self.params.get(i - 1) }
    }
    pub fn field_index(&self, key: &str) -> Option<usize> {
        self.fields().position(|f| f.key == key)
    }
}

/// The amount, how much of a node's result replaces what came in.
pub fn mix_param() -> &'static Param {
    &generated::MIX
}

/// Every node type, in share-code order (TYPES_LIST).
pub fn types() -> &'static [NodeType] {
    &generated::TYPES
}

/// TYPES[key].
pub fn node_type(key: &str) -> Option<&'static NodeType> {
    generated::TYPES.iter().find(|t| t.key == key)
}

/// BY_CODE[code].
pub fn type_by_code(code: u128) -> Option<&'static NodeType> {
    generated::TYPES.iter().find(|t| t.code as u128 == code)
}

/// One node of a normalized look, with every setting present, in range and on its step.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub ty: &'static NodeType,
    pub on: bool,
    /// One per field, mix first and then the type's params.
    pub values: Vec<FieldValue>,
}

impl Node {
    pub fn type_key(&self) -> &'static str {
        self.ty.key
    }
    pub fn mix(&self) -> f64 {
        self.values[0].num()
    }
    /// A number setting (None for an unknown key).
    pub fn get(&self, key: &str) -> Option<f64> {
        self.ty.field_index(key).map(|i| self.values[i].num())
    }
    fn n(&self, key: &str) -> f64 {
        self.get(key).unwrap_or(0.0)
    }
    /// The LUT's file name ("" for none, or for another type).
    pub fn file(&self) -> &str {
        self.ty.field_index("file").map_or("", |i| self.values[i].text())
    }
    pub fn to_value(&self) -> Value {
        let mut m = Map::new();
        m.insert("type", self.ty.key);
        m.insert("on", self.on);
        for (f, v) in self.ty.fields().zip(&self.values) {
            m.insert(f.key, v.to_value());
        }
        Value::Dict(m)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Look {
    pub name: String,
    pub nodes: Vec<Node>,
}

impl Look {
    /// The look as JSON: {"name", "nodes": [...]}.
    pub fn to_value(&self) -> Value {
        let mut m = Map::new();
        m.insert("name", self.name.as_str());
        m.insert("nodes", Value::List(self.nodes.iter().map(Node::to_value).collect()));
        Value::Dict(m)
    }
}

/// A LUT's file name without folders, so a share code or a file can't point anywhere else. It's str(v or "") with
/// Windows' folder separators taken off on every system (PureWindowsPath(p).name), stripped, and "" unless it ends
/// in .cube.
pub fn lut_name(v: &Value) -> String {
    let text = if v.truthy() { v.py_str() } else { String::new() };
    let name = py::strip(windows_name(&text));
    if name.to_lowercase().ends_with(".cube") { name.chars().take(120).collect() } else { String::new() }
}

/// PureWindowsPath(p).name, as Python 3.12 splits it.
fn windows_name(p: &str) -> &str {
    let norm: Vec<char> = p.chars().map(|c| if c == '/' { '\\' } else { c }).collect();
    // byte offsets of each char, to slice p
    let offs: Vec<usize> = p.char_indices().map(|(i, _)| i).chain(std::iter::once(p.len())).collect();
    let find = |from: usize| (from..norm.len()).find(|&i| norm[i] == '\\');
    let rel_start = if norm.first() == Some(&'\\') {
        if norm.get(1) == Some(&'\\') {
            let unc: String = norm.iter().take(8).collect::<String>().to_uppercase();
            let start = if unc == "\\\\?\\UNC\\" { 8 } else { 2 };
            match find(start) {
                None => return "",
                Some(i) => match find(i + 1) {
                    None => return "",
                    Some(j) => j + 1,
                },
            }
        } else {
            1
        }
    } else if norm.get(1) == Some(&':') {
        if norm.get(2) == Some(&'\\') { 3 } else { 2 }
    } else {
        0
    };
    let rel = &p[offs[rel_start.min(norm.len())]..];
    rel.split(['\\', '/']).rfind(|x| !x.is_empty() && *x != ".").unwrap_or("")
}

/// A clean copy, with known types only and every setting present, in range and on its step. Err(NotALook) if it isn't
/// a look at all.
pub fn normalize(look: &Value) -> Result<Look, LookError> {
    let Value::Dict(m) = look else { return Err(LookError::NotALook) };
    let empty = Value::List(Vec::new());
    let nodes_v = m.get("nodes").unwrap_or(&empty);
    let Value::List(nodes_in) = nodes_v else { return Err(LookError::NotALook) };
    let raw = match m.get("name") {
        Some(v) if v.truthy() => v.py_str(),
        _ => "My look".to_string(),
    };
    let mut name: String = py::split_ws(&raw).join(" ").chars().take(60).collect();
    if name.is_empty() {
        name = "My look".into();
    }
    let mut nodes = Vec::new();
    for n in nodes_in {
        let Value::Dict(nm) = n else { continue };
        let ty = match nm.get("type") {
            Some(Value::Str(s)) => node_type(s),
            Some(Value::List(_)) => return Err(LookError::Type("unhashable type: 'list'".into())),
            Some(Value::Dict(_)) => return Err(LookError::Type("unhashable type: 'dict'".into())),
            _ => None,
        };
        let Some(ty) = ty else { continue };
        let on = nm.get("on").is_none_or(Value::truthy);
        let mut values = Vec::with_capacity(ty.field_count());
        for f in ty.fields() {
            let v = match nm.get(f.key) {
                Some(v) => f.clean(v)?,
                None => f.clean(&f.default_value())?,
            };
            values.push(v);
        }
        nodes.push(Node { ty, on, values });
        if nodes.len() == MAX_NODES {
            break;
        }
    }
    Ok(Look { name, nodes })
}

/// node(type, on, mix, **params) as JSON, for building looks to normalize.
pub fn node(ty: &str, on: bool, mix: f64, params: &[(&str, Value)]) -> Value {
    let mut m = Map::new();
    m.insert("type", ty);
    m.insert("on", on);
    m.insert("mix", mix);
    for (k, v) in params {
        m.insert(*k, v.clone());
    }
    Value::Dict(m)
}

/// The built-in looks, in order: (id, look).
pub fn builtin() -> &'static [(&'static str, Look)] {
    static B: OnceLock<Vec<(&'static str, Look)>> = OnceLock::new();
    B.get_or_init(|| {
        generated::BUILTIN
            .iter()
            .map(|(id, name, nodes)| {
                let nodes = nodes
                    .iter()
                    .map(|(t, on, vals)| {
                        let ty = node_type(t).expect("built-in node type");
                        Node { ty, on: *on, values: vals.iter().map(|v| FieldValue::Num(*v)).collect() }
                    })
                    .collect();
                (*id, Look { name: name.to_string(), nodes })
            })
            .collect()
    })
}

/// BUILTIN[id].
pub fn builtin_look(id: &str) -> Option<&'static Look> {
    builtin().iter().find(|(k, _)| *k == id).map(|(_, l)| l)
}

// ------------------------------------------------------------------------------------------------ passes
/// The nodes that do something: on, with an amount above zero, and for a LUT, a file that's there.
pub fn active_nodes<'a>(look: &'a Look, has_lut: &dyn Fn(&str) -> bool) -> Vec<&'a Node> {
    look.nodes
        .iter()
        .filter(|n| n.on && n.mix() > 0.0 && (n.ty.key != "lut" || (!n.file().is_empty() && has_lut(n.file()))))
        .collect()
}

/// One pass. It's a node that reads other pixels of the pass before (None for the first pass), then nodes that need
/// only their own pixel.
#[derive(Debug, PartialEq)]
pub struct Stage<'a> {
    pub start: Option<&'a Node>,
    pub nodes: Vec<&'a Node>,
}

/// The passes that draw a look, or none if it does nothing. The first never starts with a node that reads other
/// pixels. It brings the picture to the output's size, so every later pass reads an image of exactly that size.
pub fn stages<'a>(look: &'a Look, has_lut: &dyn Fn(&str) -> bool) -> Vec<Stage<'a>> {
    let nodes = active_nodes(look, has_lut);
    if nodes.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Stage { start: None, nodes: Vec::new() }];
    let mut luts = 0;
    for n in nodes {
        if n.ty.key == "lut" {
            luts += 1;
            if luts > MAX_LUTS {
                continue;
            }
        }
        if n.ty.kind == "point" {
            out.last_mut().expect("a stage").nodes.push(n);
        } else {
            out.push(Stage { start: Some(n), nodes: Vec::new() });
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ shader inputs
/// A shader input, a float or a vector (3 for a vec3, 9 for a row-major mat3).
#[derive(Clone, Debug, PartialEq)]
pub enum Uniform {
    F(f64),
    V(Vec<f64>),
}

impl Uniform {
    pub fn to_value(&self) -> Value {
        match self {
            Uniform::F(f) => Value::Float(*f),
            Uniform::V(v) => Value::List(v.iter().map(|x| Value::Float(*x)).collect()),
        }
    }
}

/// 2 ** x, through the C library's pow as Python's float ** does. black_box stops the compiler turning it into exp2.
fn pow2(x: f64) -> f64 {
    std::hint::black_box(2.0f64).powf(x)
}

fn hsv(deg: f64) -> [f64; 3] {
    let h = py::fmod(deg, 360.0) / 60.0;
    let x = 1.0 - (py::fmod(h, 2.0) - 1.0).abs();
    let i = (h as i64).rem_euclid(6);
    match i {
        0 => [1.0, x, 0.0],
        1 => [x, 1.0, 0.0],
        2 => [0.0, 1.0, x],
        3 => [0.0, x, 1.0],
        4 => [x, 0.0, 1.0],
        _ => [1.0, 0.0, x],
    }
}

fn first_max(v: impl IntoIterator<Item = f64>) -> f64 {
    let mut it = v.into_iter();
    let mut m = it.next().unwrap_or(0.0);
    for x in it {
        if x > m {
            m = x;
        }
    }
    m
}

/// A push toward a hue that leaves brightness (luma) alone. Its largest part is 1.
pub fn hue_vec(deg: f64) -> [f64; 3] {
    let rgb = hsv(deg);
    let y = py::sum(rgb.iter().zip(W709).map(|(c, w)| c * w));
    let v = rgb.map(|c| c - y);
    let m = first_max(v.iter().map(|c| c.abs()));
    v.map(|c| c / m)
}

fn scale(v: &[f64], k: f64) -> Vec<f64> {
    v.iter().map(|c| k * c).collect()
}

fn add(vs: &[&[f64]]) -> Vec<f64> {
    let n = vs.iter().map(|v| v.len()).min().unwrap_or(0);
    (0..n).map(|i| py::sum(vs.iter().map(|v| v[i]))).collect()
}

const CVD: [[[f64; 3]; 3]; 3] = [
    [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
];
const SHIFT: [[[f64; 3]; 3]; 3] = [
    [[0.0, 0.0, 0.0], [0.7, 1.0, 0.0], [0.7, 0.0, 1.0]],
    [[0.0, 0.0, 0.0], [0.7, 1.0, 0.0], [0.7, 0.0, 1.0]],
    [[1.0, 0.0, 0.7], [0.0, 1.0, 0.7], [0.0, 0.0, 0.0]],
];

/// Color-blind assist (Machado et al. 2009 at severity 1, with Fidaner's daltonization). A row-major 3x3 for linear
/// RGB, I + E(I - S).
pub fn assist_matrix(kind: usize) -> [f64; 9] {
    let (s, e) = (CVD[kind.min(2)], SHIFT[kind.min(2)]);
    let id = |i: usize, j: usize| if i == j { 1.0 } else { 0.0 };
    let mut lost = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            lost[i][j] = id(i, j) - s[i][j];
        }
    }
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] = id(i, j) + py::sum((0..3).map(|k| e[i][k] * lost[k][j]));
        }
    }
    out
}

/// A node's shader inputs, in the order its shader function takes them. `lut` is its LUT, for a LUT node.
pub fn uniforms(n: &Node, lut: Option<&Lut>) -> Vec<(&'static str, Uniform)> {
    use Uniform::{F, V};
    let one = [1.0, 1.0, 1.0];
    match n.ty.key {
        "color" => {
            let (temp, tint) = (n.n("temperature"), n.n("tint"));
            let wb = [1.0 + 0.18 * temp, 1.0 - 0.12 * tint, 1.0 - 0.18 * temp];
            let y = py::sum(wb.iter().zip(W709).map(|(c, w)| c * w));
            vec![
                ("wb", V(scale(&wb, 1.0 / y))),
                ("gain", F(pow2(n.n("exposure") / 2.2))),
                ("k", F(n.n("contrast"))),
                ("sh", F(0.945 * n.n("shadows"))),
                ("hi", F(0.945 * n.n("highlights"))),
                ("fade", F(0.16 * n.n("fade"))),
                ("sat", F(n.n("saturation"))),
                ("vib", F(n.n("vibrance"))),
            ]
        }
        "wheels" => {
            let lift =
                add(&[&scale(&one, 0.25 * n.n("lift")), &scale(&hue_vec(n.n("lift_hue")), 0.15 * n.n("lift_amount"))]);
            let gain = add(&[
                &one,
                &scale(&one, 0.5 * n.n("gain")),
                &scale(&hue_vec(n.n("gain_hue")), 0.25 * n.n("gain_amount")),
            ]);
            let g =
                add(&[&scale(&one, 0.5 * n.n("gamma")), &scale(&hue_vec(n.n("gamma_hue")), 0.3 * n.n("gamma_amount"))]);
            vec![("lift", V(lift)), ("gain", V(gain)), ("power", V(g.iter().map(|c| pow2(-c)).collect()))]
        }
        "split" => vec![
            ("sh", V(scale(&hue_vec(n.n("shadow_hue")), 0.2 * n.n("shadow_amount")))),
            ("hi", V(scale(&hue_vec(n.n("highlight_hue")), 0.2 * n.n("highlight_amount")))),
            ("bal", F(n.n("balance"))),
        ],
        "mono" => {
            let w = [n.n("red"), n.n("green"), n.n("blue")];
            let total = py::sum(w);
            let w = if total > 0.05 { scale(&w, 1.0 / total) } else { W709.to_vec() };
            vec![("w", V(w)), ("tone", V(scale(&hue_vec(n.n("tone_hue")), 0.3 * n.n("tone_amount"))))]
        }
        "lut" => {
            let (dmin, dmax) = match lut {
                Some(l) => (l.dmin.clone(), l.dmax.clone()),
                None => (vec![0.0; 3], vec![1.0; 3]),
            };
            let dscale = dmin
                .iter()
                .zip(&dmax)
                .map(|(a, b)| {
                    let d = b - a;
                    1.0 / if 1e-6 > d { 1e-6 } else { d }
                })
                .collect();
            vec![("dmin", V(dmin)), ("dscale", V(dscale)), ("n", F(lut.map_or(2, |l| l.size) as f64))]
        }
        "bloom" | "halation" => {
            let tint: Vec<f64> = if n.ty.key == "bloom" {
                vec![1.0 + 0.25 * n.n("warmth"), 1.0, 1.0 - 0.25 * n.n("warmth")]
            } else {
                hsv(n.n("hue")).iter().map(|c| 0.15 + 0.85 * c).collect()
            };
            let m = first_max(tint.iter().copied());
            vec![("amount", F(n.n("amount"))), ("tint", V(scale(&tint, 1.0 / m)))]
        }
        "clarity" | "sharpen" => vec![("amount", F(n.n("amount")))],
        "ca" => vec![("k", F(0.012 * n.n("amount")))],
        "grain" => vec![("a", F(0.15 * n.n("amount"))), ("size", F(n.n("size"))), ("col", F(n.n("color")))],
        "vignette" => {
            let a = n.n("amount");
            vec![
                ("a", F(if a >= 0.0 { a } else { 0.5 * a })),
                ("r0", F(1.0 - 0.6 * n.n("size"))),
                ("soft", F(0.05 + 0.5 * n.n("softness"))),
                ("round", F(n.n("roundness"))),
            ]
        }
        "scanlines" => vec![("a", F(0.6 * n.n("amount"))), ("size", F(n.n("size")))],
        "letterbox" => vec![("aspect", F(n.n("aspect")))],
        "assist" => vec![("m", V(assist_matrix(n.n("kind") as usize).to_vec()))],
        _ => Vec::new(), // blur has everything in its blurred copy
    }
}

/// For a node that uses a blurred copy, its bright-pass threshold (None for a plain blur) and how much the wider
/// levels count.
pub fn glow(n: &Node) -> (Option<f64>, f64) {
    let thr = (n.ty.kind == "glow").then(|| n.n("threshold"));
    (thr, 0.15 + 0.8 * n.n("size"))
}
