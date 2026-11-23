//! Looks (framegen/looks.py): optional effects on the final picture, stacked like the nodes of a colour grading app.
//! This module is the data side, ported as it is: the node types and their settings ([`TYPES`]), cleaning a look up
//! from JSON ([`normalize`]), the built-in looks ([`builtin`]), splitting a look into passes ([`stages`]), what each
//! node's shader gets ([`uniforms`]), .cube LUTs ([`load_cube`]) and the user's looks on disk ([`Library`]). The GPU
//! side is [`crate::looks_gpu`]. Share codes stay with the launcher (looks.py encode/decode).
//!
//! A type's settings' order and `code` are part of the share-code format, so they follow looks.py exactly.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use serde_json::{Value, json};

pub const MAX_NODES: usize = 16;
/// LUT nodes a look can use at once (texture slots).
pub const MAX_LUTS: usize = 4;
pub const W709: [f64; 3] = [0.2126, 0.7152, 0.0722];
/// Blur levels (looks.py LEVELS).
pub const LEVELS: usize = 6;
pub const DEFAULT_LOOK: &str = "cinematic";

/// One setting of a node type.
#[derive(Debug)]
pub struct Param {
    pub key: &'static str,
    pub label: &'static str,
    pub lo: f64,
    pub hi: f64,
    pub default: f64,
    pub step: f64,
    pub unit: &'static str,
    /// A choice: the names of lo, lo + 1, ...
    pub names: &'static [&'static str],
    /// A file name (the LUT's), not a number.
    pub text: bool,
}

impl Param {
    /// In range and on its step (looks.py Param.clean).
    pub fn clean(&self, v: f64) -> f64 {
        if !v.is_finite() {
            return self.default;
        }
        let q = ((v.clamp(self.lo, self.hi) - self.lo) / self.step).round_ties_even();
        ((self.lo + q * self.step) * 1e6).round_ties_even() / 1e6
    }
}

const fn p(key: &'static str, label: &'static str, lo: f64, hi: f64, default: f64) -> Param {
    Param { key, label, lo, hi, default, step: 0.01, unit: "", names: &[], text: false }
}
const fn ps(key: &'static str, label: &'static str, lo: f64, hi: f64, default: f64, step: f64, unit: &'static str) -> Param {
    Param { key, label, lo, hi, default, step, unit, names: &[], text: false }
}
const fn hue(key: &'static str, label: &'static str, default: f64) -> Param {
    Param { key, label, lo: 0.0, hi: 360.0, default, step: 1.0, unit: "°", names: &[], text: false }
}

/// "Amount": how much of a node's result replaces what came in.
pub const MIX: Param = p("mix", "Amount", 0.0, 1.0, 1.0);

/// What a node reads: only its own pixel, a blurred copy, or neighbours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Point,
    Glow,
    Blur,
    Clarity,
    Ca,
    Sharpen,
}

#[derive(Debug)]
pub struct NodeType {
    pub key: &'static str,
    /// Its number in share codes.
    pub code: u32,
    pub label: &'static str,
    pub kind: Kind,
    pub help: &'static str,
    pub params: &'static [Param],
}

impl PartialEq for NodeType {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl NodeType {
    /// Whether this node starts a new pass with a blurred copy of its input (bloom, halation, blur, clarity).
    pub fn uses_pyramid(&self) -> bool {
        matches!(self.kind, Kind::Glow | Kind::Blur | Kind::Clarity)
    }
}

pub static TYPES: &[NodeType] = &[
    NodeType { key: "color", code: 1, label: "Color", kind: Kind::Point,
        help: "The basics: brightness, contrast, how colorful, and warm or cool.",
        params: &[ps("exposure", "Exposure", -3.0, 3.0, 0.0, 0.05, "stops"), p("contrast", "Contrast", -1.0, 1.0, 0.0),
            p("shadows", "Shadows", -1.0, 1.0, 0.0), p("highlights", "Highlights", -1.0, 1.0, 0.0),
            p("fade", "Fade", 0.0, 1.0, 0.0), p("saturation", "Saturation", 0.0, 2.0, 1.0),
            p("vibrance", "Vibrance", -1.0, 1.0, 0.0), p("temperature", "Temperature", -1.0, 1.0, 0.0),
            p("tint", "Tint", -1.0, 1.0, 0.0)] },
    NodeType { key: "wheels", code: 2, label: "Lift / gamma / gain", kind: Kind::Point,
        help: "Grading wheels: lift moves the darks, gamma the middle, gain the brights, each brighter or darker and toward a color.",
        params: &[p("lift", "Lift", -1.0, 1.0, 0.0), hue("lift_hue", "Lift color", 210.0),
            p("lift_amount", "Lift color amount", 0.0, 1.0, 0.0),
            p("gamma", "Gamma", -1.0, 1.0, 0.0), hue("gamma_hue", "Gamma color", 40.0),
            p("gamma_amount", "Gamma color amount", 0.0, 1.0, 0.0),
            p("gain", "Gain", -1.0, 1.0, 0.0), hue("gain_hue", "Gain color", 40.0),
            p("gain_amount", "Gain color amount", 0.0, 1.0, 0.0)] },
    NodeType { key: "split", code: 3, label: "Split tone", kind: Kind::Point,
        help: "One color in the shadows and another in the highlights, like teal and orange.",
        params: &[hue("shadow_hue", "Shadow color", 200.0), p("shadow_amount", "Shadow amount", 0.0, 1.0, 0.0),
            hue("highlight_hue", "Highlight color", 35.0),
            p("highlight_amount", "Highlight amount", 0.0, 1.0, 0.0), p("balance", "Balance", -1.0, 1.0, 0.0)] },
    NodeType { key: "mono", code: 4, label: "Black & white", kind: Kind::Point,
        help: "Black and white, with how bright each color comes out, and an optional tint like sepia.",
        params: &[p("red", "Reds", -1.0, 2.0, 0.30), p("green", "Greens", -1.0, 2.0, 0.59), p("blue", "Blues", -1.0, 2.0, 0.11),
            hue("tone_hue", "Tint color", 35.0), p("tone_amount", "Tint amount", 0.0, 1.0, 0.0)] },
    NodeType { key: "lut", code: 5, label: "LUT", kind: Kind::Point,
        help: "A color grade from a .cube LUT file, as made in DaVinci Resolve, Photoshop and other apps. Amount blends it with the picture.",
        params: &[Param { key: "file", label: "LUT file", lo: 0.0, hi: 0.0, default: 0.0, step: 0.01, unit: "",
            names: &[], text: true }] },
    NodeType { key: "bloom", code: 6, label: "Bloom", kind: Kind::Glow,
        help: "A soft glow around bright parts of the picture.",
        params: &[p("amount", "Strength", 0.0, 2.0, 0.5), p("threshold", "Threshold", 0.0, 1.0, 0.7),
            p("size", "Size", 0.0, 1.0, 0.6), p("warmth", "Warmth", -1.0, 1.0, 0.0)] },
    NodeType { key: "halation", code: 7, label: "Halation", kind: Kind::Glow,
        help: "The red-orange glow film gets around bright edges.",
        params: &[p("amount", "Strength", 0.0, 2.0, 0.6), p("threshold", "Threshold", 0.0, 1.0, 0.6),
            p("size", "Size", 0.0, 1.0, 0.3), hue("hue", "Color", 10.0)] },
    NodeType { key: "blur", code: 8, label: "Blur", kind: Kind::Blur,
        help: "Softens the picture. Amount blends it with the sharp one, for a dreamy, soft-focus look.",
        params: &[p("size", "Size", 0.0, 1.0, 0.3)] },
    NodeType { key: "clarity", code: 9, label: "Clarity", kind: Kind::Clarity,
        help: "Brings out texture and detail (or, below zero, smooths it), without the halos of sharpening.",
        params: &[p("amount", "Strength", -1.0, 1.0, 0.3), p("size", "Size", 0.0, 1.0, 0.4)] },
    NodeType { key: "ca", code: 10, label: "Chromatic aberration", kind: Kind::Ca,
        help: "Color fringes toward the edges of the picture, like a cheap lens.",
        params: &[p("amount", "Strength", 0.0, 1.0, 0.3)] },
    NodeType { key: "sharpen", code: 11, label: "Sharpen", kind: Kind::Sharpen,
        help: "Crisper edges. Below zero, softer.",
        params: &[p("amount", "Strength", -1.0, 2.0, 0.5)] },
    NodeType { key: "grain", code: 12, label: "Film grain", kind: Kind::Point,
        help: "Moving film grain, mostly in the mid-tones.",
        params: &[p("amount", "Strength", 0.0, 1.0, 0.3), ps("size", "Size", 0.5, 3.0, 1.2, 0.05, "px"),
            p("color", "Color", 0.0, 1.0, 0.2)] },
    NodeType { key: "vignette", code: 13, label: "Vignette", kind: Kind::Point,
        help: "Darker corners (or, below zero, brighter ones).",
        params: &[p("amount", "Strength", -1.0, 1.0, 0.35), p("size", "Size", 0.0, 1.0, 0.5),
            p("softness", "Softness", 0.0, 1.0, 0.5), p("roundness", "Roundness", 0.0, 1.0, 0.5)] },
    NodeType { key: "scanlines", code: 14, label: "Scanlines", kind: Kind::Point,
        help: "The lines of an old TV.",
        params: &[p("amount", "Strength", 0.0, 1.0, 0.3), ps("size", "Size", 1.0, 8.0, 3.0, 0.5, "px")] },
    NodeType { key: "letterbox", code: 15, label: "Letterbox", kind: Kind::Point,
        help: "Black bars for a wider, film-shaped picture. They cover part of the game's picture, HUD included.",
        params: &[ps("aspect", "Shape", 1.33, 3.0, 2.39, 0.01, ":1")] },
    NodeType { key: "assist", code: 16, label: "Color-blind assist", kind: Kind::Point,
        help: "Moves the colors you can't tell apart toward ones you can (daltonization).",
        params: &[Param { key: "kind", label: "Type", lo: 0.0, hi: 2.0, default: 1.0, step: 1.0, unit: "",
            names: &["Protanopia (red)", "Deuteranopia (green)", "Tritanopia (blue)"], text: false }] },
];

pub fn node_type(key: &str) -> Option<&'static NodeType> {
    TYPES.iter().find(|t| t.key == key)
}

/// One node of a look: its type, on/off, its amount and its settings (in the type's order; a LUT's file in `file`).
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub ty: &'static NodeType,
    pub on: bool,
    pub mix: f64,
    pub values: Vec<f64>,
    pub file: String,
}

impl Node {
    /// A node of type `ty` at its defaults (on, full amount).
    pub fn new(ty: &str) -> Option<Node> {
        let t = node_type(ty)?;
        Some(Node { ty: t, on: true, mix: 1.0, values: t.params.iter().map(|p| p.default).collect(), file: String::new() })
    }

    /// A setting by key (NaN if the type has no such setting).
    pub fn get(&self, key: &str) -> f64 {
        self.ty.params.iter().position(|p| p.key == key).map_or(f64::NAN, |i| self.values[i])
    }

    /// Sets a setting by key (cleaned to its range and step); `with` for chaining.
    pub fn with(mut self, key: &str, v: f64) -> Node {
        if key == "mix" {
            self.mix = MIX.clean(v);
        } else if let Some(i) = self.ty.params.iter().position(|p| p.key == key) {
            self.values[i] = self.ty.params[i].clean(v);
        }
        self
    }

    pub fn to_json(&self) -> Value {
        let mut m = serde_json::Map::new();
        m.insert("type".into(), json!(self.ty.key));
        m.insert("on".into(), json!(self.on));
        m.insert("mix".into(), json!(self.mix));
        for (p, v) in self.ty.params.iter().zip(&self.values) {
            if p.text {
                m.insert(p.key.into(), json!(self.file));
            } else {
                m.insert(p.key.into(), json!(v));
            }
        }
        Value::Object(m)
    }
}

/// A look: a name and its nodes, applied in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Look {
    pub name: String,
    pub nodes: Vec<Node>,
}

impl Look {
    pub fn to_json(&self) -> Value {
        json!({"name": self.name, "nodes": self.nodes.iter().map(Node::to_json).collect::<Vec<_>>()})
    }
}

/// A LUT's file name, without folders (looks.py _lut_name): a look can't point anywhere else.
pub fn lut_name(v: &str) -> String {
    let name = v.rsplit(['/', '\\']).next().unwrap_or("").trim();
    if name.to_lowercase().ends_with(".cube") { name.chars().take(120).collect() } else { String::new() }
}

/// A clean copy (looks.py normalize): known types only, every setting present, in range and on its step. Err if it
/// isn't a look at all.
pub fn normalize(look: &Value) -> Result<Look, String> {
    let obj = look.as_object().ok_or("not a look")?;
    let nodes_v = match obj.get("nodes") {
        None => &[][..],
        Some(Value::Array(a)) => &a[..],
        Some(_) => return Err("not a look".into()),
    };
    let raw_name = match obj.get("name") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        Some(Value::Null) | None => "My look".into(),
        Some(Value::String(_)) => "My look".into(),
        Some(v) => v.to_string(),
    };
    let mut name: String = raw_name.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(60).collect();
    if name.is_empty() {
        name = "My look".into();
    }
    let mut nodes = Vec::new();
    for n in nodes_v {
        let Some(o) = n.as_object() else { continue };
        let Some(t) = o.get("type").and_then(Value::as_str).and_then(node_type) else { continue };
        let num = |key: &str, p: &Param| -> f64 {
            match o.get(key) {
                None => p.default,
                Some(v) => match number(v) {
                    Some(x) => p.clean(x),
                    None => p.default,
                },
            }
        };
        let on = o.get("on").is_none_or(truthy);
        let mix = num("mix", &MIX);
        let mut values = Vec::new();
        let mut file = String::new();
        for prm in t.params {
            if prm.text {
                file = lut_name(&o.get(prm.key).map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))
                    .unwrap_or_default());
                values.push(0.0);
            } else {
                values.push(num(prm.key, prm));
            }
        }
        nodes.push(Node { ty: t, on, mix, values, file });
        if nodes.len() == MAX_NODES {
            break;
        }
    }
    Ok(Look { name, nodes })
}

fn number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(*b as i32 as f64),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Null => false,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// The nodes that do something: on, with an amount above zero, and for a LUT, a file that's there.
pub fn active_nodes<'a>(look: &'a Look, has_lut: &dyn Fn(&str) -> bool) -> Vec<&'a Node> {
    look.nodes
        .iter()
        .filter(|n| n.on && n.mix > 0.0 && (n.ty.key != "lut" || (!n.file.is_empty() && has_lut(&n.file))))
        .collect()
}

/// One pass: a node that reads other pixels of the pass before (None: the first pass), then nodes that need only
/// their own pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub start: Option<Node>,
    pub nodes: Vec<Node>,
}

impl Stage {
    /// Its nodes in shader order: the start node first.
    pub fn all(&self) -> impl Iterator<Item = &Node> {
        self.start.iter().chain(self.nodes.iter())
    }
}

/// The passes that draw a look; empty if it does nothing. The first never starts with a node that reads other
/// pixels: it brings the picture to the output's size.
pub fn stages(look: &Look, has_lut: &dyn Fn(&str) -> bool) -> Vec<Stage> {
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
        if n.ty.kind == Kind::Point {
            out.last_mut().unwrap().nodes.push(n.clone());
        } else {
            out.push(Stage { start: Some(n.clone()), nodes: Vec::new() });
        }
    }
    out
}

fn hsv(deg: f64) -> [f64; 3] {
    let h = deg.rem_euclid(360.0) / 60.0;
    let x = 1.0 - ((h % 2.0) - 1.0).abs();
    [[1.0, x, 0.0], [x, 1.0, 0.0], [0.0, 1.0, x], [0.0, x, 1.0], [x, 0.0, 1.0], [1.0, 0.0, x]][(h as usize) % 6]
}

/// A push toward a hue that leaves brightness (luma) alone; its largest part is 1.
pub fn hue_vec(deg: f64) -> [f64; 3] {
    let rgb = hsv(deg);
    let y: f64 = rgb.iter().zip(W709).map(|(c, w)| c * w).sum();
    let v = rgb.map(|c| c - y);
    let m = v.iter().fold(0.0f64, |a, c| a.max(c.abs()));
    v.map(|c| c / m)
}

fn scale(v: [f64; 3], k: f64) -> [f64; 3] {
    v.map(|c| c * k)
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
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

/// Row-major 3x3, linear RGB: I + E (I - S) (Machado et al. 2009, Fidaner et al.'s daltonization).
pub fn assist_matrix(kind: usize) -> [f64; 9] {
    let (s, e) = (&CVD[kind.min(2)], &SHIFT[kind.min(2)]);
    let id = |i: usize, j: usize| if i == j { 1.0 } else { 0.0 };
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] = id(i, j) + (0..3).map(|k| e[i][k] * (id(k, j) - s[k][j])).sum::<f64>();
        }
    }
    out
}

/// A node's shader inputs, in the order its shader function takes them: (name, 1, 3 or 9 floats). `lut`: its Lut,
/// for a LUT node.
pub fn uniforms(n: &Node, lut: Option<&Lut>) -> Vec<(&'static str, Vec<f64>)> {
    let g = |k: &str| n.get(k);
    match n.ty.key {
        "color" => {
            let (temp, tint) = (g("temperature"), g("tint"));
            let wb = [1.0 + 0.18 * temp, 1.0 - 0.12 * tint, 1.0 - 0.18 * temp];
            let y: f64 = wb.iter().zip(W709).map(|(c, w)| c * w).sum();
            vec![("wb", scale(wb, 1.0 / y).to_vec()), ("gain", vec![2f64.powf(g("exposure") / 2.2)]),
                 ("k", vec![g("contrast")]), ("sh", vec![0.945 * g("shadows")]), ("hi", vec![0.945 * g("highlights")]),
                 ("fade", vec![0.16 * g("fade")]), ("sat", vec![g("saturation")]), ("vib", vec![g("vibrance")])]
        }
        "wheels" => {
            let one = [1.0; 3];
            let lift = add(scale(one, 0.25 * g("lift")), scale(hue_vec(g("lift_hue")), 0.15 * g("lift_amount")));
            let gain = add(add(one, scale(one, 0.5 * g("gain"))), scale(hue_vec(g("gain_hue")), 0.25 * g("gain_amount")));
            let gm = add(scale(one, 0.5 * g("gamma")), scale(hue_vec(g("gamma_hue")), 0.3 * g("gamma_amount")));
            vec![("lift", lift.to_vec()), ("gain", gain.to_vec()), ("power", gm.map(|c| 2f64.powf(-c)).to_vec())]
        }
        "split" => vec![
            ("sh", scale(hue_vec(g("shadow_hue")), 0.2 * g("shadow_amount")).to_vec()),
            ("hi", scale(hue_vec(g("highlight_hue")), 0.2 * g("highlight_amount")).to_vec()),
            ("bal", vec![g("balance")]),
        ],
        "mono" => {
            let w = [g("red"), g("green"), g("blue")];
            let total: f64 = w.iter().sum();
            let w = if total > 0.05 { scale(w, 1.0 / total) } else { W709 };
            vec![("w", w.to_vec()), ("tone", scale(hue_vec(g("tone_hue")), 0.3 * g("tone_amount")).to_vec())]
        }
        "lut" => {
            let (dmin, dmax) = lut.map_or(([0.0; 3], [1.0; 3]), |l| (l.dmin, l.dmax));
            let dscale = [0, 1, 2].map(|i| 1.0 / (dmax[i] - dmin[i]).max(1e-6));
            vec![("dmin", dmin.to_vec()), ("dscale", dscale.to_vec()), ("n", vec![lut.map_or(2.0, |l| l.size as f64)])]
        }
        "bloom" | "halation" => {
            let tint = if n.ty.key == "bloom" {
                [1.0 + 0.25 * g("warmth"), 1.0, 1.0 - 0.25 * g("warmth")]
            } else {
                hsv(g("hue")).map(|c| 0.15 + 0.85 * c)
            };
            let m = tint.iter().fold(f64::MIN, |a, &c| a.max(c));
            vec![("amount", vec![g("amount")]), ("tint", scale(tint, 1.0 / m).to_vec())]
        }
        "clarity" => vec![("amount", vec![g("amount")])],
        "ca" => vec![("k", vec![0.012 * g("amount")])],
        "sharpen" => vec![("amount", vec![g("amount")])],
        "grain" => vec![("a", vec![0.15 * g("amount")]), ("size", vec![g("size")]), ("col", vec![g("color")])],
        "vignette" => {
            let a = g("amount");
            vec![("a", vec![if a >= 0.0 { a } else { 0.5 * a }]), ("r0", vec![1.0 - 0.6 * g("size")]),
                 ("soft", vec![0.05 + 0.5 * g("softness")]), ("round", vec![g("roundness")])]
        }
        "scanlines" => vec![("a", vec![0.6 * g("amount")]), ("size", vec![g("size")])],
        "letterbox" => vec![("aspect", vec![g("aspect")])],
        "assist" => vec![("m", assist_matrix(g("kind") as usize).to_vec())],
        _ => Vec::new(), // blur: everything is in its blurred copy
    }
}

/// For a node that uses a blurred copy: (its bright-pass threshold, or None for a plain blur; how much the wider
/// levels count).
pub fn glow(n: &Node) -> (Option<f64>, f64) {
    let thr = (n.ty.kind == Kind::Glow).then(|| n.get("threshold"));
    (thr, 0.15 + 0.8 * n.get("size"))
}

/// The blur levels' sizes for an image w x h: halvings, at most LEVELS.
pub fn pyramid_sizes(mut w: usize, mut h: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    while out.len() < LEVELS && w >= 2 && h >= 2 {
        w /= 2;
        h /= 2;
        out.push((w, h));
    }
    out
}

// ------------------------------------------------------------------------------------------------ built-in looks
fn nd(ty: &str, kv: &[(&str, f64)]) -> Node {
    kv.iter().fold(Node::new(ty).unwrap(), |n, &(k, v)| n.with(k, v))
}

fn off(n: Node) -> Node {
    Node { on: false, ..n }
}

/// The looks SmoovPlay comes with (looks.py BUILTIN), as (id, look), in the settings' order.
pub fn builtin() -> &'static [(&'static str, Look)] {
    static B: std::sync::OnceLock<Vec<(&'static str, Look)>> = std::sync::OnceLock::new();
    B.get_or_init(|| {
        let l = |name: &str, nodes: Vec<Node>| Look { name: name.into(), nodes };
        vec![
            ("cinematic", l("Cinematic", vec![
                nd("color", &[("contrast", 0.25), ("highlights", -0.3), ("shadows", 0.1), ("saturation", 0.9), ("temperature", 0.05)]),
                nd("split", &[("shadow_hue", 195.0), ("shadow_amount", 0.5), ("highlight_hue", 30.0), ("highlight_amount", 0.45)]),
                nd("bloom", &[("amount", 0.25), ("threshold", 0.75), ("size", 0.6)]),
                nd("vignette", &[("amount", 0.3), ("size", 0.5), ("softness", 0.6)]),
                off(nd("letterbox", &[("aspect", 2.39)]))])),
            ("film", l("Film", vec![
                nd("color", &[("contrast", 0.15), ("highlights", -0.2), ("fade", 0.4), ("saturation", 0.85), ("temperature", 0.1)]),
                nd("split", &[("shadow_hue", 200.0), ("shadow_amount", 0.2), ("highlight_hue", 40.0), ("highlight_amount", 0.25)]),
                nd("halation", &[("amount", 0.5), ("threshold", 0.65), ("size", 0.3), ("hue", 12.0)]),
                nd("grain", &[("amount", 0.35), ("size", 1.3), ("color", 0.25)]),
                nd("vignette", &[("amount", 0.2)])])),
            ("vivid", l("Vivid", vec![
                nd("color", &[("contrast", 0.2), ("shadows", 0.15), ("saturation", 1.1), ("vibrance", 0.45)]),
                nd("clarity", &[("amount", 0.35), ("size", 0.4)]),
                nd("sharpen", &[("amount", 0.3)])])),
            ("noir", l("Noir", vec![
                nd("mono", &[("red", 0.45), ("green", 0.45), ("blue", 0.1)]),
                nd("color", &[("exposure", -0.1), ("contrast", 0.5), ("shadows", -0.2)]),
                nd("bloom", &[("amount", 0.2), ("threshold", 0.8), ("size", 0.5)]),
                nd("grain", &[("amount", 0.4), ("size", 1.4), ("color", 0.0)]),
                nd("vignette", &[("amount", 0.45), ("size", 0.55), ("softness", 0.5)])])),
            ("golden", l("Golden hour", vec![
                nd("color", &[("contrast", 0.1), ("highlights", -0.15), ("saturation", 1.05), ("temperature", 0.45), ("tint", 0.1)]),
                nd("split", &[("shadow_hue", 260.0), ("shadow_amount", 0.15), ("highlight_hue", 40.0), ("highlight_amount", 0.4)]),
                nd("bloom", &[("amount", 0.45), ("threshold", 0.6), ("size", 0.75), ("warmth", 0.6)]),
                nd("vignette", &[("amount", 0.25)])])),
            ("neon", l("Neon night", vec![
                nd("color", &[("exposure", -0.15), ("contrast", 0.3), ("saturation", 1.25), ("temperature", -0.2)]),
                nd("split", &[("shadow_hue", 240.0), ("shadow_amount", 0.4), ("highlight_hue", 320.0), ("highlight_amount", 0.35)]),
                nd("bloom", &[("amount", 0.8), ("threshold", 0.55), ("size", 0.7)]),
                nd("ca", &[("amount", 0.35)]),
                nd("vignette", &[("amount", 0.35)])])),
            ("vhs", l("Retro VHS", vec![
                nd("color", &[("contrast", -0.1), ("fade", 0.5), ("saturation", 0.8), ("temperature", 0.05), ("tint", 0.15)]),
                nd("blur", &[("mix", 0.35), ("size", 0.1)]),
                nd("ca", &[("amount", 0.8)]),
                nd("scanlines", &[("amount", 0.35), ("size", 3.0)]),
                nd("grain", &[("amount", 0.5), ("size", 1.0), ("color", 0.6)])])),
            ("bleach", l("Bleach bypass", vec![
                nd("color", &[("contrast", 0.45), ("shadows", -0.15), ("highlights", 0.1), ("saturation", 0.45)]),
                nd("grain", &[("amount", 0.2)]),
                nd("sharpen", &[("amount", 0.2)])])),
            ("dreamy", l("Dreamy", vec![
                nd("color", &[("exposure", 0.1), ("contrast", -0.15), ("fade", 0.3), ("saturation", 0.9)]),
                nd("split", &[("shadow_hue", 220.0), ("shadow_amount", 0.15), ("highlight_hue", 330.0), ("highlight_amount", 0.2)]),
                nd("blur", &[("mix", 0.25), ("size", 0.5)]),
                nd("bloom", &[("amount", 0.6), ("threshold", 0.45), ("size", 0.85)])])),
            ("arctic", l("Arctic", vec![
                nd("color", &[("contrast", 0.2), ("highlights", 0.1), ("saturation", 0.75), ("temperature", -0.5)]),
                nd("split", &[("shadow_hue", 210.0), ("shadow_amount", 0.3), ("highlight_hue", 190.0), ("highlight_amount", 0.2)]),
                nd("sharpen", &[("amount", 0.25)])])),
            ("sepia", l("Sepia", vec![
                nd("mono", &[("tone_hue", 35.0), ("tone_amount", 0.7)]),
                nd("color", &[("contrast", 0.2), ("fade", 0.25)]),
                nd("grain", &[("amount", 0.35)]),
                nd("vignette", &[("amount", 0.4)])])),
            ("assist-protan", l("Color assist: protanopia", vec![nd("assist", &[("kind", 0.0)])])),
            ("assist-deutan", l("Color assist: deuteranopia", vec![nd("assist", &[("kind", 1.0)])])),
            ("assist-tritan", l("Color assist: tritanopia", vec![nd("assist", &[("kind", 2.0)])])),
        ]
    })
}

// ------------------------------------------------------------------------------------------------ LUT files
/// A 3D LUT: `table` is size^3 RGB triples, indexed [b][g][r] (red fastest), as in a .cube file.
#[derive(Debug, PartialEq)]
pub struct Lut {
    pub size: usize,
    pub table: Vec<[f32; 3]>,
    pub dmin: [f64; 3],
    pub dmax: [f64; 3],
    pub title: String,
}

/// A 3D .cube LUT (Adobe/Resolve format). Err with a message for the user.
pub fn load_cube(path: &Path) -> Result<Lut, String> {
    let text = std::fs::read(path).map_err(|e| format!("couldn't read it ({e})"))?;
    parse_cube(&String::from_utf8_lossy(&text))
}

pub fn parse_cube(text: &str) -> Result<Lut, String> {
    let (mut size, mut dmin, mut dmax, mut title) = (0usize, [0.0; 3], [1.0; 3], String::new());
    let mut rows: Vec<[f32; 3]> = Vec::new();
    let three = |words: &[&str]| -> Option<[f64; 3]> {
        Some([words.first()?.parse().ok()?, words.get(1)?.parse().ok()?, words.get(2)?.parse().ok()?])
    };
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        let word = words[0].to_uppercase();
        match word.as_str() {
            "TITLE" => title = line[5..].trim().trim_matches('"').to_string(),
            "LUT_3D_SIZE" => size = words.get(1).and_then(|s| s.parse().ok()).unwrap_or(0),
            "LUT_1D_SIZE" => return Err("it's a 1D LUT; SmoovPlay uses 3D LUTs (LUT_3D_SIZE)".into()),
            "DOMAIN_MIN" => dmin = three(&words[1..]).ok_or("its DOMAIN_MIN isn't three numbers")?,
            "DOMAIN_MAX" => dmax = three(&words[1..]).ok_or("its DOMAIN_MAX isn't three numbers")?,
            _ if word.starts_with(|c: char| c.is_ascii_digit() || "-+.".contains(c)) => {
                let v = three(&words).ok_or("some of its rows aren't three numbers")?;
                rows.push(v.map(|x| x as f32));
            }
            _ => {}
        }
    }
    if !(2..=129).contains(&size) {
        return Err("it has no LUT_3D_SIZE between 2 and 129".into());
    }
    if rows.len() != size.pow(3) {
        return Err(format!("it should have {} rows of colors for size {size}, but has {}", size.pow(3), rows.len()));
    }
    if rows.iter().flatten().any(|v| !v.is_finite()) {
        return Err("it has colors that aren't numbers".into());
    }
    Ok(Lut { size, table: rows, dmin, dmax, title })
}

/// LUT files by name from a folder, read once (and again when a file changes); None if missing or broken.
#[derive(Debug, Default)]
pub struct LutStore {
    pub folder: Option<PathBuf>,
    cache: Mutex<BTreeMap<String, (Option<SystemTime>, Option<Arc<Lut>>)>>,
}

impl LutStore {
    pub fn new(folder: Option<PathBuf>) -> LutStore {
        LutStore { folder, cache: Mutex::default() }
    }

    pub fn get(&self, name: &str) -> Option<Arc<Lut>> {
        let clean = lut_name(name);
        let path = self.folder.as_ref()?.join(&clean);
        if clean.is_empty() {
            return None;
        }
        let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok()?;
        let mut cache = self.cache.lock().unwrap();
        if let Some((t, lut)) = cache.get(name)
            && *t == Some(mtime) {
                return lut.clone();
            }
        let lut = match load_cube(&path) {
            Ok(l) => Some(Arc::new(l)),
            Err(e) => {
                eprintln!("[looks] LUT {name}: {e}");
                None
            }
        };
        cache.insert(name.to_string(), (Some(mtime), lut.clone()));
        lut
    }

    /// The .cube files in the folder.
    pub fn names(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .folder
            .as_ref()
            .and_then(|f| std::fs::read_dir(f).ok())
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.to_lowercase().ends_with(".cube"))
            .collect();
        out.sort();
        out
    }
}

/// The built-in looks and the user's (a folder of .json looks, ids "my:<file name>", and its luts/ folder), as
/// looks.py's Library. The order of [`Library::ids`] is the "Which look" setting's: built-in ones, then the user's by
/// name.
#[derive(Debug)]
pub struct Library {
    pub folder: Option<PathBuf>,
    pub luts: LutStore,
    user: Vec<(String, Look)>,
}

impl Library {
    /// `folder`: where the user's looks are (None: built-in ones only). LUTs are in its "luts" folder.
    pub fn new(folder: Option<PathBuf>) -> Library {
        let luts = LutStore::new(folder.as_ref().map(|f| f.join("luts")));
        let mut lib = Library { folder, luts, user: Vec::new() };
        lib.reload();
        lib
    }

    /// Reads the user's looks again (a broken file is skipped, and said so).
    pub fn reload(&mut self) {
        let mut user = Vec::new();
        if let Some(dir) = self.folder.as_ref().and_then(|f| std::fs::read_dir(f).ok()) {
            let mut files: Vec<PathBuf> =
                dir.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
            files.sort();
            for p in files {
                let stem = p.file_stem().unwrap_or_default().to_string_lossy().to_string();
                let parsed = std::fs::read_to_string(&p)
                    .map_err(|e| e.to_string())
                    .and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| e.to_string()))
                    .and_then(|v| normalize(&v));
                match parsed {
                    Ok(look) => user.push((format!("my:{stem}"), look)),
                    Err(e) => eprintln!("[looks] skipped {}: {e}", p.display()),
                }
            }
        }
        user.sort_by(|a, b| (a.1.name.to_lowercase(), &a.0).cmp(&(b.1.name.to_lowercase(), &b.0)));
        self.user = user;
    }

    pub fn ids(&self) -> Vec<String> {
        builtin().iter().map(|(k, _)| k.to_string()).chain(self.user.iter().map(|(k, _)| k.clone())).collect()
    }

    pub fn get(&self, id: &str) -> Option<&Look> {
        builtin().iter().find(|(k, _)| *k == id).map(|(_, l)| l).or_else(|| self.user.iter().find(|(k, _)| k == id).map(|(_, l)| l))
    }

    /// A look's name, for the menu and messages.
    pub fn name(&self, id: &str) -> String {
        self.get(id).map(|l| l.name.clone()).unwrap_or_else(|| id.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_cleans_up() {
        let v = json!({"name": "  A   look ", "nodes": [
            {"type": "color", "contrast": 7, "saturation": "x", "exposure": 0.123},
            {"type": "warp-drive"}, "junk",
            {"type": "lut", "file": "..\\..\\secret\\grade.cube"},
            {"type": "lut", "file": "notalut.png"}]});
        let mut v2 = v.clone();
        for _ in 0..30 {
            v2["nodes"].as_array_mut().unwrap().push(json!({"type": "grain"}));
        }
        let look = normalize(&v2).unwrap();
        let c = &look.nodes[0];
        assert_eq!(look.name, "A look");
        assert_eq!((c.get("contrast"), c.get("saturation")), (1.0, 1.0));
        assert!((c.get("exposure") - 0.1).abs() < 1e-9 && c.on && c.mix == 1.0);
        assert_eq!((look.nodes[1].file.as_str(), look.nodes[2].file.as_str()), ("grade.cube", ""));
        assert_eq!(look.nodes.len(), MAX_NODES);
        assert!(normalize(&json!("not a look")).is_err());
    }

    #[test]
    fn passes() {
        let dreamy = &builtin().iter().find(|(k, _)| *k == "dreamy").unwrap().1;
        let st = stages(dreamy, &|_| true);
        let starts: Vec<_> = st.iter().map(|s| s.start.as_ref().map(|n| n.ty.key)).collect();
        assert_eq!(starts, [None, Some("blur"), Some("bloom")]);
        assert_eq!(st[0].nodes.iter().map(|n| n.ty.key).collect::<Vec<_>>(), ["color", "split"]);
        let sharpen = Look { name: "x".into(), nodes: vec![Node::new("sharpen").unwrap()] };
        assert!(stages(&sharpen, &|_| true)[0].start.is_none());
        let idle = Look { name: "x".into(), nodes: vec![off(Node::new("color").unwrap()), Node::new("grain").unwrap().with("mix", 0.0)] };
        assert!(stages(&idle, &|_| true).is_empty());
    }

    #[test]
    fn builtin_looks_are_normal_and_round_trip_json() {
        for (k, look) in builtin() {
            assert_eq!(&normalize(&look.to_json()).unwrap(), look, "{k}");
        }
        assert_eq!(builtin().len(), 14);
    }

    #[test]
    fn cube_files() {
        let mut text = String::from("TITLE \"test\"\nLUT_3D_SIZE 2\n# a comment\n");
        for b in 0..2 {
            for g in 0..2 {
                for r in 0..2 {
                    text += &format!("{r} {g} {} \n", b as f64 * 0.5);
                }
            }
        }
        let lut = parse_cube(&text).unwrap();
        assert_eq!((lut.size, lut.title.as_str(), lut.table.len()), (2, "test", 8));
        assert_eq!(lut.table[1], [1.0, 0.0, 0.0]);
        assert_eq!(lut.table[4][2], 0.5);
        for (t, says) in [("LUT_1D_SIZE 4\n0 0 0\n", "1D LUT"), ("LUT_3D_SIZE 2\n0 0 0\n", "should have 8 rows"), ("hello\n", "no LUT_3D_SIZE")] {
            assert!(parse_cube(t).unwrap_err().contains(says), "{t}");
        }
    }

    #[test]
    fn hue_and_assist() {
        let v = hue_vec(0.0);
        assert!((v.iter().zip(W709).map(|(c, w)| c * w).sum::<f64>()).abs() < 1e-12);
        assert!((v.iter().fold(0.0f64, |a, c| a.max(c.abs())) - 1.0).abs() < 1e-12);
        let m = assist_matrix(1);
        // gray stays gray: each row sums to 1
        for i in 0..3 {
            assert!((m[i * 3] + m[i * 3 + 1] + m[i * 3 + 2] - 1.0).abs() < 1e-3);
        }
    }
}
