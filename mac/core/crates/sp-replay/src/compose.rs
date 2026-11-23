//! The before & after video (replay.py compose and `python -m framegen.replay compose`): the game's own frames
//! (each held until the next) and SmoovPlay's (every frame shown), at SmoovPlay's times, split down the middle or
//! side by side, with labels; the sound copied in.
//!
//! On a Mac: the tracks (MP4s the recorder writes, each placed on the replay's timeline) are decoded by VideoToolbox
//! (sp-media's VtDecoder) into NV12 pixel buffers; a Metal kernel composes each picture straight into a buffer from
//! the encoder's pool; a VTCompressionSession that isn't real time encodes it (so the live replay's sessions come
//! first on the media engine), and the MP4 is written as packets come, its moov at the front.
//!
//! [`compose_main`] is the subcommand (`smoovplay replay-compose ...`, or `sp-replay-compose ...`) with the Python
//! CLI's arguments: --after --before [--sound] --out [--layout split|side] [--labels BEFORE AFTER] [--encoder]
//! [--cleanup FOLDER] [--mark FOLDER]. It runs at low priority (the recorder starts it at nice 10; it asks for the
//! utility QoS too) and keeps FOLDER/<pid>.txt (the video's path) while it works; if it fails, the file stays, with
//! why.

use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::mpsc::channel;
use std::time::Duration;

use objc2::rc::{Retained, autoreleasepool};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureUsage,
};
use sp_core::Gpu;
use sp_media::{Decoded, VtDecoder};
use sp_tap::nal::Codec;

use crate::gpu::{self, ComposeParams, TexCache, Texture};
use crate::label::{Label, label};
use crate::mp4::{self, Mp4File, Mp4Stream, ReadTrack, TrackSpec};
use crate::recorder::Layout;
use crate::track::Format;
use crate::vtenc::{Encoded, VideoConfig, VtEncoder};
use crate::{AFTER_BPP, AFTER_NOMINAL, DIVIDER_BGR, Error, Result, TB, py_round, yuv};

/// replay.py main()'s compose arguments.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposeArgs {
    pub after: PathBuf,
    pub before: PathBuf,
    pub sound: Option<PathBuf>,
    pub out: PathBuf,
    pub layout: Layout,
    pub labels: (String, String),
    pub encoder: Option<String>,
    /// Deleted (the tracks) when done.
    pub cleanup: Option<PathBuf>,
    /// FOLDER/<process id>.txt holds the video's path while it's made, for the launcher.
    pub mark: Option<PathBuf>,
}

pub const USAGE: &str = "usage: replay-compose --after AFTER --before BEFORE [--sound SOUND] --out OUT \
[--layout {split,side}] [--labels BEFORE_LABEL AFTER_LABEL] [--encoder ENCODER] [--cleanup FOLDER] [--mark FOLDER]";

impl ComposeArgs {
    /// Parses the arguments after the subcommand (`--name value` or `--name=value`).
    pub fn parse(args: &[String]) -> std::result::Result<ComposeArgs, String> {
        let mut it: Vec<String> = Vec::new();
        for a in args {
            match a.split_once('=') {
                Some((k, v)) if k.starts_with("--") => {
                    it.push(k.to_string());
                    it.push(v.to_string());
                }
                _ => it.push(a.clone()),
            }
        }
        let (mut after, mut before, mut sound, mut out, mut encoder, mut cleanup, mut mark) = (None, None, None, None, None, None, None);
        let mut layout = Layout::Split;
        let mut labels = ("Original".to_string(), "SmoovPlay".to_string());
        let mut i = 0;
        let value = |i: &mut usize, name: &str| -> std::result::Result<String, String> {
            *i += 1;
            it.get(*i).cloned().ok_or_else(|| format!("argument {name}: expected one argument"))
        };
        while i < it.len() {
            let name = it[i].clone();
            match name.as_str() {
                "--after" => after = Some(PathBuf::from(value(&mut i, &name)?)),
                "--before" => before = Some(PathBuf::from(value(&mut i, &name)?)),
                "--sound" => sound = Some(PathBuf::from(value(&mut i, &name)?)),
                "--out" => out = Some(PathBuf::from(value(&mut i, &name)?)),
                "--encoder" => encoder = Some(value(&mut i, &name)?).filter(|e| !e.is_empty()),
                "--cleanup" => cleanup = Some(PathBuf::from(value(&mut i, &name)?)),
                "--mark" => mark = Some(PathBuf::from(value(&mut i, &name)?)),
                "--layout" => {
                    let v = value(&mut i, &name)?;
                    layout = Layout::parse(&v).ok_or_else(|| format!("argument --layout: invalid choice: '{v}' (choose from 'split', 'side')"))?;
                }
                "--labels" => {
                    let a = value(&mut i, &name)?;
                    let b = value(&mut i, &name).map_err(|_| "argument --labels: expected 2 arguments".to_string())?;
                    labels = (a, b);
                }
                other => return Err(format!("unrecognized arguments: {other}")),
            }
            i += 1;
        }
        let need = |v: Option<PathBuf>, n: &str| v.ok_or_else(|| format!("the following arguments are required: {n}"));
        Ok(ComposeArgs {
            after: need(after, "--after")?,
            before: need(before, "--before")?,
            out: need(out, "--out")?,
            sound,
            layout,
            labels,
            encoder,
            cleanup,
            mark,
        })
    }
}

fn part_path(out: &Path) -> PathBuf {
    let mut s = out.as_os_str().to_os_string();
    s.push(".part");
    PathBuf::from(s)
}

/// The subcommand: makes the video, keeping the launcher's marker meanwhile. Returns the exit status.
pub fn compose_main(args: &[String]) -> i32 {
    unsafe {
        // Below the game: the utility QoS (CPU, GPU and disk), on top of the nice 10 the recorder starts it at
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0);
    }
    let a = match ComposeArgs::parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{USAGE}\nreplay-compose: error: {e}");
            return 2;
        }
    };
    let mut mark = None;
    if let Some(dir) = &a.mark
        && std::fs::create_dir_all(dir).is_ok() {
            let m = dir.join(format!("{}.txt", std::process::id()));
            if std::fs::write(&m, format!("{}\n", a.out.display())).is_ok() {
                mark = Some(m);
            }
        }
    let labels = (a.labels.0.as_str(), a.labels.1.as_str());
    let r = std::panic::catch_unwind(|| compose(&a.after, &a.before, &a.out, a.sound.as_deref(), a.layout, labels, a.encoder.as_deref()));
    let r = match r {
        Ok(r) => r,
        Err(p) => Err(Error::Other(format!(
            "panic: {}",
            p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
        ))),
    };
    if let Some(c) = &a.cleanup {
        let _ = std::fs::remove_dir_all(c);
    }
    match r {
        Ok(secs) => {
            if let Some(m) = &mark {
                let _ = std::fs::remove_file(m);
            }
            println!("[replay] before & after: {secs:.1} s -> {}", a.out.display());
            0
        }
        Err(e) => {
            let _ = std::fs::remove_file(part_path(&a.out));
            if let Some(m) = &mark {
                let why: String = e.to_string().chars().take(300).collect();
                let kind = match &e {
                    Error::Io(_) => "IOError",
                    Error::Status { .. } => "OSStatusError",
                    _ => "Error",
                };
                let _ = std::fs::write(m, format!("{}\nfailed: {kind}: {why}\n", a.out.display()));
            }
            eprintln!("[replay] before & after failed: {e}");
            1
        }
    }
}

/// A video track's decoder: AVCC samples in, NV12 pictures out.
struct Dec {
    vt: VtDecoder,
    len_size: usize,
}

impl Dec {
    fn new(track: &ReadTrack) -> Result<Dec> {
        let Format::H264 { avcc, .. } = &track.format else { return Err(Error::Other("not an H.264 track".into())) };
        let (params, len_size) = mp4::avcc_params(avcc).ok_or_else(|| Error::Other("a bad avcC".into()))?;
        let mut vt = VtDecoder::new(Codec::H264);
        vt.set_params(&params);
        Ok(Dec { vt, len_size })
    }

    fn decode(&mut self, sample: &[u8]) -> Result<Option<Decoded>> {
        let annexb = mp4::avcc_to_annexb(sample, self.len_size);
        let mut pics = self.vt.decode(&annexb).map_err(|e| Error::Other(format!("decoding: {e}")))?;
        Ok(pics.pop())
    }
}

/// A label as an RG32Float texture of (Y, weight).
fn label_texture(l: &Label) -> Result<Retained<Texture>> {
    let gpu = Gpu::shared();
    unsafe {
        let desc = MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::RG32Float, l.w, l.h, false);
        desc.setUsage(MTLTextureUsage::ShaderRead);
        desc.setStorageMode(MTLStorageMode::Shared);
        let t = gpu.device.newTextureWithDescriptor(&desc).ok_or_else(|| Error::Other("no label texture".into()))?;
        let region = MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 }, size: MTLSize { width: l.w, height: l.h, depth: 1 } };
        t.replaceRegion_mipmapLevel_withBytes_bytesPerRow(region, 0, NonNull::new(l.data.as_ptr() as *mut _).unwrap(), l.w * 8);
        Ok(t)
    }
}

/// A before & after video of `after` (every frame shown) and `before` (the game's own frames), at after's times,
/// `sound` copied in; written to `out` (through out.part). Returns its length in seconds.
pub fn compose(
    after: &Path,
    before: &Path,
    out: &Path,
    sound: Option<&Path>,
    layout: Layout,
    labels: (&str, &str),
    encoder: Option<&str>,
) -> Result<f64> {
    if let Some(e) = encoder
        && !matches!(e, "h264_videotoolbox" | "videotoolbox") {
            eprintln!("[replay] encoder {e} isn't on a Mac: VideoToolbox's H.264 instead");
        }
    let open = |p: &Path, what: &str| Mp4File::open(p).map_err(|e| Error::Other(format!("{what} ({}): {e}", p.display())));
    let mut fa = open(after, "after")?;
    let mut fb = open(before, "before")?;
    let va = fa.video().cloned().ok_or_else(|| Error::Other("no video in the after track".into()))?;
    let vb = fb.video().cloned().ok_or_else(|| Error::Other("no video in the before track".into()))?;
    let mut sin: Option<(Mp4File, ReadTrack)> = match sound {
        Some(p) => {
            let f = open(p, "sound")?;
            f.audio().cloned().map(|t| (f, t))
        }
        None => None,
    };
    if va.samples.is_empty() {
        return Err(Error::Other("no frames in the clip".into()));
    }
    let tmp = part_path(out);
    let r = autoreleasepool(|_| run(&mut fa, &va, &mut fb, &vb, sin.as_mut(), &tmp, layout, labels));
    match r {
        Ok(secs) => {
            std::fs::rename(&tmp, out)?;
            Ok(secs)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    fa: &mut Mp4File,
    va: &ReadTrack,
    fb: &mut Mp4File,
    vb: &ReadTrack,
    mut sin: Option<&mut (Mp4File, ReadTrack)>,
    tmp: &Path,
    layout: Layout,
    labels: (&str, &str),
) -> Result<f64> {
    let cache = TexCache::new()?;
    let mut da = Dec::new(va)?;
    let mut db = Dec::new(vb)?;
    // The game's frames, each held until the next: (time, picture)
    let mut jb = 0usize;
    let mut next_b = |db: &mut Dec, fb: &mut Mp4File| -> Result<Option<(i64, Decoded)>> {
        while jb < vb.samples.len() {
            let s = vb.samples[jb];
            let t = vb.time90(jb);
            jb += 1;
            if let Some(p) = db.decode(&fb.read(&s)?)? {
                return Ok(Some((t, p)));
            }
        }
        Ok(None)
    };
    let mut cur_b: Option<(i64, Decoded)> = None;
    let mut nxt_b = next_b(&mut db, fb)?;

    struct Out {
        enc: VtEncoder,
        rx: std::sync::mpsc::Receiver<std::result::Result<Encoded, i32>>,
        mp4: Mp4Stream,
        params: ComposeParams,
        lab: [Retained<Texture>; 2],
        base: i64,
        audio: Vec<(i64, usize)>, // (time from base, 90 kHz; sample index) still to write
        a_next: usize,
        format: Option<Format>,
        written: i64,
    }
    let mut o: Option<Out> = None;
    let mut last = 0i64;
    let mut frames = 0usize;
    let mut prof = [0f64; 5]; // decoding after, before; composing; encoding; finishing (SP_REPLAY_PROFILE)
    let gpu = Gpu::shared();

    // Packets that have come out: video in, then the sound up to the last of them
    fn drain(o: &mut Out, sin: &mut Option<&mut (Mp4File, ReadTrack)>, upto: Option<i64>) -> Result<()> {
        while let Ok(r) = o.rx.try_recv() {
            match r {
                Ok(e) => {
                    if o.format.is_none() {
                        o.format = Some((*e.format).clone());
                    }
                    o.mp4.write(0, &e.data, e.pts, 0, e.key)?;
                    o.written = e.pts;
                }
                Err(-1) => {}
                Err(st) => return Err(Error::Status { what: "encoding the before & after", status: st }),
            }
        }
        if let Some(s) = sin.as_deref_mut() {
            let lim = upto.unwrap_or(o.written);
            while o.a_next < o.audio.len() && o.audio[o.a_next].0 <= lim {
                let k = o.audio[o.a_next].1;
                let smp = s.1.samples[k];
                let data = s.0.read(&smp)?;
                o.mp4.write(1, &data, smp.pts + s.1.shift, 0, true)?;
                o.a_next += 1;
            }
        }
        Ok(())
    }

    for (i, s) in va.samples.iter().enumerate() {
        let t = va.time90(i);
        let tp = std::time::Instant::now();
        let Some(fa_pic) = da.decode(&fa.read(s)?)? else { continue };
        prof[0] += tp.elapsed().as_secs_f64();
        if o.is_none() {
            let base = t;
            let (w, h) = (fa_pic.width & !1, fa_pic.height & !1);
            let (hw, ow, oh) = match layout {
                Layout::Side => {
                    let scale = (4096.0 / (2.0 * w as f64)).min(1.0);
                    let hw = ((w as f64 * scale) as usize) & !1;
                    (hw, 2 * hw, ((h as f64 * scale) as usize) & !1)
                }
                Layout::Split => (w, w, h),
            };
            let span = (va.time90(va.samples.len() - 1) - base) as f64 / TB as f64;
            let fps = if span > 0.0 { (va.samples.len() - 1) as f64 / span } else { 60.0 };
            let (tx, rx) = channel();
            let tx = std::sync::Mutex::new(tx);
            let enc = VtEncoder::new(
                VideoConfig { width: ow, height: oh, nominal: AFTER_NOMINAL, bpp: AFTER_BPP, realtime: false, expected_fps: Some(fps) },
                Box::new(move |r| {
                    let _ = tx.lock().unwrap().send(r);
                }),
            )?;
            // The sound from the first frame on
            let mut audio = Vec::new();
            let mut tracks = vec![TrackSpec {
                format: Format::H264 { width: ow as u32, height: oh as u32, avcc: vec![0; 64] },
                timescale: TB as u32,
                start: 0,
            }];
            if let Some(sn) = sin.as_deref() {
                let st = &sn.1;
                for k in 0..st.samples.len() {
                    let at = st.time90(k) - base;
                    if at >= 0 {
                        audio.push((at, k));
                    }
                }
                if let Some(&(_, k)) = audio.first() {
                    let t48 = st.samples[k].pts + st.shift;
                    let base_ts = (base as i128 * st.timescale as i128 / TB as i128) as i64;
                    tracks.push(TrackSpec { format: st.format.clone(), timescale: st.timescale, start: t48 - base_ts });
                }
            }
            let reserve = Mp4Stream::moov_bound(&tracks, va.samples.len() + audio.len());
            let mp4 = Mp4Stream::create(tmp, tracks, reserve)?;
            let m = py_round(24.0 * oh as f64 / 1080.0) as i32 & !1;
            let mid = ((ow / 2) & !1) as i32;
            let (lb, la) = (label(labels.0, oh), label(labels.1, oh));
            let amber = yuv::yuv_of(DIVIDER_BGR);
            let params = ComposeParams {
                amber: [amber[0] as f32, amber[1] as f32, amber[2] as f32, 0.0],
                ow: ow as u32,
                oh: oh as u32,
                layout: (layout == Layout::Side) as u32,
                w: w as u32,
                h: h as u32,
                hw: hw as u32,
                aw: fa_pic.width as u32,
                ah: fa_pic.height as u32,
                bw: 0,
                bh: 0,
                mid: mid as u32,
                lb: [m, m, lb.w as i32, lb.h as i32],
                la: [(mid + m) & !1, m, la.w as i32, la.h as i32],
                has_b: 0,
            };
            let lab = [label_texture(&lb)?, label_texture(&la)?];
            o = Some(Out { enc, rx, mp4, params, lab, base, audio, a_next: 0, format: None, written: -1 });
        }
        let out = o.as_mut().unwrap();
        while let Some((tb, _)) = &nxt_b {
            if *tb > t {
                break;
            }
            cur_b = nxt_b.take();
            let tp = std::time::Instant::now();
            nxt_b = next_b(&mut db, fb)?;
            prof[1] += tp.elapsed().as_secs_f64();
        }
        let fb_pic = cur_b.as_ref().or(nxt_b.as_ref()).map(|x| &x.1);
        // A buffer to compose into (waiting for the encoder if it's that far behind)
        let tp = std::time::Instant::now();
        let buf = loop {
            if let Some(b) = out.enc.pixel_buffer(24) {
                break b;
            }
            drain(out, &mut sin, None)?;
            std::thread::sleep(Duration::from_millis(1));
        };
        let dst = cache.planes(&buf)?;
        let pa = cache.planes(&fa_pic.buffer)?;
        let pb = match fb_pic {
            Some(b) => Some(cache.planes(&b.buffer)?),
            None => None,
        };
        let mut p = out.params;
        p.aw = fa_pic.width as u32;
        p.ah = fa_pic.height as u32;
        if let Some(b) = fb_pic {
            p.bw = b.width as u32;
            p.bh = b.height as u32;
            p.has_b = 1;
        }
        let cb = gpu.queue.commandBuffer().ok_or_else(|| Error::Other("no command buffer".into()))?;
        gpu::encode_compose(&cb, &pa, pb.as_ref().unwrap_or(&pa), [&out.lab[0], &out.lab[1]], &dst, &p)?;
        cb.commit();
        cb.waitUntilCompleted();
        prof[2] += tp.elapsed().as_secs_f64();
        let tp = std::time::Instant::now();
        drop((dst, pa, pb));
        last = t - out.base;
        out.enc.encode(&buf, last, false)?;
        frames += 1;
        drain(out, &mut sin, None)?;
        prof[3] += tp.elapsed().as_secs_f64();
        if frames.is_multiple_of(64) {
            cache.flush();
        }
    }
    let Some(mut out) = o else { return Err(Error::Other("no frames in the clip".into())) };
    let tp = std::time::Instant::now();
    out.enc.complete();
    prof[4] += tp.elapsed().as_secs_f64();
    if std::env::var_os("SP_REPLAY_PROFILE").is_some() {
        eprintln!(
            "[replay] compose of {frames} frames: decoding after {:.2} s, before {:.2} s, composing {:.2} s, encoding {:.2} s, finishing {:.2} s",
            prof[0], prof[1], prof[2], prof[3], prof[4]
        );
    }
    drain(&mut out, &mut sin, Some(last))?;
    let format = out.format.take().ok_or_else(|| Error::Other("the encoder gave nothing".into()))?;
    let Out { enc, mut mp4, .. } = out;
    drop(enc);
    mp4.set_format(0, format);
    mp4.finish()?;
    Ok(last as f64 / TB as f64)
}
