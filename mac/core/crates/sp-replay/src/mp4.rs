//! MP4 files of what instant replay keeps (H.264 with its avcC, AAC with its AudioSpecificConfig): written moov first
//! (replay.py's `movflags +faststart`), and read back (the before & after process reads the tracks it's given).
//!
//! Two ways to write: [`write_file`] when every sample is in memory (a saved clip: the moov is worked out first, so
//! it goes before the data), and [`Mp4Stream`] for a video being encoded (the before & after: room for the moov is
//! kept at the front and filled in at the end, the rest left as a `free` box).
//!
//! A track can start later than the movie (an empty edit) or earlier (its first part edited out), so tracks cut
//! from one timeline keep their places on it.

use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::track::Format;

/// The movie's timescale (mvhd, tkhd and edit lists): the replay's 90 kHz clock.
pub const MOVIE_TIMESCALE: u32 = 90_000;

/// A track to write.
#[derive(Clone, Debug)]
pub struct TrackSpec {
    pub format: Format,
    /// Ticks a second of its timestamps (90 000 for video, the sample rate for sound).
    pub timescale: u32,
    /// Where its first sample is shown on the movie's timeline, in its timescale: positive, after an empty edit;
    /// negative, that much of its start edited out.
    pub start: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    /// Decode time, in the track's timescale (any origin: the first sample's is the track's zero).
    pub dts: i64,
    /// Composition offset (pts - dts).
    pub cto: i32,
    pub size: u32,
    pub key: bool,
}

#[derive(Clone, Copy, Debug)]
struct Chunk {
    offset: u64,
    count: u32,
}

struct Build {
    spec: TrackSpec,
    samples: Vec<Sample>,
    chunks: Vec<Chunk>,
}

// ---- box writing ------------------------------------------------------------------------------------------------
struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }
    fn zeros(&mut self, n: usize) {
        self.0.resize(self.0.len() + n, 0);
    }
    fn boxed(&mut self, typ: &[u8; 4], f: impl FnOnce(&mut W)) {
        let at = self.0.len();
        self.u32(0);
        self.bytes(typ);
        f(self);
        let size = (self.0.len() - at) as u32;
        self.0[at..at + 4].copy_from_slice(&size.to_be_bytes());
    }
    fn full(&mut self, typ: &[u8; 4], version: u8, flags: u32, f: impl FnOnce(&mut W)) {
        self.boxed(typ, |w| {
            w.u32((version as u32) << 24 | (flags & 0xff_ffff));
            f(w);
        });
    }
    fn matrix(&mut self) {
        for v in [0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
            self.u32(v);
        }
    }
}

fn rescale(v: i64, from: u32, to: u32) -> i64 {
    ((v as i128 * to as i128 + from as i128 / 2) / from as i128) as i64
}

impl Build {
    /// The track's samples' durations: to the next one's dts, the last as the one before (or `fallback`).
    fn durations(&self) -> Vec<u32> {
        let n = self.samples.len();
        let mut d: Vec<u32> = (0..n.saturating_sub(1))
            .map(|i| (self.samples[i + 1].dts - self.samples[i].dts).clamp(0, u32::MAX as i64) as u32)
            .collect();
        let fallback = match &self.spec.format {
            Format::Aac { .. } => 1024,
            Format::H264 { .. } => (self.spec.timescale / 60).max(1),
        };
        if n > 0 {
            d.push(d.last().copied().filter(|&x| x > 0).unwrap_or(fallback));
        }
        d
    }

    fn media_duration(&self) -> i64 {
        self.durations().iter().map(|&d| d as i64).sum()
    }

    /// On the movie's timeline: (empty edit, shown duration), in the movie's timescale.
    fn edit(&self) -> (i64, i64) {
        let ts = self.spec.timescale;
        let media = self.media_duration();
        let start = self.spec.start;
        if start >= 0 {
            (rescale(start, ts, MOVIE_TIMESCALE), rescale(media, ts, MOVIE_TIMESCALE))
        } else {
            (0, rescale((media + start).max(0), ts, MOVIE_TIMESCALE))
        }
    }
}

fn ftyp() -> Vec<u8> {
    let mut w = W(Vec::new());
    w.boxed(b"ftyp", |w| {
        w.bytes(b"isom");
        w.u32(0x200);
        for b in [b"isom", b"iso2", b"avc1", b"mp41"] {
            w.bytes(b);
        }
    });
    w.0
}

fn descriptor(tag: u8, payload: &[u8]) -> Vec<u8> {
    assert!(payload.len() < 1 << 28);
    let n = payload.len() as u32;
    let mut out = vec![tag, 0x80 | ((n >> 21) & 0x7f) as u8, 0x80 | ((n >> 14) & 0x7f) as u8, 0x80 | ((n >> 7) & 0x7f) as u8, (n & 0x7f) as u8];
    out.extend_from_slice(payload);
    out
}

fn sample_entry(w: &mut W, format: &Format) {
    match format {
        Format::H264 { width, height, avcc } => w.boxed(b"avc1", |w| {
            w.zeros(6);
            w.u16(1); // data reference index
            w.zeros(16);
            w.u16(*width as u16);
            w.u16(*height as u16);
            w.u32(0x0048_0000);
            w.u32(0x0048_0000);
            w.u32(0);
            w.u16(1); // frame count
            let mut name = [0u8; 32];
            let n = b"SmoovPlay";
            name[0] = n.len() as u8;
            name[1..1 + n.len()].copy_from_slice(n);
            w.bytes(&name);
            w.u16(0x18);
            w.u16(0xffff);
            w.boxed(b"avcC", |w| w.bytes(avcc));
            // BT.709, limited range (the picture's own VUI says so too)
            w.boxed(b"colr", |w| {
                w.bytes(b"nclx");
                w.u16(1);
                w.u16(1);
                w.u16(1);
                w.u8(0);
            });
            w.boxed(b"pasp", |w| {
                w.u32(1);
                w.u32(1);
            });
        }),
        Format::Aac { rate, channels, asc, bitrate } => w.boxed(b"mp4a", |w| {
            w.zeros(6);
            w.u16(1);
            w.zeros(8);
            w.u16(*channels as u16);
            w.u16(16);
            w.u16(0);
            w.u16(0);
            w.u32(rate << 16);
            w.full(b"esds", 0, 0, |w| {
                let mut dc = vec![0x40, 0x15, 0, 0x18, 0]; // MPEG-4 audio, audio stream; 6 KB buffer
                dc.extend_from_slice(&(bitrate * 3 / 2).to_be_bytes());
                dc.extend_from_slice(&bitrate.to_be_bytes());
                dc.extend_from_slice(&descriptor(0x05, asc));
                let mut es = vec![0, 0, 0]; // ES_ID 0, no flags
                es.extend_from_slice(&descriptor(0x04, &dc));
                es.extend_from_slice(&descriptor(0x06, &[0x02]));
                w.bytes(&descriptor(0x03, &es));
            });
        }),
    }
}

/// The moov for these tracks, chunk offsets moved on by `shift`.
fn moov(tracks: &[Build], shift: u64) -> Vec<u8> {
    let movie_dur = tracks.iter().map(|t| t.edit().0 + t.edit().1).max().unwrap_or(0);
    let mut w = W(Vec::new());
    w.boxed(b"moov", |w| {
        let v1 = movie_dur > u32::MAX as i64;
        w.full(b"mvhd", v1 as u8, 0, |w| {
            if v1 {
                w.u64(0);
                w.u64(0);
                w.u32(MOVIE_TIMESCALE);
                w.u64(movie_dur as u64);
            } else {
                w.u32(0);
                w.u32(0);
                w.u32(MOVIE_TIMESCALE);
                w.u32(movie_dur as u32);
            }
            w.u32(0x0001_0000);
            w.u16(0x0100);
            w.zeros(10);
            w.matrix();
            w.zeros(24);
            w.u32(tracks.len() as u32 + 1);
        });
        for (i, t) in tracks.iter().enumerate() {
            trak(w, t, i as u32 + 1, shift);
        }
    });
    w.0
}

fn trak(w: &mut W, t: &Build, id: u32, shift: u64) {
    let (empty, shown) = t.edit();
    let video = matches!(t.spec.format, Format::H264 { .. });
    let durations = t.durations();
    let media: i64 = durations.iter().map(|&d| d as i64).sum();
    w.boxed(b"trak", |w| {
        let total = empty + shown;
        let v1 = total > u32::MAX as i64;
        w.full(b"tkhd", v1 as u8, 3, |w| {
            if v1 {
                w.u64(0);
                w.u64(0);
                w.u32(id);
                w.u32(0);
                w.u64(total as u64);
            } else {
                w.u32(0);
                w.u32(0);
                w.u32(id);
                w.u32(0);
                w.u32(total as u32);
            }
            w.zeros(8);
            w.u16(0);
            w.u16(0);
            w.u16(if video { 0 } else { 0x0100 });
            w.u16(0);
            w.matrix();
            let (tw, th) = match &t.spec.format {
                Format::H264 { width, height, .. } => (*width, *height),
                _ => (0, 0),
            };
            w.u32(tw << 16);
            w.u32(th << 16);
        });
        if t.spec.start != 0 {
            w.boxed(b"edts", |w| {
                w.full(b"elst", 1, 0, |w| {
                    if t.spec.start > 0 {
                        w.u32(2);
                        w.u64(empty as u64);
                        w.i64(-1);
                        w.u32(0x0001_0000);
                        w.u64(shown as u64);
                        w.i64(0);
                        w.u32(0x0001_0000);
                    } else {
                        w.u32(1);
                        w.u64(shown as u64);
                        w.i64(-t.spec.start);
                        w.u32(0x0001_0000);
                    }
                });
            });
        }
        w.boxed(b"mdia", |w| {
            let v1 = media > u32::MAX as i64;
            w.full(b"mdhd", v1 as u8, 0, |w| {
                if v1 {
                    w.u64(0);
                    w.u64(0);
                    w.u32(t.spec.timescale);
                    w.u64(media as u64);
                } else {
                    w.u32(0);
                    w.u32(0);
                    w.u32(t.spec.timescale);
                    w.u32(media as u32);
                }
                w.u16(0x55c4); // "und"
                w.u16(0);
            });
            w.full(b"hdlr", 0, 0, |w| {
                w.u32(0);
                w.bytes(if video { b"vide" } else { b"soun" });
                w.zeros(12);
                w.bytes(if video { b"VideoHandler\0" } else { b"SoundHandler\0" });
            });
            w.boxed(b"minf", |w| {
                if video {
                    w.full(b"vmhd", 0, 1, |w| w.zeros(8));
                } else {
                    w.full(b"smhd", 0, 0, |w| w.zeros(4));
                }
                w.boxed(b"dinf", |w| {
                    w.full(b"dref", 0, 0, |w| {
                        w.u32(1);
                        w.full(b"url ", 0, 1, |_| {});
                    });
                });
                w.boxed(b"stbl", |w| stbl(w, t, &durations, shift));
            });
        });
    });
}

fn stbl(w: &mut W, t: &Build, durations: &[u32], shift: u64) {
    w.full(b"stsd", 0, 0, |w| {
        w.u32(1);
        sample_entry(w, &t.spec.format);
    });
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &d in durations {
        match runs.last_mut() {
            Some((n, v)) if *v == d => *n += 1,
            _ => runs.push((1, d)),
        }
    }
    w.full(b"stts", 0, 0, |w| {
        w.u32(runs.len() as u32);
        for (n, d) in &runs {
            w.u32(*n);
            w.u32(*d);
        }
    });
    if t.samples.iter().any(|s| s.cto != 0) {
        let mut runs: Vec<(u32, i32)> = Vec::new();
        for s in &t.samples {
            match runs.last_mut() {
                Some((n, v)) if *v == s.cto => *n += 1,
                _ => runs.push((1, s.cto)),
            }
        }
        w.full(b"ctts", 1, 0, |w| {
            w.u32(runs.len() as u32);
            for (n, o) in &runs {
                w.u32(*n);
                w.i32(*o);
            }
        });
    }
    if !t.samples.iter().all(|s| s.key) {
        let keys: Vec<u32> = t.samples.iter().enumerate().filter(|(_, s)| s.key).map(|(i, _)| i as u32 + 1).collect();
        w.full(b"stss", 0, 0, |w| {
            w.u32(keys.len() as u32);
            for k in keys {
                w.u32(k);
            }
        });
    }
    let mut stsc: Vec<(u32, u32)> = Vec::new();
    for (i, c) in t.chunks.iter().enumerate() {
        if stsc.last().is_none_or(|&(_, n)| n != c.count) {
            stsc.push((i as u32 + 1, c.count));
        }
    }
    w.full(b"stsc", 0, 0, |w| {
        w.u32(stsc.len() as u32);
        for (first, n) in &stsc {
            w.u32(*first);
            w.u32(*n);
            w.u32(1);
        }
    });
    w.full(b"stsz", 0, 0, |w| {
        w.u32(0);
        w.u32(t.samples.len() as u32);
        for s in &t.samples {
            w.u32(s.size);
        }
    });
    w.full(b"co64", 0, 0, |w| {
        w.u32(t.chunks.len() as u32);
        for c in &t.chunks {
            w.u64(c.offset + shift);
        }
    });
}

fn normalise(samples: &mut [Sample]) {
    if let Some(first) = samples.first().map(|s| s.dts) {
        for s in samples.iter_mut() {
            s.dts -= first;
        }
    }
}

/// A track and its samples (each with its bytes), in decode order.
pub type TrackSamples<'a> = (TrackSpec, Vec<(Sample, &'a [u8])>);

/// Writes an MP4 of these tracks, moov first, the tracks interleaved by time. Writes `path` directly; returns its
/// size.
pub fn write_file(path: &Path, tracks: Vec<TrackSamples>) -> io::Result<u64> {
    // Interleave by time: each sample's dts on the movie's timeline, in seconds
    let at = |t: &TrackSpec, s: &Sample, first: i64| (s.dts - first + t.start) as f64 / t.timescale as f64;
    let mut idx = vec![0usize; tracks.len()];
    let firsts: Vec<i64> = tracks.iter().map(|(_, s)| s.first().map(|x| x.0.dts).unwrap_or(0)).collect();
    let mut order: Vec<(usize, usize)> = Vec::new();
    loop {
        let mut best: Option<(usize, f64)> = None;
        for (k, (spec, samples)) in tracks.iter().enumerate() {
            if let Some((s, _)) = samples.get(idx[k]) {
                let t = at(spec, s, firsts[k]);
                if best.is_none_or(|(_, b)| t < b) {
                    best = Some((k, t));
                }
            }
        }
        let Some((k, _)) = best else { break };
        order.push((k, idx[k]));
        idx[k] += 1;
    }
    let mut builds: Vec<Build> = tracks
        .iter()
        .map(|(spec, samples)| {
            let mut s: Vec<Sample> = samples.iter().map(|x| x.0).collect();
            normalise(&mut s);
            Build { spec: spec.clone(), samples: s, chunks: Vec::new() }
        })
        .collect();
    let mut off = 0u64;
    let mut last: Option<usize> = None;
    for &(k, i) in &order {
        let size = tracks[k].1[i].0.size as u64;
        if last != Some(k) {
            builds[k].chunks.push(Chunk { offset: off, count: 0 });
        }
        builds[k].chunks.last_mut().unwrap().count += 1;
        off += size;
        last = Some(k);
    }
    let head = ftyp();
    let moov_len = moov(&builds, 0).len() as u64;
    let data_at = head.len() as u64 + moov_len + 16;
    let m = moov(&builds, data_at);
    debug_assert_eq!(m.len() as u64, moov_len);
    let mut f = BufWriter::with_capacity(1 << 20, File::create(path)?);
    f.write_all(&head)?;
    f.write_all(&m)?;
    f.write_all(&1u32.to_be_bytes())?;
    f.write_all(b"mdat")?;
    f.write_all(&(off + 16).to_be_bytes())?;
    for &(k, i) in &order {
        f.write_all(tracks[k].1[i].1)?;
    }
    f.flush()?;
    Ok(data_at + off)
}

/// An MP4 written as its samples come (the before & after video): the moov goes in room kept at the front.
pub struct Mp4Stream {
    f: BufWriter<File>,
    builds: Vec<Build>,
    pos: u64,
    reserve_at: u64,
    reserve: u64,
    mdat_at: u64,
    last: Option<usize>,
}

impl Mp4Stream {
    /// Room for the moov: enough for this many samples (every one its own chunk, no two durations alike).
    pub fn moov_bound(tracks: &[TrackSpec], samples: usize) -> u64 {
        let per_track: u64 = tracks
            .iter()
            .map(|t| {
                1024 + match &t.format {
                    Format::H264 { avcc, .. } => avcc.len() as u64,
                    Format::Aac { asc, .. } => asc.len() as u64,
                }
            })
            .sum();
        4096 + per_track + samples as u64 * (4 + 8 + 4 + 8 + 12)
    }

    pub fn create(path: &Path, tracks: Vec<TrackSpec>, reserve: u64) -> io::Result<Mp4Stream> {
        let mut f = BufWriter::with_capacity(1 << 20, File::create(path)?);
        let head = ftyp();
        f.write_all(&head)?;
        let reserve_at = head.len() as u64;
        let reserve = if reserve > 0 { reserve.max(8) } else { 0 };
        if reserve > 0 {
            f.write_all(&(reserve as u32).to_be_bytes())?;
            f.write_all(b"free")?;
            io::copy(&mut io::repeat(0).take(reserve - 8), &mut f)?;
        }
        let mdat_at = reserve_at + reserve;
        f.write_all(&1u32.to_be_bytes())?;
        f.write_all(b"mdat")?;
        f.write_all(&0u64.to_be_bytes())?;
        let builds = tracks.into_iter().map(|spec| Build { spec, samples: Vec::new(), chunks: Vec::new() }).collect();
        Ok(Mp4Stream { f, builds, pos: mdat_at + 16, reserve_at, reserve, mdat_at, last: None })
    }

    /// The track's format, when it's only known once its first packet is out (an encoder's avcC).
    pub fn set_format(&mut self, track: usize, format: Format) {
        self.builds[track].spec.format = format;
    }

    pub fn samples(&self, track: usize) -> usize {
        self.builds[track].samples.len()
    }

    pub fn write(&mut self, track: usize, data: &[u8], dts: i64, cto: i32, key: bool) -> io::Result<()> {
        let b = &mut self.builds[track];
        if self.last != Some(track) {
            b.chunks.push(Chunk { offset: self.pos, count: 0 });
        }
        b.chunks.last_mut().unwrap().count += 1;
        b.samples.push(Sample { dts, cto, size: data.len() as u32, key });
        self.f.write_all(data)?;
        self.pos += data.len() as u64;
        self.last = Some(track);
        Ok(())
    }

    /// Fills in the moov (at the front if it fits, else at the end) and closes the file.
    pub fn finish(mut self) -> io::Result<()> {
        for b in &mut self.builds {
            normalise(&mut b.samples);
        }
        let m = moov(&self.builds, 0);
        let mut f = self.f.into_inner().map_err(|e| e.into_error())?;
        f.seek(SeekFrom::Start(self.mdat_at + 8))?;
        f.write_all(&(self.pos - self.mdat_at).to_be_bytes())?;
        let len = m.len() as u64;
        if len == self.reserve || len + 8 <= self.reserve {
            f.seek(SeekFrom::Start(self.reserve_at))?;
            f.write_all(&m)?;
            if len < self.reserve {
                f.write_all(&((self.reserve - len) as u32).to_be_bytes())?;
                f.write_all(b"free")?;
            }
        } else {
            f.seek(SeekFrom::End(0))?;
            f.write_all(&m)?;
        }
        f.flush()
    }
}

// ---- reading -----------------------------------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct ReadSample {
    pub offset: u64,
    pub size: u32,
    pub dts: i64,
    pub pts: i64,
    pub key: bool,
}

#[derive(Clone, Debug)]
pub struct ReadTrack {
    pub format: Format,
    pub timescale: u32,
    pub samples: Vec<ReadSample>,
    /// Added to a sample's pts gives when it's shown on the movie's timeline (the edit list), in the track's
    /// timescale.
    pub shift: i64,
}

impl ReadTrack {
    pub fn is_video(&self) -> bool {
        matches!(self.format, Format::H264 { .. })
    }
    /// When sample `i` is shown, in seconds on the movie's timeline.
    pub fn time(&self, i: usize) -> f64 {
        (self.samples[i].pts + self.shift) as f64 / self.timescale as f64
    }
    /// The same on the 90 kHz clock.
    pub fn time90(&self, i: usize) -> i64 {
        rescale(self.samples[i].pts + self.shift, self.timescale, crate::TB as u32)
    }
}

pub struct Mp4File {
    file: File,
    pub movie_timescale: u32,
    pub tracks: Vec<ReadTrack>,
}

fn bad(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("not an MP4 this can read: {what}"))
}

struct R<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> R<'a> {
    fn new(b: &'a [u8]) -> R<'a> {
        R { b, i: 0 }
    }
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let s = self.b.get(self.i..self.i + n).ok_or_else(|| bad("truncated box"))?;
        self.i += n;
        Ok(s)
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> io::Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn left(&self) -> usize {
        self.b.len() - self.i
    }
}

/// The boxes in `b`: (type, payload).
fn boxes(b: &[u8]) -> io::Result<Vec<([u8; 4], &[u8])>> {
    let mut out = Vec::new();
    let mut r = R::new(b);
    while r.left() >= 8 {
        let size = r.u32()? as u64;
        let typ: [u8; 4] = r.take(4)?.try_into().unwrap();
        let (head, size) = match size {
            1 => (16, r.u64()?),
            0 => (8, r.left() as u64 + 8),
            s => (8, s),
        };
        if size < head {
            return Err(bad("box size"));
        }
        out.push((typ, r.take((size - head) as usize)?));
    }
    Ok(out)
}

fn child<'a>(b: &'a [u8], typ: &[u8; 4]) -> io::Result<Option<&'a [u8]>> {
    Ok(boxes(b)?.into_iter().find(|(t, _)| t == typ).map(|(_, p)| p))
}

fn need<'a>(b: &'a [u8], typ: &[u8; 4]) -> io::Result<&'a [u8]> {
    child(b, typ)?.ok_or_else(|| bad(&format!("no {} box", String::from_utf8_lossy(typ))))
}

/// The payload of a descriptor's length, from an esds.
fn descriptor_len(r: &mut R) -> io::Result<usize> {
    let mut n = 0usize;
    for _ in 0..4 {
        let b = r.u8()?;
        n = n << 7 | (b & 0x7f) as usize;
        if b & 0x80 == 0 {
            break;
        }
    }
    Ok(n)
}

fn asc_of_esds(esds: &[u8]) -> io::Result<Vec<u8>> {
    // Walk the descriptors (ES, DecoderConfig) down to DecoderSpecificInfo (tag 5).
    let mut r = R::new(&esds[4..]);
    while r.left() > 2 {
        let tag = r.u8()?;
        let n = descriptor_len(&mut r)?;
        match tag {
            0x03 => {
                let _id = r.u16()?;
                let flags = r.u8()?;
                if flags & 0x80 != 0 {
                    r.take(2)?;
                }
                if flags & 0x40 != 0 {
                    let k = r.u8()? as usize;
                    r.take(k)?;
                }
                if flags & 0x20 != 0 {
                    r.take(2)?;
                }
            }
            0x04 => {
                r.take(13)?;
            }
            0x05 => return Ok(r.take(n)?.to_vec()),
            _ => {
                r.take(n)?;
            }
        }
    }
    Err(bad("no AudioSpecificConfig"))
}

fn read_track(trak: &[u8], movie_ts: u32) -> io::Result<Option<ReadTrack>> {
    let mdia = need(trak, b"mdia")?;
    let hdlr = need(mdia, b"hdlr")?;
    let handler: [u8; 4] = hdlr.get(8..12).ok_or_else(|| bad("hdlr"))?.try_into().unwrap();
    let mdhd = need(mdia, b"mdhd")?;
    let timescale = if mdhd[0] == 1 { u32::from_be_bytes(mdhd[20..24].try_into().unwrap()) } else { u32::from_be_bytes(mdhd[12..16].try_into().unwrap()) };
    let stbl = need(need(mdia, b"minf")?, b"stbl")?;
    let stsd = need(stbl, b"stsd")?;
    let entries = boxes(&stsd[8..])?;
    let Some((etype, entry)) = entries.first() else { return Ok(None) };
    let format = match (&handler, etype) {
        (b"vide", b"avc1") | (b"vide", b"avc3") => {
            let width = u16::from_be_bytes(entry[24..26].try_into().unwrap()) as u32;
            let height = u16::from_be_bytes(entry[26..28].try_into().unwrap()) as u32;
            let avcc = need(&entry[78..], b"avcC")?.to_vec();
            Format::H264 { width, height, avcc }
        }
        (b"soun", b"mp4a") => {
            let channels = u16::from_be_bytes(entry[16..18].try_into().unwrap()) as u32;
            let rate = u32::from_be_bytes(entry[24..28].try_into().unwrap()) >> 16;
            let asc = asc_of_esds(need(&entry[28..], b"esds")?)?;
            Format::Aac { rate, channels, asc, bitrate: 0 }
        }
        _ => return Ok(None),
    };
    // Timing
    let mut durs: Vec<u32> = Vec::new();
    let mut r = R::new(&need(stbl, b"stts")?[4..]);
    for _ in 0..r.u32()? {
        let (n, d) = (r.u32()?, r.u32()?);
        durs.extend(std::iter::repeat_n(d, n as usize));
    }
    let n = durs.len();
    let mut ctos = vec![0i64; n];
    if let Some(ctts) = child(stbl, b"ctts")? {
        let v1 = ctts[0] == 1;
        let mut r = R::new(&ctts[4..]);
        let mut i = 0;
        for _ in 0..r.u32()? {
            let c = r.u32()?;
            let o = if v1 { r.i32()? as i64 } else { r.u32()? as i64 };
            for _ in 0..c {
                if i < n {
                    ctos[i] = o;
                }
                i += 1;
            }
        }
    }
    let mut keys = vec![true; n];
    if let Some(stss) = child(stbl, b"stss")? {
        keys = vec![false; n];
        let mut r = R::new(&stss[4..]);
        for _ in 0..r.u32()? {
            let k = r.u32()? as usize;
            if k >= 1 && k <= n {
                keys[k - 1] = true;
            }
        }
    }
    let stsz = need(stbl, b"stsz")?;
    let mut r = R::new(&stsz[4..]);
    let (fixed, count) = (r.u32()?, r.u32()? as usize);
    let sizes: Vec<u32> = if fixed != 0 { vec![fixed; count] } else { (0..count).map(|_| r.u32()).collect::<io::Result<_>>()? };
    let offsets: Vec<u64> = if let Some(co64) = child(stbl, b"co64")? {
        let mut r = R::new(&co64[4..]);
        (0..r.u32()?).map(|_| r.u64()).collect::<io::Result<_>>()?
    } else {
        let mut r = R::new(&need(stbl, b"stco")?[4..]);
        (0..r.u32()?).map(|_| r.u32().map(|v| v as u64)).collect::<io::Result<_>>()?
    };
    let mut stsc: Vec<(u32, u32)> = Vec::new();
    let mut r = R::new(&need(stbl, b"stsc")?[4..]);
    for _ in 0..r.u32()? {
        let (first, per, _sdi) = (r.u32()?, r.u32()?, r.u32()?);
        stsc.push((first, per));
    }
    let mut samples = Vec::with_capacity(n);
    let (mut dts, mut si) = (0i64, 0usize);
    for (ci, &off) in offsets.iter().enumerate() {
        let chunk = ci as u32 + 1;
        let per = stsc.iter().rev().find(|(f, _)| *f <= chunk).map(|x| x.1).unwrap_or(0);
        let mut o = off;
        for _ in 0..per {
            if si >= n.min(sizes.len()) {
                break;
            }
            samples.push(ReadSample { offset: o, size: sizes[si], dts, pts: dts + ctos[si], key: keys[si] });
            o += sizes[si] as u64;
            dts += durs[si] as i64;
            si += 1;
        }
    }
    // The edit list: empty edits put it later; the first real one's media time edits its start out
    let mut shift = 0i64;
    if let Some(edts) = child(trak, b"edts")?
        && let Some(elst) = child(edts, b"elst")? {
            let v1 = elst[0] == 1;
            let mut r = R::new(&elst[4..]);
            let mut empty = 0i64;
            for _ in 0..r.u32()? {
                let (d, t) = if v1 { (r.u64()? as i64, r.u64()? as i64) } else { (r.u32()? as i64, r.i32()? as i64) };
                r.u32()?;
                if t == -1 {
                    empty += d;
                } else {
                    shift = rescale(empty, movie_ts, timescale) - t;
                    break;
                }
            }
        }
    Ok(Some(ReadTrack { format, timescale, samples, shift }))
}

impl Mp4File {
    pub fn open(path: &Path) -> io::Result<Mp4File> {
        let mut file = File::open(path)?;
        let len = file.metadata()?.len();
        let mut pos = 0u64;
        let mut moov: Option<Vec<u8>> = None;
        while pos + 8 <= len {
            file.seek(SeekFrom::Start(pos))?;
            let mut h = [0u8; 16];
            file.read_exact(&mut h[..8])?;
            let mut size = u32::from_be_bytes(h[..4].try_into().unwrap()) as u64;
            let mut head = 8;
            if size == 1 {
                file.read_exact(&mut h[8..16])?;
                size = u64::from_be_bytes(h[8..16].try_into().unwrap());
                head = 16;
            } else if size == 0 {
                size = len - pos;
            }
            if size < head {
                return Err(bad("box size"));
            }
            if &h[4..8] == b"moov" {
                let mut b = vec![0u8; (size - head) as usize];
                file.read_exact(&mut b)?;
                moov = Some(b);
                break;
            }
            pos += size;
        }
        let moov = moov.ok_or_else(|| bad("no moov"))?;
        let mvhd = need(&moov, b"mvhd")?;
        let movie_timescale =
            if mvhd[0] == 1 { u32::from_be_bytes(mvhd[20..24].try_into().unwrap()) } else { u32::from_be_bytes(mvhd[12..16].try_into().unwrap()) };
        let mut tracks = Vec::new();
        for (t, p) in boxes(&moov)? {
            if &t == b"trak"
                && let Some(tr) = read_track(p, movie_timescale)? {
                    tracks.push(tr);
                }
        }
        Ok(Mp4File { file, movie_timescale, tracks })
    }

    pub fn video(&self) -> Option<&ReadTrack> {
        self.tracks.iter().find(|t| t.is_video())
    }

    pub fn audio(&self) -> Option<&ReadTrack> {
        self.tracks.iter().find(|t| !t.is_video())
    }

    /// A sample's bytes.
    pub fn read(&mut self, s: &ReadSample) -> io::Result<Vec<u8>> {
        let mut b = vec![0u8; s.size as usize];
        self.file.seek(SeekFrom::Start(s.offset))?;
        self.file.read_exact(&mut b)?;
        Ok(b)
    }
}

/// The SPS and PPS in an avcC, as Annex-B (start codes), and the NAL length size.
pub fn avcc_params(avcc: &[u8]) -> Option<(Vec<u8>, usize)> {
    let mut r = R::new(avcc);
    r.take(4).ok()?;
    let len_size = (r.u8().ok()? & 3) as usize + 1;
    let mut out = Vec::new();
    let nsps = r.u8().ok()? & 0x1f;
    for _ in 0..nsps {
        let n = r.u16().ok()? as usize;
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(r.take(n).ok()?);
    }
    let npps = r.u8().ok()?;
    for _ in 0..npps {
        let n = r.u16().ok()? as usize;
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(r.take(n).ok()?);
    }
    Some((out, len_size))
}

/// An AVCC sample (length-prefixed NAL units) as Annex-B.
pub fn avcc_to_annexb(sample: &[u8], len_size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(sample.len() + 16);
    let mut i = 0;
    while i + len_size <= sample.len() {
        let mut n = 0usize;
        for k in 0..len_size {
            n = n << 8 | sample[i + k] as usize;
        }
        i += len_size;
        let end = (i + n).min(sample.len());
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&sample[i..end]);
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_and_reads_back() {
        let dir = std::env::temp_dir().join(format!("sp-replay-mp4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let avcc = vec![1, 0x64, 0, 0x28, 0xff, 0xe1, 0, 4, 0x67, 1, 2, 3, 1, 0, 2, 0x68, 9];
        let v = TrackSpec { format: Format::H264 { width: 64, height: 32, avcc: avcc.clone() }, timescale: 90_000, start: 0 };
        let a = TrackSpec {
            format: Format::Aac { rate: 48_000, channels: 2, asc: vec![0x11, 0x90], bitrate: 160_000 },
            timescale: 48_000,
            start: 4800,
        };
        let data: Vec<Vec<u8>> = (0..40).map(|i| vec![i as u8; 10 + i]).collect();
        let vs: Vec<(Sample, &[u8])> = (0..20)
            .map(|i| (Sample { dts: 1000 + i as i64 * 1500, cto: 0, size: data[i].len() as u32, key: i % 10 == 0 }, &data[i][..]))
            .collect();
        let as_: Vec<(Sample, &[u8])> = (20..40)
            .map(|i| (Sample { dts: (i as i64 - 20) * 1024, cto: 0, size: data[i].len() as u32, key: true }, &data[i][..]))
            .collect();
        let path = dir.join("t.mp4");
        write_file(&path, vec![(v.clone(), vs), (a, as_)]).unwrap();
        let mut f = Mp4File::open(&path).unwrap();
        let vt = f.video().unwrap().clone();
        assert_eq!(vt.samples.len(), 20);
        assert_eq!(vt.format, v.format);
        assert!(vt.samples[0].key && !vt.samples[1].key && vt.samples[10].key);
        assert_eq!(vt.time90(1), 1500);
        let at = f.audio().unwrap().clone();
        assert_eq!(at.time(0), 0.1);
        assert_eq!(f.read(&at.samples[3]).unwrap(), data[23]);
        assert_eq!(f.read(&vt.samples[7]).unwrap(), data[7]);

        // The streaming writer, its moov at the front
        let path2 = dir.join("s.mp4");
        let mut s = Mp4Stream::create(&path2, vec![v], Mp4Stream::moov_bound(&[], 20)).unwrap();
        for (i, d) in data.iter().take(20).enumerate() {
            s.write(0, d, i as i64 * 1500, 0, i == 0).unwrap();
        }
        s.finish().unwrap();
        let mut f = Mp4File::open(&path2).unwrap();
        let vt = f.video().unwrap().clone();
        assert_eq!(vt.samples.len(), 20);
        assert_eq!(f.read(&vt.samples[19]).unwrap(), data[19]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
