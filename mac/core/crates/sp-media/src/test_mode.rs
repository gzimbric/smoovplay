//! "Simulate a 30 fps game" for any 60 Hz source (framegen/sources.py TestMode). Frames 0, 2, 4 and so on arrive as
//! new content, and each odd frame is replaced by the one before it (A A B B). The frames left out are kept to score
//! the generated ones against. It's a filter that turns a tick into a tick or nothing. A source with a reader thread
//! applies it there, before its queue, so a frame dropped from a full queue never shifts which frames it keeps.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use sp_core::{Event, Frame, Source, SourceInfo, Sound, Tick};

pub struct TestMode {
    pub on: bool,
    n: i64,
    last: Option<Frame>,
    truth: HashMap<i64, Frame>,
    /// What the source said about each recent frame: Some(true) new, Some(false) a repeat, None unknown.
    game: HashMap<i64, Option<bool>>,
}

/// A TestMode shared between a source's reader thread (which applies it) and the pipeline (which asks for truth and
/// switches it on and off).
pub type SharedTestMode = Arc<Mutex<TestMode>>;

impl TestMode {
    /// Left-out frames kept for scoring. The pipeline asks within a frame or two.
    pub const KEEP: i64 = 8;

    pub fn new(on: bool) -> TestMode {
        TestMode { on, n: 0, last: None, truth: HashMap::new(), game: HashMap::new() }
    }

    pub fn shared(on: bool) -> SharedTestMode {
        Arc::new(Mutex::new(TestMode::new(on)))
    }

    /// Takes effect from the next frame.
    pub fn set_on(&mut self, on: bool) {
        if on && !self.on {
            self.last = None; // don't repeat a frame from before it was switched on
            self.truth.clear();
            self.game.clear();
        }
        self.on = on;
    }

    /// The real frame left out at `content`, or None. It's None too when the game repeated a frame there or at the
    /// kept frames either side, because below 60 fps the left-out frame isn't the real one halfway between them.
    pub fn gt(&self, content: i64) -> Option<Frame> {
        if (content - 1..=content + 1).any(|i| self.game.get(&i) == Some(&Some(false))) {
            return None;
        }
        self.truth.get(&content).cloned()
    }

    pub fn apply(&mut self, mut tick: Tick) -> Option<Tick> {
        if !self.on {
            return Some(tick);
        }
        let i = self.n;
        self.n += 1;
        // Remote Play's size check only says roughly. The pipeline makes its own call from the picture.
        let game = match tick.extra.get("size_new") {
            Some(v) => v.clone(),
            None => tick.new_hint.map(Value::Bool).unwrap_or(Value::Null),
        };
        self.game.insert(i, game.as_bool());
        tick.extra.insert("game_new".into(), game);
        self.game.remove(&(i - Self::KEEP));
        if i % 2 == 0 {
            self.last = Some(tick.frame.clone());
            (tick.content, tick.truth_new, tick.new_hint) = (Some(i), Some(true), Some(true));
        } else if let Some(last) = &self.last {
            self.truth.insert(i, std::mem::replace(&mut tick.frame, last.clone()));
            self.truth.remove(&(i - Self::KEEP));
            (tick.content, tick.truth_new, tick.new_hint) = (Some(i - 1), Some(false), Some(false));
        } else {
            return None; // switched on at an odd frame, so nothing to repeat yet
        }
        Some(tick)
    }
}

/// Applies a shared TestMode, if any, to a tick.
pub fn apply(test: &Option<SharedTestMode>, tick: Tick) -> Option<Tick> {
    match test {
        Some(t) => t.lock().unwrap().apply(tick),
        None => Some(tick),
    }
}

/// Any source with test mode applied as its ticks are taken (TestMode.wrap), for sources without a reader thread
/// of their own. Waitings pass through.
pub struct WithTestMode<S: Source> {
    pub source: S,
    pub test: SharedTestMode,
}

impl<S: Source> Source for WithTestMode<S> {
    fn info(&self) -> &SourceInfo {
        self.source.info()
    }
    fn size(&self) -> (usize, usize) {
        self.source.size()
    }
    fn next_event(&mut self) -> Option<Event> {
        loop {
            match self.source.next_event()? {
                Event::Tick(t) => {
                    if let Some(t) = self.test.lock().unwrap().apply(t) {
                        return Some(Event::Tick(t));
                    }
                }
                w => return Some(w),
            }
        }
    }
    fn ground_truth(&mut self, content: i64) -> Option<Frame> {
        self.test.lock().unwrap().gt(content)
    }
    fn read_sound(&mut self) -> Vec<Sound> {
        self.source.read_sound()
    }
    fn request_keyframe(&mut self) {
        self.source.request_keyframe()
    }
    fn set_audio_delay(&mut self, ms: f64) {
        self.source.set_audio_delay(ms)
    }
    fn close(&mut self) {
        self.source.close()
    }
}
