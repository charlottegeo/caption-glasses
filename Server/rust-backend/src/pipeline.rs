use crate::music::MusicWindower;
use crate::partial::{f32_to_i16, PartialMsg};
use crate::sfx::{Detection, YamnetClassifier};
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc as std_mpsc, Arc, Mutex,
};

pub const SPEECH_THRESHOLD: f32 = 0.5;
const PARTIAL_EVERY_CHUNKS: usize = 8;
const END_SILENCE_CHUNKS: usize = 20;
const PRE_ROLL_CHUNKS: usize = 10;
const SFX_WINDOW_SAMPLES: usize = crate::SAMPLE_RATE as usize;
const SFX_THRESHOLD: f32 = 0.4;
const MUSIC_GATE_THRESHOLD: f32 = 0.4;
const MAX_UTTERANCE_SAMPLES: usize = 5 * crate::SAMPLE_RATE as usize;
const TRAILING_SILENCE_KEEP_SAMPLES: usize = crate::SAMPLE_RATE as usize / 10;
const MIN_UTTERANCE_SAMPLES: usize = crate::SAMPLE_RATE as usize / 4;

pub struct SpeechSegmenter {
    buffer: Vec<f32>,
    pre_roll: VecDeque<Vec<f32>>,
    speaking: bool,
    silence_chunks: usize,
    trailing_silence_samples: usize,
    chunks_since_partial: usize,
}

impl SpeechSegmenter {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            pre_roll: VecDeque::with_capacity(PRE_ROLL_CHUNKS),
            speaking: false,
            silence_chunks: 0,
            trailing_silence_samples: 0,
            chunks_since_partial: 0,
        }
    }

    pub fn ingest(
        &mut self,
        partial_tx: &std_mpsc::Sender<PartialMsg>,
        chunk: Vec<f32>,
        is_speech: bool,
    ) -> Option<Vec<f32>> {
        let emit_partial = is_speech && self.partial_due();

        if is_speech && !self.speaking {
            for pre in self.pre_roll.drain(..) {
                send_to_vosk(partial_tx, &pre, false);
                self.buffer.extend(pre);
            }
        }

        if is_speech || self.speaking {
            send_to_vosk(partial_tx, &chunk, emit_partial);
            if !is_speech {
                self.trailing_silence_samples += chunk.len();
            }
            self.buffer.extend(chunk);
        } else {
            if self.pre_roll.len() == PRE_ROLL_CHUNKS {
                self.pre_roll.pop_front();
            }
            self.pre_roll.push_back(chunk);
        }

        if is_speech {
            self.speaking = true;
            self.silence_chunks = 0;
            self.trailing_silence_samples = 0;
            if self.buffer.len() >= MAX_UTTERANCE_SAMPLES {
                self.chunks_since_partial = 0;
                let _ = partial_tx.send(PartialMsg::Reset);
                return Some(std::mem::take(&mut self.buffer));
            }
            return None;
        }
        if !self.speaking {
            return None;
        }

        self.silence_chunks += 1;
        if self.silence_chunks < END_SILENCE_CHUNKS {
            return None;
        }

        self.speaking = false;
        self.silence_chunks = 0;
        let trim = self
            .trailing_silence_samples
            .saturating_sub(TRAILING_SILENCE_KEEP_SAMPLES);
        self.trailing_silence_samples = 0;
        self.buffer.truncate(self.buffer.len().saturating_sub(trim));
        self.chunks_since_partial = 0;
        let _ = partial_tx.send(PartialMsg::Reset);
        let utterance = std::mem::take(&mut self.buffer);
        if utterance.len() < MIN_UTTERANCE_SAMPLES {
            return None;
        }
        Some(utterance)
    }

    fn partial_due(&mut self) -> bool {
        self.chunks_since_partial += 1;
        if self.chunks_since_partial >= PARTIAL_EVERY_CHUNKS {
            self.chunks_since_partial = 0;
            true
        } else {
            false
        }
    }
}

fn send_to_vosk(partial_tx: &std_mpsc::Sender<PartialMsg>, chunk: &[f32], emit: bool) {
    let pcm16 = f32_to_i16(chunk);
    let _ = partial_tx.send(PartialMsg::Audio(pcm16, emit, std::time::Instant::now()));
}

pub struct SfxWindower {
    buffer: Vec<f32>,
    inflight: Arc<AtomicBool>,
    last_label: Arc<Mutex<String>>,
    music: MusicWindower,
}

impl SfxWindower {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(SFX_WINDOW_SAMPLES),
            inflight: Arc::new(AtomicBool::new(false)),
            last_label: Arc::new(Mutex::new(String::new())),
            music: MusicWindower::new(),
        }
    }

    pub fn ingest(&mut self, yamnet: &Arc<YamnetClassifier>, chunk: &[f32]) {
        self.music.feed_audio(chunk);
        self.buffer.extend_from_slice(chunk);
        if self.buffer.len() >= SFX_WINDOW_SAMPLES && !self.inflight.swap(true, Ordering::Relaxed) {
            let window = std::mem::take(&mut self.buffer);
            spawn_sfx(
                yamnet.clone(),
                window,
                self.inflight.clone(),
                self.last_label.clone(),
                self.music.clone(),
            );
        }
    }
}

fn spawn_sfx(
    yamnet: Arc<YamnetClassifier>,
    window: Vec<f32>,
    inflight: Arc<AtomicBool>,
    last_label: Arc<Mutex<String>>,
    music: MusicWindower,
) {
    tokio::spawn(async move {
        let (detection, is_music) = tokio::task::spawn_blocking(move || {
            yamnet.analyze(&window, SFX_THRESHOLD, MUSIC_GATE_THRESHOLD)
        })
        .await
        .unwrap_or((None, false));
        inflight.store(false, Ordering::Relaxed);

        music.maybe_recognize(is_music);

        if let Some(detection) = detection {
            let mut last = last_label.lock().unwrap();
            if *last != detection.label {
                emit_sfx(&detection);
                *last = detection.label;
            }
        }
    });
}

fn emit_sfx(detection: &Detection) {
    println!(
        "\n[sfx] {} ({}, {:.2})",
        detection.label, detection.category, detection.score
    );
}
