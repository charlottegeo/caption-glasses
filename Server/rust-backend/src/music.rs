use crate::partial::f32_to_i16;
use songrec::{Config, SongRec};
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

const WINDOW_SECONDS: usize = 10;
const WINDOW_SAMPLES: usize = crate::SAMPLE_RATE as usize * WINDOW_SECONDS;
const RECOGNITION_COOLDOWN: Duration = Duration::from_secs(10);
const SILENCE_RESET_AFTER: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct MusicWindower {
    ring: Arc<Mutex<VecDeque<f32>>>,
    recognizer: Arc<SongRec>,
    inflight: Arc<AtomicBool>,
    last_song: Arc<Mutex<Option<String>>>,
    last_attempt: Arc<Mutex<Instant>>,
    last_music_seen: Arc<Mutex<Instant>>,
}

impl MusicWindower {
    pub fn new() -> Self {
        let config = Config::default().with_quiet_mode(true);
        let epoch = Instant::now() - RECOGNITION_COOLDOWN;
        Self {
            ring: Arc::new(Mutex::new(VecDeque::with_capacity(WINDOW_SAMPLES))),
            recognizer: Arc::new(SongRec::new(config)),
            inflight: Arc::new(AtomicBool::new(false)),
            last_song: Arc::new(Mutex::new(None)),
            last_attempt: Arc::new(Mutex::new(epoch)),
            last_music_seen: Arc::new(Mutex::new(epoch)),
        }
    }

    pub fn feed_audio(&self, chunk: &[f32]) {
        let mut ring = self.ring.lock().unwrap();
        ring.extend(chunk.iter().copied());
        while ring.len() > WINDOW_SAMPLES {
            ring.pop_front();
        }
    }

    pub fn maybe_recognize(&self, music_detected: bool) {
        let mut last_seen = self.last_music_seen.lock().unwrap();
        if music_detected {
            *last_seen = Instant::now();
        } else if last_seen.elapsed() > SILENCE_RESET_AFTER {
            *self.last_song.lock().unwrap() = None;
        }
        drop(last_seen);

        if !music_detected || self.inflight.load(Ordering::Relaxed) {
            return;
        }

        let samples: Vec<f32> = {
            let ring = self.ring.lock().unwrap();
            if ring.len() < WINDOW_SAMPLES {
                return;
            }
            ring.iter().copied().collect()
        };

        {
            let mut last_attempt = self.last_attempt.lock().unwrap();
            if last_attempt.elapsed() < RECOGNITION_COOLDOWN {
                return;
            }
            *last_attempt = Instant::now();
        }

        self.inflight.store(true, Ordering::Relaxed);
        spawn_recognition(
            self.recognizer.clone(),
            f32_to_i16(&samples),
            self.inflight.clone(),
            self.last_song.clone(),
        );
    }
}

fn spawn_recognition(
    recognizer: Arc<SongRec>,
    samples: Vec<i16>,
    inflight: Arc<AtomicBool>,
    last_song: Arc<Mutex<Option<String>>>,
) {
    tokio::spawn(async move {
        let result = tokio::task::spawn_blocking(move || {
            recognizer.recognize_from_samples(&samples, crate::SAMPLE_RATE)
        })
        .await;
        inflight.store(false, Ordering::Relaxed);

        if let Ok(Ok(recognition)) = result {
            let label = format!("{} - {}", recognition.song_name, recognition.artist_name);
            let mut last = last_song.lock().unwrap();
            if last.as_deref() != Some(label.as_str()) {
                emit_music(&label);
                *last = Some(label);
            }
        }
    });
}

fn emit_music(label: &str) {
    println!("\n[music] {label}");
}
