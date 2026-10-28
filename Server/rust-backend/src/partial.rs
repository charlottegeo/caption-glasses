use std::io::Write;
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};
use vosk::{Model, Recognizer};

use crate::SAMPLE_RATE;

const BACKLOG_WARN_THRESHOLD: Duration = Duration::from_millis(300);
const BACKLOG_WARN_COOLDOWN: Duration = Duration::from_secs(10);

pub enum PartialMsg {
    Audio(Vec<i16>, bool, Instant),
    Reset,
}

pub fn spawn_vosk_worker(model_path: String) -> std_mpsc::Sender<PartialMsg> {
    let (tx, rx) = std_mpsc::channel::<PartialMsg>();
    std::thread::spawn(move || {
        let model = Model::new(&model_path).expect("failed to load vosk model");
        let mut recognizer = Recognizer::new(&model, SAMPLE_RATE as f32)
            .expect("failed to create vosk recognizer");

        let mut last_words: Vec<String> = Vec::new();
        let mut last_backlog_warning = Instant::now() - BACKLOG_WARN_COOLDOWN;

        for msg in rx {
            match msg {
                PartialMsg::Audio(pcm, emit, sent_at) => {
                    let queued_for = sent_at.elapsed();
                    if queued_for >= BACKLOG_WARN_THRESHOLD
                        && last_backlog_warning.elapsed() >= BACKLOG_WARN_COOLDOWN
                    {
                        eprintln!(
                            "\n[vosk] falling behind real time: chunk queued {:.0}ms before processing",
                            queued_for.as_secs_f64() * 1000.0
                        );
                        last_backlog_warning = Instant::now();
                    }
                    let _ = recognizer.accept_waveform(&pcm);
                    if emit {
                        print_partial(&mut recognizer, &mut last_words);
                    }
                }
                PartialMsg::Reset => {
                    recognizer.reset();
                    last_words.clear();
                }
            }
        }
    });
    tx
}

fn print_partial(recognizer: &mut Recognizer, last_words: &mut Vec<String>) {
    let text = recognizer.partial_result().partial.to_string();
    if text.is_empty() {
        return;
    }
    let words: Vec<String> = text.split_whitespace().map(String::from).collect();
    if *last_words == words {
        return;
    }

    let confirmed = words
        .iter()
        .zip(last_words.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let (stable, tentative) = (&words[..confirmed], &words[confirmed..]);
    print!(
        "\r\x1b[2K[partial] {} \x1b[2m{}\x1b[0m",
        stable.join(" "),
        tentative.join(" ")
    );
    let _ = std::io::stdout().flush();
    *last_words = words;
}

pub fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect()
}
