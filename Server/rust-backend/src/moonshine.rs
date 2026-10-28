use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};

const BACKLOG_WARN_THRESHOLD: Duration = Duration::from_millis(1500);
const BACKLOG_WARN_COOLDOWN: Duration = Duration::from_secs(10);

pub type Job = (Vec<f32>, Instant, oneshot::Sender<String>);

pub struct MoonshinePaths {
    pub preprocessor: String,
    pub encoder: String,
    pub uncached_decoder: String,
    pub cached_decoder: String,
    pub tokens: String,
}

pub fn spawn_moonshine_worker(
    paths: MoonshinePaths,
    n_threads: i32,
    itn_enabled: bool,
) -> std_mpsc::Sender<Job> {
    let (tx, rx) = std_mpsc::channel::<Job>();
    std::thread::spawn(move || {
        let mut config = OfflineRecognizerConfig::default();
        config.model_config.moonshine.preprocessor = Some(paths.preprocessor);
        config.model_config.moonshine.encoder = Some(paths.encoder);
        config.model_config.moonshine.uncached_decoder = Some(paths.uncached_decoder);
        config.model_config.moonshine.cached_decoder = Some(paths.cached_decoder);
        config.model_config.tokens = Some(paths.tokens);
        config.model_config.num_threads = n_threads;
        config.model_config.provider = Some("cpu".to_string());

        let recognizer =
            OfflineRecognizer::create(&config).expect("failed to load moonshine model");

        let mut last_backlog_warning = Instant::now() - BACKLOG_WARN_COOLDOWN;

        for (audio, sent_at, reply) in rx {
            let queued_for = sent_at.elapsed();
            if queued_for >= BACKLOG_WARN_THRESHOLD
                && last_backlog_warning.elapsed() >= BACKLOG_WARN_COOLDOWN
            {
                eprintln!(
                    "\n[moonshine] falling behind real time: utterance queued {:.0}ms before processing",
                    queued_for.as_secs_f64() * 1000.0
                );
                last_backlog_warning = Instant::now();
            }

            let stream = recognizer.create_stream();
            stream.accept_waveform(crate::SAMPLE_RATE as i32, &audio);
            recognizer.decode(&stream);
            let mut text = stream
                .get_result()
                .map(|result| result.text.trim().to_string())
                .unwrap_or_default();
            if itn_enabled {
                text = text_processing_rs::normalize_sentence(&text);
            }
            let _ = reply.send(text);
        }
    });
    tx
}

pub fn spawn_final(job_tx: std_mpsc::Sender<Job>, audio: Vec<f32>) {
    tokio::spawn(async move {
        let text = run_job(&job_tx, audio).await;
        if !text.is_empty() {
            emit_transcript(&text);
        }
    });
}

async fn run_job(job_tx: &std_mpsc::Sender<Job>, audio: Vec<f32>) -> String {
    let (reply_tx, reply_rx) = oneshot::channel();
    if job_tx.send((audio, Instant::now(), reply_tx)).is_err() {
        return String::new();
    }
    reply_rx.await.unwrap_or_default()
}

fn emit_transcript(text: &str) {
    println!("\r\x1b[2K{text}");
}