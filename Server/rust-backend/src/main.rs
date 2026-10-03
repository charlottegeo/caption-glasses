mod sfx;

use futures_util::StreamExt;
use kalosm_sound::*;
use rodio::{buffer::SamplesBuffer, source::UniformSourceIterator, Source};
use sfx::YamnetClassifier;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc as std_mpsc, Arc, Mutex,
};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const SPEECH_THRESHOLD: f32 = 0.5;
const PARTIAL_EVERY_CHUNKS: usize = 16;
const END_SILENCE_CHUNKS: usize = 20;
const PRE_ROLL_CHUNKS: usize = 10;
const SAMPLE_RATE: u32 = 16_000;
const SFX_WINDOW_SAMPLES: usize = SAMPLE_RATE as usize;
const SFX_THRESHOLD: f32 = 0.4;

type Job = (Vec<f32>, tokio::sync::oneshot::Sender<String>);

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let partial_model_path = std::env::var("WHISPER_PARTIAL_MODEL_PATH")
        .unwrap_or_else(|_| "models/ggml-tiny.en.bin".into());
    let final_model_path = std::env::var("WHISPER_FINAL_MODEL_PATH")
        .unwrap_or_else(|_| "models/ggml-base.en.bin".into());
    const WHISPER_THREADS: i32 = 4;
    const FINAL_BEAM_SIZE: i32 = 2;
    let partial_job_tx = spawn_whisper_worker(partial_model_path, WHISPER_THREADS, None);
    let final_job_tx = spawn_whisper_worker(final_model_path, WHISPER_THREADS, Some(FINAL_BEAM_SIZE));

    let yamnet_model_path =
        std::env::var("YAMNET_MODEL_PATH").unwrap_or_else(|_| "models/yamnet.onnx".into());
    let yamnet_class_map_path = std::env::var("YAMNET_CLASS_MAP_PATH")
        .unwrap_or_else(|_| "models/yamnet_class_map.csv".into());
    let yamnet = Arc::new(YamnetClassifier::load(&yamnet_model_path, &yamnet_class_map_path)?);

    let mic = MicInput::default();
    let mut vad_stream = mic.stream().voice_activity_stream();

    let mut buffer: Vec<f32> = Vec::new();
    let mut pre_roll: VecDeque<Vec<f32>> = VecDeque::with_capacity(PRE_ROLL_CHUNKS);
    let mut sample_rate = SAMPLE_RATE;
    let mut speaking = false;
    let mut silence_chunks = 0usize;
    let mut chunks_since_partial = 0usize;
    let partial_inflight = Arc::new(AtomicBool::new(false));

    let mut sfx_buffer: Vec<f32> = Vec::with_capacity(SFX_WINDOW_SAMPLES);
    let sfx_inflight = Arc::new(AtomicBool::new(false));
    let last_sfx_label: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));

    while let Some(output) = vad_stream.next().await {
        sample_rate = Source::sample_rate(&output.samples);
        let is_speech = output.probability > SPEECH_THRESHOLD;
        let chunk: Vec<f32> = output.samples.collect();

        sfx_buffer.extend(resample_to_16k(chunk.clone(), sample_rate));
        if sfx_buffer.len() >= SFX_WINDOW_SAMPLES && !sfx_inflight.swap(true, Ordering::Relaxed) {
            let window = std::mem::take(&mut sfx_buffer);
            spawn_sfx(
                yamnet.clone(),
                window,
                sfx_inflight.clone(),
                last_sfx_label.clone(),
            );
        }

        if is_speech && !speaking {
            for pre in pre_roll.drain(..) {
                buffer.extend(pre);
            }
        }

        if is_speech || speaking {
            buffer.extend(chunk);
        } else {
            if pre_roll.len() == PRE_ROLL_CHUNKS {
                pre_roll.pop_front();
            }
            pre_roll.push_back(chunk);
        }

        if is_speech {
            speaking = true;
            silence_chunks = 0;
            chunks_since_partial += 1;

            if chunks_since_partial >= PARTIAL_EVERY_CHUNKS
                && !partial_inflight.swap(true, Ordering::Relaxed)
            {
                chunks_since_partial = 0;
                spawn_partial(
                    partial_job_tx.clone(),
                    buffer.clone(),
                    sample_rate,
                    partial_inflight.clone(),
                );
            }
        } else if speaking {
            silence_chunks += 1;
            if silence_chunks >= END_SILENCE_CHUNKS {
                let audio = std::mem::take(&mut buffer);
                speaking = false;
                silence_chunks = 0;
                chunks_since_partial = 0;
                spawn_final(final_job_tx.clone(), audio, sample_rate);
            }
        }
    }

    Ok(())
}

fn spawn_whisper_worker(
    model_path: String,
    n_threads: i32,
    beam_size: Option<i32>,
) -> std_mpsc::Sender<Job> {
    let (tx, rx) = std_mpsc::channel::<Job>();
    std::thread::spawn(move || {
        let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
            .expect("failed to load whisper model");
        let mut state = ctx.create_state().expect("failed to create whisper state");

        for (audio, reply) in rx {
            let strategy = match beam_size {
                Some(beam_size) => SamplingStrategy::BeamSearch {
                    beam_size,
                    patience: -1.0,
                },
                None => SamplingStrategy::Greedy { best_of: 1 },
            };
            let mut params = FullParams::new(strategy);
            params.set_language(Some("en"));
            params.set_n_threads(n_threads);
            params.set_single_segment(true);
            params.set_no_context(true);
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);

            params.set_suppress_nst(true);

            let text = match state.full(params, &audio) {
                Ok(_) => state
                    .as_iter()
                    .map(|segment| segment.to_string())
                    .collect::<String>()
                    .trim()
                    .to_string(),
                Err(_) => String::new(),
            };
            let _ = reply.send(text);
        }
    });
    tx
}

fn spawn_partial(
    job_tx: std_mpsc::Sender<Job>,
    audio: Vec<f32>,
    sample_rate: u32,
    inflight: Arc<AtomicBool>,
) {
    tokio::spawn(async move {
        let text = run_job(&job_tx, audio, sample_rate).await;
        if !text.is_empty() {
            print!("\r\x1b[2K{text}");
            let _ = std::io::stdout().flush();
        }
        inflight.store(false, Ordering::Relaxed);
    });
}

fn spawn_final(job_tx: std_mpsc::Sender<Job>, audio: Vec<f32>, sample_rate: u32) {
    tokio::spawn(async move {
        let text = run_job(&job_tx, audio, sample_rate).await;
        if !text.is_empty() {
            println!("\r\x1b[2K{text}");
        }
    });
}

async fn run_job(job_tx: &std_mpsc::Sender<Job>, audio: Vec<f32>, sample_rate: u32) -> String {
    let pcm = resample_to_16k(audio, sample_rate);
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    if job_tx.send((pcm, reply_tx)).is_err() {
        return String::new();
    }
    reply_rx.await.unwrap_or_default()
}

fn spawn_sfx(
    yamnet: Arc<YamnetClassifier>,
    window: Vec<f32>,
    inflight: Arc<AtomicBool>,
    last_label: Arc<Mutex<String>>,
) {
    tokio::spawn(async move {
        let detection = tokio::task::spawn_blocking(move || yamnet.classify(&window, SFX_THRESHOLD))
            .await
            .unwrap_or(None);
        inflight.store(false, Ordering::Relaxed);

        if let Some(detection) = detection {
            let mut last = last_label.lock().unwrap();
            if *last != detection.label {
                println!(
                    "\n[sfx] {} ({}, {:.2})",
                    detection.label, detection.category, detection.score
                );
                *last = detection.label;
            }
        }
    });
}

fn resample_to_16k(audio: Vec<f32>, sample_rate: u32) -> Vec<f32> {
    if sample_rate == SAMPLE_RATE {
        return audio;
    }
    let source = SamplesBuffer::new(1, sample_rate, audio);
    UniformSourceIterator::<_, f32>::new(source, 1, SAMPLE_RATE).collect()
}
