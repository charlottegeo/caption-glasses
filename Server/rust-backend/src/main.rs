mod audio;
mod models;
mod moonshine;
mod music;
mod partial;
mod pipeline;
mod sfx;

use audio::VadChunk;
use pipeline::{SfxWindower, SpeechSegmenter, SPEECH_THRESHOLD};
use sfx::YamnetClassifier;
use std::sync::Arc;

const SAMPLE_RATE: u32 = 16_000;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let (partial_tx, final_job_tx, yamnet) = load_models().await?;

    let mut vad_stream = audio::spawn_mic_vad_stream()?;
    let mut segmenter = SpeechSegmenter::new();
    let mut sfx_windower = SfxWindower::new();

    //listens for audio and if it passes the VAD threshold it checks for SFX and transcribes it
    while let Some(VadChunk {
        samples: chunk,
        probability,
    }) = vad_stream.recv().await
    {
        let is_speech = probability > SPEECH_THRESHOLD;
        sfx_windower.ingest(&yamnet, &chunk);

        if let Some(utterance) = segmenter.ingest(&partial_tx, chunk, is_speech) {
            moonshine::spawn_final(final_job_tx.clone(), utterance);
        }
    }

    Ok(())
}

//this just loads the models and spawns the workers for partial and final transcription, plus the yamnet classifier (sfx)
async fn load_models() -> anyhow::Result<(
    std::sync::mpsc::Sender<partial::PartialMsg>,
    std::sync::mpsc::Sender<moonshine::Job>,
    Arc<YamnetClassifier>,
)> {
    let vosk_model_path = std::env::var("VOSK_MODEL_PATH")
        .unwrap_or_else(|_| "models/vosk-model-small-en-us-0.15".into());
    let moonshine_model_dir = std::env::var("MOONSHINE_MODEL_DIR")
        .unwrap_or_else(|_| "models/sherpa-onnx-moonshine-base-en-int8".into());
    models::check_for_model_archive(&vosk_model_path)?;
    models::check_for_model_archive(&moonshine_model_dir)?;

    let moonshine_threads: i32 = env_parsed("MOONSHINE_THREADS").unwrap_or(2);
    let itn_enabled: bool = env_parsed("ITN_ENABLED").unwrap_or(true);
    let partial_tx = partial::spawn_vosk_worker(vosk_model_path);
    let final_job_tx = moonshine::spawn_moonshine_worker(
        moonshine::MoonshinePaths {
            preprocessor: format!("{moonshine_model_dir}/preprocess.onnx"),
            encoder: format!("{moonshine_model_dir}/encode.int8.onnx"),
            uncached_decoder: format!("{moonshine_model_dir}/uncached_decode.int8.onnx"),
            cached_decoder: format!("{moonshine_model_dir}/cached_decode.int8.onnx"),
            tokens: format!("{moonshine_model_dir}/tokens.txt"),
        },
        moonshine_threads,
        itn_enabled,
    );

    let yamnet_model_path =
        std::env::var("YAMNET_MODEL_PATH").unwrap_or_else(|_| "models/yamnet.onnx".into());
    let yamnet_class_map_path = std::env::var("YAMNET_CLASS_MAP_PATH")
        .unwrap_or_else(|_| "models/yamnet_class_map.csv".into());
    models::check_for_model(&yamnet_model_path)?;
    models::check_for_model(&yamnet_class_map_path)?;
    let yamnet = Arc::new(YamnetClassifier::load(&yamnet_model_path, &yamnet_class_map_path)?);

    Ok((partial_tx, final_job_tx, yamnet))
}

fn env_parsed<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}
