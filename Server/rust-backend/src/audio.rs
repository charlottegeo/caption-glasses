use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rodio::{buffer::SamplesBuffer, source::UniformSourceIterator};
use std::sync::mpsc as std_mpsc;
use tokio::sync::mpsc as tokio_mpsc;
use voice_activity_detector::VoiceActivityDetector;

use crate::SAMPLE_RATE;

const VAD_CHUNK_SAMPLES: usize = 512;

pub struct VadChunk {
    pub samples: Vec<f32>,
    pub probability: f32,
}

//Spawns a thread that reads from the microphone and sends chunks of audio to a channel, along with the probability of speech detected in that chunk.
pub fn spawn_mic_vad_stream() -> anyhow::Result<tokio_mpsc::UnboundedReceiver<VadChunk>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no default input device"))?;
    let config = device.default_input_config()?;
    let sample_format = config.sample_format();
    let channels = config.channels() as usize;
    let native_rate = config.sample_rate().0;
    let stream_config: cpal::StreamConfig = config.into();
    eprintln!(
        "[mic] device={:?} format={sample_format:?} channels={channels} native_rate={native_rate}",
        device.name().unwrap_or_else(|_| "<unknown>".into()),
    );

    let (raw_tx, raw_rx) = std_mpsc::channel::<Vec<f32>>();
    let err_fn = |err| eprintln!("[mic] stream error: {err}");

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                let _ = raw_tx.send(downmix(data, channels));
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                let samples: Vec<f32> =
                    data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                let _ = raw_tx.send(downmix(&samples, channels));
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| {
                let samples: Vec<f32> = data
                    .iter()
                    .map(|&s| (s as f32 - 32768.0) / 32768.0)
                    .collect();
                let _ = raw_tx.send(downmix(&samples, channels));
            },
            err_fn,
            None,
        )?,
        other => anyhow::bail!("unsupported mic sample format: {other:?}"),
    };
    stream.play()?;
    std::mem::forget(stream);

    let (vad_tx, vad_rx) = tokio_mpsc::unbounded_channel();
    std::thread::spawn(move || {
        let mut vad = VoiceActivityDetector::builder()
            .sample_rate(SAMPLE_RATE as i64)
            .chunk_size(VAD_CHUNK_SAMPLES)
            .build()
            .expect("failed to build VAD model");
        let mut pending: Vec<f32> = Vec::with_capacity(VAD_CHUNK_SAMPLES * 2);

        for raw in raw_rx {
            pending.extend(resample_to_16k(raw, native_rate));
            while pending.len() >= VAD_CHUNK_SAMPLES {
                let chunk: Vec<f32> = pending.drain(..VAD_CHUNK_SAMPLES).collect();
                let probability = vad.predict(chunk.iter().copied());
                if vad_tx
                    .send(VadChunk {
                        samples: chunk,
                        probability,
                    })
                    .is_err()
                {
                    return;
                }
            }
        }
    });

    Ok(vad_rx)
}

//Downmixes multi-channel audio to mono by averaging the channels, the VAD model expects mono audio
fn downmix(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    data.chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

//Resamples audio to 16kHz
fn resample_to_16k(audio: Vec<f32>, sample_rate: u32) -> Vec<f32> {
    if sample_rate == SAMPLE_RATE {
        return audio;
    }
    let source = SamplesBuffer::new(1, sample_rate, audio);
    UniformSourceIterator::<_, f32>::new(source, 1, SAMPLE_RATE).collect()
}
