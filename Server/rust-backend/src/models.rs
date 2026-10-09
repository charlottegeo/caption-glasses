use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

//Holds model files and their download URLs, so it downloads them if missing
const KNOWN_MODELS: &[(&str, &str)] = &[
    (
        "models/ggml-tiny.en.bin",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
    ),
    (
        "models/ggml-base.en.bin",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
    ),
    (
        "models/yamnet_class_map.csv",
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
    ),
];


//Checks if the model file exists, and downloads it if not. Returns an error if the model is missing and there is no download URL for it.
pub fn check_for_model(path: &str) -> anyhow::Result<()> {
    if Path::new(path).exists() {
        return Ok(());
    }

    let url = KNOWN_MODELS
        .iter()
        .find(|(known_path, _)| *known_path == path)
        .map(|(_, url)| *url)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Model file '{path}' is missing and no download source is present"
            )
        })?;

    download(url, path)
}

//Downloads a file from the given URL to the given destination path, showing progress in the console
fn download(url: &str, dest: &str) -> anyhow::Result<()> {
    eprintln!("[models] {dest} not found, downloading from {url}");
    if let Some(parent) = Path::new(dest).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let response = ureq::get(url).call()?;
    let total_bytes: Option<u64> = response
        .header("Content-Length")
        .and_then(|len| len.parse().ok());

    let tmp_dest = format!("{dest}.part");
    let mut file = File::create(&tmp_dest)?;
    let mut reader = response.into_reader();
    let mut buf = [0u8; 256 * 1024];
    let mut downloaded: u64 = 0;
    let mut next_report: u64 = 10 * 1024 * 1024;

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
        if downloaded >= next_report {
            report_progress(dest, downloaded, total_bytes);
            next_report += 10 * 1024 * 1024;
        }
    }
    file.flush()?;
    drop(file);
    std::fs::rename(&tmp_dest, dest)?;
    eprintln!("[models] saved {dest} ({} MB)", downloaded / (1024 * 1024));
    Ok(())
}

//Reports download progress to the console, either as a percentage if total size is known, or in MB if not
fn report_progress(dest: &str, downloaded: u64, total: Option<u64>) {
    match total {
        Some(total) if total > 0 => {
            eprintln!(
                "[models] {dest}: {:.0}%",
                100.0 * downloaded as f64 / total as f64
            );
        }
        _ => eprintln!("[models] {dest}: {} MB", downloaded / (1024 * 1024)),
    }
}
