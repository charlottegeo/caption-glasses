use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

const KNOWN_MODELS: &[(&str, &str)] = &[
    (
        "models/yamnet_class_map.csv",
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
    ),
    (
        "models/yamnet.onnx",
        "https://huggingface.co/audiomagic/yamnet-onnx/resolve/main/yamnet.onnx",
    ),
];
const KNOWN_MODEL_ARCHIVES: &[(&str, &str)] = &[
    (
        "models/vosk-model-small-en-us-0.15",
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    ),
    (
        "models/sherpa-onnx-moonshine-tiny-en-int8",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2",
    ),
    (
        "models/sherpa-onnx-moonshine-base-en-int8",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-base-en-int8.tar.bz2",
    ),
];

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

pub fn check_for_model_archive(dir_path: &str) -> anyhow::Result<()> {
    if Path::new(dir_path).exists() {
        return Ok(());
    }

    let url = KNOWN_MODEL_ARCHIVES
        .iter()
        .find(|(known_path, _)| *known_path == dir_path)
        .map(|(_, url)| *url)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Model directory '{dir_path}' is missing and no download source is present"
            )
        })?;

    if url.ends_with(".tar.bz2") {
        download_and_extract_tar_bz2(url, dir_path)
    } else {
        download_and_extract_zip(url, dir_path)
    }
}

fn download_and_extract_zip(url: &str, dest_dir: &str) -> anyhow::Result<()> {
    eprintln!("[models] {dest_dir} not found, downloading from {url}");

    let response = ureq::get(url).call()?;
    let mut bytes = Vec::new();
    response.into_reader().read_to_end(&mut bytes)?;

    let tmp_dir = format!("{dest_dir}.part");
    std::fs::create_dir_all(&tmp_dir)?;

    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let Some(relative) = entry.name().split_once('/').map(|(_, rest)| rest) else {
            continue;
        };
        if relative.is_empty() {
            continue;
        }
        let out_path = Path::new(&tmp_dir).join(relative);
        if entry.is_dir() {
            std::fs::create_dir_all(&out_path)?;
            continue;
        }
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut out_file = File::create(&out_path)?;
        std::io::copy(&mut entry, &mut out_file)?;
    }

    std::fs::rename(&tmp_dir, dest_dir)?;
    eprintln!("[models] extracted {dest_dir}");
    Ok(())
}

fn download_and_extract_tar_bz2(url: &str, dest_dir: &str) -> anyhow::Result<()> {
    eprintln!("[models] {dest_dir} not found, downloading from {url}");

    let response = ureq::get(url).call()?;
    let mut bytes = Vec::new();
    response.into_reader().read_to_end(&mut bytes)?;

    let tmp_dir = format!("{dest_dir}.part");
    std::fs::create_dir_all(&tmp_dir)?;

    let decoder = bzip2::read::BzDecoder::new(std::io::Cursor::new(bytes));
    let mut archive = tar::Archive::new(decoder);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let Some(relative) = path.strip_prefix(path.components().next().unwrap()).ok() else {
            continue;
        };
        if relative.as_os_str().is_empty() {
            continue;
        }
        let out_path = Path::new(&tmp_dir).join(relative);
        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&out_path)?;
            continue;
        }
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        entry.unpack(&out_path)?;
    }

    std::fs::rename(&tmp_dir, dest_dir)?;
    eprintln!("[models] extracted {dest_dir}");
    Ok(())
}

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
