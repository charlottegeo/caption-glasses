use std::io::Read;
use std::path::{Path, PathBuf};

const VOSK_VERSION_LINUX: &str = "0.3.45";
const VOSK_VERSION_MACOS: &str = "0.3.42";

fn main() {
    let native_dir = native_dir();
    let lib_path = native_dir.join(lib_file_name());

    if !lib_path.exists() {
        download_and_extract(&native_dir).expect("failed to fetch libvosk");
    }

    println!("cargo:rustc-link-search=native={}", native_dir.display());
    println!("cargo:rustc-link-lib=dylib=vosk");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", native_dir.display());
    if let Some(sherpa_lib_dir) = find_sherpa_onnx_lib_dir() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", sherpa_lib_dir.display());
    }

    println!("cargo:rerun-if-changed=build.rs");
}

fn find_sherpa_onnx_lib_dir() -> Option<PathBuf> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").ok()?);
    let target_dir = out_dir.ancestors().nth(4)?;
    let prebuilt_dir = target_dir.join("sherpa-onnx-prebuilt");
    let entries = std::fs::read_dir(&prebuilt_dir).ok()?;
    for entry in entries.flatten() {
        let lib_dir = entry.path().join("lib");
        if lib_dir.join("libsherpa-onnx-c-api.so").exists()
            || lib_dir.join("libsherpa-onnx-c-api.dylib").exists()
        {
            return Some(lib_dir);
        }
    }
    None
}

fn native_dir() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    Path::new(&manifest_dir).join("native")
}

fn lib_file_name() -> &'static str {
    if target_os() == "macos" {
        "libvosk.dylib"
    } else {
        "libvosk.so"
    }
}

fn target_os() -> String {
    std::env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS not set")
}

fn release_asset() -> String {
    match target_os().as_str() {
        "macos" => format!("vosk-osx-{VOSK_VERSION_MACOS}"),
        "linux" => {
            let arch = std::env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH not set");
            match arch.as_str() {
                "aarch64" => format!("vosk-linux-aarch64-{VOSK_VERSION_LINUX}"),
                "x86_64" => format!("vosk-linux-x86_64-{VOSK_VERSION_LINUX}"),
                other => panic!("no known libvosk release for linux/{other}"),
            }
        }
        other => panic!("no known libvosk release for target_os {other}"),
    }
}

fn download_and_extract(native_dir: &Path) -> anyhow::Result<()> {
    let asset = release_asset();
    let version = if target_os() == "macos" {
        VOSK_VERSION_MACOS
    } else {
        VOSK_VERSION_LINUX
    };
    let url = format!(
        "https://github.com/alphacep/vosk-api/releases/download/v{version}/{asset}.zip"
    );
    eprintln!("[build] libvosk not found, downloading from {url}");

    let response = ureq::get(&url).call()?;
    let mut bytes = Vec::new();
    response.into_reader().read_to_end(&mut bytes)?;

    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))?;
    std::fs::create_dir_all(native_dir)?;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name();
        let Some(file_name) = name.rsplit('/').next().filter(|n| !n.is_empty()) else {
            continue;
        };
        if entry.is_dir() {
            continue;
        }
        let dest = native_dir.join(file_name);
        let mut out = std::fs::File::create(&dest)?;
        std::io::copy(&mut entry, &mut out)?;
    }

    eprintln!("[build] saved libvosk to {}", native_dir.display());
    Ok(())
}
