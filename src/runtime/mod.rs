mod llama;

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tar::Archive;
use tokio::io::AsyncWriteExt;
use zip::ZipArchive;

use crate::config::{AppConfig, Device};

pub use llama::{EncoderOutput, LlamaEncoder, LlamaLibrary};

const MANAGED_VERSION: &str = "b10964";
const MANAGED_RELEASE_URL: &str = "https://github.com/ggml-org/llama.cpp/releases/download/b10964";

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeInfo {
    pub runtime: String,
    pub source: String,
    pub version: Option<String>,
    pub path: Option<PathBuf>,
    pub platform: String,
    pub architecture: String,
    pub backend: String,
    pub compatible: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RuntimeManifest {
    runtime: String,
    version: String,
    platform: String,
    architecture: String,
    backend: String,
    archive: String,
    sha256: String,
}

#[derive(Debug, Clone, Copy)]
enum ArchiveKind {
    TarGz,
    Zip,
}

#[derive(Debug, Clone)]
struct RuntimeAsset {
    version: &'static str,
    name: &'static str,
    sha256: &'static str,
    kind: ArchiveKind,
    backend: &'static str,
}

pub struct RuntimeManager {
    cache_dir: PathBuf,
    explicit_home: Option<PathBuf>,
    client: Client,
}

impl RuntimeManager {
    pub fn new(config: &AppConfig) -> Self {
        Self {
            cache_dir: config.cache_dir.join("runtimes").join("llama.cpp"),
            explicit_home: config.llama_cpp_home.clone(),
            client: Client::builder()
                .user_agent(concat!("system-one/", env!("CARGO_PKG_VERSION")))
                .build()
                .expect("building the runtime HTTP client cannot fail"),
        }
    }

    pub fn resolve_library_path_for(&self, device: Device) -> Result<(PathBuf, String)> {
        if let Some(home) = &self.explicit_home {
            let path = llama::find_library(home).with_context(|| {
                format!(
                    "no compatible llama.cpp shared library found under {}",
                    home.display()
                )
            })?;
            return Ok((path, "external".to_owned()));
        }
        if let Some(value) = std::env::var_os("SYSTEM_ONE_LLAMA_LIBRARY") {
            let path = PathBuf::from(value);
            if path.is_file() {
                return Ok((path, "external".to_owned()));
            }
        }
        if let Some(path) = llama::find_path_library()? {
            return Ok((path, "system".to_owned()));
        }
        let managed_root = self.managed_root(device);
        if let Some(path) = llama::find_cached_library(&managed_root)? {
            return Ok((path, "managed".to_owned()));
        }
        if let Some(path) = llama::find_cached_library(&self.cache_dir)? {
            return Ok((path, "managed".to_owned()));
        }
        bail!(
            "no llama.cpp runtime found; configure --llama-cpp-home, set SYSTEM_ONE_LLAMA_LIBRARY, install a runtime, or allow managed download under {}",
            self.cache_dir.display()
        )
    }

    pub fn load(&self, device: Device) -> Result<(LlamaLibrary, RuntimeInfo)> {
        let (path, source) = self.resolve_library_path_for(device)?;
        let library = LlamaLibrary::load(&path)?;
        let info = library.runtime_info(source, device);
        Ok((library, info))
    }

    /// Resolve or provision a compatible managed runtime. This is the only method that may
    /// download llama.cpp, and callers pass the offline flag explicitly.
    pub async fn ensure(&self, device: Device, offline: bool) -> Result<()> {
        match self.resolve_library_path_for(device) {
            Ok((path, source)) => match LlamaLibrary::load(&path) {
                Ok(library) => {
                    tracing::info!(
                        source,
                        path = %path.display(),
                        version = %library.version(),
                        "using llama.cpp runtime"
                    );
                    return Ok(());
                }
                Err(error) if source == "external" || source == "system" => {
                    return Err(error).context(format!(
                        "configured llama.cpp runtime is incompatible: {}",
                        path.display()
                    ));
                }
                Err(error) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %error,
                        "cached llama.cpp runtime is incompatible; refreshing managed runtime"
                    );
                }
            },
            Err(error) if offline => {
                return Err(error)
                    .context("offline mode: no compatible llama.cpp runtime is available");
            }
            Err(error) => {
                tracing::info!(
                    error = %error,
                    "no local llama.cpp runtime found; downloading managed runtime"
                );
            }
        }

        if offline {
            bail!(
                "offline mode: no compatible llama.cpp runtime is available under {}",
                self.cache_dir.display()
            )
        }
        let path = self.install_managed(device).await?;
        let library = LlamaLibrary::load(&path).with_context(|| {
            format!(
                "managed llama.cpp runtime at {} failed compatibility checks",
                path.display()
            )
        })?;
        tracing::info!(
            source = "managed",
            path = %path.display(),
            version = %library.version(),
            "using managed llama.cpp runtime"
        );
        Ok(())
    }

    pub fn info(&self, device: Device) -> RuntimeInfo {
        let platform = std::env::consts::OS.to_owned();
        let architecture = std::env::consts::ARCH.to_owned();
        match self.resolve_library_path_for(device) {
            Ok((path, source)) => match LlamaLibrary::load(&path) {
                Ok(library) => library.runtime_info(source, device),
                Err(error) => RuntimeInfo {
                    runtime: "llama.cpp".to_owned(),
                    source,
                    version: None,
                    path: Some(path),
                    platform,
                    architecture,
                    backend: device.as_str().to_owned(),
                    compatible: false,
                    error: Some(error.to_string()),
                },
            },
            Err(error) => RuntimeInfo {
                runtime: "llama.cpp".to_owned(),
                source: "none".to_owned(),
                version: None,
                path: None,
                platform,
                architecture,
                backend: device.as_str().to_owned(),
                compatible: false,
                error: Some(error.to_string()),
            },
        }
    }

    pub fn managed_destination(&self, version: &str, backend: &str) -> PathBuf {
        self.cache_dir
            .join(version)
            .join(format!(
                "{}-{}",
                std::env::consts::OS,
                std::env::consts::ARCH
            ))
            .join(backend)
    }

    pub fn write_install_manifest(
        &self,
        destination: &Path,
        version: &str,
        backend: &str,
    ) -> Result<()> {
        let manifest = RuntimeManifest {
            runtime: "llama.cpp".to_owned(),
            version: version.to_owned(),
            platform: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            backend: backend.to_owned(),
            archive: "external-install".to_owned(),
            sha256: String::new(),
        };
        write_json_atomic(&destination.join("manifest.json"), &manifest)
    }

    fn managed_root(&self, device: Device) -> PathBuf {
        let backend = runtime_asset(device)
            .map(|asset| asset.backend)
            .unwrap_or("cpu");
        self.managed_destination(MANAGED_VERSION, backend)
    }

    async fn install_managed(&self, device: Device) -> Result<PathBuf> {
        let asset = runtime_asset(device)?;
        let destination = self.managed_destination(asset.version, asset.backend);
        fs::create_dir_all(&self.cache_dir)?;
        let lock_path = self
            .cache_dir
            .join(format!("{}.download.lock", asset.backend));
        let _lock = RuntimeLock::acquire(&lock_path)?;

        if let Some(path) = llama::find_cached_library(&destination)?
            && LlamaLibrary::load(&path).is_ok()
        {
            return Ok(path);
        }

        let archive_dir = self.cache_dir.join("downloads").join(asset.version);
        fs::create_dir_all(&archive_dir)?;
        let archive = archive_dir.join(asset.name);
        if !valid_sha256_file(&archive, asset.sha256)? {
            let part = PathBuf::from(format!("{}.part", archive.display()));
            if part.exists() {
                fs::remove_file(&part)?;
            }
            self.download_archive(&asset, &part).await?;
            if !valid_sha256_file(&part, asset.sha256)? {
                let _ = fs::remove_file(&part);
                bail!("downloaded llama.cpp archive checksum does not match the pinned release")
            }
            fs::rename(&part, &archive)?;
        }

        let staging = PathBuf::from(format!("{}.part", destination.display()));
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(&staging)?;
        if let Err(error) = extract_archive(&archive, asset.kind, &staging) {
            let _ = fs::remove_dir_all(&staging);
            return Err(error).with_context(|| format!("extracting {}", asset.name));
        }
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::rename(&staging, &destination)?;
        let manifest = RuntimeManifest {
            runtime: "llama.cpp".to_owned(),
            version: asset.version.to_owned(),
            platform: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            backend: asset.backend.to_owned(),
            archive: asset.name.to_owned(),
            sha256: asset.sha256.to_owned(),
        };
        write_json_atomic(&destination.join("manifest.json"), &manifest)?;
        llama::find_library(&destination).with_context(|| {
            format!(
                "managed llama.cpp archive did not contain a supported shared library: {}",
                destination.display()
            )
        })
    }

    async fn download_archive(&self, asset: &RuntimeAsset, part: &Path) -> Result<()> {
        let url = format!("{}/{}", MANAGED_RELEASE_URL, asset.name);
        tracing::info!(asset = asset.name, url, "downloading llama.cpp runtime");
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("requesting {url}"))?
            .error_for_status()
            .with_context(|| format!("downloading {url}"))?;
        let mut file = tokio::fs::File::create(part).await?;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            file.write_all(&chunk?).await?;
        }
        file.flush().await?;
        drop(file);
        Ok(())
    }
}

fn runtime_asset(device: Device) -> Result<RuntimeAsset> {
    let asset = match (std::env::consts::OS, std::env::consts::ARCH, device) {
        ("macos", "aarch64", Device::Auto | Device::Cpu | Device::Metal) => RuntimeAsset {
            version: MANAGED_VERSION,
            name: "llama-b10964-bin-macos-arm64.tar.gz",
            sha256: "033c845c1df9bf945ff37bb193238b40910b2244be3e1e637b2ceb5878f1a6f5",
            kind: ArchiveKind::TarGz,
            backend: "cpu",
        },
        ("macos", "x86_64", Device::Auto | Device::Cpu | Device::Metal) => RuntimeAsset {
            version: MANAGED_VERSION,
            name: "llama-b10964-bin-macos-x64.tar.gz",
            sha256: "03430a394d0a169a5e6d8f01c09f48cf58eb026af6fc95940a4a528e2e50cf38",
            kind: ArchiveKind::TarGz,
            backend: "cpu",
        },
        ("linux", "aarch64", Device::Auto | Device::Cpu) => RuntimeAsset {
            version: MANAGED_VERSION,
            name: "llama-b10964-bin-ubuntu-arm64.tar.gz",
            sha256: "5f0e9c95d970892e43380f82ebcab960edfd20a1cd0f7abffa13b29fdb924949",
            kind: ArchiveKind::TarGz,
            backend: "cpu",
        },
        ("linux", "x86_64", Device::Auto | Device::Cpu) => RuntimeAsset {
            version: MANAGED_VERSION,
            name: "llama-b10964-bin-ubuntu-x64.tar.gz",
            sha256: "9abf88aea48a55d0f80edb1ee20220b186848cca0b4e919d71518cfd7ca67443",
            kind: ArchiveKind::TarGz,
            backend: "cpu",
        },
        ("linux", "aarch64", Device::Vulkan) => RuntimeAsset {
            version: MANAGED_VERSION,
            name: "llama-b10964-bin-ubuntu-vulkan-arm64.tar.gz",
            sha256: "f7864baa0edf5a059fb42c5efb5aceb960edfd20a1cd0f7abffa13b29fdb924949",
            kind: ArchiveKind::TarGz,
            backend: "vulkan",
        },
        ("linux", "x86_64", Device::Vulkan) => RuntimeAsset {
            version: MANAGED_VERSION,
            name: "llama-b10964-bin-ubuntu-vulkan-x64.tar.gz",
            sha256: "55d1e58e14c11eedea090bf088fdeefbfe7b4b09ee03bf6dba9834651769afcf",
            kind: ArchiveKind::TarGz,
            backend: "vulkan",
        },
        ("windows", "x86_64", Device::Auto | Device::Cpu) => RuntimeAsset {
            version: MANAGED_VERSION,
            name: "llama-b10964-bin-win-cpu-x64.zip",
            sha256: "917f39c076402c421224824607397af20f53625a60defc20e8dd22446bf4c5d7",
            kind: ArchiveKind::Zip,
            backend: "cpu",
        },
        ("windows", "aarch64", Device::Auto | Device::Cpu) => RuntimeAsset {
            version: MANAGED_VERSION,
            name: "llama-b10964-bin-win-cpu-arm64.zip",
            sha256: "4b6a004b076eea47c318bea35cf1db2ff2bf037738b04645646ae8d7c3159478",
            kind: ArchiveKind::Zip,
            backend: "cpu",
        },
        ("windows", "x86_64", Device::Vulkan) => RuntimeAsset {
            version: MANAGED_VERSION,
            name: "llama-b10964-bin-win-vulkan-x64.zip",
            sha256: "1ee3ad952f4ba71f438bd6d7bebef19e1c7af04adcaa35d08b4ddabb27d4c642",
            kind: ArchiveKind::Zip,
            backend: "vulkan",
        },
        ("windows", "x86_64", Device::Cuda) => RuntimeAsset {
            version: MANAGED_VERSION,
            name: "llama-b10964-bin-win-cuda-12.4-x64.zip",
            sha256: "264f20d7ee3860aecca9ec12418357a9f3e80349a2b186f66c63859ded1a9593",
            kind: ArchiveKind::Zip,
            backend: "cuda",
        },
        _ => bail!(
            "no pinned managed llama.cpp runtime is available for {} / {} / {}; install an external runtime or choose --device cpu",
            std::env::consts::OS,
            std::env::consts::ARCH,
            device.as_str()
        ),
    };
    Ok(asset)
}

fn valid_sha256_file(path: &Path, expected: &str) -> Result<bool> {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file() {
        return Ok(false);
    }
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex::encode(hasher.finalize()) == expected)
}

fn extract_archive(archive: &Path, kind: ArchiveKind, destination: &Path) -> Result<()> {
    match kind {
        ArchiveKind::TarGz => {
            let file = File::open(archive)?;
            let decoder = GzDecoder::new(file);
            let mut archive = Archive::new(decoder);
            for entry in archive.entries()? {
                let mut entry = entry?;
                let relative = safe_archive_path(&entry.path()?)?;
                if relative.as_os_str().is_empty() {
                    continue;
                }
                let target = destination.join(relative);
                if entry.header().entry_type().is_dir() {
                    fs::create_dir_all(&target)?;
                } else if entry.header().entry_type().is_file() {
                    if let Some(parent) = target.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    entry.unpack(&target)?;
                } else if entry.header().entry_type().is_symlink() {
                    let link_name = entry.link_name()?.context("symlink entry has no target")?;
                    let link_name = safe_link_target(&link_name)?;
                    if let Some(parent) = target.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    #[cfg(unix)]
                    std::os::unix::fs::symlink(&link_name, &target)?;
                    #[cfg(windows)]
                    std::os::windows::fs::symlink_file(&link_name, &target)?;
                } else if entry.header().entry_type().is_hard_link() {
                    // Hard links are not required by the published runtime packages. Refuse
                    // them rather than allowing an archive to alias an arbitrary path.
                    bail!("refusing hard link in llama.cpp archive")
                } else {
                    bail!("refusing non-regular file in llama.cpp archive")
                }
            }
        }
        ArchiveKind::Zip => {
            let file = File::open(archive)?;
            let mut archive = ZipArchive::new(file)?;
            for index in 0..archive.len() {
                let mut entry = archive.by_index(index)?;
                let relative = safe_archive_path(Path::new(entry.name()))?;
                if relative.as_os_str().is_empty() {
                    continue;
                }
                let target = destination.join(relative);
                if entry.is_dir() {
                    fs::create_dir_all(&target)?;
                } else {
                    if let Some(parent) = target.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    let mut output = File::create(&target)?;
                    std::io::copy(&mut entry, &mut output)?;
                    output.flush()?;
                }
            }
        }
    }
    Ok(())
}

fn safe_archive_path(path: &Path) -> Result<PathBuf> {
    let mut safe = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => safe.push(value),
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) | Component::ParentDir => {
                bail!("archive contains an unsafe path: {}", path.display())
            }
        }
    }
    Ok(safe)
}

fn safe_link_target(path: &Path) -> Result<PathBuf> {
    let mut safe = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => safe.push(value),
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) | Component::ParentDir => {
                bail!(
                    "archive contains an unsafe symlink target: {}",
                    path.display()
                )
            }
        }
    }
    Ok(safe)
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    let part = PathBuf::from(format!("{}.part", path.display()));
    let mut file = File::create(&part)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(part, path)?;
    Ok(())
}

struct RuntimeLock {
    path: PathBuf,
}

impl RuntimeLock {
    fn acquire(path: &Path) -> Result<Self> {
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(mut file) => {
                writeln!(file, "pid={}", std::process::id())?;
                Ok(Self {
                    path: path.to_path_buf(),
                })
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if stale_lock(path) {
                    let _ = fs::remove_file(path);
                    return Self::acquire(path);
                }
                bail!(
                    "another process is updating the runtime cache: {}",
                    path.display()
                )
            }
            Err(error) => {
                Err(error).with_context(|| format!("creating runtime lock {}", path.display()))
            }
        }
    }
}

impl Drop for RuntimeLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn stale_lock(path: &Path) -> bool {
    let Ok(contents) = fs::read_to_string(path) else {
        return false;
    };
    let Some(pid) = contents
        .strip_prefix("pid=")
        .and_then(|value| value.trim().parse::<i32>().ok())
    else {
        return false;
    };
    #[cfg(unix)]
    {
        let result = unsafe { libc::kill(pid, 0) };
        if result == 0 {
            return false;
        }
        std::io::Error::last_os_error().raw_os_error() != Some(libc::EPERM)
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        path.metadata()
            .and_then(|metadata| metadata.modified())
            .and_then(|modified| modified.elapsed().map_err(std::io::Error::other))
            .map(|age| age > std::time::Duration::from_secs(6 * 60 * 60))
            .unwrap_or(false)
    }
}
