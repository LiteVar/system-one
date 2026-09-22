use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::config::AppConfig;

pub const MODEL_ID: &str = "laya-multilingual";
pub const MODEL_REVISION: &str = "6235959603bf3d316379404b9130a10b8a406446";
const MODEL_REPO: &str = "Weidows/laya-multilingual-GGUF";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetSpec {
    pub name: String,
    pub size: u64,
    pub sha256: String,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub id: String,
    pub repo: String,
    pub revision: String,
    pub assets: Vec<AssetSpec>,
}

impl ModelManifest {
    pub fn laya_multilingual() -> Self {
        Self {
            id: MODEL_ID.to_owned(),
            repo: MODEL_REPO.to_owned(),
            revision: MODEL_REVISION.to_owned(),
            assets: vec![
                AssetSpec {
                    name: "laya-multilingual-Q8_0.gguf".to_owned(),
                    size: 341_095_616,
                    sha256: "f97fd6808375801535c42d792fe1db14967ad067eea9cac7d6901d405d5de43e"
                        .to_owned(),
                    kind: "gguf".to_owned(),
                },
                AssetSpec {
                    name: "laya-multilingual-head.safetensors".to_owned(),
                    size: 29_942_522,
                    sha256: "b016634e9741448289e4b6786ab502e05056c57235e24d2e839adb2c218edf3f"
                        .to_owned(),
                    kind: "safetensors".to_owned(),
                },
                AssetSpec {
                    name: "tokenizer.json".to_owned(),
                    size: 34_363_188,
                    sha256: "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f"
                        .to_owned(),
                    kind: "tokenizer".to_owned(),
                },
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub root: PathBuf,
    pub manifest: ModelManifest,
    pub gguf: PathBuf,
    pub head: PathBuf,
    pub tokenizer: PathBuf,
}

impl ModelPaths {
    pub fn from_root(root: PathBuf, manifest: ModelManifest) -> Self {
        Self {
            gguf: root.join(&manifest.assets[0].name),
            head: root.join(&manifest.assets[1].name),
            tokenizer: root.join(&manifest.assets[2].name),
            root,
            manifest,
        }
    }
}

pub struct CacheManager {
    root: PathBuf,
    client: Client,
}

impl CacheManager {
    pub fn new(config: &AppConfig) -> Result<Self> {
        fs::create_dir_all(config.cache_dir.join("models"))
            .with_context(|| format!("creating cache directory {}", config.cache_dir.display()))?;
        Ok(Self {
            root: config.cache_dir.clone(),
            client: Client::builder()
                .user_agent(concat!("system-one/", env!("CARGO_PKG_VERSION")))
                .build()
                .context("building HTTP client")?,
        })
    }

    pub fn model_root(&self, id: &str) -> PathBuf {
        self.root.join("models").join(id).join(MODEL_REVISION)
    }

    pub fn model_paths(&self, id: &str) -> Result<ModelPaths> {
        let manifest = match id {
            MODEL_ID => ModelManifest::laya_multilingual(),
            _ => bail!("unknown model `{id}`"),
        };
        Ok(ModelPaths::from_root(self.model_root(id), manifest))
    }

    pub fn model_is_valid(&self, id: &str) -> Result<bool> {
        let paths = self.model_paths(id)?;
        for asset in &paths.manifest.assets {
            let path = paths.root.join(&asset.name);
            if validate_asset(&path, asset).is_err() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn list_models(&self) -> Result<Vec<ModelStatus>> {
        let mut result = Vec::new();
        let paths = self.model_paths(MODEL_ID)?;
        result.push(ModelStatus {
            id: MODEL_ID.to_owned(),
            revision: paths.manifest.revision.clone(),
            path: paths.root,
            valid: self.model_is_valid(MODEL_ID)?,
        });
        Ok(result)
    }

    pub async fn ensure_model(&self, id: &str, offline: bool) -> Result<ModelPaths> {
        let paths = self.model_paths(id)?;
        fs::create_dir_all(&paths.root)
            .with_context(|| format!("creating model cache {}", paths.root.display()))?;

        let _lock = CacheLock::acquire(&paths.root.join(".download.lock"))?;
        let mut manifest = paths.manifest.clone();
        for asset in &manifest.assets {
            let path = paths.root.join(&asset.name);
            if validate_asset(&path, asset).is_ok() {
                continue;
            }
            if path.exists() {
                fs::remove_file(&path)
                    .with_context(|| format!("removing invalid cached asset {}", path.display()))?;
            }
            if offline {
                bail!(
                    "offline mode: required model asset is missing or invalid: {}",
                    path.display()
                );
            }
            self.download_asset(&manifest, asset, &path).await?;
        }
        manifest.assets.shrink_to_fit();
        let manifest_path = paths.root.join("manifest.json");
        let bytes = serde_json::to_vec_pretty(&manifest)?;
        atomic_write(&manifest_path, &bytes)?;
        Ok(ModelPaths::from_root(paths.root, manifest))
    }

    async fn download_asset(
        &self,
        manifest: &ModelManifest,
        asset: &AssetSpec,
        path: &Path,
    ) -> Result<()> {
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}?download=true",
            manifest.repo, manifest.revision, asset.name
        );
        tracing::info!(asset = %asset.name, "downloading model asset");
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("requesting {url}"))?
            .error_for_status()
            .with_context(|| format!("downloading {}", asset.name))?;
        if let Some(length) = response.content_length()
            && length != asset.size
        {
            bail!(
                "unexpected size for {}: server reported {}, expected {}",
                asset.name,
                length,
                asset.size
            );
        }

        let part = PathBuf::from(format!("{}.part", path.display()));
        if part.exists() {
            fs::remove_file(&part)
                .with_context(|| format!("removing stale partial file {}", part.display()))?;
        }
        let mut file = tokio::fs::File::create(&part).await?;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.with_context(|| format!("reading {}", asset.name))?;
            file.write_all(&chunk).await?;
        }
        file.flush().await?;
        drop(file);

        if let Err(error) = validate_asset(&part, asset) {
            let _ = fs::remove_file(&part);
            return Err(error).with_context(|| format!("validating downloaded {}", asset.name));
        }
        fs::rename(&part, path).with_context(|| format!("committing {}", path.display()))?;
        Ok(())
    }

    pub fn remove_model(&self, id: &str) -> Result<()> {
        let path = self.root.join("models").join(id);
        if path.exists() {
            fs::remove_dir_all(&path)
                .with_context(|| format!("removing model cache {}", path.display()))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    pub id: String,
    pub revision: String,
    pub path: PathBuf,
    pub valid: bool,
}

fn validate_asset(path: &Path, asset: &AssetSpec) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("reading asset metadata {}", path.display()))?;
    if !metadata.is_file() || metadata.len() != asset.size {
        bail!("asset {} has unexpected size", path.display());
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
    let digest = hex::encode(hasher.finalize());
    if digest != asset.sha256 {
        bail!(
            "asset {} has checksum {}, expected {}",
            path.display(),
            digest,
            asset.sha256
        );
    }
    match asset.kind.as_str() {
        "gguf" => {
            let mut file = File::open(path)?;
            let mut magic = [0_u8; 4];
            file.read_exact(&mut magic)?;
            if &magic != b"GGUF" {
                bail!("{} is not a GGUF file", path.display());
            }
        }
        "tokenizer" => {
            let file = File::open(path)?;
            let _: serde_json::Value = serde_json::from_reader(file)
                .with_context(|| format!("parsing tokenizer {}", path.display()))?;
        }
        "safetensors" => {
            let bytes = fs::read(path)?;
            safetensors::SafeTensors::deserialize(&bytes)
                .map_err(|e| anyhow::anyhow!(e.to_string()))
                .with_context(|| format!("parsing safetensors {}", path.display()))?;
        }
        _ => {}
    }
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let part = path.with_extension("json.part");
    let mut file = File::create(&part)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(part, path)?;
    Ok(())
}

struct CacheLock {
    path: PathBuf,
}

impl CacheLock {
    fn acquire(path: &Path) -> Result<Self> {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(mut file) => {
                let _ = writeln!(file, "pid={}", std::process::id());
                Ok(Self {
                    path: path.to_path_buf(),
                })
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if stale_lock(path) {
                    let _ = fs::remove_file(path);
                    return Self::acquire(path);
                }
                bail!("another process is updating the cache: {}", path.display())
            }
            Err(error) => {
                Err(error).with_context(|| format!("creating cache lock {}", path.display()))
            }
        }
    }
}

impl Drop for CacheLock {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AppConfig, Device};

    #[test]
    fn validates_and_rejects_cached_assets_by_checksum() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.json");
        fs::write(&path, b"{}").unwrap();
        let digest = hex::encode(Sha256::digest(b"{}"));
        let asset = AssetSpec {
            name: "tokenizer.json".to_owned(),
            size: 2,
            sha256: digest,
            kind: "tokenizer".to_owned(),
        };
        assert!(validate_asset(&path, &asset).is_ok());
        fs::write(&path, b"broken").unwrap();
        assert!(validate_asset(&path, &asset).is_err());
    }

    #[tokio::test]
    async fn offline_missing_model_fails_without_creating_partial_assets() {
        let directory = tempfile::tempdir().unwrap();
        let config = AppConfig {
            cache_dir: directory.path().to_path_buf(),
            offline: true,
            llama_cpp_home: None,
            device: Device::Cpu,
            model: MODEL_ID.to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 8080,
        };
        let cache = CacheManager::new(&config).unwrap();
        let error = cache.ensure_model(MODEL_ID, true).await.unwrap_err();
        assert!(error.to_string().contains("offline mode"));
        let root = cache.model_root(MODEL_ID);
        assert!(!root.join("laya-multilingual-Q8_0.gguf.part").exists());
    }
}
