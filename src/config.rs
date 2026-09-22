use std::env;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::ValueEnum;
use directories::BaseDirs;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Device {
    Auto,
    Cpu,
    Metal,
    Vulkan,
    Cuda,
}

impl Device {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub cache_dir: PathBuf,
    pub offline: bool,
    pub llama_cpp_home: Option<PathBuf>,
    pub device: Device,
    pub model: String,
    pub host: String,
    pub port: u16,
}

pub fn default_cache_dir() -> Result<PathBuf> {
    if let Some(value) = env::var_os("SYSTEM_ONE_CACHE") {
        return Ok(PathBuf::from(value));
    }
    let base = BaseDirs::new().context("could not determine the platform cache directory")?;
    Ok(base.cache_dir().join("system-one"))
}

pub fn default_config_path() -> Option<PathBuf> {
    BaseDirs::new().map(|base| base.config_dir().join("system-one").join("config.toml"))
}
