mod api;
mod cache;
mod cli;
mod config;
mod domain;
mod models;
mod runtime;
mod service;

use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Map, Value};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

use crate::cache::CacheManager;
use crate::cli::{Cli, Command, ModelCommand, RuntimeCommand};
use crate::config::{AppConfig, Device, default_cache_dir, default_config_path};
use crate::domain::{Question, SystemOneRequest};
use crate::runtime::RuntimeManager;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    let config = build_config(&cli)?;
    dispatch(cli.command, config).await
}

fn init_logging(verbose: bool) {
    let default = if verbose { "info" } else { "warn" };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .json()
        .try_init();
}

fn build_config(cli: &Cli) -> Result<AppConfig> {
    let file = load_file_config(cli.config.as_deref())?;
    let cache_dir = cli
        .cache_dir
        .clone()
        .or(file.cache_dir)
        .unwrap_or(default_cache_dir()?);
    let llama_cpp_home = cli.llama_cpp_home.clone().or_else(|| {
        file.llama_cpp_home.or_else(|| {
            file.runtime
                .as_ref()
                .and_then(|runtime| runtime.llama_cpp.as_ref())
                .and_then(|llama_cpp| llama_cpp.home.clone())
        })
    });
    let device = match cli.device {
        Some(device) => device,
        None => file
            .device
            .as_deref()
            .map(parse_device)
            .transpose()?
            .unwrap_or(Device::Auto),
    };
    Ok(AppConfig {
        cache_dir,
        offline: cli.offline || file.offline.unwrap_or(false),
        llama_cpp_home,
        device,
        model: cli
            .model
            .clone()
            .or(file.model)
            .unwrap_or_else(|| "laya-multilingual".to_owned()),
        host: cli
            .host
            .clone()
            .or(file.host)
            .unwrap_or_else(|| "127.0.0.1".to_owned()),
        port: cli.port.or(file.port).unwrap_or(8080),
    })
}

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    cache_dir: Option<std::path::PathBuf>,
    offline: Option<bool>,
    llama_cpp_home: Option<std::path::PathBuf>,
    device: Option<String>,
    model: Option<String>,
    host: Option<String>,
    port: Option<u16>,
    runtime: Option<FileRuntimeConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct FileRuntimeConfig {
    llama_cpp: Option<FileLlamaCppConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct FileLlamaCppConfig {
    home: Option<std::path::PathBuf>,
}

fn load_file_config(path: Option<&Path>) -> Result<FileConfig> {
    let (path, explicit) = match path {
        Some(path) => (path.to_path_buf(), true),
        None => {
            let Some(path) = default_config_path() else {
                return Ok(FileConfig::default());
            };
            (path, false)
        }
    };
    if !path.exists() {
        if explicit {
            bail!("configuration file does not exist: {}", path.display());
        }
        return Ok(FileConfig::default());
    }
    let text = fs::read_to_string(&path)
        .with_context(|| format!("reading configuration file {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing configuration file {}", path.display()))
}

fn parse_device(value: &str) -> Result<Device> {
    Device::from_str(value, true)
        .map_err(|error| anyhow::anyhow!("invalid configured device `{value}`: {error}"))
}

async fn dispatch(command: Command, config: AppConfig) -> Result<()> {
    match command {
        Command::Model { command } => model_command(command, &config).await,
        Command::Runtime { command } => runtime_command(command, &config).await,
        Command::Serve => serve(config).await,
        Command::Decide(args) => decide(config, args).await,
        Command::Bench(args) => bench(config, args).await,
    }
}

async fn model_command(command: ModelCommand, config: &AppConfig) -> Result<()> {
    let cache = CacheManager::new(config)?;
    match command {
        ModelCommand::Pull(args) => {
            let model = args.model.as_deref().unwrap_or(&config.model);
            let paths = cache.ensure_model(model, config.offline).await?;
            println!("Model cached at {}", paths.root.display());
        }
        ModelCommand::List(_args) => {
            println!("{}", serde_json::to_string_pretty(&cache.list_models()?)?);
        }
        ModelCommand::Path(args) => {
            let model = args.model.as_deref().unwrap_or(&config.model);
            println!("{}", cache.model_paths(model)?.root.display());
        }
        ModelCommand::Remove(args) => {
            let model = args.model.as_deref().unwrap_or(&config.model);
            cache.remove_model(model)?;
            println!("Removed cached model {}", model);
        }
    }
    Ok(())
}

async fn runtime_command(command: RuntimeCommand, config: &AppConfig) -> Result<()> {
    let manager = RuntimeManager::new(config);
    match command {
        RuntimeCommand::Info => println!(
            "{}",
            serde_json::to_string_pretty(&manager.info(config.device))?
        ),
        RuntimeCommand::Path => println!(
            "{}",
            manager.resolve_library_path_for(config.device)?.0.display()
        ),
        RuntimeCommand::Install(args) | RuntimeCommand::Update(args) => {
            if let Some(from_home) = args.from_home {
                let destination = manager.managed_destination(&args.version, &args.backend);
                copy_dir_recursive(&from_home, &destination)?;
                manager.write_install_manifest(&destination, &args.version, &args.backend)?;
                println!("Installed llama.cpp runtime at {}", destination.display());
            } else {
                manager.ensure(config.device, config.offline).await?;
                let (path, source) = manager.resolve_library_path_for(config.device)?;
                println!("Using {source} llama.cpp runtime at {}", path.display());
            }
        }
    }
    Ok(())
}

async fn serve(config: AppConfig) -> Result<()> {
    let runtime_manager = RuntimeManager::new(&config);
    let loaded = async {
        let paths = CacheManager::new(&config)?
            .ensure_model(&config.model, config.offline)
            .await?;
        runtime_manager
            .ensure(config.device, config.offline)
            .await?;
        service::InferenceService::load(&config, paths)
    }
    .await;
    let service: Arc<dyn service::DecisionModel> = match loaded {
        Ok(service) => service,
        Err(error) => {
            let message = format!("{error:#}");
            tracing::error!(model = %config.model, error = %message, "model is unavailable");
            service::UnavailableDecisionModel::new(
                config.model.clone(),
                runtime_manager.info(config.device),
                message,
            )
        }
    };
    let app = api::router(service);
    let address: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .with_context(|| format!("invalid listen address {}:{}", config.host, config.port))?;
    let listener = TcpListener::bind(address).await?;
    tracing::info!(address = %address, "server listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("HTTP server failed")?;
    Ok(())
}

async fn decide(config: AppConfig, args: cli::DecideArgs) -> Result<()> {
    if args.choice.is_empty() {
        bail!("--choice requires at least one comma-separated option")
    }
    let paths = CacheManager::new(&config)?
        .ensure_model(&config.model, config.offline)
        .await?;
    RuntimeManager::new(&config)
        .ensure(config.device, config.offline)
        .await?;
    let service = service::InferenceService::load(&config, paths)?;
    let mut criteria = Map::new();
    for option in &args.choice {
        if option.trim().is_empty() {
            bail!("choice options must not be empty")
        }
        criteria.insert(option.clone(), Value::Null);
    }
    let mut questions = IndexMap::new();
    questions.insert(
        "decision".to_owned(),
        Question {
            kind: "choice".to_owned(),
            instructions: Value::String(args.instructions),
            criteria: Some(Value::Object(criteria)),
        },
    );
    let response = service.decide(&SystemOneRequest {
        model: Some(config.model),
        state: Value::String(args.state),
        questions,
    })?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}

async fn bench(config: AppConfig, args: cli::BenchArgs) -> Result<()> {
    if args.iterations == 0 || args.questions == 0 {
        bail!("iterations and questions must be positive")
    }
    let paths = CacheManager::new(&config)?
        .ensure_model(&config.model, config.offline)
        .await?;
    RuntimeManager::new(&config)
        .ensure(config.device, config.offline)
        .await?;
    let service = service::InferenceService::load(&config, paths)?;
    let mut questions = IndexMap::new();
    for index in 0..args.questions {
        let mut criteria = Map::new();
        criteria.insert("execute".to_owned(), Value::String("execute".to_owned()));
        criteria.insert("confirm".to_owned(), Value::String("confirm".to_owned()));
        questions.insert(
            format!("q{index}"),
            Question {
                kind: "choice".to_owned(),
                instructions: Value::String("What should happen next?".to_owned()),
                criteria: Some(Value::Object(criteria)),
            },
        );
    }
    let request = SystemOneRequest {
        model: Some(config.model),
        state: Value::String(args.state),
        questions,
    };
    let mut samples = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        let (_, timings) = service.decide_with_timings(&request)?;
        samples.push(timings);
    }
    let mut tokenization = samples
        .iter()
        .map(|sample| sample.tokenization_ms)
        .collect::<Vec<_>>();
    let mut encoder = samples
        .iter()
        .map(|sample| sample.encoder_ms)
        .collect::<Vec<_>>();
    let mut head = samples
        .iter()
        .map(|sample| sample.head_ms)
        .collect::<Vec<_>>();
    let mut total = samples
        .iter()
        .map(|sample| sample.total_ms)
        .collect::<Vec<_>>();
    let p50 = |values: &mut Vec<f64>| percentile(values, 50);
    let p95 = |values: &mut Vec<f64>| percentile(values, 95);
    let total_average = total.iter().sum::<f64>() / total.len() as f64;
    println!(
        "iterations={} questions={} tokenization_p50_ms={:.3} encoder_p50_ms={:.3} head_p50_ms={:.3} total_p50_ms={:.3} total_p95_ms={:.3} requests_per_sec={:.2}",
        args.iterations,
        args.questions,
        p50(&mut tokenization),
        p50(&mut encoder),
        p50(&mut head),
        p50(&mut total),
        p95(&mut total),
        1000.0 / total_average
    );
    Ok(())
}

fn percentile(values: &mut [f64], percentile: usize) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) * percentile) / 100]
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> Result<()> {
    if !source.is_dir() {
        bail!("runtime source is not a directory: {}", source.display())
    }
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let target_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            copy_dir_recursive(&source_path, &target_path)?;
        } else {
            fs::copy(&source_path, &target_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_file_values_are_used_and_cli_values_win() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("config.toml");
        fs::write(
            &config_path,
            r#"
cache_dir = "/tmp/configured-system-one"
model = "laya-multilingual"
device = "cpu"
offline = true

[runtime.llama_cpp]
home = "/tmp/llama-configured"
"#,
        )
        .unwrap();
        let cli = Cli::try_parse_from([
            "system-one",
            "--config",
            config_path.to_str().unwrap(),
            "--device",
            "metal",
            "model",
            "list",
        ])
        .unwrap();
        let config = build_config(&cli).unwrap();
        assert_eq!(config.cache_dir, Path::new("/tmp/configured-system-one"));
        assert_eq!(config.device, Device::Metal);
        assert_eq!(
            config.llama_cpp_home.as_deref(),
            Some(Path::new("/tmp/llama-configured"))
        );
        assert!(config.offline);
    }
}
