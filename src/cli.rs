use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::config::Device;

#[derive(Debug, Parser)]
#[command(
    name = "system-one",
    version,
    about = "Local-first System One decision runtime"
)]
pub struct Cli {
    #[arg(long, global = true, env = "SYSTEM_ONE_CONFIG")]
    pub config: Option<PathBuf>,
    #[arg(long, global = true, env = "SYSTEM_ONE_CACHE")]
    pub cache_dir: Option<PathBuf>,
    #[arg(long, global = true)]
    pub offline: bool,
    #[arg(long, global = true, env = "SYSTEM_ONE_LLAMA_CPP_HOME")]
    pub llama_cpp_home: Option<PathBuf>,
    #[arg(long, global = true, env = "SYSTEM_ONE_DEVICE", value_enum)]
    pub device: Option<Device>,
    #[arg(long, global = true, env = "SYSTEM_ONE_MODEL")]
    pub model: Option<String>,
    #[arg(long, global = true)]
    pub host: Option<String>,
    #[arg(long, global = true)]
    pub port: Option<u16>,
    #[arg(short, long, global = true)]
    pub verbose: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Serve,
    Decide(DecideArgs),
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    Runtime {
        #[command(subcommand)]
        command: RuntimeCommand,
    },
    Bench(BenchArgs),
}

#[derive(Debug, Subcommand)]
pub enum ModelCommand {
    Pull(ModelNameArgs),
    List(ModelNameArgs),
    Path(ModelNameArgs),
    Remove(ModelNameArgs),
}

#[derive(Debug, Args)]
pub struct ModelNameArgs {
    pub model: Option<String>,
}

#[derive(Debug, Subcommand)]
pub enum RuntimeCommand {
    Info,
    Path,
    Install(RuntimeInstallArgs),
    Update(RuntimeInstallArgs),
}

#[derive(Debug, Args)]
pub struct RuntimeInstallArgs {
    #[arg(long, value_name = "DIR")]
    pub from_home: Option<PathBuf>,
    #[arg(long, default_value = "local")]
    pub version: String,
    #[arg(long, default_value = "cpu")]
    pub backend: String,
}

#[derive(Debug, Args)]
pub struct DecideArgs {
    #[arg(long)]
    pub state: String,
    #[arg(long, value_delimiter = ',')]
    pub choice: Vec<String>,
    #[arg(long, default_value = "Choose the best option for this state.")]
    pub instructions: String,
}

#[derive(Debug, Args)]
pub struct BenchArgs {
    #[arg(long, default_value_t = 10)]
    pub iterations: usize,
    #[arg(long, default_value_t = 1)]
    pub questions: usize,
    #[arg(long, default_value = "The user wants to restart nginx.")]
    pub state: String,
}
