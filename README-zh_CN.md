# System One

[English](README.md) | 简体中文

System One 是一个开源、本地优先、跨平台的 System One 决策模型运行时。它提供兼容 Jev 形状的 `POST /v1/systemone` API 和原生 CLI，不需要 Python、PyTorch、Transformers、Conda 或虚拟环境。

当前首个后端是 Laya Multilingual。公共 API 有意与具体模型解耦；Laya 只是 `DecisionModel` 抽象的一个实现。

本项目独立开发，与 TypeSafe AI、Jev 或其他上游项目没有隶属、背书或官方认证关系。

## 当前状态

当前为 alpha 版本（`0.1.0-alpha.1`）。运行时包含原生 tokenizer、Laya 决策头、模型缓存、Jev 形状的 API、CLI、受管理的 llama.cpp 运行时配置，以及动态 llama.cpp FFI。首次运行会下载固定版本的模型资源；当本机找不到兼容库时，还会为当前平台下载固定版本的 CPU/Metal/Vulkan llama.cpp 运行时。选中的运行时会在进程生命周期内常驻。

原生路径已经在 macOS arm64 上完成端到端 smoke test。跨平台 CI 会检查支持目标的源码编译和 release 构建；硬件后端和 reference vector 验证仍属于发布前检查项。

具体的参考行为和兼容性说明见 [docs/laya-runtime.md](docs/laya-runtime.md)。

## 快速开始

```sh
cargo run -- model pull laya-multilingual
cargo run -- serve
```

服务器默认绑定 `127.0.0.1:8080`。

如果启动时缺少模型或运行时资源，服务器仍会提供 `/health` 和 `/v1/models`，并返回 `model: "unavailable"`；推理请求会返回 HTTP 503 和诊断信息。这样可以用于离线环境或容器健康检查，同时不会隐藏缺失的模型或运行时。

如果要使用已有的 llama.cpp 构建：

```sh
system-one --llama-cpp-home /path/to/llama.cpp serve
```

运行时按以下顺序查找：显式指定的 `--llama-cpp-home`、`SYSTEM_ONE_LLAMA_CPP_HOME`、`SYSTEM_ONE_LLAMA_LIBRARY`、`PATH` 中的库、本地运行时缓存，最后才是固定版本的受管理下载。使用 `system-one runtime info` 可以查看当前选中的库。

可选的 TOML 配置文件位于平台配置目录下的 `system-one/config.toml`；也可以通过 `--config PATH` 或 `SYSTEM_ONE_CONFIG` 指定。命令行参数和环境变量会覆盖配置文件中的值：

```toml
cache_dir = "/var/tmp/system-one"
model = "laya-multilingual"
device = "cpu"

[runtime.llama_cpp]
home = "/opt/llama.cpp"
```

```sh
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "model": "laya-multilingual",
    "state": "The user wants to restart nginx",
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "What should happen next?",
        "criteria": {
          "execute": "Execute the restart",
          "confirm": "Ask for confirmation",
          "reject": "Reject the request"
        }
      }
    }
  }'
```

## 离线模式

```sh
system-one serve --offline
```

离线模式不会尝试下载模型文件或 llama.cpp。缺失的资源会被明确报告。

## 模型和运行时缓存

默认缓存目录使用平台缓存目录。可以通过 `--cache-dir /path/to/cache` 或 `SYSTEM_ONE_CACHE` 覆盖。模型文件存放在带版本的目录下，复用前会进行 SHA-256 校验。下载过程使用 `.part` 文件，损坏或中断的文件不会被当作完整资源使用。

模型由三个独立资源组成：Q8_0 encoder GGUF、Laya 决策头和 `tokenizer.json`。GGUF 不会被当作完整的决策模型。`scripts/download-model.sh` 是内置模型缓存命令的便捷包装脚本。

## API 格式和差异

`POST /v1/systemone` 接受字符串、对象或数组形式的 `state`，并将问题定义映射为 Laya 的 `choice`、`score` 和 `noul` 类型。Choice 既支持数组形式的 `criteria`/`options`，也支持“标签到描述”的对象形式。结果包含概率、choice/score 的归一化熵置信度、Noul 概率和置信度，以及 Laya action 概率。

`GET /health` 会将进程存活状态与模型可用性分开报告。模型已加载时返回 `status: "ok"`；模型不可用时返回 `status: "degraded"` 和启动诊断信息。`GET /v1/models` 会包含相同的加载/错误状态。

这是一个 Jev 形状的适配器，不代表获得了官方 Jev 协议认证。适配器与模型后端隔离，因此可以在不修改公共推理契约的情况下添加其他决策模型。

## 架构

```text
HTTP / CLI
   -> 与模型无关的决策适配器
   -> Laya 后端
      -> Hugging Face Rust tokenizer
      -> llama.cpp encoder FFI -> 每个 token 的 hidden states
      -> 原生 safetensors 决策头
      -> 校准 -> 带类型的概率答案
```

encoder 上下文通过互斥锁串行化，因为不能假设 llama.cpp 上下文支持无限制并发访问；独立的决策头计算和 HTTP 请求处理仍然可以围绕这段临界区异步进行。

当前常驻上下文每个请求最多接受 16 个问题，每个问题最多 1024 个 token；超过限制的请求会返回结构化校验错误。

## 支持的平台

基线支持 macOS arm64/x86_64、Windows x86_64 和 Linux x86_64/arm64 的 CPU 推理。只要上游提供相应二进制，受管理运行时也会提供支持 Metal 的 macOS 包和 Vulkan 包。CUDA 可以通过显式安装的运行时以及固定版本的 Windows CUDA 包使用；CPU 始终是回退方案。

使用以下命令构建原生可执行文件：

```sh
cargo build --release
```

## 开发与测试

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --locked
cargo build --release --locked
```

常规测试套件不需要联网。要运行原生模型 smoke test，请将 `SYSTEM_ONE_E2E_CACHE` 设置为包含已验证模型和 llama.cpp 运行时的缓存目录：

```sh
SYSTEM_ONE_E2E_CACHE="$HOME/.cache/system-one" cargo test --test e2e_smoke -- --nocapture
```

发布验证清单见 [docs/testing.md](docs/testing.md)，贡献要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 发布

推送 `v*` tag 会运行发布工作流，并为 macOS arm64、macOS x86_64、Linux x86_64、Linux arm64 和 Windows x86_64 发布归档，同时生成 `SHA256SUMS`。可执行文件不会打包模型权重；首次运行时会将模型和运行时放入用户缓存。

## 许可证

本项目源码采用 MIT License，版权所有者为 LiteVar。详见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。模型文件和运行时库沿用各自上游许可证，不随本仓库一同分发。

项目地址：<https://github.com/LiteVar/system-one>。
