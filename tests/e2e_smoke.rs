use std::env;
use std::process::Command;

use serde_json::Value;

#[test]
fn native_model_decision_smoke() {
    let Some(cache) = env::var_os("SYSTEM_ONE_E2E_CACHE") else {
        eprintln!("skipping native model smoke test: SYSTEM_ONE_E2E_CACHE is not set");
        return;
    };
    let executable = env!("CARGO_BIN_EXE_system-one");
    let output = Command::new(executable)
        .args([
            "--cache-dir",
            cache.to_str().expect("cache path must be valid UTF-8"),
            "--offline",
            "--device",
            "cpu",
            "decide",
            "--state",
            "The user wants to restart nginx",
            "--choice",
            "execute,confirm,reject",
            "--instructions",
            "What should happen next?",
        ])
        .env_remove("SYSTEM_ONE_CONFIG")
        .env_remove("SYSTEM_ONE_LLAMA_CPP_HOME")
        .env_remove("SYSTEM_ONE_LLAMA_LIBRARY")
        .env_remove("SYSTEM_ONE_CACHE")
        .env_remove("SYSTEM_ONE_DEVICE")
        .env_remove("SYSTEM_ONE_MODEL")
        .output()
        .expect("failed to start system-one");
    assert!(
        output.status.success(),
        "system-one failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let response: Value = serde_json::from_slice(&output.stdout).expect("invalid JSON response");
    assert_eq!(response["model"], "laya-multilingual");
    assert!(response["usage"]["input_tokens"].as_u64().unwrap_or(0) > 0);

    let answer = &response["answers"]["decision"];
    assert_eq!(answer["type"], "choice");
    assert!(answer["choice"].is_string());
    let probabilities = answer["probabilities"]
        .as_object()
        .expect("choice probabilities must be an object");
    assert_eq!(probabilities.len(), 3);
    let sum = probabilities
        .values()
        .map(|value| value.as_f64().expect("probability must be numeric"))
        .sum::<f64>();
    assert!((sum - 1.0).abs() < 0.002, "probabilities sum to {sum}");
}
