use std::env;
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

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

struct ServerChild(Child);

impl Drop for ServerChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn multi_question_http_request_keeps_server_alive_for_follow_up_requests() {
    let Some(cache) = env::var_os("SYSTEM_ONE_E2E_CACHE") else {
        eprintln!("skipping multi-question HTTP smoke test: SYSTEM_ONE_E2E_CACHE is not set");
        return;
    };
    let port = TcpListener::bind(("127.0.0.1", 0))
        .expect("failed to reserve a local port")
        .local_addr()
        .unwrap()
        .port();
    let port_arg = port.to_string();
    let executable = env!("CARGO_BIN_EXE_system-one");
    let child = Command::new(executable)
        .args([
            "--cache-dir",
            cache.to_str().expect("cache path must be valid UTF-8"),
            "--offline",
            "--device",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            &port_arg,
            "serve",
        ])
        .env_remove("SYSTEM_ONE_CONFIG")
        .env_remove("SYSTEM_ONE_LLAMA_CPP_HOME")
        .env_remove("SYSTEM_ONE_LLAMA_LIBRARY")
        .env_remove("SYSTEM_ONE_CACHE")
        .env_remove("SYSTEM_ONE_DEVICE")
        .env_remove("SYSTEM_ONE_MODEL")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to start system-one HTTP server");
    let mut server = ServerChild(child);
    let base_url = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();

    let mut health = None;
    for _ in 0..1200 {
        if let Some(status) = server
            .0
            .try_wait()
            .expect("failed to inspect server process")
        {
            panic!("system-one server exited during startup with {status}");
        }
        if let Ok(response) = client.get(format!("{base_url}/health")).send().await
            && response.status().is_success()
        {
            let body: Value = serde_json::from_slice(
                &response.bytes().await.expect("failed to read /health JSON"),
            )
            .expect("invalid /health JSON");
            if body["status"] == "ok" {
                health = Some(body);
                break;
            }
            if body["status"] == "degraded" {
                panic!("system-one model did not load: {body}");
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        health.is_some(),
        "system-one HTTP server did not become ready"
    );

    let state = std::iter::repeat_n("the agent has completed a system update", 700)
        .collect::<Vec<_>>()
        .join(" ");
    let request = serde_json::json!({
        "state": state,
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "Choose the next action.",
                "options": ["execute", "confirm", "reject"]
            },
            "noul-1": {"type": "noul", "instructions": "The update completed."},
            "noul-2": {"type": "noul", "instructions": "The service is healthy."},
            "noul-3": {"type": "noul", "instructions": "The action is safe."}
        }
    });
    let request_body = serde_json::to_vec(&request).expect("failed to serialize request");

    let first = client
        .post(format!("{base_url}/v1/systemone"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(request_body.clone())
        .send()
        .await
        .expect("first /v1/systemone request failed");
    assert_eq!(first.status(), reqwest::StatusCode::OK);
    let first: Value =
        serde_json::from_slice(&first.bytes().await.expect("failed to read first response"))
            .expect("invalid first response JSON");
    assert_eq!(first["answers"].as_object().unwrap().len(), 4);
    assert_eq!(first["answers"]["decision"]["type"], "choice");
    assert!(first["answers"]["decision"]["choice"].is_string());
    for id in ["noul-1", "noul-2", "noul-3"] {
        assert_eq!(first["answers"][id]["type"], "noul");
        assert!(first["answers"][id]["noul"].is_number());
    }
    let input_tokens = first["usage"]["input_tokens"]
        .as_u64()
        .expect("usage.input_tokens must be an integer");
    assert!(
        input_tokens > 1024,
        "request only used {input_tokens} tokens"
    );
    assert!(
        server
            .0
            .try_wait()
            .expect("failed to inspect server process")
            .is_none(),
        "system-one server exited after the multi-question request"
    );

    let health_response = client
        .get(format!("{base_url}/health"))
        .send()
        .await
        .expect("/health failed after the multi-question request")
        .error_for_status()
        .expect("/health returned an error status");
    let health: Value = serde_json::from_slice(
        &health_response
            .bytes()
            .await
            .expect("failed to read follow-up /health response"),
    )
    .expect("invalid follow-up /health JSON");
    assert_eq!(health["status"], "ok");
    assert_eq!(health["process"], "alive");

    let second = client
        .post(format!("{base_url}/v1/systemone"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(request_body)
        .send()
        .await
        .expect("second /v1/systemone request failed");
    assert_eq!(second.status(), reqwest::StatusCode::OK);
    let second: Value = serde_json::from_slice(
        &second
            .bytes()
            .await
            .expect("failed to read second response"),
    )
    .expect("invalid second response JSON");
    assert_eq!(second["answers"].as_object().unwrap().len(), 4);
    assert_eq!(second["usage"]["input_tokens"], input_tokens);
}
