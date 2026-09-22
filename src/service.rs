use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde::Serialize;

use crate::cache::{MODEL_ID, ModelPaths};
use crate::config::AppConfig;
use crate::domain::{
    NormalizedQuestion, SystemOneRequest, SystemOneResponse, Usage, answer_from_distribution,
};
use crate::models::laya::LayaModel;
use crate::runtime::{LlamaEncoder, RuntimeInfo, RuntimeManager};

pub trait DecisionModel: Send + Sync {
    fn decide(&self, request: &SystemOneRequest) -> Result<SystemOneResponse>;
    fn model_card(&self) -> ModelCard;
}

pub struct InferenceService {
    model_id: String,
    model: Arc<LayaModel>,
    encoder: Mutex<LlamaEncoder>,
    runtime: RuntimeInfo,
}

#[derive(Debug, Clone, Copy)]
pub struct DecisionTimings {
    pub tokenization_ms: f64,
    pub encoder_ms: f64,
    pub head_ms: f64,
    pub total_ms: f64,
}

impl InferenceService {
    pub fn load(config: &AppConfig, paths: ModelPaths) -> Result<Arc<Self>> {
        let load_start = Instant::now();
        let model = LayaModel::load(&paths.tokenizer, &paths.head)?;
        let runtime_manager = RuntimeManager::new(config);
        let (library, runtime) = runtime_manager
            .load(config.device)
            .context("resolving a compatible llama.cpp runtime")?;
        let encoder = LlamaEncoder::load(
            library,
            &paths.gguf,
            config.device,
            model.config.max_len,
            16,
        )?;
        if encoder.embedding_dim() != model.head.hidden_size() {
            bail!(
                "encoder output dimension {} does not match Laya head dimension {}",
                encoder.embedding_dim(),
                model.head.hidden_size()
            )
        }
        tracing::info!(
            model = %MODEL_ID,
            backend = %runtime.backend,
            device = %config.device.as_str(),
            model_load_ms = load_start.elapsed().as_secs_f64() * 1000.0,
            "decision model loaded"
        );
        Ok(Arc::new(Self {
            model_id: MODEL_ID.to_owned(),
            model: Arc::new(model),
            encoder: Mutex::new(encoder),
            runtime,
        }))
    }

    pub fn decide(&self, request: &SystemOneRequest) -> Result<SystemOneResponse> {
        self.decide_with_timings(request)
            .map(|(response, _)| response)
    }

    pub fn decide_with_timings(
        &self,
        request: &SystemOneRequest,
    ) -> Result<(SystemOneResponse, DecisionTimings)> {
        if let Some(model) = &request.model
            && model != &self.model_id
        {
            bail!(
                "model `{model}` is not loaded; loaded model is `{}`",
                self.model_id
            )
        }
        if request.questions.is_empty() {
            bail!("questions must contain at least one question")
        }
        let start = Instant::now();
        let normalized = request
            .questions
            .iter()
            .map(|(id, question)| NormalizedQuestion::from_api(id.clone(), question))
            .collect::<Result<Vec<_>>>()?;
        let tokenized = self.model.build_inputs(&request.state, &normalized)?;
        let tokenization_time = start.elapsed();
        let token_ids = tokenized
            .iter()
            .map(|item| item.ids.clone())
            .collect::<Vec<_>>();
        let encoder_start = Instant::now();
        let encoded = self.encoder.lock().encode(&token_ids)?;
        let encoder_time = encoder_start.elapsed();

        let head_start = Instant::now();
        let mut answers = IndexMap::new();
        for ((question, input), encoder_output) in normalized.iter().zip(&tokenized).zip(&encoded) {
            let head_output = self.model.classify(question, input, encoder_output)?;
            let answer = answer_from_distribution(
                question,
                &head_output.probabilities,
                head_output.confidence,
                head_output.action_probability,
            )?;
            answers.insert(question.id.clone(), answer);
        }
        let head_time = head_start.elapsed();
        let input_tokens = tokenized.iter().map(|item| item.ids.len()).sum();
        tracing::info!(
            model = %self.model_id,
            backend = %self.runtime.runtime,
            device = %self.runtime.backend,
            questions = normalized.len(),
            input_tokens,
            tokenization_ms = tokenization_time.as_secs_f64() * 1000.0,
            encoder_ms = encoder_time.as_secs_f64() * 1000.0,
            head_ms = head_time.as_secs_f64() * 1000.0,
            total_ms = start.elapsed().as_secs_f64() * 1000.0,
            "decision completed"
        );
        let timings = DecisionTimings {
            tokenization_ms: tokenization_time.as_secs_f64() * 1000.0,
            encoder_ms: encoder_time.as_secs_f64() * 1000.0,
            head_ms: head_time.as_secs_f64() * 1000.0,
            total_ms: start.elapsed().as_secs_f64() * 1000.0,
        };
        Ok((
            SystemOneResponse {
                model: self.model_id.clone(),
                answers,
                usage: Usage {
                    input_tokens,
                    output_tokens: 0,
                },
            },
            timings,
        ))
    }

    pub fn model_card(&self) -> ModelCard {
        ModelCard {
            id: self.model_id.clone(),
            loaded: true,
            runtime: self.runtime.clone(),
            error: None,
        }
    }
}

impl DecisionModel for InferenceService {
    fn decide(&self, request: &SystemOneRequest) -> Result<SystemOneResponse> {
        InferenceService::decide(self, request)
    }

    fn model_card(&self) -> ModelCard {
        InferenceService::model_card(self)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub loaded: bool,
    pub runtime: RuntimeInfo,
    pub error: Option<String>,
}

pub struct UnavailableDecisionModel {
    model_id: String,
    runtime: RuntimeInfo,
    error: String,
}

impl UnavailableDecisionModel {
    pub fn new(model_id: String, runtime: RuntimeInfo, error: String) -> Arc<Self> {
        Arc::new(Self {
            model_id,
            runtime,
            error,
        })
    }
}

impl DecisionModel for UnavailableDecisionModel {
    fn decide(&self, _request: &SystemOneRequest) -> Result<SystemOneResponse> {
        bail!("model `{}` is unavailable: {}", self.model_id, self.error)
    }

    fn model_card(&self) -> ModelCard {
        ModelCard {
            id: self.model_id.clone(),
            loaded: false,
            runtime: self.runtime.clone(),
            error: Some(self.error.clone()),
        }
    }
}
