mod head;
mod tokenizer;

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::domain::NormalizedQuestion;
use crate::runtime::EncoderOutput;

pub use head::{HeadOutput, LayaHead};
pub use tokenizer::{LayaTokenizer, TokenizedQuestion};

#[derive(Debug, Clone)]
pub struct LayaConfig {
    pub max_len: usize,
    pub head_max_len: usize,
}

impl Default for LayaConfig {
    fn default() -> Self {
        Self {
            max_len: 1024,
            head_max_len: 256,
        }
    }
}

pub struct LayaModel {
    pub tokenizer: LayaTokenizer,
    pub head: LayaHead,
    pub config: LayaConfig,
}

impl LayaModel {
    pub fn load(tokenizer_path: &Path, head_path: &Path) -> Result<Self> {
        let tokenizer = LayaTokenizer::load(tokenizer_path)
            .with_context(|| format!("loading tokenizer {}", tokenizer_path.display()))?;
        let head = LayaHead::load(head_path)
            .with_context(|| format!("loading Laya head {}", head_path.display()))?;
        let config = head
            .config()
            .map(parse_config)
            .transpose()?
            .unwrap_or_default();
        if head.hidden_size() != 768 {
            anyhow::bail!(
                "Laya multilingual head expects hidden size 768, got {}",
                head.hidden_size()
            );
        }
        Ok(Self {
            tokenizer,
            head,
            config,
        })
    }

    pub fn build_inputs(
        &self,
        state: &Value,
        questions: &[NormalizedQuestion],
    ) -> Result<Vec<TokenizedQuestion>> {
        questions
            .iter()
            .map(|question| {
                self.tokenizer.build_sequence(
                    state,
                    question,
                    self.config.max_len,
                    self.config.head_max_len,
                )
            })
            .collect()
    }

    pub fn classify(
        &self,
        question: &NormalizedQuestion,
        input: &TokenizedQuestion,
        encoder: &EncoderOutput,
    ) -> Result<HeadOutput> {
        self.head.forward(
            &encoder.hidden,
            encoder.tokens,
            encoder.width,
            &input.marker_positions,
            question.kind,
        )
    }
}

fn parse_config(value: Value) -> Result<LayaConfig> {
    Ok(LayaConfig {
        max_len: value.get("max_len").and_then(Value::as_u64).unwrap_or(1024) as usize,
        head_max_len: value
            .get("head_max_len")
            .and_then(Value::as_u64)
            .unwrap_or(256) as usize,
    })
}
