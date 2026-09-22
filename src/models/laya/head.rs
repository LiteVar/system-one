use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use half::{bf16, f16};
use safetensors::tensor::{Dtype, SafeTensors};
use serde_json::Value;

use crate::domain::{QuestionKind, entropy_confidence};

#[derive(Debug, Clone)]
struct Tensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl Tensor {
    fn matrix(self, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
        if self.shape != [rows, cols] {
            bail!(
                "tensor {name} has shape {:?}, expected [{rows}, {cols}]",
                self.shape
            )
        }
        Ok(Matrix {
            rows,
            cols,
            values: self.values,
        })
    }

    fn vector(self, name: &str, len: usize) -> Result<Vec<f32>> {
        if self.shape != [len] {
            bail!("tensor {name} has shape {:?}, expected [{len}]", self.shape)
        }
        Ok(self.values)
    }
}

#[derive(Debug, Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl Matrix {
    fn matvec(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.cols);
        debug_assert_eq!(output.len(), self.rows);
        for (row, output_value) in output.iter_mut().enumerate() {
            let start = row * self.cols;
            let weights = &self.values[start..start + self.cols];
            *output_value = weights
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum();
        }
    }
}

#[derive(Debug, Clone)]
struct Linear {
    weight: Matrix,
    bias: Vec<f32>,
}

impl Linear {
    fn apply(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.weight.rows];
        self.weight.matvec(input, &mut output);
        for (value, bias) in output.iter_mut().zip(&self.bias) {
            *value += *bias;
        }
        output
    }
}

#[derive(Debug, Clone)]
struct TransformerLayer {
    qkv: Linear,
    out: Linear,
    ffn1: Linear,
    ffn2: Linear,
    norm1_weight: Vec<f32>,
    norm1_bias: Vec<f32>,
    norm2_weight: Vec<f32>,
    norm2_bias: Vec<f32>,
    heads: usize,
}

#[derive(Debug, Clone)]
pub struct HeadOutput {
    pub probabilities: Vec<f32>,
    pub confidence: f32,
    pub action_probability: f32,
}

pub struct LayaHead {
    hidden_size: usize,
    type_emb: Matrix,
    layers: Vec<TransformerLayer>,
    scorer_norm_weight: Vec<f32>,
    scorer_norm_bias: Vec<f32>,
    scorer1: Linear,
    scorer2: Linear,
    act1: Linear,
    act2: Linear,
    temperatures: Vec<f32>,
    temperature_by_options: serde_json::Map<String, Value>,
    config: Option<Value>,
}

impl LayaHead {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let config = read_config_metadata(&bytes)?;
        let tensors = SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .context("decoding safetensors")?;
        let hidden_size = 768;
        let type_emb =
            load_tensor(&tensors, "type_emb.weight")?.matrix("type_emb.weight", 3, hidden_size)?;

        let mut layers = Vec::new();
        for layer_index in 0..2 {
            let prefix = format!("head.layers.{layer_index}");
            let qkv = Linear {
                weight: load_tensor(&tensors, &format!("{prefix}.self_attn.in_proj_weight"))?
                    .matrix("qkv weight", 3 * hidden_size, hidden_size)?,
                bias: load_tensor(&tensors, &format!("{prefix}.self_attn.in_proj_bias"))?
                    .vector("qkv bias", 3 * hidden_size)?,
            };
            let out = Linear {
                weight: load_tensor(&tensors, &format!("{prefix}.self_attn.out_proj.weight"))?
                    .matrix("attention output weight", hidden_size, hidden_size)?,
                bias: load_tensor(&tensors, &format!("{prefix}.self_attn.out_proj.bias"))?
                    .vector("attention output bias", hidden_size)?,
            };
            let ffn1 = Linear {
                weight: load_tensor(&tensors, &format!("{prefix}.linear1.weight"))?.matrix(
                    "ffn1 weight",
                    4 * hidden_size,
                    hidden_size,
                )?,
                bias: load_tensor(&tensors, &format!("{prefix}.linear1.bias"))?
                    .vector("ffn1 bias", 4 * hidden_size)?,
            };
            let ffn2 = Linear {
                weight: load_tensor(&tensors, &format!("{prefix}.linear2.weight"))?.matrix(
                    "ffn2 weight",
                    hidden_size,
                    4 * hidden_size,
                )?,
                bias: load_tensor(&tensors, &format!("{prefix}.linear2.bias"))?
                    .vector("ffn2 bias", hidden_size)?,
            };
            layers.push(TransformerLayer {
                qkv,
                out,
                ffn1,
                ffn2,
                norm1_weight: load_tensor(&tensors, &format!("{prefix}.norm1.weight"))?
                    .vector("norm1 weight", hidden_size)?,
                norm1_bias: load_tensor(&tensors, &format!("{prefix}.norm1.bias"))?
                    .vector("norm1 bias", hidden_size)?,
                norm2_weight: load_tensor(&tensors, &format!("{prefix}.norm2.weight"))?
                    .vector("norm2 weight", hidden_size)?,
                norm2_bias: load_tensor(&tensors, &format!("{prefix}.norm2.bias"))?
                    .vector("norm2 bias", hidden_size)?,
                heads: hidden_size / 64,
            });
        }

        let scorer_norm_weight =
            load_tensor(&tensors, "scorer.0.weight")?.vector("scorer norm weight", hidden_size)?;
        let scorer_norm_bias =
            load_tensor(&tensors, "scorer.0.bias")?.vector("scorer norm bias", hidden_size)?;
        let scorer1 = Linear {
            weight: load_tensor(&tensors, "scorer.1.weight")?.matrix(
                "scorer1 weight",
                hidden_size,
                hidden_size,
            )?,
            bias: load_tensor(&tensors, "scorer.1.bias")?.vector("scorer1 bias", hidden_size)?,
        };
        let scorer2 = Linear {
            weight: load_tensor(&tensors, "scorer.3.weight")?.matrix(
                "scorer2 weight",
                1,
                hidden_size,
            )?,
            bias: load_tensor(&tensors, "scorer.3.bias")?.vector("scorer2 bias", 1)?,
        };
        let act1 = Linear {
            weight: load_tensor(&tensors, "act_head.0.weight")?.matrix(
                "action weight",
                256,
                hidden_size + 4,
            )?,
            bias: load_tensor(&tensors, "act_head.0.bias")?.vector("action bias", 256)?,
        };
        let act2 = Linear {
            weight: load_tensor(&tensors, "act_head.2.weight")?.matrix(
                "action output weight",
                2,
                256,
            )?,
            bias: load_tensor(&tensors, "act_head.2.bias")?.vector("action output bias", 2)?,
        };
        let temperatures = load_tensor(&tensors, "temperature")?.vector("temperature", 3)?;
        let temperature_by_options = config
            .as_ref()
            .and_then(|value| value.get("temperature_by_options"))
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        tracing::info!(
            hidden_size,
            layers = layers.len(),
            "Laya decision head loaded"
        );
        Ok(Self {
            hidden_size,
            type_emb,
            layers,
            scorer_norm_weight,
            scorer_norm_bias,
            scorer1,
            scorer2,
            act1,
            act2,
            temperatures,
            temperature_by_options,
            config,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn config(&self) -> Option<Value> {
        self.config.clone()
    }

    pub fn forward(
        &self,
        hidden: &[f32],
        tokens: usize,
        width: usize,
        markers: &[usize],
        kind: QuestionKind,
    ) -> Result<HeadOutput> {
        if width != self.hidden_size || hidden.len() != tokens * width {
            bail!(
                "encoder output shape is [{tokens}, {width}], expected [{tokens}, {}]",
                self.hidden_size
            )
        }
        if markers.is_empty() {
            bail!("question has no option markers")
        }
        if markers.iter().any(|position| *position >= tokens) {
            bail!("question marker is outside the encoder sequence")
        }

        let mut sequence = hidden.to_vec();
        for token in sequence.chunks_exact_mut(width) {
            for (value, embedding) in token
                .iter_mut()
                .zip(self.type_emb_row(kind.code() as usize))
            {
                *value += embedding;
            }
        }
        for layer in &self.layers {
            self.apply_layer(layer, &mut sequence, tokens)?;
        }

        let mut logits = Vec::with_capacity(markers.len());
        for marker in markers {
            let row = &sequence[marker * width..(marker + 1) * width];
            let normalized = layer_norm(row, &self.scorer_norm_weight, &self.scorer_norm_bias);
            let hidden_score = gelu_vec(&self.scorer1.apply(&normalized));
            logits.push(self.scorer2.apply(&hidden_score)[0]);
        }
        let temperature = self.temperature(kind, markers.len());
        let probabilities = softmax(
            &logits
                .iter()
                .map(|logit| *logit / temperature)
                .collect::<Vec<_>>(),
        );
        let confidence = entropy_confidence(&probabilities);

        let first = sequence[..width].to_vec();
        let top1 = probabilities.iter().copied().fold(0.0_f32, f32::max);
        let mut sorted = probabilities.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let top2 = sorted.get(1).copied().unwrap_or(0.0);
        let entropy = if probabilities.len() > 1 {
            1.0 - confidence
        } else {
            0.0
        };
        let mut action_input = first;
        action_input.extend([top1, top1 - top2, entropy, markers.len() as f32 / 255.0]);
        let action_hidden = gelu_vec(&self.act1.apply(&action_input));
        let action = softmax(&self.act2.apply(&action_hidden));

        Ok(HeadOutput {
            probabilities,
            confidence,
            action_probability: action[0],
        })
    }

    fn type_emb_row(&self, row: usize) -> &[f32] {
        let start = row * self.hidden_size;
        &self.type_emb.values[start..start + self.hidden_size]
    }

    fn temperature(&self, kind: QuestionKind, options: usize) -> f32 {
        let type_name = kind.as_str();
        let bucket = if options <= 2 {
            "2"
        } else if options <= 5 {
            "3-5"
        } else if options <= 10 {
            "6-10"
        } else {
            "11+"
        };
        let key = format!("{type_name}:{bucket}");
        self.temperature_by_options
            .get(&key)
            .and_then(Value::as_f64)
            .map(|value| (value as f32).clamp(0.5, 5.0))
            .unwrap_or_else(|| self.temperatures[kind.code() as usize].clamp(0.5, 5.0))
    }

    fn apply_layer(
        &self,
        layer: &TransformerLayer,
        sequence: &mut [f32],
        tokens: usize,
    ) -> Result<()> {
        let width = self.hidden_size;
        let mut normalized = vec![0.0; sequence.len()];
        for (source, target) in sequence
            .chunks_exact(width)
            .zip(normalized.chunks_exact_mut(width))
        {
            target.copy_from_slice(&layer_norm(source, &layer.norm1_weight, &layer.norm1_bias));
        }
        let mut attention = vec![0.0; sequence.len()];
        self.self_attention(layer, &normalized, &mut attention, tokens)?;
        for (value, update) in sequence.iter_mut().zip(attention) {
            *value += update;
        }

        for (source, target) in sequence
            .chunks_exact(width)
            .zip(normalized.chunks_exact_mut(width))
        {
            target.copy_from_slice(&layer_norm(source, &layer.norm2_weight, &layer.norm2_bias));
        }
        let mut ff = vec![0.0; sequence.len()];
        for (source, target) in normalized
            .chunks_exact(width)
            .zip(ff.chunks_exact_mut(width))
        {
            let hidden = relu_vec(&layer.ffn1.apply(source));
            target.copy_from_slice(&layer.ffn2.apply(&hidden));
        }
        for (value, update) in sequence.iter_mut().zip(ff) {
            *value += update;
        }
        Ok(())
    }

    fn self_attention(
        &self,
        layer: &TransformerLayer,
        input: &[f32],
        output: &mut [f32],
        tokens: usize,
    ) -> Result<()> {
        let width = self.hidden_size;
        let mut qkv = vec![0.0; width * 3];
        let mut all_qkv = vec![0.0; tokens * width * 3];
        let mut projected = vec![0.0; width];
        let mut attention = vec![0.0; width];
        let head_dim = width / layer.heads;
        for token_index in 0..tokens {
            let query = &input[token_index * width..(token_index + 1) * width];
            layer.qkv.weight.matvec(query, &mut qkv);
            for (value, bias) in qkv.iter_mut().zip(&layer.qkv.bias) {
                *value += *bias;
            }
            all_qkv[token_index * width * 3..(token_index + 1) * width * 3].copy_from_slice(&qkv);
        }
        for query_index in 0..tokens {
            let query_qkv = &all_qkv[query_index * width * 3..(query_index + 1) * width * 3];
            attention.fill(0.0);
            for head in 0..layer.heads {
                let q_start = head * head_dim;
                let mut scores = vec![0.0; tokens];
                for key_index in 0..tokens {
                    let key_qkv = &all_qkv[key_index * width * 3..(key_index + 1) * width * 3];
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        score += query_qkv[q_start + d] * key_qkv[q_start + d + width];
                    }
                    scores[key_index] = score / (head_dim as f32).sqrt();
                }
                let probs = softmax(&scores);
                for d in 0..head_dim {
                    let mut value = 0.0;
                    for key_index in 0..tokens {
                        let key_qkv = &all_qkv[key_index * width * 3..(key_index + 1) * width * 3];
                        value += probs[key_index] * key_qkv[q_start + d + 2 * width];
                    }
                    projected[q_start + d] = value;
                }
            }
            layer.out.weight.matvec(&projected, &mut attention);
            for (value, bias) in attention.iter_mut().zip(&layer.out.bias) {
                *value += *bias;
            }
            output[query_index * width..(query_index + 1) * width].copy_from_slice(&attention);
        }
        Ok(())
    }
}

fn load_tensor(tensors: &SafeTensors<'_>, name: &str) -> Result<Tensor> {
    let view = tensors
        .tensor(name)
        .map_err(|e| anyhow::anyhow!(e.to_string()))
        .with_context(|| format!("loading tensor {name}"))?;
    let values = match view.dtype() {
        Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|bytes| f16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
            .collect(),
        Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|bytes| bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
            .collect(),
        Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect(),
        dtype => bail!("unsupported tensor dtype {dtype:?} for {name}"),
    };
    Ok(Tensor {
        shape: view.shape().to_vec(),
        values,
    })
}

fn read_config_metadata(bytes: &[u8]) -> Result<Option<Value>> {
    if bytes.len() < 8 {
        bail!("safetensors file is truncated")
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
    if bytes.len() < 8 + header_len {
        bail!("safetensors header is truncated")
    }
    let header: Value = serde_json::from_slice(&bytes[8..8 + header_len])?;
    let config = header
        .get("__metadata__")
        .and_then(|metadata| metadata.get("laya.config"))
        .and_then(Value::as_str)
        .map(serde_json::from_str)
        .transpose()?;
    Ok(config)
}

fn layer_norm(input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let mean = input.iter().sum::<f32>() / input.len() as f32;
    let variance = input
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / input.len() as f32;
    input
        .iter()
        .zip(weight)
        .zip(bias)
        .map(|((value, weight), bias)| ((*value - mean) / (variance + 1e-5).sqrt()) * weight + bias)
        .collect()
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erff(value / std::f32::consts::SQRT_2))
}

fn gelu_vec(values: &[f32]) -> Vec<f32> {
    values.iter().map(|value| gelu(*value)).collect()
}

fn relu_vec(values: &[f32]) -> Vec<f32> {
    values.iter().map(|value| value.max(0.0)).collect()
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = values
        .iter()
        .map(|value| (*value - max).exp())
        .collect::<Vec<_>>();
    let sum = probabilities.iter().sum::<f32>().max(1e-12);
    for probability in &mut probabilities {
        *probability /= sum;
    }
    probabilities
}
