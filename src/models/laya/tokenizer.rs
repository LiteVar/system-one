use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::domain::{NormalizedQuestion, render_state};

#[derive(Debug, Clone)]
pub struct TokenizedQuestion {
    pub ids: Vec<i32>,
    pub marker_positions: Vec<usize>,
}

#[derive(Clone)]
pub struct LayaTokenizer {
    tokenizer: Tokenizer,
    cls_id: i32,
    sep_id: i32,
    mask_id: i32,
}

impl LayaTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let cls_id = token_id(&tokenizer, "<bos>")
            .or_else(|| token_id(&tokenizer, "<cls>"))
            .context("tokenizer has no <bos>/<cls> token")?;
        let sep_id = token_id(&tokenizer, "<eos>")
            .or_else(|| token_id(&tokenizer, "<sep>"))
            .context("tokenizer has no <eos>/<sep> token")?;
        let mask_id = token_id(&tokenizer, "<mask>").context("tokenizer has no <mask> token")?;
        Ok(Self {
            tokenizer,
            cls_id: cls_id as i32,
            sep_id: sep_id as i32,
            mask_id: mask_id as i32,
        })
    }

    pub fn encode_without_special_tokens(&self, text: &str) -> Result<Vec<i32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(encoding.get_ids().iter().map(|id| *id as i32).collect())
    }

    pub fn build_sequence(
        &self,
        state: &Value,
        question: &NormalizedQuestion,
        max_len: usize,
        head_max_len: usize,
    ) -> Result<TokenizedQuestion> {
        let mask_token = "<mask>";
        let instructions = question.instructions.replace(mask_token, " ");
        let head_text = format!("{} question: {}", question.kind.as_str(), instructions);
        let mut head_ids = self.encode_without_special_tokens(&head_text)?;

        let mut option_ids = Vec::with_capacity(question.option_texts.len());
        for option in &question.option_texts {
            let option = option.replace(mask_token, " ");
            let mut ids = Vec::with_capacity(49);
            ids.push(self.mask_id);
            let mut encoded = self.encode_without_special_tokens(&format!(" {option}"))?;
            encoded.truncate(48);
            ids.extend(encoded);
            option_ids.push(ids);
        }

        let option_tokens: usize = option_ids.iter().map(Vec::len).sum();
        let mut option_budget = head_max_len.saturating_sub(option_tokens);
        if option_budget < 16 {
            let per = std::cmp::max(
                4,
                head_max_len.saturating_sub(16) / std::cmp::max(1, option_ids.len()),
            );
            for option in &mut option_ids {
                option.truncate(per);
            }
            option_budget = head_max_len.saturating_sub(option_ids.iter().map(Vec::len).sum());
        }
        head_ids.truncate(std::cmp::max(8, option_budget));

        let mut ids = Vec::with_capacity(max_len);
        ids.push(self.cls_id);
        ids.extend(head_ids);
        ids.push(self.sep_id);

        let mut markers = Vec::with_capacity(option_ids.len());
        for option in option_ids {
            markers.push(ids.len());
            ids.extend(option);
        }
        ids.push(self.sep_id);

        let room = max_len.saturating_sub(ids.len() + 1);
        let state_text = render_state(state)?.replace(mask_token, " ");
        let mut state_ids = self.encode_without_special_tokens(&state_text)?;
        state_ids.truncate(room);
        ids.extend(state_ids);
        ids.push(self.sep_id);
        ids.truncate(max_len);
        markers.retain(|position| *position < max_len);
        if markers.len() != question.option_count() {
            bail!(
                "question `{}` options exceed head budget or max length",
                question.id
            )
        }
        Ok(TokenizedQuestion {
            ids,
            marker_positions: markers,
        })
    }
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Option<u32> {
    tokenizer.get_vocab(true).get(token).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Question;

    #[test]
    fn preserves_marker_positions_and_special_token_layout() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.json");
        std::fs::write(
            &path,
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/minimal-tokenizer.json"
            )),
        )
        .unwrap();
        let tokenizer = LayaTokenizer::load(&path).unwrap();
        let question = NormalizedQuestion::from_api(
            "q",
            &Question {
                kind: "choice".to_owned(),
                instructions: Value::String("hello".to_owned()),
                criteria: Some(serde_json::json!(["yes", "no"])),
            },
        )
        .unwrap();
        let result = tokenizer
            .build_sequence(&Value::String("hello world".to_owned()), &question, 64, 32)
            .unwrap();
        assert_eq!(result.ids[0], 2);
        assert_eq!(result.marker_positions.len(), 2);
        assert_eq!(
            result
                .marker_positions
                .iter()
                .map(|position| result.ids[*position])
                .collect::<Vec<_>>(),
            vec![4, 4]
        );
        assert_eq!(result.ids.last().copied(), Some(1));
    }
}
