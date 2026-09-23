use anyhow::{Context, Result, bail};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub type JsonMap<T> = IndexMap<String, T>;

#[derive(Debug, Clone, Deserialize)]
pub struct SystemOneRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub state: Value,
    pub questions: JsonMap<Question>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default = "default_instructions")]
    pub instructions: Value,
    #[serde(default, alias = "options")]
    pub criteria: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: JsonMap<Value>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<Vec<ValidationError>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationError {
    pub loc: Vec<String>,
    pub msg: String,
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionKind {
    Choice,
    Score,
    Noul,
}

impl QuestionKind {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "choice" => Ok(Self::Choice),
            "score" => Ok(Self::Score),
            "noul" => Ok(Self::Noul),
            other => bail!("unsupported question type `{other}`; expected choice, score, or noul"),
        }
    }

    pub fn code(self) -> u32 {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormalizedQuestion {
    pub id: String,
    pub kind: QuestionKind,
    pub instructions: String,
    pub labels: Vec<String>,
    pub option_texts: Vec<String>,
    pub criteria: Option<Value>,
}

impl NormalizedQuestion {
    pub fn from_api(id: impl Into<String>, question: &Question) -> Result<Self> {
        let id = id.into();
        if id.is_empty() {
            bail!("question ids must not be empty");
        }
        let kind =
            QuestionKind::parse(&question.kind).with_context(|| format!("question `{id}`"))?;
        let instructions = render_instruction(&question.instructions)
            .with_context(|| format!("question `{id}` instructions"))?;

        let (labels, option_values) = match kind {
            QuestionKind::Choice => normalize_choice(question.criteria.as_ref())?,
            QuestionKind::Score => normalize_score(question.criteria.as_ref())?,
            QuestionKind::Noul => normalize_noul(question.criteria.as_ref())?,
        };

        let option_texts = match kind {
            QuestionKind::Choice => labels
                .iter()
                .zip(option_values.iter())
                .map(|(label, value)| {
                    if value.is_none() || value == &Some(Value::String(String::new())) {
                        label.clone()
                    } else {
                        format!("{label}: {}", render_value(value.as_ref().unwrap()))
                    }
                })
                .collect(),
            QuestionKind::Score => option_values
                .iter()
                .enumerate()
                .map(|(i, value)| format!("level {i}: {}", render_value(value.as_ref().unwrap())))
                .collect(),
            QuestionKind::Noul => option_values
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    let label = if i == 0 { "false" } else { "true" };
                    let fallback = if i == 0 {
                        "no, the statement does not hold"
                    } else {
                        "yes, the statement holds"
                    };
                    if value.is_none() || value == &Some(Value::String(String::new())) {
                        format!("{label}: {fallback}")
                    } else {
                        format!("{label}: {}", render_value(value.as_ref().unwrap()))
                    }
                })
                .collect(),
        };

        Ok(Self {
            id,
            kind,
            instructions,
            labels,
            option_texts,
            criteria: question.criteria.clone(),
        })
    }

    pub fn option_count(&self) -> usize {
        self.labels.len()
    }
}

fn render_instruction(value: &Value) -> Result<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Object(_) | Value::Array(_) => Ok(render_json_ascii(value)),
        Value::Null => Ok(String::new()),
        _ => bail!("instructions must be a string, object, or array"),
    }
}

fn render_json_ascii(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => ascii_json_string(value),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(render_json_ascii)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| {
                    format!("{}: {}", ascii_json_string(key), render_json_ascii(value))
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn ascii_json_string(value: &str) -> String {
    let serialized = serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_owned());
    let mut output = String::with_capacity(serialized.len());
    for character in serialized.chars() {
        if character.is_ascii() {
            output.push(character);
        } else {
            let code = character as u32;
            if code <= 0xffff {
                output.push_str(&format!("\\u{code:04x}"));
            } else {
                let adjusted = code - 0x1_0000;
                let high = 0xd800 + ((adjusted >> 10) & 0x3ff);
                let low = 0xdc00 + (adjusted & 0x3ff);
                output.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
            }
        }
    }
    output
}

fn render_value(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(render_structured_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}: {}",
                        serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_owned()),
                        render_structured_value(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn render_structured_value(value: &Value) -> String {
    match value {
        Value::String(value) => serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_owned()),
        _ => render_value(value),
    }
}

fn default_instructions() -> Value {
    Value::String(String::new())
}

fn normalize_choice(criteria: Option<&Value>) -> Result<(Vec<String>, Vec<Option<Value>>)> {
    let Some(criteria) = criteria else {
        bail!("choice questions require a criteria object")
    };
    match criteria {
        Value::Object(map) => {
            if map.is_empty() || map.len() > 255 {
                bail!("choice criteria must contain between 1 and 255 options")
            }
            let mut labels = Vec::with_capacity(map.len());
            let mut values = Vec::with_capacity(map.len());
            for (label, value) in map {
                if label.is_empty() {
                    bail!("choice option labels must not be empty")
                }
                labels.push(label.clone());
                values.push(if value.is_null() {
                    None
                } else {
                    Some(value.clone())
                });
            }
            Ok((labels, values))
        }
        Value::Array(values) => {
            if values.is_empty() || values.len() > 255 {
                bail!("choice criteria must contain between 1 and 255 options")
            }
            let mut labels = Vec::with_capacity(values.len());
            let mut descriptions = Vec::with_capacity(values.len());
            for value in values {
                let label = match value {
                    Value::String(s) => s.clone(),
                    _ => render_value(value),
                };
                if label.is_empty() {
                    bail!("choice option labels must not be empty")
                }
                labels.push(label);
                descriptions.push(None);
            }
            Ok((labels, descriptions))
        }
        _ => bail!("choice criteria must be an object or array"),
    }
}

fn normalize_score(criteria: Option<&Value>) -> Result<(Vec<String>, Vec<Option<Value>>)> {
    let Some(Value::Array(values)) = criteria else {
        bail!("score questions require a criteria array")
    };
    if !(2..=10).contains(&values.len()) {
        bail!("score criteria must contain between 2 and 10 levels")
    }
    let labels = values
        .iter()
        .enumerate()
        .map(|(i, _)| i.to_string())
        .collect();
    Ok((labels, values.iter().cloned().map(Some).collect()))
}

fn normalize_noul(criteria: Option<&Value>) -> Result<(Vec<String>, Vec<Option<Value>>)> {
    let (false_value, true_value) = match criteria {
        None | Some(Value::Null) => (None, None),
        Some(Value::Object(map)) => {
            let false_value = map.get("false").filter(|v| !v.is_null()).cloned();
            let true_value = map.get("true").filter(|v| !v.is_null()).cloned();
            (false_value, true_value)
        }
        Some(_) => bail!("noul criteria must be an object with optional true/false fields"),
    };
    Ok((
        vec!["false".to_owned(), "true".to_owned()],
        vec![false_value, true_value],
    ))
}

pub fn render_state(value: &Value) -> Result<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Object(_) | Value::Array(_) => Ok(render_value(value)),
        _ => bail!("state must be a string, object, or array"),
    }
}

pub fn answer_from_distribution(
    question: &NormalizedQuestion,
    probabilities: &[f32],
    confidence: f32,
    action_probability: f32,
) -> Result<Value> {
    if probabilities.len() != question.option_count() {
        bail!("probability count does not match question option count")
    }
    let mut answer = Map::new();
    answer.insert(
        "type".to_owned(),
        Value::String(question.kind.as_str().to_owned()),
    );
    let mut action = Map::new();
    action.insert(
        "act_probability".to_owned(),
        Value::from(round4(action_probability)),
    );
    match question.kind {
        QuestionKind::Choice => {
            let choice_index = probabilities
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .context("choice has no options")?;
            answer.insert(
                "choice".to_owned(),
                Value::String(question.labels[choice_index].clone()),
            );
            let mut map = Map::new();
            for (label, probability) in question.labels.iter().zip(probabilities) {
                map.insert(label.clone(), Value::from(round4(*probability)));
            }
            answer.insert("probabilities".to_owned(), Value::Object(map));
            answer.insert("confidence".to_owned(), Value::from(round4(confidence)));
            answer.insert("action".to_owned(), Value::Object(action));
        }
        QuestionKind::Score => {
            let score = probabilities
                .iter()
                .enumerate()
                .map(|(i, p)| i as f32 * p)
                .sum::<f32>();
            answer.insert("score".to_owned(), Value::from(round4(score)));
            let mut legend = Map::new();
            let mut map = Map::new();
            if let Some(Value::Array(criteria)) = &question.criteria {
                for (i, value) in criteria.iter().enumerate() {
                    legend.insert(i.to_string(), value.clone());
                    map.insert(i.to_string(), Value::from(round4(probabilities[i])));
                }
            }
            answer.insert("legend".to_owned(), Value::Object(legend));
            answer.insert("probabilities".to_owned(), Value::Object(map));
            answer.insert("confidence".to_owned(), Value::from(round4(confidence)));
            answer.insert("action".to_owned(), Value::Object(action));
        }
        QuestionKind::Noul => {
            answer.insert("noul".to_owned(), Value::from(round4(probabilities[1])));
            answer.insert(
                "confidence".to_owned(),
                Value::from(round4(probabilities[1].max(1.0 - probabilities[1]))),
            );
            answer.insert("action".to_owned(), Value::Object(action));
        }
    }
    Ok(Value::Object(answer))
}

pub fn round4(value: f32) -> f64 {
    (value as f64 * 10_000.0).round() / 10_000.0
}

pub fn entropy_confidence(probabilities: &[f32]) -> f32 {
    if probabilities.len() < 2 {
        return 1.0;
    }
    let k = probabilities.len() as f32;
    let entropy = probabilities
        .iter()
        .map(|p| {
            let p = p.max(1e-9);
            -p * p.ln()
        })
        .sum::<f32>();
    (1.0 - entropy / k.ln()).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_json_like_state_and_criteria() {
        let state = serde_json::json!({"message": "你好", "attempts": [1, true]});
        assert_eq!(
            render_state(&state).unwrap(),
            r#"{"message": "你好", "attempts": [1, true]}"#
        );
        let question = Question {
            kind: "choice".to_owned(),
            instructions: Value::String("pick one".to_owned()),
            criteria: Some(serde_json::json!({"a": {"why": "x"}, "b": null})),
        };
        let normalized = NormalizedQuestion::from_api("q", &question).unwrap();
        assert_eq!(normalized.option_texts[0], r#"a: {"why": "x"}"#);
        assert_eq!(normalized.option_texts[1], "b");
    }

    #[test]
    fn accepts_options_alias_and_default_instructions() {
        let request: SystemOneRequest = serde_json::from_value(serde_json::json!({
            "state": "hello",
            "questions": {"q": {"type": "choice", "options": ["yes", "no"]}}
        }))
        .unwrap();
        let question =
            NormalizedQuestion::from_api("q", request.questions.get("q").unwrap()).unwrap();
        assert_eq!(question.instructions, "");
        assert_eq!(question.labels, ["yes", "no"]);
    }

    #[test]
    fn accepts_null_instructions_like_standard_system_one_clients() {
        let request: SystemOneRequest = serde_json::from_value(serde_json::json!({
            "state": "hello",
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": null
                }
            }
        }))
        .unwrap();
        let normalized =
            NormalizedQuestion::from_api("q", request.questions.get("q").unwrap()).unwrap();
        assert_eq!(normalized.instructions, "");
    }

    #[test]
    fn returns_typed_answers_with_action_and_noul_confidence() {
        let question = Question {
            kind: "noul".to_owned(),
            instructions: Value::String("is it true?".to_owned()),
            criteria: None,
        };
        let normalized = NormalizedQuestion::from_api("q", &question).unwrap();
        let answer = answer_from_distribution(&normalized, &[0.2, 0.8], 0.3, 0.7).unwrap();
        assert_eq!(answer["noul"], 0.8);
        assert_eq!(answer["confidence"], 0.8);
        assert_eq!(answer["action"]["act_probability"], 0.7);
    }

    #[test]
    fn returns_answer_shapes_for_choice_score_and_noul() {
        let questions = [
            (
                "choice",
                Question {
                    kind: "choice".to_owned(),
                    instructions: Value::String("pick one".to_owned()),
                    criteria: Some(serde_json::json!(["allow", "deny"])),
                },
                vec![0.75, 0.25],
            ),
            (
                "score",
                Question {
                    kind: "score".to_owned(),
                    instructions: Value::String("rate it".to_owned()),
                    criteria: Some(serde_json::json!(["low", "medium", "high"])),
                },
                vec![0.2, 0.3, 0.5],
            ),
            (
                "noul",
                Question {
                    kind: "noul".to_owned(),
                    instructions: Value::String("is it true?".to_owned()),
                    criteria: None,
                },
                vec![0.1, 0.9],
            ),
        ];

        for (id, question, probabilities) in questions {
            let normalized = NormalizedQuestion::from_api(id, &question).unwrap();
            let answer = answer_from_distribution(&normalized, &probabilities, 0.7, 0.6).unwrap();
            assert_eq!(answer["type"], id);
            assert!(answer["action"]["act_probability"].is_number());
            match id {
                "choice" => {
                    assert_eq!(answer["choice"], "allow");
                    assert_eq!(answer["probabilities"].as_object().unwrap().len(), 2);
                }
                "score" => {
                    assert_eq!(answer["score"], 1.3);
                    assert_eq!(answer["legend"].as_object().unwrap().len(), 3);
                }
                "noul" => assert_eq!(answer["noul"], 0.9),
                _ => unreachable!(),
            }
        }
    }
}
