use std::sync::Arc;

use axum::extract::{State, rejection::JsonRejection};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use crate::domain::{ErrorResponse, SystemOneRequest, ValidationError};
use crate::service::DecisionModel;

#[derive(Clone)]
pub struct ApiState {
    pub service: Arc<dyn DecisionModel>,
}

pub fn router(service: Arc<dyn DecisionModel>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(system_one))
        .layer(RequestBodyLimitLayer::new(16 * 1024 * 1024))
        .layer(TraceLayer::new_for_http())
        .with_state(ApiState { service })
}

async fn health(State(state): State<ApiState>) -> impl IntoResponse {
    let card = state.service.model_card();
    Json(HealthResponse {
        status: if card.loaded { "ok" } else { "degraded" },
        process: "alive",
        model: if card.loaded { "loaded" } else { "unavailable" },
        model_id: Some(card.id),
        runtime: Some(card.runtime),
        error: card.error,
    })
}

async fn models(State(state): State<ApiState>) -> impl IntoResponse {
    let card = state.service.model_card();
    Json(ModelsResponse {
        models: vec![StandardModelMetadata {
            name: card.id.clone(),
            description: "A local System One decision model.".to_owned(),
            // The local model manifest currently carries a revision, not a
            // public release date. Keep the standard field present without
            // inventing a date; detailed local metadata remains in `data`.
            release_date: "unknown".to_owned(),
        }],
        data: vec![card],
    })
}

async fn system_one(
    State(state): State<ApiState>,
    request: Result<Json<SystemOneRequest>, JsonRejection>,
) -> Result<Json<crate::domain::SystemOneResponse>, ApiError> {
    let card = state.service.model_card();
    if !card.loaded {
        return Err(ApiError::unavailable(
            card.error
                .unwrap_or_else(|| "model is unavailable".to_owned()),
        ));
    }
    let Json(request) = request.map_err(ApiError::from_json)?;
    tokio::task::spawn_blocking(move || state.service.decide(&request))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map(Json)
        .map_err(ApiError::from_anyhow)
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    process: &'static str,
    model: &'static str,
    model_id: Option<String>,
    runtime: Option<crate::runtime::RuntimeInfo>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    models: Vec<StandardModelMetadata>,
    data: Vec<crate::service::ModelCard>,
}

#[derive(Debug, Serialize)]
struct StandardModelMetadata {
    name: String,
    description: String,
    release_date: String,
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    body: ErrorResponse,
}

impl ApiError {
    fn from_json(error: JsonRejection) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ErrorResponse {
                error: crate::domain::ErrorBody {
                    code: "invalid_json".to_owned(),
                    message: error.body_text(),
                },
                detail: None,
            },
        }
    }

    fn internal(message: String) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: ErrorResponse {
                error: crate::domain::ErrorBody {
                    code: "internal_error".to_owned(),
                    message,
                },
                detail: None,
            },
        }
    }

    fn unavailable(message: String) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: ErrorResponse {
                error: crate::domain::ErrorBody {
                    code: "model_unavailable".to_owned(),
                    message,
                },
                detail: None,
            },
        }
    }

    fn from_anyhow(error: anyhow::Error) -> Self {
        let message = error.to_string();
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            body: ErrorResponse {
                error: crate::domain::ErrorBody {
                    code: "invalid_request".to_owned(),
                    message: message.clone(),
                },
                detail: Some(vec![ValidationError {
                    loc: vec!["body".to_owned()],
                    msg: message,
                    kind: "invalid_request".to_owned(),
                }]),
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(self.body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_response_has_standard_and_local_views() {
        let response = ModelsResponse {
            models: vec![StandardModelMetadata {
                name: "laya-multilingual".to_owned(),
                description: "A local System One decision model.".to_owned(),
                release_date: "unknown".to_owned(),
            }],
            data: vec![crate::service::ModelCard {
                id: "laya-multilingual".to_owned(),
                loaded: true,
                runtime: crate::runtime::RuntimeInfo {
                    runtime: "llama.cpp".to_owned(),
                    source: "managed".to_owned(),
                    version: None,
                    path: None,
                    platform: "test".to_owned(),
                    architecture: "test".to_owned(),
                    backend: "cpu".to_owned(),
                    compatible: true,
                    error: None,
                },
                error: None,
            }],
        };
        let json = serde_json::to_value(response).unwrap();
        assert_eq!(json["models"][0]["name"], "laya-multilingual");
        assert_eq!(json["data"][0]["id"], "laya-multilingual");
        assert_eq!(json["data"][0]["loaded"], true);
    }

    #[test]
    fn invalid_request_error_has_standard_detail_and_local_error() {
        let error = ApiError::from_anyhow(anyhow::anyhow!("questions must not be empty"));
        let json = serde_json::to_value(error.body).unwrap();
        assert_eq!(json["error"]["code"], "invalid_request");
        assert_eq!(json["detail"][0]["type"], "invalid_request");
    }
}
