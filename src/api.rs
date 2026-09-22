use std::sync::Arc;

use axum::extract::{State, rejection::JsonRejection};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use crate::domain::{ErrorResponse, SystemOneRequest};
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
    Json(ModelsResponse {
        data: vec![state.service.model_card()],
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
    data: Vec<crate::service::ModelCard>,
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
            },
        }
    }

    fn from_anyhow(error: anyhow::Error) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            body: ErrorResponse {
                error: crate::domain::ErrorBody {
                    code: "invalid_request".to_owned(),
                    message: error.to_string(),
                },
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(self.body)).into_response()
    }
}
