//! Vitrine Onboarding backend (ADR-015 / D-015.2, D-015.4).
//!
//! A single Axum binary that serves a vanilla-JS wizard and round-trips the
//! curator's input into an ADR-013 / D-013.1 `exhibit.toml` manifest.
//!
//! Secret containment (D-015.4): the `[secrets]` block of the emitted TOML
//! contains ONLY `env:NAME` indirection references — never a raw token value.
//! If the client posts a raw `hf_token_value`, it is written to a separate
//! server-side `<output>/.secrets.env` file with `0600` permissions and is
//! NEVER echoed back in any HTTP response.

use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

// ---------------------------------------------------------------------------
// Wire model — JSON posted by the browser wizard.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ManifestRequest {
    exhibit: ExhibitInput,
    drive: DriveInput,
    #[serde(default)]
    objects: Vec<ObjectInput>,
    secrets: SecretsInput,
    pipeline: PipelineInput,
    oversight: OversightInput,
}

#[derive(Debug, Deserialize)]
struct ExhibitInput {
    id: String,
    name: String,
    #[serde(default)]
    venue: String,
    #[serde(default)]
    date: String,
    #[serde(default)]
    curator: String,
    #[serde(default)]
    description: String,
}

#[derive(Debug, Deserialize)]
struct DriveInput {
    #[serde(default)]
    url: String,
    #[serde(default = "default_rclone_remote")]
    rclone_remote: String,
    #[serde(default = "default_true")]
    recursive: bool,
}

fn default_rclone_remote() -> String {
    "gdrive".to_string()
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct ObjectInput {
    id: String,
    name: String,
    #[serde(default)]
    sam3_concept: String,
    #[serde(default)]
    description: String,
    #[serde(default = "default_priority")]
    priority: String,
    #[serde(default = "default_expected_count")]
    expected_count: u32,
}

fn default_priority() -> String {
    "standard".to_string()
}
fn default_expected_count() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
struct SecretsInput {
    /// Env-var NAME holding the HF token, e.g. "HF_TOKEN".
    #[serde(default = "default_hf_env")]
    hf_token_env: String,
    /// Env-var NAME pointing at the gcloud service-account json path.
    #[serde(default = "default_gcloud_env")]
    gcloud_credentials_env: String,
    #[serde(default)]
    gcloud_project: String,
    /// OPTIONAL raw token paste — contained server-side, never echoed, never in TOML.
    #[serde(default)]
    hf_token_value: Option<String>,
}

fn default_hf_env() -> String {
    "HF_TOKEN".to_string()
}
fn default_gcloud_env() -> String {
    "GOOGLE_APPLICATION_CREDENTIALS".to_string()
}

#[derive(Debug, Deserialize)]
struct PipelineInput {
    #[serde(default = "default_mesh_backend")]
    mesh_backend: String,
    #[serde(default = "default_matcher")]
    matcher: String,
}

fn default_mesh_backend() -> String {
    "auto".to_string()
}
fn default_matcher() -> String {
    "aliked_lightglue".to_string()
}

#[derive(Debug, Deserialize)]
struct OversightInput {
    #[serde(default = "default_oversight_backend")]
    backend: String,
    #[serde(default = "default_artifact_vlm")]
    artifact_vlm: String,
}

fn default_oversight_backend() -> String {
    "claude_code".to_string()
}
fn default_artifact_vlm() -> String {
    "gemma_local".to_string()
}

// ---------------------------------------------------------------------------
// Persisted model — serialises to ADR-013 D-013.1 `exhibit.toml`.
// Field order and section names match `exhibit.example.toml` exactly.
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct Manifest {
    schema_version: String,
    exhibit: Exhibit,
    drive: Drive,
    objects: Vec<Object>,
    secrets: Secrets,
    pipeline: Pipeline,
    oversight: Oversight,
}

#[derive(Debug, Serialize)]
struct Exhibit {
    id: String,
    name: String,
    venue: String,
    date: String,
    curator: String,
    description: String,
}

#[derive(Debug, Serialize)]
struct Drive {
    url: String,
    rclone_remote: String,
    recursive: bool,
}

#[derive(Debug, Serialize)]
struct Object {
    id: String,
    name: String,
    sam3_concept: String,
    description: String,
    priority: String,
    expected_count: u32,
}

#[derive(Debug, Serialize)]
struct Secrets {
    /// `env:NAME` indirection ONLY — never a raw value (ADR-013 secret rule).
    hf_token: String,
    gcloud_credentials: String,
    gcloud_project: String,
}

#[derive(Debug, Serialize)]
struct Pipeline {
    mesh_backend: String,
    matcher: String,
}

#[derive(Debug, Serialize)]
struct Oversight {
    backend: String,
    artifact_vlm: String,
}

/// Normalise an env-var NAME into an `env:NAME` reference. Accepts a name with
/// or without the `env:` prefix; strips/re-applies so the TOML is canonical and
/// can never contain a raw secret value.
fn env_ref(name: &str) -> String {
    let trimmed = name.trim();
    let bare = trimmed.strip_prefix("env:").unwrap_or(trimmed);
    format!("env:{bare}")
}

impl From<ManifestRequest> for Manifest {
    fn from(req: ManifestRequest) -> Self {
        let objects = req
            .objects
            .into_iter()
            .map(|o| Object {
                id: o.id,
                name: o.name,
                sam3_concept: o.sam3_concept,
                description: o.description,
                priority: o.priority,
                expected_count: o.expected_count,
            })
            .collect();

        Manifest {
            schema_version: "1.0".to_string(),
            exhibit: Exhibit {
                id: req.exhibit.id,
                name: req.exhibit.name,
                venue: req.exhibit.venue,
                date: req.exhibit.date,
                curator: req.exhibit.curator,
                description: req.exhibit.description,
            },
            drive: Drive {
                url: req.drive.url,
                rclone_remote: req.drive.rclone_remote,
                recursive: req.drive.recursive,
            },
            objects,
            secrets: Secrets {
                // CONTAINMENT: only env: references reach the manifest.
                hf_token: env_ref(&req.secrets.hf_token_env),
                gcloud_credentials: env_ref(&req.secrets.gcloud_credentials_env),
                gcloud_project: req.secrets.gcloud_project,
            },
            pipeline: Pipeline {
                mesh_backend: req.pipeline.mesh_backend,
                matcher: req.pipeline.matcher,
            },
            oversight: Oversight {
                backend: req.oversight.backend,
                artifact_vlm: req.oversight.artifact_vlm,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Server state & handlers.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    output_dir: PathBuf,
    static_dir: PathBuf,
}

async fn index(State(state): State<AppState>) -> impl IntoResponse {
    let path = state.static_dir.join("index.html");
    match std::fs::read_to_string(&path) {
        Ok(body) => Html(body).into_response(),
        Err(e) => {
            tracing::error!(error = %e, path = %path.display(), "failed to read index.html");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "index.html not found",
            )
                .into_response()
        }
    }
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn create_manifest(
    State(state): State<AppState>,
    Json(req): Json<ManifestRequest>,
) -> impl IntoResponse {
    // Capture the optional raw token BEFORE conversion consumes the request.
    let raw_hf_token = req.secrets.hf_token_value.clone();
    let hf_env_name = req
        .secrets
        .hf_token_env
        .trim()
        .strip_prefix("env:")
        .unwrap_or(req.secrets.hf_token_env.trim())
        .to_string();

    let manifest: Manifest = req.into();

    let toml_body = match toml::to_string_pretty(&manifest) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!(error = %e, "TOML serialisation failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "ok": false, "error": "serialisation failed" })),
            )
                .into_response();
        }
    };

    if let Err(e) = std::fs::create_dir_all(&state.output_dir) {
        tracing::error!(error = %e, "failed to create output dir");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "ok": false, "error": "could not create output dir" })),
        )
            .into_response();
    }

    let header = "# exhibit.toml — generated by Vitrine Onboarding (ADR-015 / ADR-013 D-013.1).\n\
                  # Secrets are env: references only; raw token values are contained server-side.\n\n";
    let toml_path = state.output_dir.join("exhibit.toml");
    if let Err(e) = std::fs::write(&toml_path, format!("{header}{toml_body}")) {
        tracing::error!(error = %e, "failed to write exhibit.toml");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "ok": false, "error": "could not write manifest" })),
        )
            .into_response();
    }

    // CONTAINMENT: a pasted raw token is written ONLY to .secrets.env (0600),
    // never to the manifest, never returned to the client.
    let mut secret_contained = false;
    if let Some(token) = raw_hf_token {
        if !token.trim().is_empty() {
            match write_secret_env(&state.output_dir, &hf_env_name, token.trim()) {
                Ok(()) => secret_contained = true,
                Err(e) => {
                    tracing::error!(error = %e, "failed to contain pasted secret");
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "ok": false, "error": "secret containment failed" })),
                    )
                        .into_response();
                }
            }
        }
    }

    tracing::info!(path = %toml_path.display(), secret_contained, "manifest written");

    // Response carries NO secret values — only paths and a containment flag.
    (
        StatusCode::OK,
        Json(json!({
            "ok": true,
            "toml_path": toml_path.display().to_string(),
            "secret_contained": secret_contained,
        })),
    )
        .into_response()
}

/// Write (or upsert) a `NAME=value` line into `<output>/.secrets.env` with
/// `0600` perms. The file is created restricted before any secret is written.
fn write_secret_env(output_dir: &Path, env_name: &str, value: &str) -> std::io::Result<()> {
    let path = output_dir.join(".secrets.env");

    // Read any existing lines, dropping a prior entry for this var.
    let mut lines: Vec<String> = match std::fs::read_to_string(&path) {
        Ok(existing) => existing
            .lines()
            .filter(|l| {
                let key = l.split('=').next().unwrap_or("").trim();
                key != env_name && !l.trim().is_empty()
            })
            .map(|l| l.to_string())
            .collect(),
        Err(_) => Vec::new(),
    };
    lines.push(format!("{env_name}={value}"));
    let body = format!("{}\n", lines.join("\n"));

    std::fs::write(&path, body)?;
    let mut perms = std::fs::metadata(&path)?.permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(&path, perms)?;
    Ok(())
}

fn router(state: AppState) -> Router {
    let serve_dir = ServeDir::new(&state.static_dir);
    Router::new()
        .route("/", get(index))
        .route("/api/health", get(health))
        .route("/api/manifest", post(create_manifest))
        .nest_service("/static", serve_dir)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "vitrine_onboarding=info,tower_http=info".into()),
        )
        .init();

    let output_dir =
        PathBuf::from(std::env::var("VITRINE_OUTPUT_DIR").unwrap_or_else(|_| "./out".to_string()));
    let static_dir =
        PathBuf::from(std::env::var("VITRINE_STATIC_DIR").unwrap_or_else(|_| "./static".to_string()));

    let state = AppState {
        output_dir: output_dir.clone(),
        static_dir: static_dir.clone(),
    };

    let app = router(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8088));
    tracing::info!(
        %addr,
        output_dir = %output_dir.display(),
        static_dir = %static_dir.display(),
        "Vitrine Onboarding listening"
    );

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind 0.0.0.0:8088");
    axum::serve(listener, app)
        .await
        .expect("server error");
}
