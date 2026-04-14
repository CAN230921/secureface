use std::time::Duration;

use anyhow::Context;
use tokio::process::Command;
use tokio::time;
use tracing::{error, info, warn};
use zbus::{interface, ConnectionBuilder};

const SERVICE_NAME: &str = "io.secureface.FaceAuth";
const OBJECT_PATH: &str = "/io/secureface/FaceAuth";
const INTERFACE_NAME: &str = "io.secureface.FaceAuth";

#[derive(Debug, Clone, Copy)]
enum EngineResult {
    Pass,
    Unknown,
    NotLive,
    Error,
}

impl EngineResult {
    fn as_str(self) -> &'static str {
        match self {
            EngineResult::Pass => "PASS",
            EngineResult::Unknown => "UNKNOWN",
            EngineResult::NotLive => "NOT_LIVE",
            EngineResult::Error => "ERROR",
        }
    }

    fn parse(raw: &str) -> Self {
        match raw.trim() {
            "PASS" => EngineResult::Pass,
            "UNKNOWN" => EngineResult::Unknown,
            "NOT_LIVE" => EngineResult::NotLive,
            "ERROR" => EngineResult::Error,
            _ => EngineResult::Error,
        }
    }

    fn is_success(self) -> bool {
        matches!(self, EngineResult::Pass)
    }
}

struct FaceAuthService;

#[interface(name = "io.secureface.FaceAuth")]
impl FaceAuthService {
    /// Returns `(engine_result, fallback_to_password)`.
    async fn authenticate(&self, user: &str, reason: &str, timeout_ms: u32) -> (String, bool) {
        match run_engine(user, reason, timeout_ms).await {
            Ok(result) if result.is_success() => (result.as_str().to_owned(), false),
            Ok(result) => (result.as_str().to_owned(), true),
            Err(e) => {
                // Critical rule: errors must NEVER block password authentication.
                error!(error = %e, "engine invocation failed");
                (EngineResult::Error.as_str().to_owned(), true)
            }
        }
    }
}

async fn run_engine(user: &str, reason: &str, timeout_ms: u32) -> anyhow::Result<EngineResult> {
    let mut cmd = Command::new("faceauth-engine");
    cmd.arg("--user")
        .arg(user)
        .arg("--reason")
        .arg(reason)
        .arg("--timeout-ms")
        .arg(timeout_ms.to_string());

    let output = time::timeout(Duration::from_millis(timeout_ms as u64), cmd.output())
        .await
        .context("engine timeout")?
        .context("failed to execute faceauth-engine")?;

    if !output.status.success() {
        warn!(status = %output.status, "engine returned non-zero status");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(EngineResult::parse(&stdout))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    info!(service = SERVICE_NAME, object_path = OBJECT_PATH, interface = INTERFACE_NAME, "starting daemon");

    let _connection = ConnectionBuilder::system()?
        .name(SERVICE_NAME)?
        .serve_at(OBJECT_PATH, FaceAuthService)?
        .build()
        .await?;

    info!("faceauthd is ready");
    std::future::pending::<()>().await;
    Ok(())
}
