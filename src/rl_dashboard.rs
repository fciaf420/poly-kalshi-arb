//! RL Dashboard Module
//!
//! Provides WebSocket-based real-time updates and API endpoints for
//! visualizing PPO training, model decisions, and trading performance.

use axum::{
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::{Html, IntoResponse},
    routing::get,
    Json, Router,
};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

use crate::rl::{
    RlMetricsCollector, RlStatusResponse, RlPerformanceResponse, RlLossesResponse,
    RlWsMessage, RlMetricsUpdate, LiveInference,
};

// =============================================================================
// New State Structs for 6 Panels
// =============================================================================

/// Binance metrics for a single asset (BTC/ETH/SOL/XRP)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssetMetricsSnapshot {
    pub price: Option<f64>,
    pub return_1m: Option<f64>,
    pub return_5m: Option<f64>,
    pub return_10m: Option<f64>,
    pub cvd: f64,
    pub cvd_acceleration: f64,
    pub trade_flow_imbalance: f64,
    pub trade_intensity: f64,
    pub large_trade: bool,
    pub vol_expansion: f64,
    pub vol_regime: f64,
    pub trend_regime: f64,
    pub order_book_imbalance_l1: f64,
    pub order_book_imbalance_l5: f64,
    pub spread_pct: f64,
}

/// Binance state for all 4 assets
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BinanceState {
    pub btc: AssetMetricsSnapshot,
    pub eth: AssetMetricsSnapshot,
    pub sol: AssetMetricsSnapshot,
    pub xrp: AssetMetricsSnapshot,
    pub last_update: Option<DateTime<Utc>>,
}

/// Polymarket market state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PolymarketState {
    pub market_id: String,
    pub question: String,
    pub asset: String,
    pub yes_ask: Option<f64>,
    pub no_ask: Option<f64>,
    pub yes_bid: Option<f64>,
    pub no_bid: Option<f64>,
    pub yes_ask_size: f64,
    pub no_ask_size: f64,
    pub spread_pct: f64,
    pub orderbook_imbalance_l1: f64,
    pub orderbook_imbalance_l5: f64,
    pub mid_price: Option<f64>,
    pub time_to_expiry_mins: Option<f64>,
    pub last_update: Option<DateTime<Utc>>,
}

/// Single feature entry for the 18-feature grid
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEntry {
    pub index: usize,
    pub name: String,
    pub source: String,  // "Binance", "Polymarket", "Internal"
    pub raw_value: f32,
    pub normalized_value: f32,
}

/// Full 18-feature observation state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeaturesState {
    pub features: Vec<FeatureEntry>,
    pub last_update: Option<DateTime<Utc>>,
}

impl FeaturesState {
    /// Create features state from raw observation array
    pub fn from_observation(raw: &[f32; 18], normalized: &[f32; 18]) -> Self {
        let feature_info: [(usize, &str, &str); 18] = [
            (0, "returns_1m", "Binance"),
            (1, "returns_5m", "Binance"),
            (2, "returns_10m", "Binance"),
            (3, "order_book_imbalance_l1", "Polymarket"),
            (4, "order_book_imbalance_l5", "Polymarket"),
            (5, "trade_flow_imbalance", "Binance"),
            (6, "cvd_acceleration", "Binance"),
            (7, "spread_pct", "Polymarket"),
            (8, "trade_intensity", "Binance"),
            (9, "large_trade_flag", "Binance"),
            (10, "volatility", "Polymarket"),
            (11, "vol_expansion", "Binance"),
            (12, "has_position", "Internal"),
            (13, "position_side", "Internal"),
            (14, "position_pnl", "Internal"),
            (15, "time_remaining", "Internal"),
            (16, "vol_regime", "Binance"),
            (17, "trend_regime", "Binance"),
        ];

        let features = feature_info.iter().map(|(i, name, source)| {
            FeatureEntry {
                index: *i,
                name: name.to_string(),
                source: source.to_string(),
                raw_value: raw[*i],
                normalized_value: normalized[*i],
            }
        }).collect();

        Self {
            features,
            last_update: Some(Utc::now()),
        }
    }
}

/// Position summary for dashboard
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PositionsSummary {
    pub total_cost_basis: f64,
    pub guaranteed_profit: f64,
    pub unmatched_exposure: f64,
    pub open_count: usize,
    pub resolved_count: usize,
    pub total_contracts: i64,
    pub daily_pnl: f64,
    pub all_time_pnl: f64,
}

/// Single position entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionEntry {
    pub market_id: String,
    pub description: String,
    pub kalshi_yes: i64,
    pub kalshi_no: i64,
    pub poly_yes: i64,
    pub poly_no: i64,
    pub total_cost: f64,
    pub guaranteed_profit: f64,
    pub unmatched_exposure: f64,
}

/// Positions state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PositionsState {
    pub summary: PositionsSummary,
    pub positions: Vec<PositionEntry>,
    pub last_update: Option<DateTime<Utc>>,
}

/// Circuit breaker state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CircuitBreakerState {
    pub enabled: bool,
    pub halted: bool,
    pub trip_reason: Option<String>,
    pub consecutive_errors: u32,
    pub daily_pnl: f64,
    pub max_daily_loss: f64,
    pub total_position: i64,
    pub max_position: i64,
    pub market_count: usize,
    pub peak_equity: f64,
    pub current_drawdown_pct: f64,
    pub max_drawdown_pct: f64,
    pub last_update: Option<DateTime<Utc>>,
}

/// Model info state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelInfoState {
    pub model_path: String,
    pub safetensors_path: Option<String>,
    pub device: String,
    pub lr_actor: f64,
    pub lr_critic: f64,
    pub gamma: f32,
    pub lambda: f32,
    pub clip_epsilon: f32,
    pub value_coef: f32,
    pub entropy_coef: f32,
    pub ppo_epochs: usize,
    pub batch_size: usize,
    pub model_version: u32,
    pub loaded_at: Option<DateTime<Utc>>,
}

/// P&L metrics for a single asset (BTC, ETH, SOL, XRP)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssetPnL {
    pub asset: String,
    pub total_pnl: f64,
    pub total_trades: u64,
    pub winning_trades: u64,
    pub win_rate: f32,
    pub avg_return: f64,
    pub best_trade: f64,
    pub worst_trade: f64,
}

/// Performance breakdown by asset
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceByAsset {
    pub btc: AssetPnL,
    pub eth: AssetPnL,
    pub sol: AssetPnL,
    pub xrp: AssetPnL,
    pub total_pnl: f64,
    pub last_update: Option<DateTime<Utc>>,
}

impl PerformanceByAsset {
    /// Initialize with asset names
    pub fn new() -> Self {
        Self {
            btc: AssetPnL { asset: "BTC".to_string(), ..Default::default() },
            eth: AssetPnL { asset: "ETH".to_string(), ..Default::default() },
            sol: AssetPnL { asset: "SOL".to_string(), ..Default::default() },
            xrp: AssetPnL { asset: "XRP".to_string(), ..Default::default() },
            total_pnl: 0.0,
            last_update: None,
        }
    }

    /// Record a trade for a specific asset
    pub fn record_trade(&mut self, asset: &str, pnl: f64) {
        let asset_pnl = match asset.to_uppercase().as_str() {
            "BTC" | "BTCUSDT" => &mut self.btc,
            "ETH" | "ETHUSDT" => &mut self.eth,
            "SOL" | "SOLUSDT" => &mut self.sol,
            "XRP" | "XRPUSDT" => &mut self.xrp,
            _ => return, // Unknown asset
        };

        asset_pnl.total_pnl += pnl;
        asset_pnl.total_trades += 1;
        if pnl > 0.0 {
            asset_pnl.winning_trades += 1;
        }

        // Update best/worst
        if pnl > asset_pnl.best_trade {
            asset_pnl.best_trade = pnl;
        }
        if pnl < asset_pnl.worst_trade || asset_pnl.total_trades == 1 {
            asset_pnl.worst_trade = pnl;
        }

        // Recalculate derived metrics
        if asset_pnl.total_trades > 0 {
            asset_pnl.win_rate = asset_pnl.winning_trades as f32 / asset_pnl.total_trades as f32;
            asset_pnl.avg_return = asset_pnl.total_pnl / asset_pnl.total_trades as f64;
        }

        // Update total
        self.total_pnl = self.btc.total_pnl + self.eth.total_pnl + self.sol.total_pnl + self.xrp.total_pnl;
        self.last_update = Some(Utc::now());
    }

    /// Extract asset from market_id (e.g., "btc-15m-123" -> "BTC")
    pub fn extract_asset(market_id: &str) -> Option<&str> {
        let lower = market_id.to_lowercase();
        if lower.contains("btc") || lower.contains("bitcoin") {
            Some("BTC")
        } else if lower.contains("eth") || lower.contains("ethereum") {
            Some("ETH")
        } else if lower.contains("sol") || lower.contains("solana") {
            Some("SOL")
        } else if lower.contains("xrp") || lower.contains("ripple") {
            Some("XRP")
        } else {
            None
        }
    }
}

// =============================================================================
// Dashboard State
// =============================================================================

/// State for the RL dashboard
#[derive(Clone)]
pub struct RlDashboardState {
    /// Metrics collector
    pub metrics: RlMetricsCollector,

    /// Broadcast channel for WebSocket updates
    pub broadcast_tx: broadcast::Sender<RlWsMessage>,

    // New fields for 6 panels
    /// Binance metrics state
    pub binance_state: Arc<RwLock<BinanceState>>,
    /// Polymarket state
    pub polymarket_state: Arc<RwLock<PolymarketState>>,
    /// Features state (18-dim observation)
    pub features_state: Arc<RwLock<FeaturesState>>,
    /// Positions state
    pub positions_state: Arc<RwLock<PositionsState>>,
    /// Circuit breaker state
    pub circuit_breaker_state: Arc<RwLock<CircuitBreakerState>>,
    /// Model info state
    pub model_info_state: Arc<RwLock<ModelInfoState>>,
    /// Performance by asset (BTC, ETH, SOL, XRP)
    pub performance_by_asset: Arc<RwLock<PerformanceByAsset>>,
}

impl RlDashboardState {
    /// Create new dashboard state
    pub fn new(metrics: RlMetricsCollector) -> Self {
        let (broadcast_tx, _) = broadcast::channel(100);
        Self {
            metrics,
            broadcast_tx,
            binance_state: Arc::new(RwLock::new(BinanceState::default())),
            polymarket_state: Arc::new(RwLock::new(PolymarketState::default())),
            features_state: Arc::new(RwLock::new(FeaturesState::default())),
            positions_state: Arc::new(RwLock::new(PositionsState::default())),
            circuit_breaker_state: Arc::new(RwLock::new(CircuitBreakerState::default())),
            model_info_state: Arc::new(RwLock::new(ModelInfoState::default())),
            performance_by_asset: Arc::new(RwLock::new(PerformanceByAsset::new())),
        }
    }

    /// Create new dashboard state with pre-existing shared state
    pub fn new_with_state(
        metrics: RlMetricsCollector,
        binance_state: Arc<RwLock<BinanceState>>,
        polymarket_state: Arc<RwLock<PolymarketState>>,
        features_state: Arc<RwLock<FeaturesState>>,
        positions_state: Arc<RwLock<PositionsState>>,
        circuit_breaker_state: Arc<RwLock<CircuitBreakerState>>,
        model_info_state: Arc<RwLock<ModelInfoState>>,
        performance_by_asset: Arc<RwLock<PerformanceByAsset>>,
    ) -> Self {
        let (broadcast_tx, _) = broadcast::channel(100);
        Self {
            metrics,
            broadcast_tx,
            binance_state,
            polymarket_state,
            features_state,
            positions_state,
            circuit_breaker_state,
            model_info_state,
            performance_by_asset,
        }
    }

    /// Broadcast a metrics update to all connected clients
    pub fn broadcast_metrics(&self, policy_loss: f32, value_loss: f32, entropy: f32) {
        if let Some(m) = self.metrics.get_metrics_snapshot() {
            let update = RlWsMessage::Metrics(RlMetricsUpdate {
                policy_loss,
                value_loss,
                entropy,
                buffer_size: m.buffer_size,
                total_updates: m.total_updates,
            });
            let _ = self.broadcast_tx.send(update);
        }
    }

    /// Broadcast a status update
    pub fn broadcast_status(&self) {
        if let Some(m) = self.metrics.get_metrics_snapshot() {
            let status = RlStatusResponse::from(&m);
            let _ = self.broadcast_tx.send(RlWsMessage::Status(status));
        }
    }

    /// Broadcast live inference update
    pub fn broadcast_inference(&self, inference: LiveInference) {
        let _ = self.broadcast_tx.send(RlWsMessage::Inference(inference));
    }

    /// Update Binance state
    pub async fn update_binance(&self, state: BinanceState) {
        let mut bs = self.binance_state.write().await;
        *bs = state;
    }

    /// Update Polymarket state
    pub async fn update_polymarket(&self, state: PolymarketState) {
        let mut ps = self.polymarket_state.write().await;
        *ps = state;
    }

    /// Update features state
    pub async fn update_features(&self, state: FeaturesState) {
        let mut fs = self.features_state.write().await;
        *fs = state;
    }

    /// Update positions state
    pub async fn update_positions(&self, state: PositionsState) {
        let mut ps = self.positions_state.write().await;
        *ps = state;
    }

    /// Update circuit breaker state
    pub async fn update_circuit_breaker(&self, state: CircuitBreakerState) {
        let mut cbs = self.circuit_breaker_state.write().await;
        *cbs = state;
    }

    /// Update model info state
    pub async fn update_model_info(&self, state: ModelInfoState) {
        let mut mis = self.model_info_state.write().await;
        *mis = state;
    }

    /// Record a trade outcome for an asset (BTC, ETH, SOL, XRP)
    pub async fn record_asset_trade(&self, market_id: &str, pnl: f64) {
        if let Some(asset) = PerformanceByAsset::extract_asset(market_id) {
            let mut pba = self.performance_by_asset.write().await;
            pba.record_trade(asset, pnl);
        }
    }

    /// Get performance by asset snapshot
    pub async fn get_performance_by_asset(&self) -> PerformanceByAsset {
        self.performance_by_asset.read().await.clone()
    }
}

// =============================================================================
// Routes
// =============================================================================

/// Create router for RL dashboard endpoints
pub fn rl_router(state: RlDashboardState) -> Router {
    Router::new()
        .route("/", get(rl_dashboard_handler))
        .route("/api/status", get(rl_status_handler))
        .route("/api/losses", get(rl_losses_handler))
        .route("/api/performance", get(rl_performance_handler))
        .route("/api/episodes", get(rl_episodes_handler))
        .route("/api/inference", get(rl_inference_handler))
        // New API endpoints for 6 panels
        .route("/api/binance", get(rl_binance_handler))
        .route("/api/polymarket", get(rl_polymarket_handler))
        .route("/api/features", get(rl_features_handler))
        .route("/api/positions", get(rl_positions_handler))
        .route("/api/circuit-breaker", get(rl_circuit_breaker_handler))
        .route("/api/model-info", get(rl_model_info_handler))
        // Performance by asset (BTC, ETH, SOL, XRP)
        .route("/api/performance-by-asset", get(rl_performance_by_asset_handler))
        .route("/ws", get(rl_websocket_handler))
        .with_state(state)
}

// =============================================================================
// HTML Handler
// =============================================================================

/// Serve the RL dashboard HTML
async fn rl_dashboard_handler() -> impl IntoResponse {
    Html(include_str!("rl_dashboard.html"))
}

// =============================================================================
// API Handlers
// =============================================================================

/// GET /api/status - Training status
async fn rl_status_handler(State(state): State<RlDashboardState>) -> impl IntoResponse {
    match state.metrics.get_metrics_snapshot() {
        Some(m) => Json(serde_json::to_value(RlStatusResponse::from(&m)).unwrap()),
        None => Json(serde_json::json!({"error": "No metrics available"})),
    }
}

/// GET /api/losses - Loss history for charts
async fn rl_losses_handler(State(state): State<RlDashboardState>) -> impl IntoResponse {
    match state.metrics.get_loss_history() {
        Some(history) => {
            let snapshot = state.metrics.get_metrics_snapshot();
            let response = RlLossesResponse {
                avg_policy_loss: snapshot.as_ref().map(|m| m.avg_policy_loss()).unwrap_or(0.0),
                avg_value_loss: snapshot.as_ref().map(|m| m.avg_value_loss()).unwrap_or(0.0),
                avg_entropy: snapshot.as_ref().map(|m| m.avg_entropy()).unwrap_or(0.0),
                history,
            };
            Json(serde_json::to_value(response).unwrap())
        }
        None => Json(serde_json::json!({"error": "No loss history available"})),
    }
}

/// GET /api/performance - Performance metrics
async fn rl_performance_handler(State(state): State<RlDashboardState>) -> impl IntoResponse {
    match state.metrics.get_metrics_snapshot() {
        Some(m) => Json(serde_json::to_value(RlPerformanceResponse::from(&m)).unwrap()),
        None => Json(serde_json::json!({"error": "No metrics available"})),
    }
}

/// GET /api/episodes - Recent episodes
async fn rl_episodes_handler(State(state): State<RlDashboardState>) -> impl IntoResponse {
    let episodes = state.metrics.get_recent_episodes(50);
    Json(serde_json::json!({
        "count": episodes.len(),
        "episodes": episodes
    }))
}

/// GET /api/inference - Current live inference
async fn rl_inference_handler(State(state): State<RlDashboardState>) -> impl IntoResponse {
    match state.metrics.get_live_inference() {
        Some(inference) => Json(serde_json::to_value(inference).unwrap()),
        None => Json(serde_json::json!({"active": false})),
    }
}

// =============================================================================
// New API Handlers for 6 Panels
// =============================================================================

/// GET /api/binance - Binance metrics for BTC/ETH/SOL/XRP
async fn rl_binance_handler(State(state): State<RlDashboardState>) -> impl IntoResponse {
    let binance = state.binance_state.read().await;
    Json(serde_json::to_value(&*binance).unwrap_or_else(|_| serde_json::json!({"error": "Failed to serialize"})))
}

/// GET /api/polymarket - Polymarket market state
async fn rl_polymarket_handler(State(state): State<RlDashboardState>) -> impl IntoResponse {
    let polymarket = state.polymarket_state.read().await;
    Json(serde_json::to_value(&*polymarket).unwrap_or_else(|_| serde_json::json!({"error": "Failed to serialize"})))
}

/// GET /api/features - Full 18-feature observation grid
async fn rl_features_handler(State(state): State<RlDashboardState>) -> impl IntoResponse {
    let features = state.features_state.read().await;
    Json(serde_json::to_value(&*features).unwrap_or_else(|_| serde_json::json!({"error": "Failed to serialize"})))
}

/// GET /api/positions - Current positions and P&L
async fn rl_positions_handler(State(state): State<RlDashboardState>) -> impl IntoResponse {
    let positions = state.positions_state.read().await;
    Json(serde_json::to_value(&*positions).unwrap_or_else(|_| serde_json::json!({"error": "Failed to serialize"})))
}

/// GET /api/circuit-breaker - Risk management status
async fn rl_circuit_breaker_handler(State(state): State<RlDashboardState>) -> impl IntoResponse {
    let circuit_breaker = state.circuit_breaker_state.read().await;
    Json(serde_json::to_value(&*circuit_breaker).unwrap_or_else(|_| serde_json::json!({"error": "Failed to serialize"})))
}

/// GET /api/model-info - Model path and hyperparameters
async fn rl_model_info_handler(State(state): State<RlDashboardState>) -> impl IntoResponse {
    let model_info = state.model_info_state.read().await;
    Json(serde_json::to_value(&*model_info).unwrap_or_else(|_| serde_json::json!({"error": "Failed to serialize"})))
}

/// GET /api/performance-by-asset - P&L breakdown by BTC, ETH, SOL, XRP
async fn rl_performance_by_asset_handler(State(state): State<RlDashboardState>) -> impl IntoResponse {
    // Try to get from metrics collector first (source of truth)
    if let Some(pba) = state.metrics.get_performance_by_asset() {
        return Json(serde_json::to_value(&pba).unwrap_or_else(|_| serde_json::json!({"error": "Failed to serialize"})));
    }
    // Fallback to dashboard state
    let pba = state.performance_by_asset.read().await;
    Json(serde_json::to_value(&*pba).unwrap_or_else(|_| serde_json::json!({"error": "Failed to serialize"})))
}

// =============================================================================
// WebSocket Handler
// =============================================================================

/// WebSocket upgrade handler
async fn rl_websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<RlDashboardState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

/// Handle WebSocket connection
async fn handle_websocket(socket: WebSocket, state: RlDashboardState) {
    let (mut sender, mut receiver) = socket.split();

    // Subscribe to broadcast channel
    let mut broadcast_rx = state.broadcast_tx.subscribe();

    // Send initial status
    if let Some(m) = state.metrics.get_metrics_snapshot() {
        let status = RlStatusResponse::from(&m);
        let msg = RlWsMessage::Status(status);
        if let Ok(json) = serde_json::to_string(&msg) {
            let _ = sender.send(Message::Text(json.into())).await;
        }
    }

    // Spawn task to receive broadcast messages and send to client
    let send_task = tokio::spawn(async move {
        while let Ok(msg) = broadcast_rx.recv().await {
            if let Ok(json) = serde_json::to_string(&msg) {
                if sender.send(Message::Text(json.into())).await.is_err() {
                    break;
                }
            }
        }
    });

    // Receive messages from client (ping/pong handling)
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Ping(_)) => {
                    // Pong is handled automatically by axum
                    tracing::trace!("[RL-WS] Received ping");
                }
                Ok(Message::Close(_)) => {
                    tracing::debug!("[RL-WS] Client disconnected");
                    break;
                }
                Err(e) => {
                    tracing::warn!("[RL-WS] Error receiving message: {}", e);
                    break;
                }
                _ => {}
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }

    tracing::debug!("[RL-WS] WebSocket connection closed");
}

// =============================================================================
// Background Broadcast Loop
// =============================================================================

/// Start background loop that broadcasts metrics at regular intervals
pub async fn start_broadcast_loop(state: RlDashboardState, interval_secs: u64) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));

    loop {
        interval.tick().await;

        // Broadcast current status
        state.broadcast_status();

        // If there's live inference, broadcast it
        if let Some(inference) = state.metrics.get_live_inference() {
            state.broadcast_inference(inference);
        }
    }
}

// =============================================================================
// Persistence
// =============================================================================

/// Path for persisting RL metrics
const RL_METRICS_PATH: &str = "rl_metrics.json";

/// Save metrics to disk
pub async fn save_metrics(metrics: &RlMetricsCollector) -> anyhow::Result<()> {
    if let Some(snapshot) = metrics.get_metrics_snapshot() {
        let json = serde_json::to_string_pretty(&snapshot)?;
        tokio::fs::write(RL_METRICS_PATH, json).await?;
        tracing::debug!("[RL] Saved metrics to {}", RL_METRICS_PATH);
    }
    Ok(())
}

/// Load metrics from disk
pub async fn load_metrics() -> Option<crate::rl::RlMetrics> {
    match tokio::fs::read_to_string(RL_METRICS_PATH).await {
        Ok(json) => {
            match serde_json::from_str(&json) {
                Ok(metrics) => {
                    tracing::info!("[RL] Loaded metrics from {}", RL_METRICS_PATH);
                    Some(metrics)
                }
                Err(e) => {
                    tracing::warn!("[RL] Failed to parse metrics file: {}", e);
                    None
                }
            }
        }
        Err(_) => None,
    }
}

/// Start background auto-save loop
pub async fn start_autosave_loop(metrics: RlMetricsCollector, interval_secs: u64) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));

    loop {
        interval.tick().await;
        if let Err(e) = save_metrics(&metrics).await {
            tracing::warn!("[RL] Failed to auto-save metrics: {}", e);
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_state_creation() {
        let metrics = RlMetricsCollector::new(256);
        let state = RlDashboardState::new(metrics);

        // Should be able to subscribe to broadcast
        let _rx = state.broadcast_tx.subscribe();
    }

    #[tokio::test]
    async fn test_broadcast_metrics() {
        let metrics = RlMetricsCollector::new(256);
        let state = RlDashboardState::new(metrics.clone());

        // Record some data
        metrics.record_update(0.5, 0.3, 0.8);
        metrics.set_buffer_size(128);

        // Subscribe before broadcast
        let mut rx = state.broadcast_tx.subscribe();

        // Broadcast
        state.broadcast_metrics(0.4, 0.2, 0.7);

        // Should receive message
        let msg = rx.try_recv();
        assert!(msg.is_ok());
    }
}
