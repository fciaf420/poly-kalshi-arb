// src/dashboard.rs
// Web-based dashboard on localhost
//
// NOTE: This module provides a basic dashboard structure.
// The full implementation with position tracking is in the binary.

use crate::position_tracker::PositionTracker;
use axum::{
    response::{Html, IntoResponse},
    routing::get,
    Router,
    Json,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

/// Shared dashboard state
#[derive(Clone)]
pub struct DashboardState {
    pub tracker: Arc<RwLock<PositionTracker>>,
    pub balances: Arc<RwLock<Balances>>,
}

#[derive(Clone, Default)]
pub struct Balances {
    pub kalshi_balance: f64,
    pub poly_balance: f64,
    /// Reserved balance for in-flight orders (deducted from available)
    pub kalshi_reserved: f64,
    pub poly_reserved: f64,
}

impl Balances {
    /// Get available balance after reservations
    pub fn kalshi_available(&self) -> f64 {
        (self.kalshi_balance - self.kalshi_reserved).max(0.0)
    }

    pub fn poly_available(&self) -> f64 {
        (self.poly_balance - self.poly_reserved).max(0.0)
    }

    /// Reserve funds for a pending order
    pub fn reserve(&mut self, kalshi_amount: f64, poly_amount: f64) {
        self.kalshi_reserved += kalshi_amount;
        self.poly_reserved += poly_amount;
    }

    /// Release reserved funds (after order completes or fails)
    pub fn release(&mut self, kalshi_amount: f64, poly_amount: f64) {
        self.kalshi_reserved = (self.kalshi_reserved - kalshi_amount).max(0.0);
        self.poly_reserved = (self.poly_reserved - poly_amount).max(0.0);
    }

    /// Deduct actual spent funds from balance (call after fills)
    pub fn deduct_spent(&mut self, kalshi_spent: f64, poly_spent: f64) {
        self.kalshi_balance = (self.kalshi_balance - kalshi_spent).max(0.0);
        self.poly_balance = (self.poly_balance - poly_spent).max(0.0);
    }
}

/// Start web dashboard server on localhost
pub async fn start_web_dashboard(state: DashboardState, port: u16) {
    let app = Router::new()
        .route("/", get(dashboard_handler))
        .route("/api/data", get(api_data_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("127.0.0.1:{}", port);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(
                "[DASHBOARD] Failed to bind http://{} ({:?}). Is another process already using this port?",
                addr,
                e
            );
            return;
        }
    };

    tracing::info!("[DASHBOARD] running at http://{}", addr);

    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!("[DASHBOARD] Server stopped unexpectedly: {}", e);
    }
}

/// Serve the HTML dashboard
async fn dashboard_handler() -> impl IntoResponse {
    Html(include_str!("dashboard.html"))
}

/// API endpoint for live data
async fn api_data_handler(
    axum::extract::State(state): axum::extract::State<DashboardState>,
) -> impl IntoResponse {
    let tracker = state.tracker.read().await;
    let balances = state.balances.read().await;

    let summary = tracker.summary();
    let open_positions = tracker.open_positions();

    let data = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "balances": {
            "kalshi": balances.kalshi_balance,
            "polymarket": balances.poly_balance,
            "total": balances.kalshi_balance + balances.poly_balance,
        },
        "pnl": {
            "daily": tracker.daily_pnl(),
            "all_time": tracker.all_time_pnl,
        },
        "summary": {
            "open_positions": summary.open_positions,
            "total_cost": summary.total_cost_basis,
            "total_contracts": summary.total_contracts,
            "unmatched_risk": summary.total_unmatched_exposure,
            "resolved_positions": summary.resolved_positions,
        },
        "positions": open_positions.iter().map(|pos| {
            serde_json::json!({
                "market_id": pos.market_id,
                "description": pos.description,
                "opened_at": pos.opened_at,
                "kalshi_yes": {
                    "contracts": pos.kalshi_yes.contracts,
                    "avg_price": pos.kalshi_yes.avg_price,
                    "cost_basis": pos.kalshi_yes.cost_basis
                },
                "kalshi_no": {
                    "contracts": pos.kalshi_no.contracts,
                    "avg_price": pos.kalshi_no.avg_price,
                    "cost_basis": pos.kalshi_no.cost_basis
                },
                "poly_yes": {
                    "contracts": pos.poly_yes.contracts,
                    "avg_price": pos.poly_yes.avg_price,
                    "cost_basis": pos.poly_yes.cost_basis
                },
                "poly_no": {
                    "contracts": pos.poly_no.contracts,
                    "avg_price": pos.poly_no.avg_price,
                    "cost_basis": pos.poly_no.cost_basis
                },
                "total_fees": pos.total_fees,
                "unmatched_exposure": pos.unmatched_exposure(),
            })
        }).collect::<Vec<_>>(),
    });

    Json(data)
}
