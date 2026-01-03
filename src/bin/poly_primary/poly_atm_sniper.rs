//! Polymarket ATM Sniper - Delta 0.50 Strategy
//!
//! STRATEGY:
//! - Only trade when spot price is within 0.0015% of strike (delta ≈ 0.50)
//! - When ATM, bid 45¢ or less on both YES and NO
//! - If both fill: pay 90¢ or less, receive $1 = guaranteed profit
//! - If one fills: hold with ~50% win probability (fair value)
//!
//! This targets the "sweet spot" where binary options are at-the-money,
//! meaning both YES and NO have approximately equal fair value (50¢).
//!
//! Usage:
//!   cargo run --release --bin poly_atm_sniper
//!
//! Environment:
//!   POLY_PRIVATE_KEY - Your Polymarket/Polygon wallet private key
//!   POLY_FUNDER - Your funder address (proxy wallet)
//!   POLYGON_API_KEY - Polygon.io API key for price feed

use anyhow::{Context, Result};
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::RwLock;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

use arb_bot::polymarket_clob::{
    PolymarketAsyncClient, SharedAsyncClient, PreparedCreds,
};
use arb_bot::polymarket_markets::{discover_all_markets, PolyMarket};

// RL imports (always available, but PPO training requires --features rl)
use arb_bot::rl::{
    Action, Experience, ExperienceBuffer, BinanceMetrics, PolymarketMetrics, Observation, PpoConfig, PpoTrainer,
    build_observation_normalized as rl_build_observation, compute_share_reward, compute_spread_adjusted_reward,
    calculate_vol_5m, calculate_orderbook_imbalance, calculate_spread_pct,
    RlMetricsCollector, LiveInference,
};
use arb_bot::rl_dashboard::{
    RlDashboardState, rl_router, start_broadcast_loop,
    BinanceState, PolymarketState, FeaturesState, PositionsState,
    CircuitBreakerState, ModelInfoState, AssetMetricsSnapshot, PerformanceByAsset,
};
use std::collections::VecDeque;

/// Polymarket ATM Sniper - Delta 0.50 Strategy
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Size per trade in dollars
    #[arg(short, long, default_value_t = 20.0)]
    size: f64,

    /// Max bid price in cents (bid at this or lower when ATM)
    #[arg(short, long, default_value_t = 45)]
    bid: i64,

    /// Number of contracts per trade
    #[arg(short, long, default_value_t = 1.0)]
    contracts: f64,

    /// ATM threshold: max % distance from strike to be considered ATM (default: 0.0015%)
    /// Price must be within this percentage of strike for delta ≈ 0.50
    #[arg(long, default_value_t = 0.0015)]
    atm_threshold: f64,

    /// Live trading mode (default is dry run)
    #[arg(short, long, default_value_t = false)]
    live: bool,

    /// Specific market slug to monitor (optional, monitors all crypto if not set)
    #[arg(long)]
    market: Option<String>,

    /// Asset symbol filter: btc, eth, sol, xrp (optional, monitors all if not set)
    #[arg(long)]
    sym: Option<String>,

    /// Maximum total dollars to spend across all positions
    #[arg(long, default_value_t = 10.0)]
    max_dollars: f64,

    /// Minimum minutes remaining to trade (default: 2)
    #[arg(long, default_value_t = 2)]
    min_minutes: i64,

    /// Maximum minutes remaining to trade (default: 15)
    #[arg(long, default_value_t = 15)]
    max_minutes: i64,

    /// Connect directly to Polygon.io instead of local price server
    #[arg(short, long, default_value_t = false)]
    direct: bool,

    // ========== BONDS MODE ==========
    /// Enable bonds mode - buy confirmed winners near expiry
    #[arg(long, default_value_t = false)]
    bonds: bool,

    /// Bonds: Max minutes to expiry for entry (default: 3)
    #[arg(long, default_value_t = 3)]
    bond_minutes: i64,

    /// Bonds: Min % distance from strike required (default: 0.3%)
    #[arg(long, default_value_t = 0.3)]
    bond_min_distance: f64,

    /// Bonds: Max ask price in cents (default: 95 = 5%+ profit)
    #[arg(long, default_value_t = 95)]
    bond_max_price: i64,

    /// Bonds: Min contracts available at ask (default: 1)
    #[arg(long, default_value_t = 1.0)]
    bond_min_size: f64,

    /// Bonds: Max price staleness in seconds (using Binance futures)
    #[arg(long, default_value_t = BOND_MAX_PRICE_STALE_SECS_DEFAULT)]
    bond_price_stale_secs: u64,

    /// Bonds: CSV log path
    #[arg(long, default_value = "logs/bonds_capture.csv")]
    bonds_log: String,

    // ========== RL MODE ==========
    /// Enable RL mode - use PPO agent for trade decisions (requires --features rl)
    #[arg(long, default_value_t = false)]
    rl_mode: bool,

    /// RL: Path to save/load model weights
    #[arg(long, default_value = "models/ppo_model.pt")]
    rl_model_path: String,

    /// RL: Path to Python safetensors model (for importing pre-trained weights)
    #[arg(long)]
    rl_safetensors: Option<String>,

    /// RL: Enable training (otherwise inference only)
    #[arg(long, default_value_t = false)]
    rl_train: bool,

    /// RL: Enable RL dashboard at http://localhost:<port>/rl
    #[arg(long, default_value_t = false)]
    rl_dashboard: bool,

    /// RL: Dashboard port (default: 3001)
    #[arg(long, default_value_t = 3001)]
    rl_dashboard_port: u16,

    /// Path to .env file (default: .env in current directory)
    #[arg(long, default_value = ".env")]
    dotenv: String,
}

const POLYMARKET_WS_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const POLYGON_WS_URL: &str = "wss://socket.polygon.io/crypto";
const LOCAL_PRICE_SERVER: &str = "ws://127.0.0.1:9999";
const BOND_MAX_PRICE_STALE_SECS_DEFAULT: u64 = 20;
const BOND_MAX_STRIKE_DELAY_SECS: i64 = 10;

/// Market state
#[derive(Debug, Clone)]
struct Market {
    condition_id: String,
    question: String,
    event_slug: Option<String>,
    yes_token: String,
    no_token: String,
    asset: String, // "BTC", "ETH", "SOL", "XRP"
    expiry_minutes: Option<f64>,
    discovered_at: std::time::Instant,
    /// Unix timestamp when the 15-minute window starts (from slug)
    window_start_ts: Option<i64>,
    /// Strike price - captured from price feed when window starts
    strike_price: Option<f64>,
    /// Seconds late (or on-time) when strike was captured
    strike_capture_delay_secs: Option<i64>,
    // Orderbook L1 (top of book)
    yes_ask: Option<i64>,
    yes_bid: Option<i64>,
    no_ask: Option<i64>,
    no_bid: Option<i64>,
    yes_ask_size: f64,
    no_ask_size: f64,
    // Full orderbook depth for L5 imbalance (from Polymarket CLOB)
    yes_bids: Vec<(f64, f64)>,  // [(price, size), ...] sorted high to low
    yes_asks: Vec<(f64, f64)>,  // [(price, size), ...] sorted low to high
    // Probability history for vol_5m calculation (mid price history)
    prob_history: VecDeque<f64>,
}

impl Market {
    fn from_polymarket(pm: PolyMarket) -> Self {
        Self {
            condition_id: pm.condition_id,
            question: pm.question,
            event_slug: pm.event_slug,
            yes_token: pm.yes_token,
            no_token: pm.no_token,
            asset: pm.asset,
            expiry_minutes: pm.expiry_minutes,
            discovered_at: std::time::Instant::now(),
            window_start_ts: pm.window_start_ts,
            strike_price: None, // Will be captured from price feed
            strike_capture_delay_secs: None,
            yes_ask: None,
            yes_bid: None,
            no_ask: None,
            no_bid: None,
            yes_ask_size: 0.0,
            no_ask_size: 0.0,
            yes_bids: Vec::new(),
            yes_asks: Vec::new(),
            prob_history: VecDeque::with_capacity(100),
        }
    }

    /// Update orderbook depth from Polymarket CLOB snapshot
    fn update_orderbook(&mut self, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) {
        // Normalize ordering: CLOB snapshots can arrive unsorted.
        let mut bids = bids;
        let mut asks = asks;
        bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        self.yes_bids = bids;
        self.yes_asks = asks;

        // Update L1 from full book
        if let Some((price, size)) = self.yes_bids.first() {
            self.yes_bid = Some((price * 100.0) as i64);  // Convert to cents
            // Note: yes_bid_size not stored currently
        }
        if let Some((price, size)) = self.yes_asks.first() {
            self.yes_ask = Some((price * 100.0) as i64);
            self.yes_ask_size = *size;
        }

        // Update probability history with mid price
        if let (Some(&(bid_price, _)), Some(&(ask_price, _))) = (self.yes_bids.first(), self.yes_asks.first()) {
            let mid_price = (bid_price + ask_price) / 2.0;
            self.prob_history.push_back(mid_price);
            // Keep last 100 values (~5 mins at 3s intervals)
            while self.prob_history.len() > 100 {
                self.prob_history.pop_front();
            }
        }
    }

    /// Calculate L1 orderbook imbalance from YES token book
    fn orderbook_imbalance_l1(&self) -> f64 {
        let bid_vol = self.yes_bids.first().map(|(_, s)| *s).unwrap_or(0.0);
        let ask_vol = self.yes_asks.first().map(|(_, s)| *s).unwrap_or(0.0);
        calculate_orderbook_imbalance(bid_vol, ask_vol)
    }

    /// Calculate L5 orderbook imbalance from YES token book
    fn orderbook_imbalance_l5(&self) -> f64 {
        let bid_vol: f64 = self.yes_bids.iter().take(5).map(|(_, s)| s).sum();
        let ask_vol: f64 = self.yes_asks.iter().take(5).map(|(_, s)| s).sum();
        calculate_orderbook_imbalance(bid_vol, ask_vol)
    }

    /// Calculate spread percentage
    fn spread_pct(&self) -> f64 {
        let bid = self.yes_bids.first().map(|(p, _)| *p).unwrap_or(0.0);
        let ask = self.yes_asks.first().map(|(p, _)| *p).unwrap_or(0.0);
        calculate_spread_pct(bid, ask)
    }

    /// Calculate vol_5m from probability history
    fn vol_5m(&self) -> f64 {
        let history: Vec<f64> = self.prob_history.iter().copied().collect();
        calculate_vol_5m(&history, 30)  // ~30 samples = 5 mins at 10s intervals
    }

    /// Get mid price from YES token orderbook (for Phase 4 PnL calculation)
    fn mid_price(&self) -> Option<f64> {
        if let (Some(&(bid_price, _)), Some(&(ask_price, _))) = (self.yes_bids.first(), self.yes_asks.first()) {
            Some((bid_price + ask_price) / 2.0)
        } else if let Some(&last_prob) = self.prob_history.back() {
            Some(last_prob)  // Fall back to last known price
        } else {
            None
        }
    }

    fn time_remaining_mins(&self) -> Option<f64> {
        self.expiry_minutes.map(|exp| {
            let elapsed_mins = self.discovered_at.elapsed().as_secs_f64() / 60.0;
            (exp - elapsed_mins).max(0.0)
        })
    }
}

/// Position tracking
#[derive(Debug, Clone, Default)]
struct Position {
    yes_qty: f64,
    no_qty: f64,
    yes_cost: f64,
    no_cost: f64,
    // Bonds mode tracking
    bonds_qty: f64,
    bonds_cost: f64,
    bonds_side: Option<String>,
    // RL episode tracking for dashboard
    episode_id: Option<u64>,
}

#[derive(Debug, Clone)]
struct RlDecision {
    market_id: String,
    asset: String,
    action: Action,
    mid_price: f64,
    yes_bid: Option<f64>,
    yes_ask: Option<f64>,
    no_bid: Option<f64>,
    no_ask: Option<f64>,
    yes_token: String,
    no_token: String,
}

#[allow(dead_code)]
impl Position {
    fn matched(&self) -> f64 {
        self.yes_qty.min(self.no_qty)
    }
    fn profit(&self) -> f64 {
        // Locked profit from matched pairs (payout $1 each)
        self.matched() - self.yes_cost - self.no_cost
    }
}

/// Our resting orders
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct Orders {
    yes_order_id: Option<String>,
    yes_price: Option<i64>,
    no_order_id: Option<String>,
    no_price: Option<i64>,
}

/// Binance-sourced metrics for RL observation
/// Matches Python cross-market-state-fusion run.py lines 337-352
/// Note: Orderbook imbalance, spread, and vol_5m now come from Polymarket CLOB
#[derive(Debug, Clone, Default)]
struct AssetMetrics {
    // Basic metrics (for logging/debug, not used in observation)
    cvd: f64,

    // [0-2] Momentum (from Binance futures klines)
    return_1m: Option<f64>,
    return_5m: Option<f64>,
    return_10m: Option<f64>,

    // [5-6] Trade flow (from Binance futures)
    trade_flow_imbalance: f64,
    cvd_acceleration: f64,

    // [8-9] Microstructure (from Binance futures trades)
    trade_intensity: f64,
    large_trade: bool,

    // [11] Volatility expansion (from Binance)
    vol_expansion: f64,

    // [16-17] Regime (derived from Binance)
    vol_regime: f64,
    trend_regime: f64,
}

/// Price feed state
#[derive(Debug, Default, Clone)]
struct PriceState {
    btc_price: Option<f64>,
    eth_price: Option<f64>,
    sol_price: Option<f64>,
    xrp_price: Option<f64>,
    last_update: Option<std::time::Instant>,
    // Order flow metrics per asset (from Binance futures via price_feed.rs)
    btc_metrics: AssetMetrics,
    eth_metrics: AssetMetrics,
    sol_metrics: AssetMetrics,
    xrp_metrics: AssetMetrics,
}

/// Global state (must be Send + Sync for async sharing)
/// Note: PPO trainer is kept separate in main() due to PyTorch thread-safety constraints
struct State {
    markets: HashMap<String, Market>,
    positions: HashMap<String, Position>,
    orders: HashMap<String, Orders>,
    prices: PriceState,
    /// Flag to signal WebSocket needs to resubscribe with new tokens
    needs_resubscribe: bool,
    // RL config flags (trainer is separate)
    rl_enabled: bool,
    rl_train: bool,

    // === RL Training State (Phase 4) ===
    /// Previous observations for experience tuples (market_id -> obs)
    prev_observations: HashMap<String, Observation>,
    /// Previous actions for experience tuples (market_id -> (action, log_prob, value, history))
    prev_actions: HashMap<String, (Action, f32, f32, Vec<f32>)>,
    /// Pending rewards from closed positions (market_id -> pnl)
    /// Phase 4: Share-based PnL computed with compute_share_reward()
    pending_rewards: HashMap<String, f64>,
    /// Experience buffer for PPO training (uses library type)
    experience_buffer: ExperienceBuffer,
    /// Running reward stats for normalization
    reward_mean: f64,
    reward_std: f64,
    reward_count: u64,

    // === Phase 5: Temporal Features ===
    /// Observation history for temporal encoder (market_id -> last 5 observations)
    observation_history: HashMap<String, VecDeque<Observation>>,
}

impl State {
    fn new() -> Self {
        Self {
            markets: HashMap::new(),
            positions: HashMap::new(),
            orders: HashMap::new(),
            prices: PriceState::default(),
            needs_resubscribe: false,
            rl_enabled: false,
            rl_train: false,
            prev_observations: HashMap::new(),
            prev_actions: HashMap::new(),
            pending_rewards: HashMap::new(),
            experience_buffer: ExperienceBuffer::new(256),  // Phase 5: 256 buffer (was 1024)
            reward_mean: 0.0,
            reward_std: 1.0,
            reward_count: 0,
            observation_history: HashMap::new(),  // Phase 5: temporal history
        }
    }

    /// Store RL experience with reward normalization (matches Python Phase 5)
    fn store_experience(
        &mut self,
        market_id: &str,
        reward: f64,
        next_obs: Observation,
        next_history: Vec<f32>,
        done: bool,
    ) {
        // Get previous state and action (with temporal history)
        let Some(prev_obs) = self.prev_observations.get(market_id) else { return };
        let Some((action, log_prob, value, history)) = self.prev_actions.get(market_id) else { return };

        // Update running reward stats for normalization (Welford's algorithm)
        self.reward_count += 1;
        let delta = reward - self.reward_mean;
        self.reward_mean += delta / self.reward_count as f64;
        let delta2 = reward - self.reward_mean;
        self.reward_std = ((self.reward_count as f64 - 1.0) * self.reward_std.powi(2) + delta * delta2).sqrt()
            / (self.reward_count as f64).max(1.0).sqrt();

        // Normalize reward (z-score)
        let norm_reward = ((reward - self.reward_mean) / (self.reward_std + 1e-8)) as f32;

        // Store using library Experience type with temporal history
        let exp = Experience {
            obs: prev_obs.clone(),
            next_obs,
            action: *action as u8,
            log_prob: *log_prob,
            reward: norm_reward,
            value: *value,
            done,
            history: Some(history.clone()),
            next_history: Some(next_history),
        };
        self.experience_buffer.push(exp);
    }

    /// Get pending reward and clear it (Phase 4 share-based PnL)
    fn pop_reward(&mut self, market_id: &str) -> f64 {
        self.pending_rewards.remove(market_id).unwrap_or(0.0)
    }

    // === Phase 5: Temporal History Methods ===

    /// Push observation to history buffer for temporal encoder
    /// Maintains a rolling window of the last 5 observations per market
    fn push_observation(&mut self, market_id: &str, obs: Observation) {
        let history = self.observation_history
            .entry(market_id.to_string())
            .or_insert_with(|| VecDeque::with_capacity(6));

        history.push_back(obs);

        // Keep only the last 5 observations
        while history.len() > 5 {
            history.pop_front();
        }
    }

    /// Push current observation and return padded stacked history (90-dim).
    /// Matches CMSF Phase 5: pad with zeros until we have 5 observations.
    fn push_and_get_history(&mut self, market_id: &str, obs: &Observation) -> Vec<f32> {
        self.push_observation(market_id, obs.clone());

        let history = self.observation_history.get(market_id);
        let mut stacked = Vec::with_capacity(90);

        if let Some(history) = history {
            let missing = 5usize.saturating_sub(history.len());
            for _ in 0..missing {
                stacked.extend_from_slice(&[0.0; 18]);
            }
            for past in history.iter() {
                stacked.extend_from_slice(&past.features);
            }
        } else {
            // No history yet - all zeros
            stacked.extend_from_slice(&[0.0; 90]);
        }

        stacked
    }

    /// Get stacked history with zero padding (does not mutate).
    fn get_stacked_history(&self, market_id: &str) -> Vec<f32> {
        let mut stacked = Vec::with_capacity(90);
        let history = self.observation_history.get(market_id);

        if let Some(history) = history {
            let missing = 5usize.saturating_sub(history.len());
            for _ in 0..missing {
                stacked.extend_from_slice(&[0.0; 18]);
            }
            for past in history.iter() {
                stacked.extend_from_slice(&past.features);
            }
        } else {
            stacked.extend_from_slice(&[0.0; 90]);
        }

        stacked
    }

    /// Check if we have enough history for temporal encoder
    fn has_enough_history(&self, market_id: &str) -> bool {
        self.observation_history
            .get(market_id)
            .map(|h| h.len() >= 5)
            .unwrap_or(false)
    }
}


// === Polymarket WebSocket ===

#[derive(Deserialize, Debug)]
struct BookSnapshot {
    asset_id: String,
    bids: Vec<PriceLevel>,
    asks: Vec<PriceLevel>,
}

#[derive(Deserialize, Debug)]
struct PriceLevel {
    price: String,
    size: String,
}

/// Price change message format from Polymarket WebSocket
#[derive(Deserialize, Debug)]
struct PriceChangeMessage {
    #[serde(default)]
    price_changes: Vec<PriceChange>,
}

#[derive(Deserialize, Debug)]
struct PriceChange {
    asset_id: String,
    price: String,
}

#[derive(Serialize)]
struct SubscribeCmd {
    assets_ids: Vec<String>,
    #[serde(rename = "type")]
    sub_type: &'static str,
}

fn parse_price_cents(s: &str) -> i64 {
    s.parse::<f64>()
        .map(|p| (p * 100.0).round() as i64)
        .unwrap_or(0)
}

fn parse_size(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or(0.0)
}

// === Price Feed (local server or direct Polygon) ===

#[derive(Deserialize, Debug)]
struct PolygonMessage {
    ev: Option<String>,
    pair: Option<String>,
    p: Option<f64>,
}

/// Order flow metrics received from price_feed (matches price_feed.rs::AssetMetrics)
/// Updated to include Python cross-market-state-fusion features
#[derive(Deserialize, Debug, Clone, Default)]
struct PriceFeedMetrics {
    // Basic metrics
    #[serde(default)]
    cvd: f64,
    #[serde(default)]
    trade_intensity: f64,
    return_1m: Option<f64>,
    return_5m: Option<f64>,
    return_10m: Option<f64>,
    volatility: Option<f64>,
    #[serde(default)]
    large_trade: bool,
    // Order flow features (matches Python)
    #[serde(default)]
    order_book_imbalance_l1: f64,
    #[serde(default)]
    order_book_imbalance_l5: f64,
    #[serde(default)]
    trade_flow_imbalance: f64,
    #[serde(default)]
    cvd_acceleration: f64,
    // Microstructure
    spread_pct: Option<f64>,
    // Volatility features
    #[serde(default)]
    vol_expansion: f64,
    #[serde(default)]
    vol_regime: f64,
    #[serde(default)]
    trend_regime: f64,
}

#[derive(Deserialize, Debug)]
struct LocalPriceUpdate {
    btc_price: Option<f64>,
    eth_price: Option<f64>,
    sol_price: Option<f64>,
    xrp_price: Option<f64>,
    #[serde(default)]
    timestamp: i64,
    // Order flow metrics (optional, from enhanced price_feed)
    btc_metrics: Option<PriceFeedMetrics>,
    eth_metrics: Option<PriceFeedMetrics>,
    sol_metrics: Option<PriceFeedMetrics>,
    xrp_metrics: Option<PriceFeedMetrics>,
}

// NOTE: RtdsMessage and RtdsPayload structs removed - Chainlink RTDS feed no longer used

fn ensure_bonds_log(path: &str) -> Result<()> {
    let log_path = std::path::Path::new(path);
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let needs_header = match std::fs::metadata(log_path) {
        Ok(meta) => meta.len() == 0,
        Err(_) => true,
    };
    if needs_header {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;
        let header = [
            "ts_ms",
            "market_id",
            "asset",
            "expiry_mins",
            "binance_spot",
            "strike",
            "strike_delay_secs",
            "dist_pct",
            "winner",
            "yes_ask",
            "yes_ask_size",
            "no_ask",
            "no_ask_size",
            "effective_min_distance",
            "bond_max_price",
            "bond_min_size",
            "decision",
            "reason",
            "ask_price",
            "side",
            "contracts",
            "remaining",
            "total_spent",
        ]
        .join(",");
        writeln!(file, "{header}")?;
    }
    Ok(())
}

fn append_bonds_log(path: &str, row: &str) {
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{row}");
    }
}

fn fmt_opt_f64(value: Option<f64>, precision: usize) -> String {
    match value {
        Some(v) => format!("{:.*}", precision, v),
        None => String::new(),
    }
}

fn fmt_opt_i64(value: Option<i64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

/// Fetch resolved outcome prices (YES/NO) for a market, if available.
async fn fetch_resolved_outcome_prices(
    event_slug: &str,
    condition_id: &str,
) -> Result<Option<(f64, f64)>> {
    let url = format!(
        "https://gamma-api.polymarket.com/events?slug={}",
        event_slug
    );
    let resp = reqwest::get(&url).await?.error_for_status()?;
    let events: serde_json::Value = resp.json().await?;
    let Some(event) = events.as_array().and_then(|arr| arr.first()) else {
        return Ok(None);
    };
    let Some(markets) = event.get("markets").and_then(|v| v.as_array()) else {
        return Ok(None);
    };

    for market in markets {
        let Some(cond_id) = market.get("conditionId").and_then(|v| v.as_str()) else {
            continue;
        };
        if cond_id != condition_id {
            continue;
        }
        let Some(outcomes) = market.get("outcomePrices").and_then(|v| v.as_array()) else {
            return Ok(None);
        };
        if outcomes.len() < 2 {
            return Ok(None);
        }
        let parse_price = |value: &serde_json::Value| {
            value
                .as_str()
                .and_then(|s| s.parse::<f64>().ok())
                .or_else(|| value.as_f64())
        };
        let Some(yes_price) = parse_price(&outcomes[0]) else {
            return Ok(None);
        };
        let Some(no_price) = parse_price(&outcomes[1]) else {
            return Ok(None);
        };

        let resolved_yes = yes_price >= 0.999 && no_price <= 0.001;
        let resolved_no = yes_price <= 0.001 && no_price >= 0.999;
        if resolved_yes {
            return Ok(Some((1.0, 0.0)));
        }
        if resolved_no {
            return Ok(Some((0.0, 1.0)));
        }

        return Ok(None);
    }

    Ok(None)
}

/// Try local price server first, fallback to direct Polygon
async fn run_polygon_feed(
    state: Arc<RwLock<State>>,
    api_key: &str,
    dashboard_binance: Option<Arc<RwLock<BinanceState>>>,
) {
    loop {
        // Try local price server first
        info!("[PRICES] Trying local price server {}...", LOCAL_PRICE_SERVER);
        match connect_async(LOCAL_PRICE_SERVER).await {
            Ok((ws, _)) => {
                info!("[PRICES] ✅ Connected to local price server");
                if run_local_price_feed(state.clone(), ws, dashboard_binance.clone()).await.is_err() {
                    warn!("[PRICES] Local server disconnected");
                }
            }
            Err(_) => {
                info!("[PRICES] Local server not available, connecting to Polygon directly...");
                if let Err(e) = run_direct_polygon_feed(state.clone(), api_key).await {
                    error!("[POLYGON] Connection error: {}", e);
                }
            }
        }

        warn!("[PRICES] Reconnecting in 3s...");
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}

/// Connect to local price_feed server
async fn run_local_price_feed(
    state: Arc<RwLock<State>>,
    ws: tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    dashboard_binance: Option<Arc<RwLock<BinanceState>>>,
) -> Result<()> {
    let (mut write, mut read) = ws.split();

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if let Ok(update) = serde_json::from_str::<LocalPriceUpdate>(&text) {
                    let mut s = state.write().await;

                    // Current time for strike capture
                    let now_ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);

                    // Process each price and capture strikes
                    // Update Binance-sourced metrics (orderbook/spread/vol_5m now from Polymarket)
                    if let Some(m) = update.btc_metrics {
                        s.prices.btc_metrics = AssetMetrics {
                            cvd: m.cvd,
                            return_1m: m.return_1m,
                            return_5m: m.return_5m,
                            return_10m: m.return_10m,
                            trade_flow_imbalance: m.trade_flow_imbalance,
                            cvd_acceleration: m.cvd_acceleration,
                            trade_intensity: m.trade_intensity,
                            large_trade: m.large_trade,
                            vol_expansion: m.vol_expansion,
                            vol_regime: m.vol_regime,
                            trend_regime: m.trend_regime,
                        };
                    }
                    if let Some(m) = update.eth_metrics {
                        s.prices.eth_metrics = AssetMetrics {
                            cvd: m.cvd,
                            return_1m: m.return_1m,
                            return_5m: m.return_5m,
                            return_10m: m.return_10m,
                            trade_flow_imbalance: m.trade_flow_imbalance,
                            cvd_acceleration: m.cvd_acceleration,
                            trade_intensity: m.trade_intensity,
                            large_trade: m.large_trade,
                            vol_expansion: m.vol_expansion,
                            vol_regime: m.vol_regime,
                            trend_regime: m.trend_regime,
                        };
                    }
                    if let Some(m) = update.sol_metrics {
                        s.prices.sol_metrics = AssetMetrics {
                            cvd: m.cvd,
                            return_1m: m.return_1m,
                            return_5m: m.return_5m,
                            return_10m: m.return_10m,
                            trade_flow_imbalance: m.trade_flow_imbalance,
                            cvd_acceleration: m.cvd_acceleration,
                            trade_intensity: m.trade_intensity,
                            large_trade: m.large_trade,
                            vol_expansion: m.vol_expansion,
                            vol_regime: m.vol_regime,
                            trend_regime: m.trend_regime,
                        };
                    }
                    if let Some(m) = update.xrp_metrics {
                        s.prices.xrp_metrics = AssetMetrics {
                            cvd: m.cvd,
                            return_1m: m.return_1m,
                            return_5m: m.return_5m,
                            return_10m: m.return_10m,
                            trade_flow_imbalance: m.trade_flow_imbalance,
                            cvd_acceleration: m.cvd_acceleration,
                            trade_intensity: m.trade_intensity,
                            large_trade: m.large_trade,
                            vol_expansion: m.vol_expansion,
                            vol_regime: m.vol_regime,
                            trend_regime: m.trend_regime,
                        };
                    }

                    // Update prices
                    if let Some(p) = update.btc_price { s.prices.btc_price = Some(p); }
                    if let Some(p) = update.eth_price { s.prices.eth_price = Some(p); }
                    if let Some(p) = update.sol_price { s.prices.sol_price = Some(p); }
                    if let Some(p) = update.xrp_price { s.prices.xrp_price = Some(p); }

                    let prices: Vec<(&str, Option<f64>)> = vec![
                        ("BTC", update.btc_price),
                        ("ETH", update.eth_price),
                        ("SOL", update.sol_price),
                        ("XRP", update.xrp_price),
                    ];

                    for (asset, price_opt) in &prices {
                        let Some(price) = *price_opt else { continue };

                        // Capture strike price for markets where window has started
                        for market in s.markets.values_mut() {
                            if market.asset == *asset && market.strike_price.is_none() {
                                if let Some(start_ts) = market.window_start_ts {
                                    if now_ts >= start_ts {
                                        let delay_secs = now_ts - start_ts;
                                        market.strike_price = Some(price);
                                        market.strike_capture_delay_secs = Some(delay_secs);
                                        if delay_secs > 10 {
                                            warn!("[STRIKE] ⚠️ {} captured: ${:.2} ({}s LATE)",
                                                  asset, price, delay_secs);
                                        } else {
                                            info!("[STRIKE] {} captured: ${:.2} ({}s after window start)",
                                                  asset, price, delay_secs);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    s.prices.last_update = Some(std::time::Instant::now());

                    // Update dashboard Binance state if available
                    if let Some(ref dash_binance) = dashboard_binance {
                        let binance_update = BinanceState {
                            btc: AssetMetricsSnapshot {
                                price: s.prices.btc_price,
                                return_1m: s.prices.btc_metrics.return_1m,
                                return_5m: s.prices.btc_metrics.return_5m,
                                return_10m: s.prices.btc_metrics.return_10m,
                                cvd: s.prices.btc_metrics.cvd,
                                cvd_acceleration: s.prices.btc_metrics.cvd_acceleration,
                                trade_flow_imbalance: s.prices.btc_metrics.trade_flow_imbalance,
                                trade_intensity: s.prices.btc_metrics.trade_intensity,
                                large_trade: s.prices.btc_metrics.large_trade,
                                vol_expansion: s.prices.btc_metrics.vol_expansion,
                                vol_regime: s.prices.btc_metrics.vol_regime,
                                trend_regime: s.prices.btc_metrics.trend_regime,
                                ..Default::default()
                            },
                            eth: AssetMetricsSnapshot {
                                price: s.prices.eth_price,
                                return_1m: s.prices.eth_metrics.return_1m,
                                return_5m: s.prices.eth_metrics.return_5m,
                                return_10m: s.prices.eth_metrics.return_10m,
                                cvd: s.prices.eth_metrics.cvd,
                                cvd_acceleration: s.prices.eth_metrics.cvd_acceleration,
                                trade_flow_imbalance: s.prices.eth_metrics.trade_flow_imbalance,
                                trade_intensity: s.prices.eth_metrics.trade_intensity,
                                large_trade: s.prices.eth_metrics.large_trade,
                                vol_expansion: s.prices.eth_metrics.vol_expansion,
                                vol_regime: s.prices.eth_metrics.vol_regime,
                                trend_regime: s.prices.eth_metrics.trend_regime,
                                ..Default::default()
                            },
                            sol: AssetMetricsSnapshot {
                                price: s.prices.sol_price,
                                return_1m: s.prices.sol_metrics.return_1m,
                                return_5m: s.prices.sol_metrics.return_5m,
                                return_10m: s.prices.sol_metrics.return_10m,
                                cvd: s.prices.sol_metrics.cvd,
                                cvd_acceleration: s.prices.sol_metrics.cvd_acceleration,
                                trade_flow_imbalance: s.prices.sol_metrics.trade_flow_imbalance,
                                trade_intensity: s.prices.sol_metrics.trade_intensity,
                                large_trade: s.prices.sol_metrics.large_trade,
                                vol_expansion: s.prices.sol_metrics.vol_expansion,
                                vol_regime: s.prices.sol_metrics.vol_regime,
                                trend_regime: s.prices.sol_metrics.trend_regime,
                                ..Default::default()
                            },
                            xrp: AssetMetricsSnapshot {
                                price: s.prices.xrp_price,
                                return_1m: s.prices.xrp_metrics.return_1m,
                                return_5m: s.prices.xrp_metrics.return_5m,
                                return_10m: s.prices.xrp_metrics.return_10m,
                                cvd: s.prices.xrp_metrics.cvd,
                                cvd_acceleration: s.prices.xrp_metrics.cvd_acceleration,
                                trade_flow_imbalance: s.prices.xrp_metrics.trade_flow_imbalance,
                                trade_intensity: s.prices.xrp_metrics.trade_intensity,
                                large_trade: s.prices.xrp_metrics.large_trade,
                                vol_expansion: s.prices.xrp_metrics.vol_expansion,
                                vol_regime: s.prices.xrp_metrics.vol_regime,
                                trend_regime: s.prices.xrp_metrics.trend_regime,
                                ..Default::default()
                            },
                            last_update: Some(chrono::Utc::now()),
                        };
                        drop(s); // Release state lock before dashboard lock
                        let mut db = dash_binance.write().await;
                        *db = binance_update;
                    }
                }
            }
            Ok(Message::Ping(data)) => {
                let _ = write.send(Message::Pong(data)).await;
            }
            Ok(Message::Close(_)) | Err(_) => break,
            _ => {}
        }
    }
    Ok(())
}

/// Connect directly to Polygon.io
async fn run_direct_polygon_feed(state: Arc<RwLock<State>>, api_key: &str) -> Result<()> {
    let url = format!("{}?apiKey={}", POLYGON_WS_URL, api_key);
    let (ws, _) = connect_async(&url).await?;
    let (mut write, mut read) = ws.split();

    // Subscribe to all crypto pairs
    let sub = serde_json::json!({
        "action": "subscribe",
        "params": "XT.BTC-USD,XT.ETH-USD,XT.SOL-USD,XT.XRP-USD"
    });
    let _ = write.send(Message::Text(sub.to_string())).await;
    info!("[POLYGON] Subscribed to BTC-USD, ETH-USD, SOL-USD, XRP-USD");

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if let Ok(messages) = serde_json::from_str::<Vec<PolygonMessage>>(&text) {
                    for m in messages {
                        if m.ev.as_deref() != Some("XT") {
                            continue;
                        }

                        let Some(pair) = m.pair.as_ref() else { continue };
                        let Some(price) = m.p else { continue };

                        let asset = match pair.as_str() {
                            "BTC-USD" => "BTC",
                            "ETH-USD" => "ETH",
                            "SOL-USD" => "SOL",
                            "XRP-USD" => "XRP",
                            _ => continue,
                        };

                        let mut s = state.write().await;

                        // Update price state
                        match asset {
                            "BTC" => s.prices.btc_price = Some(price),
                            "ETH" => s.prices.eth_price = Some(price),
                            "SOL" => s.prices.sol_price = Some(price),
                            "XRP" => s.prices.xrp_price = Some(price),
                            _ => {}
                        }

                        // Current time for strike capture
                        let now_ts = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs() as i64)
                            .unwrap_or(0);

                        // Capture strike price for markets where window has started
                        for market in s.markets.values_mut() {
                            if market.asset == asset && market.strike_price.is_none() {
                                if let Some(start_ts) = market.window_start_ts {
                                    if now_ts >= start_ts {
                                        let delay_secs = now_ts - start_ts;
                                        market.strike_price = Some(price);
                                        if delay_secs > 10 {
                                            warn!("[STRIKE] ⚠️ {} captured: ${:.2} ({}s LATE)",
                                                  asset, price, delay_secs);
                                        } else {
                                            info!("[STRIKE] {} captured: ${:.2} ({}s after window start)",
                                                  asset, price, delay_secs);
                                        }
                                    }
                                }
                            }
                        }

                        s.prices.last_update = Some(std::time::Instant::now());
                    }
                }
            }
            Ok(Message::Ping(data)) => {
                let _ = write.send(Message::Pong(data)).await;
            }
            Err(e) => {
                error!("[POLYGON] WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }
    Ok(())
}

// NOTE: Chainlink RTDS feed removed - now using Binance futures via price_feed.rs

/// Calculate the distance from strike as a percentage
fn distance_from_strike_pct(spot: f64, strike: f64) -> f64 {
    ((spot - strike) / strike).abs() * 100.0
}

/// Check if market qualifies for bonds entry
/// Returns Some((token_id, side, ask_price, distance_pct)) if all conditions met
#[allow(dead_code)]
fn check_bonds_entry(
    market: &Market,
    spot: f64,
    bond_min_distance: f64,
    bond_max_price: i64,
    bond_min_size: f64,
) -> Option<(String, &'static str, i64, f64)> {
    let strike = market.strike_price?;

    // Calculate distance from strike
    let distance_pct = ((spot - strike) / strike).abs() * 100.0;

    // Must be far enough from strike
    if distance_pct < bond_min_distance {
        return None;
    }

    // Determine winner based on spot vs strike
    if spot >= strike {
        // YES wins - check YES ask
        let ask = market.yes_ask.unwrap_or(100);
        let size = market.yes_ask_size;
        if ask <= bond_max_price && size >= bond_min_size {
            return Some((market.yes_token.clone(), "YES", ask, distance_pct));
        }
    } else {
        // NO wins - check NO ask
        let ask = market.no_ask.unwrap_or(100);
        let size = market.no_ask_size;
        if ask <= bond_max_price && size >= bond_min_size {
            return Some((market.no_token.clone(), "NO", ask, distance_pct));
        }
    }

    None
}

/// Build an observation vector for PPO from current state
/// Uses the new Python-matching 18-dim layout with correct data sources:
/// - Binance: momentum, trade flow, microstructure (intensity/large_trade), vol_expansion, regimes
/// - Polymarket CLOB: orderbook imbalance, spread, vol_5m
fn build_observation(
    prices: &PriceState,
    market: &Market,
    position: Option<&Position>,
    now_ts: i64,
    _max_dollars: f64,
) -> Observation {
    // Get Binance metrics for this asset
    let metrics = match market.asset.as_str() {
        "BTC" => &prices.btc_metrics,
        "ETH" => &prices.eth_metrics,
        "SOL" => &prices.sol_metrics,
        "XRP" => &prices.xrp_metrics,
        _ => return Observation::default(),
    };

    // BinanceMetrics: momentum, trade flow, microstructure, regimes
    // (Matches Python run.py lines 337-352)
    let binance_metrics = BinanceMetrics {
        return_1m: metrics.return_1m,
        return_5m: metrics.return_5m,
        return_10m: metrics.return_10m,
        trade_flow_imbalance: metrics.trade_flow_imbalance,
        cvd_acceleration: metrics.cvd_acceleration,
        trade_intensity: metrics.trade_intensity,
        large_trade: metrics.large_trade,
        vol_expansion: metrics.vol_expansion,
        vol_regime: metrics.vol_regime,
        trend_regime: metrics.trend_regime,
    };

    // PolymarketMetrics: orderbook imbalance, spread, vol_5m from CLOB data
    // (Matches Python run.py lines 298-319 and base.py lines 87-90)
    let polymarket_metrics = PolymarketMetrics {
        order_book_imbalance_l1: market.orderbook_imbalance_l1(),
        order_book_imbalance_l5: market.orderbook_imbalance_l5(),
        spread_pct: market.spread_pct(),
        vol_5m: market.vol_5m(),
    };

    // Compute position state
    let has_position = position.map(|p| p.yes_qty > 0.0 || p.no_qty > 0.0).unwrap_or(false);

    let position_side = position.and_then(|p| {
        if p.yes_qty > p.no_qty {
            Some("YES")
        } else if p.no_qty > p.yes_qty {
            Some("NO")
        } else {
            None
        }
    });

    // Compute unrealized P&L using Phase 4 share-based formula (raw dollars)
    // Python: shares = pos.size / pos.entry_price; pnl = (current - entry) * shares
    let position_pnl = position.map(|p| {
        // Get current YES price from orderbook (mid price or ask)
        let current_yes_price = market.yes_bids.first()
            .and_then(|b| market.yes_asks.first().map(|a| (b.0 + a.0) / 2.0))
            .or_else(|| market.yes_ask.map(|a| a as f64 / 100.0))
            .unwrap_or(0.5);

        let mut pnl = 0.0;

        // YES position P&L (share-based)
        if p.yes_qty > 0.0 && p.yes_cost > 0.0 {
            let entry_price = p.yes_cost / p.yes_qty;  // Cost per share
            let shares = p.yes_qty;
            pnl += (current_yes_price - entry_price) * shares;
        }

        // NO position P&L (share-based)
        // NO price = 1 - YES price
        if p.no_qty > 0.0 && p.no_cost > 0.0 {
            let current_no_price = 1.0 - current_yes_price;
            let entry_price = p.no_cost / p.no_qty;  // Cost per share
            let shares = p.no_qty;
            pnl += (current_no_price - entry_price) * shares;
        }

        pnl
    }).unwrap_or(0.0);

    // Time remaining as fraction (0-1), derived from window_start_ts when available
    let time_remaining = if let Some(start_ts) = market.window_start_ts {
        let end_ts = start_ts + 15 * 60;
        let secs_left = (end_ts - now_ts) as f64;
        (secs_left / 900.0).clamp(0.0, 1.0)
    } else {
        market.expiry_minutes
            .map(|m| (m / 15.0).clamp(0.0, 1.0))
            .unwrap_or(1.0)
    };

    // Use the Python-matching observation builder with both data sources
    rl_build_observation(
        &binance_metrics,
        &polymarket_metrics,
        has_position,
        position_side,
        position_pnl,
        time_remaining,
    )
}

/// Get PPO action recommendation (without temporal history - legacy)
#[cfg(feature = "rl")]
fn get_ppo_action(trainer: &PpoTrainer, obs: &Observation) -> (Action, f32, f32) {
    trainer.model.select_action(obs)
}

#[cfg(not(feature = "rl"))]
fn get_ppo_action(_trainer: &PpoTrainer, _obs: &Observation) -> (Action, f32, f32) {
    (Action::Hold, 0.0, 0.0)
}

/// Get PPO action with Phase 5 temporal history
/// Uses the temporal encoder to process 5 past observations (90-dim) for momentum detection
#[cfg(feature = "rl")]
fn get_ppo_action_with_history(
    trainer: &PpoTrainer,
    obs: &Observation,
    history: Option<&[f32]>,
) -> (Action, f32, f32) {
    match history {
        Some(h) if h.len() == 90 => trainer.model.select_action_with_history(obs, h),
        _ => trainer.model.select_action(obs),  // Fallback without history
    }
}

#[cfg(not(feature = "rl"))]
fn get_ppo_action_with_history(
    _trainer: &PpoTrainer,
    _obs: &Observation,
    _history: Option<&[f32]>,
) -> (Action, f32, f32) {
    (Action::Hold, 0.0, 0.0)
}

// === Main ===

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("poly_atm_sniper=info".parse().unwrap())
                .add_directive("arb_bot=info".parse().unwrap()),
        )
        .init();

    let args = Args::parse();

    if args.bonds {
        ensure_bonds_log(&args.bonds_log)?;
        // ========== BONDS MODE BANNER ==========
        info!("═══════════════════════════════════════════════════════════════════════");
        info!("💎 POLYMARKET BONDS SNIPER - Near-Expiry Winner Strategy");
        info!("═══════════════════════════════════════════════════════════════════════");
        info!("");
        info!("📥 ENTRY CONDITIONS (all must be true):");
        info!("");
        info!("   1. Time:     <= {}m to expiry", args.bond_minutes);
        info!("   2. Distance: >= {:.2}% from strike (outcome locked in)", args.bond_min_distance);
        info!("   3. Winner:   Spot > strike = YES, Spot < strike = NO");
        info!("   4. Cost:     Ask <= {}¢ ({}%+ profit)", args.bond_max_price, 100 - args.bond_max_price);
        info!("   5. Size:     >= {} contracts at ask", args.bond_min_size);
        info!("   6. Budget:   Total cost < ${:.2}", args.max_dollars);
        info!("");
        info!("📤 ACTION:");
        info!("   • Buy ONE side (the winner) at ask price");
        info!("   • Hold until expiry → receive $1 per contract");
        info!("   • Profit = $1 - ask price");
        info!("");
        info!("═══════════════════════════════════════════════════════════════════════");
        info!("CONFIG:");
        info!("   Mode: {}", if args.live { "🚀 LIVE" } else { "🔍 DRY RUN" });
        info!("   Strategy: 💎 BONDS");
        info!("   Time window: <= {}m to expiry", args.bond_minutes);
        info!("   Min distance: >= {:.2}%", args.bond_min_distance);
        info!("   Max price: <= {}¢", args.bond_max_price);
        info!("   Min size: >= {} contracts", args.bond_min_size);
        if let Some(ref sym) = args.sym {
            info!("   Asset filter: {}", sym.to_uppercase());
        }
        info!("   Max dollars: ${:.2}", args.max_dollars);
        info!("═══════════════════════════════════════════════════════════════════════");
    } else {
        // ========== ATM MODE BANNER ==========
        info!("═══════════════════════════════════════════════════════════════════════");
        info!("🎯 POLYMARKET ATM SNIPER - Delta 0.50 Strategy");
        info!("═══════════════════════════════════════════════════════════════════════");
        info!("");
        info!("📥 ENTRY CONDITIONS (either triggers entry):");
        info!("");
        info!("   🔔 ATM ENTRY (delta ≈ 0.50):");
        info!("      • Spot within {:.4}% of strike", args.atm_threshold);
        info!("      • Time: {}m - {}m before expiry", args.min_minutes, args.max_minutes);
        info!("      • Action: BID {}¢ on both YES and NO", args.bid);
        info!("");
        info!("   💰 ARB ENTRY (guaranteed profit):");
        info!("      • Combined asks < 100¢ (YES + NO < $1)");
        info!("      • Time: {}m - {}m before expiry", args.min_minutes, args.max_minutes);
        info!("      • Action: BUY at ask prices immediately");
        info!("");
        info!("   Both require: Cost < ${:.2}, Order >= $1", args.max_dollars);
        info!("");
        info!("📤 EXIT CONDITIONS:");
        info!("   • Hold until expiry (auto-settles at $1 for winner, $0 for loser)");
        info!("   • Matched pairs (YES+NO) = guaranteed $1 payout");
        info!("");
        info!("═══════════════════════════════════════════════════════════════════════");
        info!("CONFIG:");
        info!("   Mode: {}", if args.live { "🚀 LIVE" } else { "🔍 DRY RUN" });
        info!("   Strategy: 🎯 ATM/ARB");
        info!("   Price Feed: {}", if args.direct { "Direct Polygon.io" } else { "Local server (ws://127.0.0.1:9999)" });
        info!("   Contracts: {} per trade (min for $1)", args.contracts);
        info!("   Max bid: {}¢", args.bid);
        info!("   ATM threshold: {:.4}%", args.atm_threshold);
        info!("   Time window: {}m - {}m before expiry", args.min_minutes, args.max_minutes);
        if let Some(ref market) = args.market {
            info!("   Market filter: {}", market);
        }
        if let Some(ref sym) = args.sym {
            info!("   Asset filter: {}", sym.to_uppercase());
        }
        info!("   Max dollars: ${:.2}", args.max_dollars);
        info!("═══════════════════════════════════════════════════════════════════════");
    }

    // Load Polymarket credentials
    dotenvy::from_path(&args.dotenv).ok();
    let private_key = std::env::var("POLY_PRIVATE_KEY")
        .context("POLY_PRIVATE_KEY not set")?;
    let funder = std::env::var("POLY_FUNDER")
        .context("POLY_FUNDER not set")?;
    let polygon_api_key = std::env::var("POLYGON_API_KEY")
        .unwrap_or_else(|_| "o2Jm26X52_0tRq2W7V5JbsCUXdMjL7qk".to_string());

    // Initialize Polymarket client
    let poly_client = PolymarketAsyncClient::new(
        "https://clob.polymarket.com",
        137,
        &private_key,
        &funder,
    )?;

    // Derive API credentials
    info!("[POLY] Deriving API credentials...");
    let api_creds = poly_client.derive_api_key(0).await?;
    let prepared_creds = PreparedCreds::from_api_creds(&api_creds)?;
    let shared_client = Arc::new(SharedAsyncClient::new(poly_client, prepared_creds, 137));
    info!("[POLY] API credentials ready");

    if args.live {
        warn!("⚠️  LIVE MODE - Real money!");
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
    if args.rl_mode && args.live {
        warn!("[RL] ⚠️  LIVE RL MODE - Real money trades based on PPO agent decisions!");
    }

    // Discover markets using shared module
    info!("[DISCOVER] Searching for crypto markets...");
    let mut discovered = discover_all_markets(args.market.as_deref()).await?;

    // Filter by asset symbol if specified
    if let Some(ref sym) = args.sym {
        let sym_upper = sym.to_uppercase();
        discovered.retain(|m| m.asset == sym_upper);
        info!("[DISCOVER] Filtered to {} {} markets", discovered.len(), sym_upper);
    } else {
        info!("[DISCOVER] Found {} markets", discovered.len());
    }

    for m in &discovered {
        let start_info = m.window_start_ts
            .map(|ts| format!("start_ts={}", ts))
            .unwrap_or_else(|| "no_start".into());
        info!("  • {} | {} | Asset: {} | Expiry: {:.1?}min",
              &m.question[..m.question.len().min(50)],
              start_info, m.asset, m.expiry_minutes);
    }

    if discovered.is_empty() {
        warn!("No markets found! Try different search terms.");
        return Ok(());
    }

    // Initialize state - convert PolyMarket to Market
    let state = Arc::new(RwLock::new({
        let mut s = State::new();
        for pm in discovered {
            let id = pm.condition_id.clone();
            s.positions.insert(id.clone(), Position::default());
            s.orders.insert(id.clone(), Orders::default());
            s.markets.insert(id, Market::from_polymarket(pm));
        }
        // Initialize RL mode from args
        s.rl_enabled = args.rl_mode;
        s.rl_train = args.rl_train;
        if args.rl_mode {
            #[cfg(feature = "rl")]
            {
                info!("[RL] PPO observation mode enabled (inference happens in main loop)");
                if args.rl_train {
                    info!("[RL] Training mode active");
                }
            }
            #[cfg(not(feature = "rl"))]
            {
                warn!("[RL] RL mode requested but 'rl' feature not enabled!");
                warn!("[RL] Build with: cargo build --features rl");
                s.rl_enabled = false;
            }
        }
        s
    }));

    // Create PPO trainer for RL mode (wrapped in Arc<Mutex<>> for thread-safe sharing)
    // The trainer runs on a blocking thread due to PyTorch thread-safety requirements
    // NOTE: Trainer is created for BOTH inference and training modes (need model for action selection)
    let ppo_trainer: Option<Arc<Mutex<PpoTrainer>>> = if args.rl_mode {
        #[cfg(feature = "rl")]
        {
            use tch::Device;
            // Phase 5 config: temporal architecture with momentum features
            let config = PpoConfig {
                lr_actor: 1e-4,            // lr_actor from Python
                lr_critic: 3e-4,           // lr_critic from Python
                gamma: 0.95,               // Phase 5: shorter horizon for 15-min markets
                lambda: 0.95,              // GAE lambda
                clip_epsilon: 0.2,         // PPO clip
                value_coef: 0.5,           // Value loss coefficient
                entropy_coef: 0.03,        // Phase 5: sparse HOLD policy
                max_grad_norm: 0.5,        // Gradient clipping
                ppo_epochs: 10,            // n_epochs from Python
                batch_size: 64,            // batch_size from Python
                history_len: 5,            // Phase 5: 5 past states for temporal encoder
                temporal_dim: 32,          // Phase 5: 32-dim temporal features
            };
            let device = Device::cuda_if_available();
            info!("[RL] Creating PPO trainer on device: {:?}", device);
            info!("[RL] Config: lr_actor={}, lr_critic={}, gamma={}, lambda={}, clip_eps={}, entropy_coef={}",
                  config.lr_actor, config.lr_critic, config.gamma, config.lambda, config.clip_epsilon, config.entropy_coef);

            // Try to load existing model weights
            let mut trainer = PpoTrainer::new(config, device);

            // First try safetensors (Python pre-trained), then PyTorch format
            let mut loaded = false;
            if let Some(ref safetensors_path) = args.rl_safetensors {
                if std::path::Path::new(safetensors_path).exists() {
                    match trainer.model.load_safetensors(safetensors_path) {
                        Ok(_) => {
                            info!("[RL] ✅ Loaded Python pre-trained weights from {}", safetensors_path);
                            loaded = true;
                        }
                        Err(e) => warn!("[RL] Failed to load safetensors: {} - trying PyTorch format", e),
                    }
                } else {
                    warn!("[RL] Safetensors file not found: {}", safetensors_path);
                }
            }

            // Fallback to PyTorch format if safetensors not loaded
            if !loaded {
                let model_path = &args.rl_model_path;
                if std::path::Path::new(model_path).exists() {
                    match trainer.load(model_path) {
                        Ok(_) => info!("[RL] Loaded model weights from {}", model_path),
                        Err(e) => {
                            // Try partial transfer for 3-action → 7-action upgrade
                            warn!("[RL] Failed to load model: {} - attempting partial transfer for upgrade...", e);
                            match trainer.model.load_partial_for_upgrade(model_path) {
                                Ok(_) => info!("[RL] ✅ Partial transfer complete - kept learned features, reinit actor_head for 7 actions"),
                                Err(e2) => warn!("[RL] Partial transfer also failed: {} - starting fresh", e2),
                            }
                        }
                    }
                } else {
                    info!("[RL] No existing model at {} - starting fresh", model_path);
                }
            }

            Some(Arc::new(Mutex::new(trainer)))
        }
        #[cfg(not(feature = "rl"))]
        {
            warn!("[RL] Training requested but 'rl' feature not enabled!");
            None
        }
    } else {
        None
    };

    // Create RL metrics collector (for dashboard visualization)
    // RlMetricsCollector is Clone and uses Arc internally for thread-safety
    let rl_metrics: Option<RlMetricsCollector> = if args.rl_mode {
        let collector = RlMetricsCollector::new(256);
        collector.set_rl_mode(true, args.rl_train);
        Some(collector)
    } else {
        None
    };

    // Create shared dashboard state for 6 panels (if dashboard enabled)
    let dashboard_binance_state: Option<Arc<RwLock<BinanceState>>> = if args.rl_dashboard && args.rl_mode {
        Some(Arc::new(RwLock::new(BinanceState::default())))
    } else {
        None
    };
    let dashboard_polymarket_state: Option<Arc<RwLock<PolymarketState>>> = if args.rl_dashboard && args.rl_mode {
        Some(Arc::new(RwLock::new(PolymarketState::default())))
    } else {
        None
    };
    let dashboard_features_state: Option<Arc<RwLock<FeaturesState>>> = if args.rl_dashboard && args.rl_mode {
        Some(Arc::new(RwLock::new(FeaturesState::default())))
    } else {
        None
    };
    let dashboard_positions_state: Option<Arc<RwLock<PositionsState>>> = if args.rl_dashboard && args.rl_mode {
        Some(Arc::new(RwLock::new(PositionsState::default())))
    } else {
        None
    };
    let dashboard_cb_state: Option<Arc<RwLock<CircuitBreakerState>>> = if args.rl_dashboard && args.rl_mode {
        Some(Arc::new(RwLock::new(CircuitBreakerState::default())))
    } else {
        None
    };
    let dashboard_model_state: Option<Arc<RwLock<ModelInfoState>>> = if args.rl_dashboard && args.rl_mode {
        Some(Arc::new(RwLock::new(ModelInfoState::default())))
    } else {
        None
    };

    // Start RL dashboard if enabled
    if args.rl_dashboard {
        if let Some(ref metrics) = rl_metrics {
            // Use new_with_state to pass shared state
            let dashboard_state = RlDashboardState::new_with_state(
                metrics.clone(),
                dashboard_binance_state.clone().unwrap_or_else(|| Arc::new(RwLock::new(BinanceState::default()))),
                dashboard_polymarket_state.clone().unwrap_or_else(|| Arc::new(RwLock::new(PolymarketState::default()))),
                dashboard_features_state.clone().unwrap_or_else(|| Arc::new(RwLock::new(FeaturesState::default()))),
                dashboard_positions_state.clone().unwrap_or_else(|| Arc::new(RwLock::new(PositionsState::default()))),
                dashboard_cb_state.clone().unwrap_or_else(|| Arc::new(RwLock::new(CircuitBreakerState::default()))),
                dashboard_model_state.clone().unwrap_or_else(|| Arc::new(RwLock::new(ModelInfoState::default()))),
                Arc::new(RwLock::new(PerformanceByAsset::new())),  // Performance by asset - data comes from metrics collector
            );
            let port = args.rl_dashboard_port;

            // Spawn dashboard server
            let dashboard_state_for_server = dashboard_state.clone();
            tokio::spawn(async move {
                use axum::Router;
                use tower_http::cors::CorsLayer;

                let app = Router::new()
                    .nest("/rl", rl_router(dashboard_state_for_server))
                    .layer(CorsLayer::permissive());

                let addr = format!("127.0.0.1:{}", port);
                let listener = match tokio::net::TcpListener::bind(&addr).await {
                    Ok(l) => l,
                    Err(e) => {
                        error!("[RL-DASHBOARD] Failed to bind http://{}: {}", addr, e);
                        return;
                    }
                };

                info!("[RL-DASHBOARD] Running at http://{}/rl", addr);

                if let Err(e) = axum::serve(listener, app).await {
                    error!("[RL-DASHBOARD] Server error: {}", e);
                }
            });

            // Spawn broadcast loop for real-time updates
            let broadcast_state = dashboard_state.clone();
            tokio::spawn(async move {
                start_broadcast_loop(broadcast_state, 1).await;
            });

            // Initialize model info state
            if let Some(ref model_state) = dashboard_model_state {
                let mut ms = model_state.write().await;
                ms.model_path = args.rl_model_path.clone();
                ms.safetensors_path = args.rl_safetensors.clone();
                #[cfg(feature = "rl")]
                {
                    use tch::Device;
                    ms.device = if Device::cuda_if_available() == Device::Cpu {
                        "CPU".to_string()
                    } else {
                        "CUDA".to_string()
                    };
                }
                #[cfg(not(feature = "rl"))]
                {
                    ms.device = "CPU".to_string();
                }
                // Config values (hardcoded to match the config created above)
                ms.lr_actor = 1e-4;
                ms.lr_critic = 3e-4;
                ms.gamma = 0.95;
                ms.lambda = 0.95;
                ms.clip_epsilon = 0.2;
                ms.value_coef = 0.5;
                ms.entropy_coef = 0.03;
                ms.ppo_epochs = 10;
                ms.batch_size = 64;
                ms.model_version = 1;
                ms.loaded_at = Some(chrono::Utc::now());
            }

            // Spawn periodic dashboard update task (for Polymarket, Features, Positions)
            let state_for_dash = state.clone();
            let dash_poly = dashboard_polymarket_state.clone();
            let dash_features = dashboard_features_state.clone();
            let dash_positions = dashboard_positions_state.clone();
            let dash_cb = dashboard_cb_state.clone();
            let metrics_for_dash = rl_metrics.clone();
            let max_dollars_for_dash = args.max_dollars;
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(1));
                loop {
                    interval.tick().await;
                    let s = state_for_dash.read().await;

                    // Update Polymarket state and Features from first active market
                    if let Some((mid, market)) = s.markets.iter().next() {
                        // Polymarket panel
                        if let Some(ref poly_state) = dash_poly {
                            let poly_update = PolymarketState {
                                market_id: mid.clone(),
                                question: market.question.clone(),
                                asset: market.asset.clone(),
                                yes_ask: market.yes_ask.map(|c| c as f64 / 100.0),
                                no_ask: market.no_ask.map(|c| c as f64 / 100.0),
                                yes_bid: market.yes_bid.map(|c| c as f64 / 100.0),
                                no_bid: market.no_bid.map(|c| c as f64 / 100.0),
                                yes_ask_size: market.yes_ask_size,
                                no_ask_size: market.no_ask_size,
                                spread_pct: {
                                    let bid = market.yes_bid.unwrap_or(0) as f64 / 100.0;
                                    let ask = market.yes_ask.unwrap_or(100) as f64 / 100.0;
                                    calculate_spread_pct(bid, ask)
                                },
                                orderbook_imbalance_l1: {
                                    let bid_vol: f64 = market.yes_bids.iter().take(1).map(|(_, s)| s).sum();
                                    let ask_vol: f64 = market.yes_asks.iter().take(1).map(|(_, s)| s).sum();
                                    calculate_orderbook_imbalance(bid_vol, ask_vol)
                                },
                                orderbook_imbalance_l5: {
                                    let bid_vol: f64 = market.yes_bids.iter().take(5).map(|(_, s)| s).sum();
                                    let ask_vol: f64 = market.yes_asks.iter().take(5).map(|(_, s)| s).sum();
                                    calculate_orderbook_imbalance(bid_vol, ask_vol)
                                },
                                mid_price: market.mid_price(),
                                time_to_expiry_mins: market.time_remaining_mins(),
                                last_update: Some(chrono::Utc::now()),
                            };
                            let mut ps = poly_state.write().await;
                            *ps = poly_update;
                        }

                        // Features panel (18-dim observation)
                        if let Some(ref feat_state) = dash_features {
                            let pos = s.positions.get(mid).cloned().unwrap_or_default();
                            let now_ts = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs() as i64)
                                .unwrap_or(0);

                            // Build raw observation (not normalized yet)
                            let obs = build_observation(&s.prices, &market, Some(&pos), now_ts, 10.0);

                            // Create features state from observation
                            let features_update = FeaturesState::from_observation(&obs.features, &obs.features);
                            let mut fs = feat_state.write().await;
                            *fs = features_update;
                        }
                    }

                    // Update Positions state
                    if let Some(ref pos_state) = dash_positions {
                        let mut total_cost = 0.0;
                        let mut guaranteed_profit = 0.0;
                        let mut unmatched_exposure = 0.0;
                        let mut positions = Vec::new();

                        for (mid, pos) in &s.positions {
                            let matched = pos.yes_qty.min(pos.no_qty);
                            let matched_cost = if matched > 0.0 {
                                let avg_yes = if pos.yes_qty > 0.0 { pos.yes_cost / pos.yes_qty } else { 0.0 };
                                let avg_no = if pos.no_qty > 0.0 { pos.no_cost / pos.no_qty } else { 0.0 };
                                matched * (avg_yes + avg_no)
                            } else { 0.0 };
                            let profit = matched - matched_cost;

                            total_cost += pos.yes_cost + pos.no_cost;
                            guaranteed_profit += profit;
                            unmatched_exposure += (pos.yes_qty - matched).max(0.0) + (pos.no_qty - matched).max(0.0);

                            if pos.yes_qty > 0.0 || pos.no_qty > 0.0 {
                                let desc = s.markets.get(mid).map(|m| m.question.clone()).unwrap_or_else(|| mid.clone());
                                positions.push(arb_bot::rl_dashboard::PositionEntry {
                                    market_id: mid.clone(),
                                    description: desc,
                                    kalshi_yes: 0,
                                    kalshi_no: 0,
                                    poly_yes: pos.yes_qty as i64,
                                    poly_no: pos.no_qty as i64,
                                    total_cost: pos.yes_cost + pos.no_cost,
                                    guaranteed_profit: profit,
                                    unmatched_exposure: (pos.yes_qty - matched).max(0.0) + (pos.no_qty - matched).max(0.0),
                                });
                            }
                        }

                        // Get P&L from metrics if available
                        let (daily_pnl, all_time_pnl, total_trades) = if let Some(ref m) = metrics_for_dash {
                            if let Some(snapshot) = m.get_metrics_snapshot() {
                                (snapshot.cumulative_pnl, snapshot.cumulative_pnl, snapshot.total_trades)
                            } else {
                                (0.0, 0.0, 0)
                            }
                        } else {
                            (0.0, 0.0, 0)
                        };

                        let mut ps = pos_state.write().await;
                        ps.summary = arb_bot::rl_dashboard::PositionsSummary {
                            total_cost_basis: total_cost,
                            guaranteed_profit,
                            unmatched_exposure,
                            open_count: positions.len(),
                            resolved_count: total_trades as usize,
                            total_contracts: positions.iter().map(|p| p.poly_yes + p.poly_no).sum(),
                            daily_pnl,
                            all_time_pnl,
                        };
                        ps.positions = positions;
                        ps.last_update = Some(chrono::Utc::now());
                    }

                    // Update Circuit Breaker state from real metrics
                    if let Some(ref cb_state) = dash_cb {
                        let mut cbs = cb_state.write().await;
                        cbs.enabled = true;
                        cbs.halted = false;
                        cbs.market_count = s.markets.len();
                        cbs.total_position = s.positions.values()
                            .map(|p| (p.yes_qty + p.no_qty) as i64)
                            .sum();
                        cbs.max_position = max_dollars_for_dash as i64;

                        // Get P&L from metrics collector if available
                        if let Some(ref m) = metrics_for_dash {
                            if let Some(snapshot) = m.get_metrics_snapshot() {
                                cbs.daily_pnl = snapshot.cumulative_pnl;
                            }
                        }

                        cbs.last_update = Some(chrono::Utc::now());
                    }
                }
            });
        } else {
            warn!("[RL-DASHBOARD] Dashboard requested but RL mode not enabled!");
        }
    }

    // Start price feed
    let state_clone = state.clone();
    let polygon_key = polygon_api_key.clone();
    let use_direct = args.direct;
    let dash_binance_for_feed = dashboard_binance_state.clone();
    if use_direct {
        tokio::spawn(async move {
            loop {
                if let Err(e) = run_direct_polygon_feed(state_clone.clone(), &polygon_key).await {
                    tracing::error!("[POLYGON] Connection error: {}", e);
                }
                tracing::warn!("[POLYGON] Reconnecting in 3s...");
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
        });
    } else {
        tokio::spawn(async move {
            run_polygon_feed(state_clone, &polygon_key, dash_binance_for_feed).await;
        });
    }

    // NOTE: Chainlink RTDS feed removed - now using Binance futures via price_feed.rs

    // Start periodic market discovery task
    let state_for_discovery = state.clone();
    let discovery_filter = args.sym.clone();
    let market_filter = args.market.clone();
    let discovery_sleep = if args.bonds { 2 } else { 15 };
    tokio::spawn(async move {
        // Wait a bit before first refresh (we just discovered markets)
        tokio::time::sleep(Duration::from_secs(discovery_sleep)).await;

        loop {
            debug!("[DISCOVER] Checking for new markets...");
            match discover_all_markets(market_filter.as_deref()).await {
                Ok(mut discovered) => {
                    // Filter by asset symbol if specified
                    if let Some(ref sym) = discovery_filter {
                        let sym_upper = sym.to_uppercase();
                        discovered.retain(|m| m.asset == sym_upper);
                    }

                    let mut s = state_for_discovery.write().await;
                    let mut new_count = 0;
                    let mut expired_count = 0;

                    // Remove expired markets (expiry <= 0)
                    let expired_ids: Vec<String> = s.markets.iter()
                        .filter(|(_, m)| m.time_remaining_mins().unwrap_or(0.0) <= 0.0)
                        .map(|(id, _)| id.clone())
                        .collect();

                    for id in expired_ids {
                        // RL Training: Store terminal experience for expired markets
                        if s.rl_enabled && s.rl_train {
                            // Clone what we need before mutating (avoid borrow issues)
                            let pos_opt = s.positions.get(&id).cloned();
                            let market_opt = s.markets.get(&id).cloned();
                            let prices = s.prices.clone();
                            let has_prev = s.prev_observations.contains_key(&id);

                            if let (Some(pos), Some(market)) = (pos_opt, market_opt) {
                                let now_ts = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|d| d.as_secs() as i64)
                                    .unwrap_or(0);
                                if has_prev && (pos.yes_qty > 0.0 || pos.no_qty > 0.0) {
                                    let mut terminal_pnl = 0.0;

                                    // 1. Matched pairs: settle at $1 (guaranteed profit)
                                    let matched = pos.yes_qty.min(pos.no_qty);
                                    if matched > 0.0 {
                                        let avg_yes_price = if pos.yes_qty > 0.0 { pos.yes_cost / pos.yes_qty } else { 0.0 };
                                        let avg_no_price = if pos.no_qty > 0.0 { pos.no_cost / pos.no_qty } else { 0.0 };
                                        let matched_cost = matched * (avg_yes_price + avg_no_price);
                                        terminal_pnl += matched - matched_cost;
                                    }

                                    // Unmatched YES
                                    let unmatched_yes = pos.yes_qty - matched;
                                    let unmatched_no = pos.no_qty - matched;
                                    let mut resolved_prices: Option<(f64, f64)> = None;
                                    if unmatched_yes > 0.0 || unmatched_no > 0.0 {
                                        if let Some(event_slug) = market.event_slug.clone() {
                                            let condition_id = market.condition_id.clone();
                                            match fetch_resolved_outcome_prices(&event_slug, &condition_id).await {
                                                Ok(prices) => {
                                                    resolved_prices = prices;
                                                }
                                                Err(e) => {
                                                    warn!("[RL] Resolution fetch failed for {}: {}", market.asset, e);
                                                }
                                            }
                                        }
                                    }
                                    // Use resolved outcome prices when available; otherwise mark-to-market.
                                    let (exit_yes_price, exit_no_price, used_resolution) = if let Some((yes_price, no_price)) = resolved_prices {
                                        (yes_price, no_price, true)
                                    } else {
                                        let mid_price = market.mid_price().unwrap_or(0.5);
                                        (mid_price, 1.0 - mid_price, false)
                                    };

                                    // 2. Unmatched inventory: Phase 4 share-based PnL
                                    if unmatched_yes > 0.0 {
                                        let entry_price = if pos.yes_qty > 0.0 { pos.yes_cost / pos.yes_qty } else { 0.0 };
                                        // Phase 4: shares = dollars / entry_price
                                        let yes_pnl = compute_share_reward(entry_price, exit_yes_price, unmatched_yes * entry_price);
                                        terminal_pnl += yes_pnl;
                                    }

                                    // Unmatched NO
                                    if unmatched_no > 0.0 {
                                        let entry_price = if pos.no_qty > 0.0 { pos.no_cost / pos.no_qty } else { 0.0 };
                                        // Phase 4: NO profits when price goes DOWN
                                        let no_pnl = compute_share_reward(entry_price, exit_no_price, unmatched_no * entry_price);
                                        terminal_pnl += no_pnl;
                                    }

                                    // Store terminal experience with done=true
                                    let obs = build_observation(&prices, &market, Some(&pos), now_ts, 10.0);
                                    let history = s.push_and_get_history(&id, &obs);
                                    s.store_experience(&id, terminal_pnl, obs, history, true);
                                    info!("[RL] Terminal experience for {}: matched={:.2} unmatched_yes={:.2} unmatched_no={:.2} pnl={:.4} resolved={}",
                                          market.asset, matched, unmatched_yes, unmatched_no, terminal_pnl, used_resolution);
                                }
                            }
                            // Clean up RL state for this market
                            s.prev_observations.remove(&id);
                            s.prev_actions.remove(&id);
                            s.pending_rewards.remove(&id);
                            s.observation_history.remove(&id);
                        }

                        s.markets.remove(&id);
                        s.positions.remove(&id);
                        s.orders.remove(&id);
                        expired_count += 1;
                    }

                    // Add new markets
                    for pm in discovered {
                        let id = pm.condition_id.clone();
                        if !s.markets.contains_key(&id) {
                            info!("[DISCOVER] 🆕 New market: {} | {} | {:.1?}min",
                                  pm.asset, &pm.question[..pm.question.len().min(40)], pm.expiry_minutes);
                            s.positions.insert(id.clone(), Position::default());
                            s.orders.insert(id.clone(), Orders::default());
                            s.markets.insert(id, Market::from_polymarket(pm));
                            new_count += 1;
                        }
                    }

                    if new_count > 0 || expired_count > 0 {
                        info!("[DISCOVER] Added {} new markets, removed {} expired", new_count, expired_count);
                        if new_count > 0 {
                            s.needs_resubscribe = true;
                        }
                    }
                }
                Err(e) => {
                    warn!("[DISCOVER] Market discovery failed: {}", e);
                }
            }

            // Check periodically for new markets
            tokio::time::sleep(Duration::from_secs(discovery_sleep)).await;
        }
    });

    // Main WebSocket loop
    let bid_price = args.bid;
    let contracts = args.contracts;
    let atm_threshold = args.atm_threshold;
    let dry_run = !args.live;
    let rl_paper = !args.live;  // RL respects --live flag for real trading
    let min_minutes = args.min_minutes as f64;
    let max_minutes = args.max_minutes as f64;
    let max_dollars = args.max_dollars;

    // Clone trainer for WebSocket loop (used for training when buffer is full)
    let trainer_for_loop = ppo_trainer.clone();
    let rl_model_path = args.rl_model_path.clone();

    // Clone metrics collector for recording actions/rewards in the loop
    let metrics_for_loop = rl_metrics.clone();

    // Bonds mode variables
    let bonds = args.bonds;
    let bond_minutes = args.bond_minutes;
    let bond_min_distance = args.bond_min_distance;
    let bond_max_price = args.bond_max_price;
    let bond_min_size = args.bond_min_size;
    let bonds_log = args.bonds_log.clone();
    let bond_price_stale_secs = args.bond_price_stale_secs;

    loop {
        // Get current token IDs from state (may have new markets)
        let tokens: Vec<String> = {
            let mut s = state.write().await;
            s.needs_resubscribe = false; // Clear flag
            s.markets.values()
                .flat_map(|m| vec![m.yes_token.clone(), m.no_token.clone()])
                .collect()
        };

        if tokens.is_empty() {
            info!("[WS] No markets to subscribe to, waiting...");
            tokio::time::sleep(Duration::from_secs(5)).await;
            continue;
        }

        info!("[WS] Connecting to Polymarket...");

        let (ws, _) = match connect_async(POLYMARKET_WS_URL).await {
            Ok(ws) => ws,
            Err(e) => {
                error!("[WS] Connect failed: {}", e);
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        let (mut write, mut read) = ws.split();

        // Subscribe
        let sub = SubscribeCmd {
            assets_ids: tokens.clone(),
            sub_type: "market",
        };
        let _ = write.send(Message::Text(serde_json::to_string(&sub)?)).await;
        info!("[WS] Subscribed to {} tokens ({} markets)", tokens.len(), tokens.len() / 2);

        let mut ping_interval = tokio::time::interval(Duration::from_secs(30));
        let status_interval_ms = if args.rl_mode { 500 } else { 1000 };
        let mut status_interval = tokio::time::interval(Duration::from_millis(status_interval_ms));
        let mut resub_check_interval = tokio::time::interval(Duration::from_secs(2));

        loop {
            tokio::select! {
                _ = resub_check_interval.tick() => {
                    // Check if we need to resubscribe for new markets
                    let needs_resub = {
                        let s = state.read().await;
                        s.needs_resubscribe
                    };
                    if needs_resub {
                        info!("[WS] New markets discovered, reconnecting to subscribe...");
                        break;
                    }
                }

                _ = ping_interval.tick() => {
                    if let Err(e) = write.send(Message::Ping(vec![])).await {
                        error!("[WS] Ping failed: {}", e);
                        break;
                    }
                }

                _ = status_interval.tick() => {
                    let now_ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);
                    {
                        let mut s = state.write().await;
                        // Use Binance futures prices (via local price feed)
                        let btc = s.prices.btc_price;
                        let eth = s.prices.eth_price;
                        let sol = s.prices.sol_price;
                        let xrp = s.prices.xrp_price;

                        for market in s.markets.values_mut() {
                            if market.strike_price.is_some() {
                                continue;
                            }
                            let Some(start_ts) = market.window_start_ts else { continue };
                            if now_ts < start_ts {
                                continue;
                            }
                            let price_opt = match market.asset.as_str() {
                                "BTC" => btc,
                                "ETH" => eth,
                                "SOL" => sol,
                                "XRP" => xrp,
                                _ => None,
                            };
                            let Some(price) = price_opt else { continue };
                            let delay_secs = now_ts - start_ts;
                            market.strike_price = Some(price);
                            market.strike_capture_delay_secs = Some(delay_secs);
                            if delay_secs > BOND_MAX_STRIKE_DELAY_SECS {
                                warn!("[STRIKE] ⚠️ {} captured (Binance): ${:.2} ({}s LATE)",
                                      market.asset, price, delay_secs);
                            } else {
                                info!("[STRIKE] {} captured (Binance): ${:.2} ({}s after window start)",
                                      market.asset, price, delay_secs);
                            }
                        }
                    }

                    let mut s = state.write().await;

                    // Build price string based on which assets we're trading
                    let price_str = if let Some(ref sym) = args.sym {
                        match sym.to_uppercase().as_str() {
                            "BTC" => format!("BTC=${}", s.prices.btc_price.map(|p| format!("{:.2}", p)).unwrap_or("-".into())),
                            "ETH" => format!("ETH=${}", s.prices.eth_price.map(|p| format!("{:.2}", p)).unwrap_or("-".into())),
                            "SOL" => format!("SOL=${}", s.prices.sol_price.map(|p| format!("{:.2}", p)).unwrap_or("-".into())),
                            "XRP" => format!("XRP=${}", s.prices.xrp_price.map(|p| format!("{:.4}", p)).unwrap_or("-".into())),
                            _ => "?".into(),
                        }
                    } else {
                        format!("BTC=${} ETH={}",
                            s.prices.btc_price.map(|p| format!("{:.2}", p)).unwrap_or("-".into()),
                            s.prices.eth_price.map(|p| format!("{:.2}", p)).unwrap_or("-".into()))
                    };

                    // Calculate total position and unrealized P&L
                    let total_yes: f64 = s.positions.values().map(|p| p.yes_qty).sum();
                    let total_no: f64 = s.positions.values().map(|p| p.no_qty).sum();
                    let total_cost: f64 = s.positions.values().map(|p| p.yes_cost + p.no_cost).sum();
                    let matched = total_yes.min(total_no);

                    // Calculate mark-to-market value using current bid prices
                    let mut mtm_value = matched; // Matched pairs are worth $1 guaranteed
                    for (market_id, pos) in &s.positions {
                        if let Some(market) = s.markets.get(market_id) {
                            // Unmatched positions valued at current bid (what we could sell for)
                            let unmatched_yes = (pos.yes_qty - matched.min(pos.yes_qty)).max(0.0);
                            let unmatched_no = (pos.no_qty - matched.min(pos.no_qty)).max(0.0);

                            let yes_bid_price = market.yes_bid.unwrap_or(50) as f64 / 100.0;
                            let no_bid_price = market.no_bid.unwrap_or(50) as f64 / 100.0;

                            mtm_value += unmatched_yes * yes_bid_price;
                            mtm_value += unmatched_no * no_bid_price;
                        }
                    }
                    let unrealized_pnl = mtm_value - total_cost;

                    let mode_str = if dry_run { "🔍 DRY" } else { "🚀 LIVE" };

                    // ===== BONDS MODE =====
                    if bonds {
                        // Bonds-specific status display
                        let bonds_qty: f64 = s.positions.values().map(|p| p.bonds_qty).sum();
                        let bonds_cost: f64 = s.positions.values().map(|p| p.bonds_cost).sum();
                        let expected_payout = bonds_qty;
                        let expected_profit = expected_payout - bonds_cost;

                        // Use Binance futures prices for bonds mode display
                        let price_str_bonds = if let Some(ref sym) = args.sym {
                            match sym.to_uppercase().as_str() {
                                "BTC" => format!("BN BTC=${}", s.prices.btc_price.map(|p| format!("{:.2}", p)).unwrap_or("-".into())),
                                "ETH" => format!("BN ETH=${}", s.prices.eth_price.map(|p| format!("{:.2}", p)).unwrap_or("-".into())),
                                "SOL" => format!("BN SOL=${}", s.prices.sol_price.map(|p| format!("{:.2}", p)).unwrap_or("-".into())),
                                "XRP" => format!("BN XRP=${}", s.prices.xrp_price.map(|p| format!("{:.4}", p)).unwrap_or("-".into())),
                                _ => "?".into(),
                            }
                        } else {
                            format!(
                                "BN BTC=${} ETH=${}",
                                s.prices.btc_price.map(|p| format!("{:.2}", p)).unwrap_or("-".into()),
                                s.prices.eth_price.map(|p| format!("{:.2}", p)).unwrap_or("-".into()),
                            )
                        };

                        info!("[{}] 💎 BONDS | {} | Holdings: {} | Cost: ${:.2} | Payout: ${:.2} | Profit: ${:+.2}",
                              mode_str, price_str_bonds, bonds_qty, bonds_cost, expected_payout, expected_profit);

                        // Bonds trading logic
                        for (id, market) in &s.markets {
                            let expiry = market.time_remaining_mins().unwrap_or(0.0);

                            // Time filter: must be in final bond_minutes
                            if expiry > bond_minutes as f64 || expiry <= 0.0 {
                                continue;
                            }

                            let ts_ms = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis() as i64)
                                .unwrap_or(0);

                            let yes_ask = market.yes_ask;
                            let no_ask = market.no_ask;
                            let yes_ask_size = market.yes_ask_size;
                            let no_ask_size = market.no_ask_size;

                            let mut decision = "skip";
                            let mut reason = "ok";
                            let mut ask_price: Option<i64> = None;
                            let mut side: Option<&'static str> = None;
                            let mut contracts: Option<f64> = None;
                            let mut remaining: Option<f64> = None;
                            let mut total_spent: Option<f64> = None;

                            // Get Binance futures spot price (via local price feed)
                            let spot = match market.asset.as_str() {
                                "BTC" => s.prices.btc_price,
                                "ETH" => s.prices.eth_price,
                                "SOL" => s.prices.sol_price,
                                "XRP" => s.prices.xrp_price,
                                _ => None,
                            };
                            let Some(spot_val) = spot else {
                                reason = "no_price";
                                let row = vec![
                                    ts_ms.to_string(),
                                    id.to_string(),
                                    market.asset.clone(),
                                    format!("{:.3}", expiry),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                    fmt_opt_i64(yes_ask),
                                    format!("{:.4}", yes_ask_size),
                                    fmt_opt_i64(no_ask),
                                    format!("{:.4}", no_ask_size),
                                    String::new(),
                                    bond_max_price.to_string(),
                                    format!("{:.4}", bond_min_size),
                                    decision.to_string(),
                                    reason.to_string(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                ]
                                .join(",");
                                append_bonds_log(&bonds_log, &row);
                                continue;
                            };

                            // Skip if price feed is stale (avoid wrong-side fills near expiry)
                            let stale = s.prices.last_update
                                .map(|t| t.elapsed() > Duration::from_secs(bond_price_stale_secs))
                                .unwrap_or(true);
                            if stale {
                                reason = "price_stale";
                                let row = vec![
                                    ts_ms.to_string(),
                                    id.to_string(),
                                    market.asset.clone(),
                                    format!("{:.3}", expiry),
                                    format!("{:.4}", spot_val),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                    fmt_opt_i64(yes_ask),
                                    format!("{:.4}", yes_ask_size),
                                    fmt_opt_i64(no_ask),
                                    format!("{:.4}", no_ask_size),
                                    String::new(),
                                    bond_max_price.to_string(),
                                    format!("{:.4}", bond_min_size),
                                    decision.to_string(),
                                    reason.to_string(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                ]
                                .join(",");
                                append_bonds_log(&bonds_log, &row);
                                continue;
                            }

                            let strike = market.strike_price;
                            if strike.is_none() {
                                reason = "no_strike";
                                let row = vec![
                                    ts_ms.to_string(),
                                    id.to_string(),
                                    market.asset.clone(),
                                    format!("{:.3}", expiry),
                                    format!("{:.4}", spot_val),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                    fmt_opt_i64(yes_ask),
                                    format!("{:.4}", yes_ask_size),
                                    fmt_opt_i64(no_ask),
                                    format!("{:.4}", no_ask_size),
                                    String::new(),
                                    bond_max_price.to_string(),
                                    format!("{:.4}", bond_min_size),
                                    decision.to_string(),
                                    reason.to_string(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                ]
                                .join(",");
                                append_bonds_log(&bonds_log, &row);
                                continue;
                            }
                            let strike = strike.unwrap();

                            // Skip if strike capture was late
                            let strike_delay = market.strike_capture_delay_secs;
                            if let Some(delay) = strike_delay {
                                if delay > BOND_MAX_STRIKE_DELAY_SECS {
                                    reason = "strike_late";
                                    let row = vec![
                                        ts_ms.to_string(),
                                        id.to_string(),
                                        market.asset.clone(),
                                        format!("{:.3}", expiry),
                                        format!("{:.4}", spot_val),
                                        format!("{:.4}", strike),
                                        fmt_opt_i64(strike_delay),
                                        String::new(),
                                        String::new(),
                                        fmt_opt_i64(yes_ask),
                                        format!("{:.4}", yes_ask_size),
                                        fmt_opt_i64(no_ask),
                                        format!("{:.4}", no_ask_size),
                                        String::new(),
                                        bond_max_price.to_string(),
                                        format!("{:.4}", bond_min_size),
                                        decision.to_string(),
                                        reason.to_string(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                    ]
                                    .join(",");
                                    append_bonds_log(&bonds_log, &row);
                                    continue;
                                }
                            }

                            // Adapt distance threshold as expiry approaches
                            let time_frac = (expiry / bond_minutes as f64).clamp(0.2, 1.0);
                            let effective_min_distance = bond_min_distance * time_frac;

                            let dist = distance_from_strike_pct(spot_val, strike);
                            let winner = if spot_val >= strike { "YES" } else { "NO" };
                            let (winner_ask, winner_size, token) = if winner == "YES" {
                                (market.yes_ask.unwrap_or(100), market.yes_ask_size, market.yes_token.clone())
                            } else {
                                (market.no_ask.unwrap_or(100), market.no_ask_size, market.no_token.clone())
                            };

                            ask_price = Some(winner_ask);
                            side = Some(winner);

                            // Log market status
                            let strike_str = format!("${:.0}", strike);
                            info!("  💎 [{}] {:.1}m | spot=${:.2} strike={} | dist={:.2}% | {} ask={}¢",
                                  market.asset, expiry, spot_val, strike_str, dist, winner, winner_ask);

                            if dist < effective_min_distance {
                                reason = "dist";
                                let row = vec![
                                    ts_ms.to_string(),
                                    id.to_string(),
                                    market.asset.clone(),
                                    format!("{:.3}", expiry),
                                    format!("{:.4}", spot_val),
                                    format!("{:.4}", strike),
                                    fmt_opt_i64(strike_delay),
                                    format!("{:.6}", dist),
                                    winner.to_string(),
                                    fmt_opt_i64(yes_ask),
                                    format!("{:.4}", yes_ask_size),
                                    fmt_opt_i64(no_ask),
                                    format!("{:.4}", no_ask_size),
                                    format!("{:.6}", effective_min_distance),
                                    bond_max_price.to_string(),
                                    format!("{:.4}", bond_min_size),
                                    decision.to_string(),
                                    reason.to_string(),
                                    fmt_opt_i64(ask_price),
                                    side.unwrap_or("").to_string(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                ]
                                .join(",");
                                append_bonds_log(&bonds_log, &row);
                                continue;
                            }

                            if winner_ask > bond_max_price {
                                reason = "price";
                                let row = vec![
                                    ts_ms.to_string(),
                                    id.to_string(),
                                    market.asset.clone(),
                                    format!("{:.3}", expiry),
                                    format!("{:.4}", spot_val),
                                    format!("{:.4}", strike),
                                    fmt_opt_i64(strike_delay),
                                    format!("{:.6}", dist),
                                    winner.to_string(),
                                    fmt_opt_i64(yes_ask),
                                    format!("{:.4}", yes_ask_size),
                                    fmt_opt_i64(no_ask),
                                    format!("{:.4}", no_ask_size),
                                    format!("{:.6}", effective_min_distance),
                                    bond_max_price.to_string(),
                                    format!("{:.4}", bond_min_size),
                                    decision.to_string(),
                                    reason.to_string(),
                                    fmt_opt_i64(ask_price),
                                    side.unwrap_or("").to_string(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                ]
                                .join(",");
                                append_bonds_log(&bonds_log, &row);
                                continue;
                            }

                            if winner_size < bond_min_size {
                                reason = "size";
                                let row = vec![
                                    ts_ms.to_string(),
                                    id.to_string(),
                                    market.asset.clone(),
                                    format!("{:.3}", expiry),
                                    format!("{:.4}", spot_val),
                                    format!("{:.4}", strike),
                                    fmt_opt_i64(strike_delay),
                                    format!("{:.6}", dist),
                                    winner.to_string(),
                                    fmt_opt_i64(yes_ask),
                                    format!("{:.4}", yes_ask_size),
                                    fmt_opt_i64(no_ask),
                                    format!("{:.4}", no_ask_size),
                                    format!("{:.6}", effective_min_distance),
                                    bond_max_price.to_string(),
                                    format!("{:.4}", bond_min_size),
                                    decision.to_string(),
                                    reason.to_string(),
                                    fmt_opt_i64(ask_price),
                                    side.unwrap_or("").to_string(),
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                ]
                                .join(",");
                                append_bonds_log(&bonds_log, &row);
                                continue;
                            }

                            // Allow multiple entries per market until max_dollars is hit

                            // Budget check
                            let total_spent_val: f64 = s.positions.values().map(|p| p.bonds_cost).sum();
                            total_spent = Some(total_spent_val);
                            if total_spent_val >= max_dollars {
                                reason = "budget";
                                let row = vec![
                                    ts_ms.to_string(),
                                    id.to_string(),
                                    market.asset.clone(),
                                    format!("{:.3}", expiry),
                                    format!("{:.4}", spot_val),
                                    format!("{:.4}", strike),
                                    fmt_opt_i64(strike_delay),
                                    format!("{:.6}", dist),
                                    winner.to_string(),
                                    fmt_opt_i64(yes_ask),
                                    format!("{:.4}", yes_ask_size),
                                    fmt_opt_i64(no_ask),
                                    format!("{:.4}", no_ask_size),
                                    format!("{:.6}", effective_min_distance),
                                    bond_max_price.to_string(),
                                    format!("{:.4}", bond_min_size),
                                    decision.to_string(),
                                    reason.to_string(),
                                    fmt_opt_i64(ask_price),
                                    side.unwrap_or("").to_string(),
                                    String::new(),
                                    String::new(),
                                    fmt_opt_f64(total_spent, 4),
                                ]
                                .join(",");
                                append_bonds_log(&bonds_log, &row);
                                info!("    ⏭️ Budget exhausted");
                                continue;
                            }

                            // Calculate order - must meet $1 minimum
                            let price = winner_ask as f64 / 100.0;
                            let remaining_val = max_dollars - total_spent_val;
                            remaining = Some(remaining_val);

                            // Minimum contracts needed for $1 order
                            let min_contracts = (1.0 / price).ceil();
                            let min_cost = min_contracts * price;

                            // Skip if we can't afford the minimum order
                            if remaining_val < min_cost {
                                reason = "min_order_budget";
                                let row = vec![
                                    ts_ms.to_string(),
                                    id.to_string(),
                                    market.asset.clone(),
                                    format!("{:.3}", expiry),
                                    format!("{:.4}", spot_val),
                                    format!("{:.4}", strike),
                                    fmt_opt_i64(strike_delay),
                                    format!("{:.6}", dist),
                                    winner.to_string(),
                                    fmt_opt_i64(yes_ask),
                                    format!("{:.4}", yes_ask_size),
                                    fmt_opt_i64(no_ask),
                                    format!("{:.4}", no_ask_size),
                                    format!("{:.6}", effective_min_distance),
                                    bond_max_price.to_string(),
                                    format!("{:.4}", bond_min_size),
                                    decision.to_string(),
                                    reason.to_string(),
                                    fmt_opt_i64(ask_price),
                                    side.unwrap_or("").to_string(),
                                    String::new(),
                                    fmt_opt_f64(remaining, 4),
                                    fmt_opt_f64(total_spent, 4),
                                ]
                                .join(",");
                                append_bonds_log(&bonds_log, &row);
                                info!("    ⏭️ Need ${:.2} for min order, only ${:.2} remaining", min_cost, remaining_val);
                                continue;
                            }

                            // Size proportional to edge and distance
                            let depth = winner_size;
                            let depth_contracts = depth.floor();
                            if depth_contracts < min_contracts {
                                reason = "depth_min";
                                let row = vec![
                                    ts_ms.to_string(),
                                    id.to_string(),
                                    market.asset.clone(),
                                    format!("{:.3}", expiry),
                                    format!("{:.4}", spot_val),
                                    format!("{:.4}", strike),
                                    fmt_opt_i64(strike_delay),
                                    format!("{:.6}", dist),
                                    winner.to_string(),
                                    fmt_opt_i64(yes_ask),
                                    format!("{:.4}", yes_ask_size),
                                    fmt_opt_i64(no_ask),
                                    format!("{:.4}", no_ask_size),
                                    format!("{:.6}", effective_min_distance),
                                    bond_max_price.to_string(),
                                    format!("{:.4}", bond_min_size),
                                    decision.to_string(),
                                    reason.to_string(),
                                    fmt_opt_i64(ask_price),
                                    side.unwrap_or("").to_string(),
                                    String::new(),
                                    fmt_opt_f64(remaining, 4),
                                    fmt_opt_f64(total_spent, 4),
                                ]
                                .join(",");
                                append_bonds_log(&bonds_log, &row);
                                info!("    ⏭️ Top-of-book size {:.2} < min contracts {:.0}", depth, min_contracts);
                                continue;
                            }

                            let max_contracts = (remaining_val / price).floor().min(depth_contracts);
                            let edge_ratio = (100 - winner_ask) as f64 / 100.0;
                            let dist_ratio = if effective_min_distance > 0.0 {
                                (dist / effective_min_distance).clamp(1.0, 3.0)
                            } else {
                                1.0
                            };
                            let spend_frac = (edge_ratio * dist_ratio).clamp(0.2, 1.0);
                            let target_contracts = (remaining_val * spend_frac / price).floor();
                            let contracts_val = max_contracts.min(target_contracts).max(min_contracts);
                            contracts = Some(contracts_val);
                            ask_price = Some(winner_ask);
                            side = Some(winner);
                            let profit_per = 100 - winner_ask;
                            let total_profit = contracts_val * (1.0 - price);

                            decision = if dry_run { "would_buy" } else { "attempt_buy" };
                            let row = vec![
                                ts_ms.to_string(),
                                id.to_string(),
                                market.asset.clone(),
                                format!("{:.3}", expiry),
                                format!("{:.4}", spot_val),
                                format!("{:.4}", strike),
                                fmt_opt_i64(strike_delay),
                                format!("{:.6}", dist),
                                winner.to_string(),
                                fmt_opt_i64(yes_ask),
                                format!("{:.4}", yes_ask_size),
                                fmt_opt_i64(no_ask),
                                format!("{:.4}", no_ask_size),
                                format!("{:.6}", effective_min_distance),
                                bond_max_price.to_string(),
                                format!("{:.4}", bond_min_size),
                                decision.to_string(),
                                "ok".to_string(),
                                fmt_opt_i64(ask_price),
                                side.unwrap_or("").to_string(),
                                fmt_opt_f64(contracts, 4),
                                fmt_opt_f64(remaining, 4),
                                fmt_opt_f64(total_spent, 4),
                            ]
                            .join(",");
                            append_bonds_log(&bonds_log, &row);

                            warn!("  💎💎💎 [BONDS] {} {} | {:.1}m | dist={:.2}% | ask={}¢ | profit={}¢/contract | ${:.2} total 💎💎💎",
                                  market.asset, winner, expiry, dist, winner_ask, profit_per, total_profit);

                            if dry_run {
                                warn!("  💎 Would BUY {:.0} {} @{}¢ (${:.2}) | expected profit ${:.2}",
                                      contracts_val, winner, winner_ask, contracts_val * price, total_profit);
                            } else {
                                // Execute FAK order at ask
                                let market_id = id.clone();
                                drop(s);

                                warn!("  💎 BUY {:.0} {} @{}¢ (${:.2}) | target profit ${:.2}",
                                      contracts_val, winner, winner_ask, contracts_val * price, total_profit);

                                match shared_client.buy_fak(&token, price, contracts_val).await {
                                    Ok(fill) => {
                                        if fill.filled_size > 0.0 {
                                            let fill_price_cents = (fill.fill_cost / fill.filled_size * 100.0).round() as i64;
                                            let locked_profit = fill.filled_size - fill.fill_cost;
                                            warn!("  💎 ✅ FILLED {:.2} {} @{}¢ | cost=${:.2} | locked profit=${:.2}",
                                                  fill.filled_size, winner, fill_price_cents, fill.fill_cost, locked_profit);

                                            let mut st = state.write().await;
                                            if let Some(p) = st.positions.get_mut(&market_id) {
                                                p.bonds_qty += fill.filled_size;
                                                p.bonds_cost += fill.fill_cost;
                                                p.bonds_side = Some(winner.to_string());
                                            }
                                        }
                                    }
                                    Err(e) => error!("  💎 ❌ Order failed: {}", e),
                                }
                                break; // Only trade one market per tick
                            }
                        }
                    } else if s.rl_enabled {
                    // ===== RL MODE (Phase 5 CMSF parity) =====
                    let mut decisions: Vec<RlDecision> = Vec::new();
                    let market_ids: Vec<String> = s.markets.keys().cloned().collect();
                    for mid in market_ids {
                        let Some(market) = s.markets.get(&mid).cloned() else { continue };
                        let pos = s.positions.get(&mid).cloned().unwrap_or_default();

                        let expiry = market.time_remaining_mins().unwrap_or(0.0);
                        if expiry <= 0.0 {
                            continue;
                        }

                        let obs = build_observation(&s.prices, &market, Some(&pos), now_ts, max_dollars);
                        let history = s.push_and_get_history(&mid, &obs);

                        let (action, log_prob, value, action_probs) = if let Some(ref trainer) = trainer_for_loop {
                            let guard = trainer.lock().unwrap();
                            let result = guard.model.select_action_with_probs(&obs, Some(&history));
                            drop(guard);
                            result
                        } else {
                            (Action::Hold, 0.0, 0.0, [1.0/7.0; 7])
                        };

                        // Record action to metrics collector for dashboard
                        if let Some(ref metrics) = metrics_for_loop {
                            metrics.record_action(action);

                            // Set live inference for dashboard
                            let market_name = format!("{} ({})", market.asset, market.question.chars().take(30).collect::<String>());
                            let inference = LiveInference::new(
                                mid.clone(),
                                market_name,
                                obs.as_slice().to_vec(),
                                action_probs,
                                value,
                            );
                            metrics.set_live_inference(inference);
                        }

                        if s.rl_train {
                            let reward = s.pop_reward(&mid);
                            // Record reward to metrics collector for dashboard
                            if let Some(ref metrics) = metrics_for_loop {
                                metrics.record_reward(reward);
                            }
                            if s.prev_observations.contains_key(&mid) {
                                s.store_experience(&mid, reward, obs.clone(), history.clone(), false);
                            }
                        }

                        s.prev_observations.insert(mid.clone(), obs.clone());
                        s.prev_actions.insert(mid.clone(), (action, log_prob, value, history));

                        let mid_price = market.mid_price().unwrap_or(0.5);      
                        let yes_bid = market.yes_bid.map(|c| c as f64 / 100.0);
                        let yes_ask = market.yes_ask.map(|c| c as f64 / 100.0);
                        let no_bid = market.no_bid.map(|c| c as f64 / 100.0);
                        let no_ask = market.no_ask.map(|c| c as f64 / 100.0);
                        decisions.push(RlDecision {
                            market_id: mid,
                            asset: market.asset.clone(),
                            action,
                            mid_price,
                            yes_bid,
                            yes_ask,
                            no_bid,
                            no_ask,
                            yes_token: market.yes_token.clone(),
                            no_token: market.no_token.clone(),
                        });
                    }

                    // Kick off PPO update when buffer is full
                    let mut training_batch: Option<(Vec<Observation>, Vec<Vec<f32>>, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>)> = None;
                    if s.rl_train && s.experience_buffer.len() >= 256 {
                        let mut next_value = 0.0;
                        if let Some(last_exp) = s.experience_buffer.last() {
                            if !last_exp.done {
                                if let Some(ref trainer) = trainer_for_loop {
                                    let guard = trainer.lock().unwrap();
                                    let next_history = last_exp
                                        .next_history
                                        .clone()
                                        .unwrap_or_else(|| vec![0.0; 90]);
                                    next_value = guard
                                        .model
                                        .get_value_with_history(&last_exp.next_obs, &next_history);
                                }
                            }
                        }

                        let (obs_refs, histories, actions, log_probs, advantages, returns) =
                            s.experience_buffer.get_batch_with_history(0.95, 0.95, next_value);
                        let obs: Vec<Observation> = obs_refs.into_iter().cloned().collect();
                        s.experience_buffer.clear();
                        training_batch = Some((obs, histories, actions, log_probs, advantages, returns));
                    }

                    drop(s);

                    let rl_max_amount = args.size;  // Max trade amount (sized by action multiplier)
                    for decision in decisions {
                        let mid_price = decision.mid_price;
                        let yes_bid = decision.yes_bid;
                        let yes_ask = decision.yes_ask;
                        let no_bid = decision.no_bid;
                        let no_ask = decision.no_ask;
                        if mid_price <= 0.0 || mid_price >= 1.0 {
                            continue;
                        }

                        let yes_price = mid_price;
                        let no_price = 1.0 - mid_price;

                        // Get size multiplier from action (0.33, 0.66, or 1.0)
                        let size_mult = decision.action.size_multiplier();
                        // Calculate absolute spread from bid/ask (LIVE MODE)
                        // spread = ask - bid (e.g. bid=0.45, ask=0.55, spread=0.10)
                        let spread = match (decision.yes_bid, decision.yes_ask) {
                            (Some(bid), Some(ask)) if bid > 0.0 && ask > 0.0 => {
                                (ask - bid).max(0.0)  // Absolute spread in price terms
                            }
                            _ => 0.02, // Default 2 cent spread if data unavailable
                        };

                        match decision.action {
                            Action::Hold => {}
                            a if a.is_buy_yes() => {
                                let mut st = state.write().await;
                                let pos = st.positions.get(&decision.market_id).cloned().unwrap_or_default();
                                drop(st);

                                if pos.no_qty > 0.0 {
                                    // Close NO position (flip)
                                    if !rl_paper && pos.no_qty < 1.0 {
                                        let mut st = state.write().await;
                                        if let Some(p) = st.positions.get_mut(&decision.market_id) {
                                            p.no_qty = 0.0;
                                            p.no_cost = 0.0;
                                        }
                                        warn!(
                                            "[RL] Close NO skipped: position size < 1 ({:.6})",
                                            pos.no_qty
                                        );
                                        continue;
                                    }
                                    let exit_price = if rl_paper {
                                        no_price
                                    } else {
                                        no_bid.unwrap_or(0.0)
                                    };
                                    if exit_price <= 0.0 {
                                        continue;
                                    }
                                    if rl_paper {
                                        let entry_price = pos.no_cost / pos.no_qty;
                                        let pnl = compute_spread_adjusted_reward(entry_price, exit_price, pos.no_cost, spread);
                                        let mut st = state.write().await;
                                        let episode_id = if let Some(p) = st.positions.get_mut(&decision.market_id) {
                                            let eid = p.episode_id.take();
                                            p.no_qty = 0.0;
                                            p.no_cost = 0.0;
                                            eid
                                        } else { None };
                                        st.pending_rewards.insert(decision.market_id.clone(), pnl);
                                        info!("[RL] CLOSE NO {} @${:.3} PnL=${:+.2} (spread={:.0}¢)", decision.asset, exit_price, pnl, spread * 100.0);
                                        // Close episode for dashboard
                                        if let Some(ref metrics) = metrics_for_loop {
                                            if let Some(eid) = episode_id {
                                                metrics.close_episode(eid, exit_price, pnl, pnl);
                                            } else {
                                                metrics.record_trade_outcome(pnl, pnl > 0.0);
                                                metrics.record_asset_pnl(&decision.market_id, pnl);
                                            }
                                        }
                                    } else {
                                        match shared_client.sell_fak(&decision.no_token, exit_price, pos.no_qty).await {
                                            Ok(fill) if fill.filled_size > 0.0 => {
                                                let exit_price = fill.fill_cost / fill.filled_size;
                                                let slip = no_price - exit_price;
                                                info!(
                                                    "[RL] SLIPPAGE SELL NO {} mid=${:.3} fill=${:.3} slip=${:+.4}",
                                                    decision.asset,
                                                    no_price,
                                                    exit_price,
                                                    slip
                                                );
                                                let entry_price = pos.no_cost / pos.no_qty;
                                                let cost_basis = entry_price * fill.filled_size;
                                                let pnl = compute_spread_adjusted_reward(entry_price, exit_price, cost_basis, spread);
                                                let mut st = state.write().await;
                                                if let Some(p) = st.positions.get_mut(&decision.market_id) {
                                                    let remaining = (p.no_qty - fill.filled_size).max(0.0);
                                                    let remaining_cost = (p.no_cost - cost_basis).max(0.0);
                                                    p.no_qty = remaining;
                                                    p.no_cost = remaining_cost;
                                                }
                                                st.pending_rewards.insert(decision.market_id.clone(), pnl);
                                                info!("[RL] CLOSE NO {} @${:.3} PnL=${:+.2} (spread={:.0}¢)", decision.asset, exit_price, pnl, spread * 100.0);
                                                // Record trade outcome for dashboard
                                                if let Some(ref metrics) = metrics_for_loop {
                                                    metrics.record_trade_outcome(pnl, pnl > 0.0);
                                                }
                                            }
                                            Ok(_) => {}
                                            Err(e) => error!("[RL] Close NO failed: {}", e),
                                        }
                                    }
                                } else if pos.yes_qty == 0.0 {
                                    // Open YES position
                                    let entry_price = if rl_paper {
                                        yes_price
                                    } else {
                                        yes_ask.unwrap_or(0.0)
                                    };
                                    if entry_price <= 0.0 {
                                        continue;
                                    }
                                    let trade_dollars = rl_max_amount * size_mult;
                                    let contracts = trade_dollars / entry_price;
                                    if contracts <= 0.0 {
                                        continue;
                                    }
                                    if rl_paper {
                                        let cost = contracts * entry_price;
                                        let mut st = state.write().await;
                                        if let Some(p) = st.positions.get_mut(&decision.market_id) {
                                            // Record episode entry if this is a new position
                                            if p.yes_qty == 0.0 && p.episode_id.is_none() {
                                                if let Some(ref metrics) = metrics_for_loop {
                                                    let market_name = st.markets.get(&decision.market_id)
                                                        .map(|m| format!("{} - {}", m.asset, m.question.chars().take(30).collect::<String>()))
                                                        .unwrap_or_else(|| decision.market_id.clone());
                                                    let episode_id = metrics.record_episode_entry(
                                                        decision.market_id.clone(),
                                                        market_name,
                                                        vec![0.0; 18], // Placeholder - obs not available here
                                                        decision.action,
                                                        [1.0/7.0; 7], // Placeholder - probs not available here
                                                        0.0,
                                                        Some(entry_price),
                                                    );
                                                    p.episode_id = Some(episode_id);
                                                }
                                            }
                                            p.yes_qty += contracts;
                                            p.yes_cost += cost;
                                        }
                                        info!("[RL] OPEN YES {} {:.2} @${:.3} (size={})", decision.asset, contracts, entry_price, decision.action.name());
                                    } else {
                                        match shared_client.buy_fak(&decision.yes_token, entry_price, contracts).await {
                                            Ok(fill) if fill.filled_size > 0.0 => {
                                                let fill_price = fill.fill_cost / fill.filled_size;
                                                let slip = fill_price - yes_price;
                                                info!(
                                                    "[RL] SLIPPAGE BUY YES {} mid=${:.3} fill=${:.3} slip=${:+.4}",
                                                    decision.asset,
                                                    yes_price,
                                                    fill_price,
                                                    slip
                                                );
                                                let mut st = state.write().await;
                                                if let Some(p) = st.positions.get_mut(&decision.market_id) {
                                                    p.yes_qty += fill.filled_size;
                                                    p.yes_cost += fill.fill_cost;
                                                }
                                                info!("[RL] OPEN YES {} {:.2} @${:.3}", decision.asset, fill.filled_size, entry_price);
                                            }
                                            Ok(_) => {}
                                            Err(e) => error!("[RL] Open YES failed: {}", e),
                                        }
                                    }
                                }
                            }
                            a if a.is_buy_no() => {
                                let mut st = state.write().await;
                                let pos = st.positions.get(&decision.market_id).cloned().unwrap_or_default();
                                drop(st);

                                if pos.yes_qty > 0.0 {
                                    // Close YES position (flip)
                                    if !rl_paper && pos.yes_qty < 1.0 {
                                        let mut st = state.write().await;
                                        if let Some(p) = st.positions.get_mut(&decision.market_id) {
                                            p.yes_qty = 0.0;
                                            p.yes_cost = 0.0;
                                        }
                                        warn!(
                                            "[RL] Close YES skipped: position size < 1 ({:.6})",
                                            pos.yes_qty
                                        );
                                        continue;
                                    }
                                    let exit_price = if rl_paper {
                                        yes_price
                                    } else {
                                        yes_bid.unwrap_or(0.0)
                                    };
                                    if exit_price <= 0.0 {
                                        continue;
                                    }
                                    if rl_paper {
                                        let entry_price = pos.yes_cost / pos.yes_qty;
                                        let pnl = compute_spread_adjusted_reward(entry_price, exit_price, pos.yes_cost, spread);
                                        let mut st = state.write().await;
                                        let episode_id = if let Some(p) = st.positions.get_mut(&decision.market_id) {
                                            let eid = p.episode_id.take();
                                            p.yes_qty = 0.0;
                                            p.yes_cost = 0.0;
                                            eid
                                        } else { None };
                                        st.pending_rewards.insert(decision.market_id.clone(), pnl);
                                        info!("[RL] CLOSE YES {} @${:.3} PnL=${:+.2} (spread={:.0}¢)", decision.asset, exit_price, pnl, spread * 100.0);
                                        // Close episode for dashboard (includes record_trade_outcome and asset P&L)
                                        if let Some(ref metrics) = metrics_for_loop {
                                            if let Some(eid) = episode_id {
                                                metrics.close_episode(eid, exit_price, pnl, pnl);
                                            } else {
                                                metrics.record_trade_outcome(pnl, pnl > 0.0);
                                                metrics.record_asset_pnl(&decision.market_id, pnl);
                                            }
                                        }
                                    } else {
                                        match shared_client.sell_fak(&decision.yes_token, exit_price, pos.yes_qty).await {
                                            Ok(fill) if fill.filled_size > 0.0 => {
                                                let exit_price = fill.fill_cost / fill.filled_size;
                                                let slip = yes_price - exit_price;
                                                info!(
                                                    "[RL] SLIPPAGE SELL YES {} mid=${:.3} fill=${:.3} slip=${:+.4}",
                                                    decision.asset,
                                                    yes_price,
                                                    exit_price,
                                                    slip
                                                );
                                                let entry_price = pos.yes_cost / pos.yes_qty;
                                                let cost_basis = entry_price * fill.filled_size;
                                                let pnl = compute_spread_adjusted_reward(entry_price, exit_price, cost_basis, spread);
                                                let mut st = state.write().await;
                                                if let Some(p) = st.positions.get_mut(&decision.market_id) {
                                                    let remaining = (p.yes_qty - fill.filled_size).max(0.0);
                                                    let remaining_cost = (p.yes_cost - cost_basis).max(0.0);
                                                    p.yes_qty = remaining;
                                                    p.yes_cost = remaining_cost;
                                                }
                                                st.pending_rewards.insert(decision.market_id.clone(), pnl);
                                                info!("[RL] CLOSE YES {} @${:.3} PnL=${:+.2} (spread={:.0}¢)", decision.asset, exit_price, pnl, spread * 100.0);
                                                // Record trade outcome for dashboard
                                                if let Some(ref metrics) = metrics_for_loop {
                                                    metrics.record_trade_outcome(pnl, pnl > 0.0);
                                                }
                                            }
                                            Ok(_) => {}
                                            Err(e) => error!("[RL] Close YES failed: {}", e),
                                        }
                                    }
                                } else if pos.no_qty == 0.0 {
                                    // Open NO position
                                    let entry_price = if rl_paper {
                                        no_price
                                    } else {
                                        no_ask.unwrap_or(0.0)
                                    };
                                    if entry_price <= 0.0 {
                                        continue;
                                    }
                                    let trade_dollars = rl_max_amount * size_mult;
                                    let contracts = trade_dollars / entry_price;
                                    if contracts <= 0.0 {
                                        continue;
                                    }
                                    if rl_paper {
                                        let cost = contracts * entry_price;
                                        let mut st = state.write().await;
                                        if let Some(p) = st.positions.get_mut(&decision.market_id) {
                                            // Record episode entry if this is a new position
                                            if p.no_qty == 0.0 && p.episode_id.is_none() {
                                                if let Some(ref metrics) = metrics_for_loop {
                                                    let market_name = st.markets.get(&decision.market_id)
                                                        .map(|m| format!("{} - {}", m.asset, m.question.chars().take(30).collect::<String>()))
                                                        .unwrap_or_else(|| decision.market_id.clone());
                                                    let episode_id = metrics.record_episode_entry(
                                                        decision.market_id.clone(),
                                                        market_name,
                                                        vec![0.0; 18],
                                                        decision.action,
                                                        [1.0/7.0; 7],
                                                        0.0,
                                                        Some(entry_price),
                                                    );
                                                    p.episode_id = Some(episode_id);
                                                }
                                            }
                                            p.no_qty += contracts;
                                            p.no_cost += cost;
                                        }
                                        info!("[RL] OPEN NO {} {:.2} @${:.3} (size={})", decision.asset, contracts, entry_price, decision.action.name());
                                    } else {
                                        match shared_client.buy_fak(&decision.no_token, entry_price, contracts).await {
                                            Ok(fill) if fill.filled_size > 0.0 => {
                                                let fill_price = fill.fill_cost / fill.filled_size;
                                                let slip = fill_price - no_price;
                                                info!(
                                                    "[RL] SLIPPAGE BUY NO {} mid=${:.3} fill=${:.3} slip=${:+.4}",
                                                    decision.asset,
                                                    no_price,
                                                    fill_price,
                                                    slip
                                                );
                                                let mut st = state.write().await;
                                                if let Some(p) = st.positions.get_mut(&decision.market_id) {
                                                    p.no_qty += fill.filled_size;
                                                    p.no_cost += fill.fill_cost;
                                                }
                                                info!("[RL] OPEN NO {} {:.2} @${:.3}", decision.asset, fill.filled_size, entry_price);
                                            }
                                            Ok(_) => {}
                                            Err(e) => error!("[RL] Open NO failed: {}", e),
                                        }
                                    }
                                }
                            }
                            _ => {} // Catchall for exhaustiveness (guards don't guarantee exhaustive match)
                        }
                    }

                    if let Some((obs, histories, actions, log_probs, advantages, returns)) = training_batch {
                        if let Some(ref trainer) = trainer_for_loop {
                            let trainer_lock = trainer.clone();
                            let model_path = rl_model_path.clone();
                            let result = tokio::task::spawn_blocking(move || {
                                let mut guard = trainer_lock.lock().unwrap();
                                let obs_refs: Vec<&Observation> = obs.iter().collect();
                                let (p_loss, v_loss, entropy) = guard.update_with_history(
                                    &obs_refs, &histories, &actions, &log_probs, &advantages, &returns
                                );
                                if let Err(e) = guard.save(&model_path) {
                                    tracing::warn!("[RL] Failed to save model: {}", e);
                                }
                                tracing::info!(
                                    "[RL] PPO Update: policy_loss={:.4}, value_loss={:.4}, entropy={:.4}",
                                    p_loss, v_loss, entropy
                                );
                                (p_loss, v_loss, entropy)
                            }).await;

                            // Record training update to metrics collector for dashboard
                            if let Ok((p_loss, v_loss, entropy)) = result {
                                if let Some(ref metrics) = metrics_for_loop {
                                    metrics.record_update(p_loss, v_loss, entropy);
                                }
                            }
                        }
                    }
                    } else {
                    // ===== ATM/ARB MODE =====
                    info!("[{}] {} | Pos: Y={:.1} N={:.1} matched={:.1} | Cost=${:.2} MTM=${:.2} PnL=${:+.2} | ${:.2}/${:.2}",
                          mode_str, price_str, total_yes, total_no, matched, total_cost, mtm_value, unrealized_pnl, total_cost, max_dollars);

                    for (id, market) in &s.markets {
                        let pos = s.positions.get(id).cloned().unwrap_or_default();
                        let orders = s.orders.get(id).cloned().unwrap_or_default();

                        // Get spot price for this asset
                        let spot = match market.asset.as_str() {
                            "BTC" => s.prices.btc_price,
                            "ETH" => s.prices.eth_price,
                            "SOL" => s.prices.sol_price,
                            "XRP" => s.prices.xrp_price,
                            _ => None,
                        };

                        let expiry = market.time_remaining_mins().unwrap_or(0.0);
                        if expiry < min_minutes || expiry > max_minutes {
                            continue;
                        }

                        // Check ATM status using captured strike price
                        // For Up/Down markets: YES wins if spot >= strike at expiry
                        // ITM = spot > strike (YES favored), OTM = spot < strike (NO favored)
                        let (atm_status, dist_pct, is_atm) = match (spot, market.strike_price) {
                            (Some(s), Some(k)) => {
                                let dist = distance_from_strike_pct(s, k);
                                let is_atm_now = dist <= atm_threshold;
                                let status = if is_atm_now {
                                    "✅ ATM"
                                } else if s > k {
                                    "📈 ITM" // spot > strike, YES (Up) is in-the-money
                                } else {
                                    "📉 OTM" // spot < strike, YES (Up) is out-of-the-money
                                };
                                (status, format!("{:.4}%", dist), is_atm_now)
                            }
                            (_, None) => ("⏳ WAIT", "-".into(), false), // Waiting for strike to be captured
                            _ => ("❓ NO PRICE", "-".into(), false),
                        };

                        let yes_ask = market.yes_ask.unwrap_or(100);
                        let no_ask = market.no_ask.unwrap_or(100);

                        let order_status = format!(
                            "Y:{} N:{}",
                            orders.yes_order_id.as_ref().map(|_| "📝").unwrap_or("-"),
                            orders.no_order_id.as_ref().map(|_| "📝").unwrap_or("-")
                        );

                        let strike_str = market.strike_price
                            .map(|s| format!("${:.0}", s))
                            .unwrap_or_else(|| "?".into());

                        // Calculate combined cost and profit/loss for EVERY market
                        let combined = yes_ask + no_ask;
                        let profit_str = if combined < 100 {
                            format!("✅+{}¢", 100 - combined)
                        } else if combined == 100 {
                            "⚖️0¢".to_string()
                        } else {
                            format!("❌-{}¢", combined - 100)
                        };

                        // Check all entry conditions
                        let total_cost: f64 = s.positions.values().map(|p| p.yes_cost + p.no_cost).sum();
                        let has_capacity = total_cost < max_dollars;
                        let in_time_window = expiry >= min_minutes && expiry <= max_minutes;
                        // Remaining budget in dollars
                        let remaining_dollars = max_dollars - total_cost;
                        // Can we afford at least $1 order?
                        let can_meet_min = remaining_dollars >= 1.0;

                        // Build condition status string
                        let cond_str = format!("ATM:{} Time:{} Cap:{} $1:{}",
                            if is_atm { "✓" } else { "✗" },
                            if in_time_window { "✓" } else { "✗" },
                            if has_capacity { "✓" } else { "✗" },
                            if can_meet_min { "✓" } else { "✗" }
                        );

                        // Determine if ALL conditions met
                        let all_conditions = is_atm && in_time_window && has_capacity && can_meet_min;

                        // Always show combined cost analysis with entry conditions
                        info!("  [{}] {:.1}m | {} dist={} | Y={}¢+N={}¢={}¢ {} | {} | ${:.2}/${:.2}",
                              market.asset,
                              expiry,
                              atm_status, dist_pct,
                              yes_ask, no_ask, combined, profit_str,
                              cond_str,
                              total_cost, max_dollars);

                        // If ALL conditions met, show action
                        if all_conditions {
                            warn!("  🎯 {} ENTRY CONDITIONS MET! BUY BOTH: Y@{}¢ + N@{}¢ = {}¢ {}",
                                  market.asset, yes_ask, no_ask, combined, profit_str);
                        }
                    }

                    // Proactive trading: attempt trades during status tick, not just on orderbook updates
                    // TWO entry conditions:
                    // 1. ATM Entry: When ATM, bid 45¢ on both sides
                    // 2. Arb Entry: When combined asks < 100¢, buy at asks (guaranteed profit)
                    let total_spent: f64 = s.positions.values().map(|p| p.yes_cost + p.no_cost).sum();
                    if total_spent < max_dollars {
                        // Collect opportunities: (market_id, yes_token, no_token, yes_ask, no_ask, asset, is_arb)
                        let trade_ops: Vec<(String, String, String, i64, i64, String, bool)> = s.markets.iter()
                            .filter_map(|(id, m)| {
                                let exp = m.time_remaining_mins().unwrap_or(0.0);
                                if exp < min_minutes || exp > max_minutes { return None; }
                                let spot = match m.asset.as_str() {
                                    "BTC" => s.prices.btc_price, "ETH" => s.prices.eth_price,
                                    "SOL" => s.prices.sol_price, "XRP" => s.prices.xrp_price, _ => None,
                                }?;
                                let (ya, na) = (m.yes_ask.unwrap_or(100), m.no_ask.unwrap_or(100));
                                let combined = ya + na;
                                let ord = s.orders.get(id).cloned().unwrap_or_default();
                                if ord.yes_order_id.is_some() && ord.no_order_id.is_some() { return None; }

                                // Check ATM condition
                                let dist = distance_from_strike_pct(spot, m.strike_price?);
                                let is_atm = dist <= atm_threshold;

                                // Check Arb condition: combined < 100¢
                                let is_arb = combined < 100;

                                // Enter if ATM OR if Arb opportunity
                                if is_atm || is_arb {
                                    Some((id.clone(), m.yes_token.clone(), m.no_token.clone(), ya, na, m.asset.clone(), is_arb))
                                } else { None }
                            }).collect();
                        drop(s);

                        for (mid, ytok, ntok, yask, nask, asset, is_arb) in trade_ops {
                            let s = state.read().await;
                            let ord = s.orders.get(&mid).cloned().unwrap_or_default();
                            let (need_y, need_n) = (ord.yes_order_id.is_none(), ord.no_order_id.is_none());
                            let combined = yask + nask;

                            let ppo_action: Option<(Action, f32, f32, Observation, Option<Vec<f32>>)> = None;

                            drop(s);

                            // If RL mode is active and PPO says Hold (and it's not arb), skip this trade
                            if let Some((Action::Hold, _, _, _, _)) = &ppo_action {
                                debug!("[RL] PPO says HOLD for {} - skipping ATM entry", asset);
                                continue;
                            }

                            let entry_type = match &ppo_action {
                                Some((a, _, _, _, _)) if a.is_buy_yes() => "RL-YES",
                                Some((a, _, _, _, _)) if a.is_buy_no() => "RL-NO",
                                _ if is_arb => "ARB",
                                _ => "ATM",
                            };

                            // For ARB: buy at ask price (guaranteed fill for arb profit)
                            // For ATM/RL: bid at our bid_price
                            let y_price = if is_arb { yask } else { bid_price };
                            let n_price = if is_arb { nask } else { bid_price };

                            if dry_run {
                                if is_arb {
                                    warn!("💰💰💰 [{}] Would BUY {} YES@{}¢ + NO@{}¢ = {}¢ ({}¢ profit) | {} 💰💰💰",
                                          entry_type, contracts, yask, nask, combined, 100 - combined, asset);
                                } else {
                                    if need_y { warn!("🔔🔔🔔 [{}] Would BID {} YES @{}¢ | ask={}¢ | {} 🔔🔔🔔", entry_type, contracts, bid_price, yask, asset); }
                                    if need_n { warn!("🔔🔔🔔 [{}] Would BID {} NO @{}¢ | ask={}¢ | {} 🔔🔔🔔", entry_type, contracts, bid_price, nask, asset); }
                                }
                            } else {
                                // Calculate remaining budget and min contracts for $1 minimum
                                let remaining_budget = max_dollars - total_spent;

                                // Skip if not enough budget for $1 minimum order
                                if remaining_budget < 1.0 {
                                    debug!("[{}] {} need $1 min, only ${:.2} budget", entry_type, asset, remaining_budget);
                                    continue;
                                }

                                // Calculate contracts based on price
                                let max_price = (y_price.max(n_price) as f64) / 100.0;
                                let min_c = (1.0 / max_price).ceil();
                                // How many contracts can we afford with remaining budget?
                                let max_afford = (remaining_budget / max_price).floor();
                                let act_c = min_c.min(max_afford).max(min_c);

                                if is_arb {
                                    // ARB: Buy BOTH at ask prices
                                    let y_pr = yask as f64 / 100.0;
                                    let n_pr = nask as f64 / 100.0;
                                    let total_cost = act_c * (y_pr + n_pr);
                                    let profit = act_c * 1.0 - total_cost;
                                    warn!("💰💰💰 [ARB] 📝 BUY {:.0} YES@{}¢ + NO@{}¢ = ${:.2} (profit ${:.2}) | {} 💰💰💰",
                                          act_c, yask, nask, total_cost, profit, asset);

                                    // Buy YES
                                    if let Ok(f) = shared_client.buy_fak(&ytok, y_pr, act_c).await {
                                        if f.filled_size > 0.0 {
                                            let fp = (f.fill_cost / f.filled_size * 100.0).round() as i64;
                                            warn!("💰 [ARB] ✅ YES Filled {:.2} @{}¢ (${:.2})", f.filled_size, fp, f.fill_cost);
                                            let mut st = state.write().await;
                                            if let Some(p) = st.positions.get_mut(&mid) { p.yes_qty += f.filled_size; p.yes_cost += f.fill_cost; }
                                        }
                                    }
                                    // Buy NO
                                    if let Ok(f) = shared_client.buy_fak(&ntok, n_pr, act_c).await {
                                        if f.filled_size > 0.0 {
                                            let fp = (f.fill_cost / f.filled_size * 100.0).round() as i64;
                                            warn!("💰 [ARB] ✅ NO Filled {:.2} @{}¢ (${:.2})", f.filled_size, fp, f.fill_cost);
                                            let mut st = state.write().await;
                                            if let Some(p) = st.positions.get_mut(&mid) { p.no_qty += f.filled_size; p.no_cost += f.fill_cost; }
                                        }
                                    }
                                } else {
                                    // ATM/RL: Bid at our bid_price
                                    let pr = bid_price as f64 / 100.0;

                                    // In RL mode, only buy the side PPO recommends
                                    let (buy_yes, buy_no) = match &ppo_action {
                                        Some((a, _, _, _, _)) if a.is_buy_yes() => (true, false),
                                        Some((a, _, _, _, _)) if a.is_buy_no() => (false, true),
                                        _ => (need_y, need_n), // Standard ATM: buy both if needed
                                    };

                                    if buy_yes {
                                        warn!("🔔🔔🔔 [{}] 📝 BID {:.0} YES @{}¢ (${:.2}) | ask={}¢ | {} 🔔🔔🔔", entry_type, act_c, bid_price, act_c * pr, yask, asset);
                                        if let Ok(f) = shared_client.buy_fak(&ytok, pr, act_c).await {
                                            if f.filled_size > 0.0 {
                                                let fp = (f.fill_cost / f.filled_size * 100.0).round() as i64;
                                                warn!("🔔 [{}] ✅ YES Filled {:.2} @{}¢ (${:.2})", entry_type, f.filled_size, fp, f.fill_cost);
                                                let mut st = state.write().await;
                                                if let Some(p) = st.positions.get_mut(&mid) { p.yes_qty += f.filled_size; p.yes_cost += f.fill_cost; }
                                                // RL Training: Store previous state and action, compute reward if matched
                                                if let Some((action, log_prob, value, obs, _)) = &ppo_action {
                                                    // Check for newly matched pairs (reward on position close)
                                                    if let Some(p) = st.positions.get(&mid) {
                                                        let prev_matched = (p.yes_qty - f.filled_size).min(p.no_qty).max(0.0);
                                                        let new_matched = p.yes_qty.min(p.no_qty);
                                                        let delta_matched = new_matched - prev_matched;

                                                        if delta_matched > 0.0 {
                                                            // Phase 4: Share-based reward for matched pairs
                                                            // Each matched pair settles at $1, compute profit
                                                            let avg_yes_price = if p.yes_qty > 0.0 { p.yes_cost / p.yes_qty } else { 0.0 };
                                                            let avg_no_price = if p.no_qty > 0.0 { p.no_cost / p.no_qty } else { 0.0 };
                                                            let matched_cost = delta_matched * (avg_yes_price + avg_no_price);
                                                            let matched_value = delta_matched; // $1 per matched pair
                                                            let pnl = matched_value - matched_cost;
                                                            st.pending_rewards.insert(mid.clone(), pnl);
                                                            debug!("[RL] Phase 4 reward: matched={:.2} cost=${:.2} pnl=${:.2}", delta_matched, matched_cost, pnl);
                                                        }
                                                    }

                                                    // Store current state as previous for next experience
                                                    st.prev_observations.insert(mid.clone(), obs.clone());
                                                    st.prev_actions.insert(mid.clone(), (*action, *log_prob, *value, Vec::new()));
                                                }
                                            }
                                        }
                                    }
                                    if buy_no {
                                        warn!("🔔🔔🔔 [{}] 📝 BID {:.0} NO @{}¢ (${:.2}) | ask={}¢ | {} 🔔🔔🔔", entry_type, act_c, bid_price, act_c * pr, nask, asset);
                                        if let Ok(f) = shared_client.buy_fak(&ntok, pr, act_c).await {
                                            if f.filled_size > 0.0 {
                                                let fp = (f.fill_cost / f.filled_size * 100.0).round() as i64;
                                                warn!("🔔 [{}] ✅ NO Filled {:.2} @{}¢ (${:.2})", entry_type, f.filled_size, fp, f.fill_cost);
                                                let mut st = state.write().await;
                                                if let Some(p) = st.positions.get_mut(&mid) { p.no_qty += f.filled_size; p.no_cost += f.fill_cost; }
                                                // RL Training: Store previous state and action, compute reward if matched
                                                if let Some((action, log_prob, value, obs, _)) = &ppo_action {
                                                    // Check for newly matched pairs (reward on position close)
                                                    if let Some(p) = st.positions.get(&mid) {
                                                        let prev_matched = p.yes_qty.min(p.no_qty - f.filled_size).max(0.0);
                                                        let new_matched = p.yes_qty.min(p.no_qty);
                                                        let delta_matched = new_matched - prev_matched;

                                                        if delta_matched > 0.0 {
                                                            // Phase 4: Share-based reward for matched pairs
                                                            let avg_yes_price = if p.yes_qty > 0.0 { p.yes_cost / p.yes_qty } else { 0.0 };
                                                            let avg_no_price = if p.no_qty > 0.0 { p.no_cost / p.no_qty } else { 0.0 };
                                                            let matched_cost = delta_matched * (avg_yes_price + avg_no_price);
                                                            let matched_value = delta_matched; // $1 per matched pair
                                                            let pnl = matched_value - matched_cost;
                                                            st.pending_rewards.insert(mid.clone(), pnl);
                                                            debug!("[RL] Phase 4 reward: matched={:.2} cost=${:.2} pnl=${:.2}", delta_matched, matched_cost, pnl);
                                                        }
                                                    }

                                                    // Store current state as previous for next experience
                                                    st.prev_observations.insert(mid.clone(), obs.clone());
                                                    st.prev_actions.insert(mid.clone(), (*action, *log_prob, *value, Vec::new()));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    } // end else (ATM/ARB mode)
                }

                msg = read.next() => {
                    let Some(msg) = msg else { break; };

                    match msg {
                        Ok(Message::Text(text)) => {

                            // Try to parse as book snapshot array
                            if let Ok(books) = serde_json::from_str::<Vec<BookSnapshot>>(&text) {
                                for book in books {
                                    let mut s = state.write().await;

                                    // Find which market this token belongs to
                                    let market_id = s.markets.iter()
                                        .find(|(_, m)| m.yes_token == book.asset_id || m.no_token == book.asset_id)
                                        .map(|(id, _)| id.clone());

                                    let Some(market_id) = market_id else { continue };

                                    // Find best ask and bid from orderbook
                                    let best_ask = book.asks.iter()
                                        .filter_map(|l| {
                                            let price = parse_price_cents(&l.price);
                                            if price > 0 { Some((price, parse_size(&l.size))) } else { None }
                                        })
                                        .min_by_key(|(p, _)| *p);

                                    let best_bid = book.bids.iter()
                                        .filter_map(|l| {
                                            let price = parse_price_cents(&l.price);
                                            if price > 0 { Some((price, parse_size(&l.size))) } else { None }
                                        })
                                        .max_by_key(|(p, _)| *p);

                                    // Get prices and market data first
                                    let btc_price = s.prices.btc_price;
                                    let eth_price = s.prices.eth_price;
                                    let sol_price = s.prices.sol_price;
                                    let xrp_price = s.prices.xrp_price;
                                    let rl_enabled = s.rl_enabled;

                                    let Some(market) = s.markets.get_mut(&market_id) else { continue };

                                    let is_yes = book.asset_id == market.yes_token;

                                    if is_yes {
                                        market.yes_ask = best_ask.map(|(p, _)| p);
                                        market.yes_bid = best_bid.map(|(p, _)| p);
                                        market.yes_ask_size = best_ask.map(|(_, s)| s).unwrap_or(0.0);

                                        // Parse full orderbook depth for RL features (L5 imbalance, spread, vol_5m)
                                        let bids: Vec<(f64, f64)> = book.bids.iter()
                                            .filter_map(|l| {
                                                let price = l.price.parse::<f64>().ok()?;
                                                let size = l.size.parse::<f64>().ok()?;
                                                if price > 0.0 && size > 0.0 { Some((price, size)) } else { None }
                                            })
                                            .collect();
                                        let asks: Vec<(f64, f64)> = book.asks.iter()
                                            .filter_map(|l| {
                                                let price = l.price.parse::<f64>().ok()?;
                                                let size = l.size.parse::<f64>().ok()?;
                                                if price > 0.0 && size > 0.0 { Some((price, size)) } else { None }
                                            })
                                            .collect();
                                        market.update_orderbook(bids, asks);
                                    } else {
                                        market.no_ask = best_ask.map(|(p, _)| p);
                                        market.no_bid = best_bid.map(|(p, _)| p);
                                        market.no_ask_size = best_ask.map(|(_, s)| s).unwrap_or(0.0);
                                    }

                                    // Skip ATM/ARB logic when RL mode is active
                                    if rl_enabled {
                                        continue;
                                    }

                                    // Extract values for trading logic
                                    let asset = market.asset.clone();
                                    let strike = market.strike_price;
                                    let expiry = market.time_remaining_mins();
                                    let yes_ask_price = market.yes_ask;
                                    let no_ask_price = market.no_ask;
                                    let yes_token = market.yes_token.clone();
                                    let no_token = market.no_token.clone();
                                    let question = market.question.clone();

                                    // Get current orders
                                    let orders = s.orders.get(&market_id).cloned().unwrap_or_default();

                                    // Get spot price based on asset
                                    let spot = match asset.as_str() {
                                        "BTC" => btc_price,
                                        "ETH" => eth_price,
                                        "SOL" => sol_price,
                                        "XRP" => xrp_price,
                                        _ => None,
                                    };

                                    // Skip WebSocket ATM trading when in bonds mode
                                    if bonds {
                                        continue;
                                    }

                                    // Check if within time window
                                    let mins = expiry.unwrap_or(0.0);
                                    if mins < min_minutes || mins > max_minutes {
                                        continue;
                                    }

                                    // Check ATM condition: spot within threshold of strike
                                    let Some(spot_price) = spot else { continue };
                                    let Some(strike_price) = strike else { continue };

                                    let dist_pct = distance_from_strike_pct(spot_price, strike_price);
                                    let is_atm_now = dist_pct <= atm_threshold;

                                    if !is_atm_now {
                                        // Not ATM - don't trade
                                        debug!("[SKIP] {} not ATM: dist={:.4}% > {:.4}%",
                                               asset, dist_pct, atm_threshold);
                                        continue;
                                    }

                                    // ATM! Check if we should bid
                                    let market_id_clone = market_id.clone();
                                    drop(s);

                                    // Check if we've hit max dollars limit
                                    let total_spent = {
                                        let s = state.read().await;
                                        s.positions.values()
                                            .map(|p| p.yes_cost + p.no_cost)
                                            .sum::<f64>()
                                    };

                                    if total_spent >= max_dollars {
                                        debug!("[SKIP] Max dollars reached: ${:.2} >= ${:.2}", total_spent, max_dollars);
                                        continue;
                                    }

                                    // Calculate remaining budget
                                    let remaining_budget = max_dollars - total_spent;

                                    // Skip if not enough budget for $1 minimum order
                                    if remaining_budget < 1.0 {
                                        debug!("[SKIP] {} need $1 min, only ${:.2} budget",
                                               asset, remaining_budget);
                                        continue;
                                    }

                                    // Calculate minimum contracts needed for $1 order
                                    let price = bid_price as f64 / 100.0;
                                    let min_contracts = (1.0 / price).ceil();

                                    // Determine if we need to place orders
                                    let need_yes = orders.yes_order_id.is_none();
                                    let need_no = orders.no_order_id.is_none();

                                    if !need_yes && !need_no {
                                        continue;
                                    }

                                    // Check combined cost - only worth it if YES + NO asks <= 100¢
                                    // But we use FAK at our bid price, so we'll fill at ask or not at all
                                    let yes_ask = yes_ask_price.unwrap_or(100);
                                    let no_ask = no_ask_price.unwrap_or(100);
                                    let combined = yes_ask + no_ask;

                                    let spot_str = format!("{:.2}", spot_price);
                                    let strike_str = format!("{:.0}", strike_price);

                                    info!("════════════════════════════════════════════════════════════════");
                                    info!("[ATM] 🎯 {} is AT-THE-MONEY! Spot=${} Strike=${} dist={:.4}%",
                                          asset, spot_str, strike_str, dist_pct);
                                    info!("[ATM] Market: {}", &question[..question.len().min(60)]);
                                    info!("[ATM] {:.1}m remaining | Ask Y={}¢ N={}¢ | Combined={}¢ {}",
                                          mins, yes_ask, no_ask, combined,
                                          if combined < 100 { "✅ PROFITABLE" } else { "⚠️ no arb" });
                                    info!("════════════════════════════════════════════════════════════════");

                                    if dry_run {
                                        // In dry run, show what we'd do - buy both sides if ATM
                                        if need_yes {
                                            warn!("🔔🔔🔔 [ORDER] Would BUY YES @{}¢ (ask={}¢) | {} 🔔🔔🔔",
                                                  bid_price.min(yes_ask as i64), yes_ask, asset);
                                        }
                                        if need_no {
                                            warn!("🔔🔔🔔 [ORDER] Would BUY NO @{}¢ (ask={}¢) | {} 🔔🔔🔔",
                                                  bid_price.min(no_ask as i64), no_ask, asset);
                                        }
                                    } else {
                                        // Place YES order - use lower of bid or ask as our limit
                                        if need_yes {
                                            let current_ask = yes_ask_price.unwrap_or(0);

                                            // Use minimum contracts needed for $1 order, capped by budget
                                            let max_afford = (remaining_budget / price).floor();
                                            let actual_contracts = min_contracts.min(max_afford).max(min_contracts);
                                            let cost = actual_contracts * price;

                                            warn!("[TRADE] 📝 BID {:.0} contracts YES @{}¢ (${:.2}) | ask={}¢ | delta≈0.50 | {}",
                                                  actual_contracts, bid_price, cost, current_ask, asset);

                                            match shared_client.buy_fak(&yes_token, price, actual_contracts).await {
                                                Ok(fill) => {
                                                    if fill.filled_size > 0.0 {
                                                        let fill_price_cents = (fill.fill_cost / fill.filled_size * 100.0).round() as i64;
                                                        warn!("[TRADE] ✅ YES Filled {:.2} @{}¢ (total ${:.2}) | ask was {}¢ | order_id={}",
                                                              fill.filled_size, fill_price_cents, fill.fill_cost, current_ask, fill.order_id);
                                                        // Update position
                                                        let mut s = state.write().await;
                                                        if let Some(pos) = s.positions.get_mut(&market_id_clone) {
                                                            pos.yes_qty += fill.filled_size;
                                                            pos.yes_cost += fill.fill_cost;
                                                        }
                                                    } else {
                                                        info!("[TRADE] ⏳ YES order placed (no immediate fill)");
                                                    }
                                                }
                                                Err(e) => error!("[TRADE] ❌ YES bid failed: {}", e),
                                            }
                                        }

                                        // Place NO order
                                        if need_no {
                                            let current_ask = no_ask_price.unwrap_or(0);

                                            // Use minimum contracts needed for $1 order, capped by budget
                                            let max_afford = (remaining_budget / price).floor();
                                            let actual_contracts = min_contracts.min(max_afford).max(min_contracts);
                                            let cost = actual_contracts * price;

                                            warn!("[TRADE] 📝 BID {:.0} contracts NO @{}¢ (${:.2}) | ask={}¢ | delta≈0.50 | {}",
                                                  actual_contracts, bid_price, cost, current_ask, asset);

                                            match shared_client.buy_fak(&no_token, price, actual_contracts).await {
                                                Ok(fill) => {
                                                    if fill.filled_size > 0.0 {
                                                        let fill_price_cents = (fill.fill_cost / fill.filled_size * 100.0).round() as i64;
                                                        warn!("[TRADE] ✅ NO Filled {:.2} @{}¢ (total ${:.2}) | ask was {}¢ | order_id={}",
                                                              fill.filled_size, fill_price_cents, fill.fill_cost, current_ask, fill.order_id);
                                                        // Update position
                                                        let mut s = state.write().await;
                                                        if let Some(pos) = s.positions.get_mut(&market_id_clone) {
                                                            pos.no_qty += fill.filled_size;
                                                            pos.no_cost += fill.fill_cost;
                                                        }
                                                    } else {
                                                        info!("[TRADE] ⏳ NO order placed (no immediate fill)");
                                                    }
                                                }
                                                Err(e) => error!("[TRADE] ❌ NO bid failed: {}", e),
                                            }
                                        }
                                    }
                                }
                            }
                            // Handle price change messages (real-time updates) - ALSO triggers trades!
                            else if let Ok(price_msg) = serde_json::from_str::<PriceChangeMessage>(&text) {
                                if !price_msg.price_changes.is_empty() {
                                    // Collect ATM/ARB opportunities after updating prices
                                    // (market_id, yes_token, no_token, yes_ask, no_ask, asset, is_arb, dist)
                                    let mut atm_trades: Vec<(String, String, String, i64, i64, String, bool, f64)> = Vec::new();

                                    {
                                        let mut s = state.write().await;
                                        let total_spent: f64 = s.positions.values().map(|p| p.yes_cost + p.no_cost).sum();

                                        for pc in &price_msg.price_changes {
                                            // Find and update market
                                            let market_info = s.markets.iter_mut()
                                                .find(|(_, m)| m.yes_token == pc.asset_id || m.no_token == pc.asset_id)
                                                .map(|(id, m)| {
                                                    let is_yes = pc.asset_id == m.yes_token;
                                                    let price_cents = pc.price.parse::<f64>()
                                                        .map(|p| (p * 100.0).round() as i64)
                                                        .unwrap_or(0);

                                                    // Only update bid/ask from price changes if we don't have book data yet
                                                    // Book snapshots give real bid/ask; price changes are last trade prices
                                                    if is_yes {
                                                        if m.yes_bid.is_none() {
                                                            m.yes_bid = Some(price_cents.saturating_sub(1).max(1));
                                                        }
                                                        if m.yes_ask.is_none() {
                                                            m.yes_ask = Some((price_cents + 1).min(99));
                                                        }
                                                    } else {
                                                        if m.no_bid.is_none() {
                                                            m.no_bid = Some(price_cents.saturating_sub(1).max(1));
                                                        }
                                                        if m.no_ask.is_none() {
                                                            m.no_ask = Some((price_cents + 1).min(99));
                                                        }
                                                    }

                                                    (id.clone(), m.clone())
                                                });

                                            if let Some((mid, market)) = market_info {
                                                // Check ATM condition
                                                let exp = market.time_remaining_mins().unwrap_or(0.0);
                                                if exp < min_minutes || exp > max_minutes { continue; }

                                                let spot = match market.asset.as_str() {
                                                    "BTC" => s.prices.btc_price,
                                                    "ETH" => s.prices.eth_price,
                                                    "SOL" => s.prices.sol_price,
                                                    "XRP" => s.prices.xrp_price,
                                                    _ => None,
                                                };
                                                let Some(spot_val) = spot else { continue };
                                                let Some(strike) = market.strike_price else { continue };

                                                let dist = distance_from_strike_pct(spot_val, strike);
                                                let is_atm = dist <= atm_threshold;

                                                let (ya, na) = (market.yes_ask.unwrap_or(100), market.no_ask.unwrap_or(100));
                                                let combined = ya + na;
                                                let is_arb = combined < 100;

                                                // Only trade if ATM or ARB opportunity
                                                if !is_atm && !is_arb { continue; }

                                                // Skip WS-PRICE ATM/ARB trading when in bonds mode
                                                if bonds { continue; }

                                                if total_spent >= max_dollars { continue; }

                                                let ord = s.orders.get(&mid).cloned().unwrap_or_default();
                                                // Buy both sides when ATM or ARB
                                                if ord.yes_order_id.is_none() || ord.no_order_id.is_none() {
                                                    atm_trades.push((mid, market.yes_token.clone(), market.no_token.clone(), ya, na, market.asset.clone(), is_arb, dist));
                                                }
                                            }
                                        }
                                    }

                                    // Execute trades outside the lock
                                    for (mid, ytok, ntok, yask, nask, asset, is_arb, dist) in atm_trades {
                                        let ord = state.read().await.orders.get(&mid).cloned().unwrap_or_default();
                                        let (need_y, need_n) = (ord.yes_order_id.is_none(), ord.no_order_id.is_none());
                                        let combined = yask + nask;
                                        let reason = if is_arb {
                                            format!("ARB: {}¢+{}¢={}¢ ({}¢ profit)", yask, nask, combined, 100 - combined)
                                        } else {
                                            format!("ATM: dist={:.4}%", dist)
                                        };

                                        if dry_run {
                                            if need_y { warn!("🔔🔔🔔 [WS-PRICE] Would BID {} YES @{}¢ | ask={}¢ | {} | {} 🔔🔔🔔", contracts, bid_price, yask, asset, reason); }
                                            if need_n { warn!("🔔🔔🔔 [WS-PRICE] Would BID {} NO @{}¢ | ask={}¢ | {} | {} 🔔🔔🔔", contracts, bid_price, nask, asset, reason); }
                                        } else {
                                            let total_spent: f64 = state.read().await.positions.values().map(|p| p.yes_cost + p.no_cost).sum();
                                            let remaining_budget = max_dollars - total_spent;
                                            if remaining_budget < 1.0 { continue; }

                                            if is_arb {
                                                // ARB: Buy BOTH at ask prices (guaranteed profit)
                                                let y_pr = yask as f64 / 100.0;
                                                let n_pr = nask as f64 / 100.0;
                                                let min_c = (1.0 / y_pr.max(n_pr)).ceil();
                                                let max_afford = (remaining_budget / (y_pr + n_pr)).floor();
                                                let act_c = min_c.min(max_afford).max(min_c);
                                                let total_cost = act_c * (y_pr + n_pr);
                                                let profit = act_c - total_cost;

                                                warn!("💰💰💰 [WS-PRICE ARB] 📝 BUY {:.0} YES@{}¢ + NO@{}¢ = ${:.2} (profit ${:.2}) | {} 💰💰💰",
                                                      act_c, yask, nask, total_cost, profit, asset);

                                                if let Ok(f) = shared_client.buy_fak(&ytok, y_pr, act_c).await {
                                                    if f.filled_size > 0.0 {
                                                        let fp = (f.fill_cost / f.filled_size * 100.0).round() as i64;
                                                        warn!("💰 [ARB] ✅ YES Filled {:.2} @{}¢ (${:.2})", f.filled_size, fp, f.fill_cost);
                                                        let mut st = state.write().await;
                                                        if let Some(p) = st.positions.get_mut(&mid) { p.yes_qty += f.filled_size; p.yes_cost += f.fill_cost; }
                                                    }
                                                }
                                                if let Ok(f) = shared_client.buy_fak(&ntok, n_pr, act_c).await {
                                                    if f.filled_size > 0.0 {
                                                        let fp = (f.fill_cost / f.filled_size * 100.0).round() as i64;
                                                        warn!("💰 [ARB] ✅ NO Filled {:.2} @{}¢ (${:.2})", f.filled_size, fp, f.fill_cost);
                                                        let mut st = state.write().await;
                                                        if let Some(p) = st.positions.get_mut(&mid) { p.no_qty += f.filled_size; p.no_cost += f.fill_cost; }
                                                    }
                                                }
                                            } else {
                                                // ATM: Bid at our bid_price
                                                let pr = bid_price as f64 / 100.0;
                                                let min_c = (1.0 / pr).ceil();
                                                let max_afford = (remaining_budget / pr).floor();
                                                let act_c = min_c.min(max_afford).max(min_c);

                                                if need_y {
                                                    warn!("🔔🔔🔔 [WS-PRICE ATM] 📝 BID {:.0} YES @{}¢ | ask={}¢ | {} | dist={:.4}% 🔔🔔🔔", act_c, bid_price, yask, asset, dist);
                                                    if let Ok(f) = shared_client.buy_fak(&ytok, pr, act_c).await {
                                                        if f.filled_size > 0.0 {
                                                            let fp = (f.fill_cost / f.filled_size * 100.0).round() as i64;
                                                            warn!("🔔 [ATM] ✅ YES Filled {:.2} @{}¢ (${:.2})", f.filled_size, fp, f.fill_cost);
                                                            let mut st = state.write().await;
                                                            if let Some(p) = st.positions.get_mut(&mid) { p.yes_qty += f.filled_size; p.yes_cost += f.fill_cost; }
                                                        }
                                                    }
                                                }
                                                if need_n {
                                                    warn!("🔔🔔🔔 [WS-PRICE ATM] 📝 BID {:.0} NO @{}¢ | ask={}¢ | {} | dist={:.4}% 🔔🔔🔔", act_c, bid_price, nask, asset, dist);
                                                    if let Ok(f) = shared_client.buy_fak(&ntok, pr, act_c).await {
                                                        if f.filled_size > 0.0 {
                                                            let fp = (f.fill_cost / f.filled_size * 100.0).round() as i64;
                                                            warn!("🔔 [ATM] ✅ NO Filled {:.2} @{}¢ (${:.2})", f.filled_size, fp, f.fill_cost);
                                                            let mut st = state.write().await;
                                                            if let Some(p) = st.positions.get_mut(&mid) { p.no_qty += f.filled_size; p.no_cost += f.fill_cost; }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Ok(Message::Ping(data)) => {
                            let _ = write.send(Message::Pong(data)).await;
                        }
                        Ok(Message::Close(_)) | Err(_) => break,
                        _ => {}
                    }
                }
            }
        }

        info!("[WS] Disconnected, reconnecting in 5s...");
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}
