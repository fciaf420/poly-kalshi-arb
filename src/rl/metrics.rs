//! RL Dashboard Metrics Module
//!
//! Provides thread-safe metrics collection for PPO training visualization.
//! Tracks training progress, episode outcomes, and live inference data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use super::Action;

// =============================================================================
// Constants
// =============================================================================

/// Maximum history size for loss/reward tracking
const MAX_HISTORY_SIZE: usize = 100;

/// Maximum episodes to retain
const MAX_EPISODES: usize = 500;

// =============================================================================
// Episode Status
// =============================================================================

/// Status of an RL episode (trade)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EpisodeStatus {
    /// Trade is open, waiting for exit
    Open,
    /// Trade closed with profit/loss
    Closed,
    /// Trade expired (e.g., market closed)
    Expired,
}

impl Default for EpisodeStatus {
    fn default() -> Self {
        Self::Open
    }
}

// =============================================================================
// RL Episode - Trade-level tracking
// =============================================================================

/// Represents a single RL episode (trade decision and outcome)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlEpisode {
    /// Unique episode ID
    pub id: u64,

    /// Market identifier
    pub market_id: String,

    /// Human-readable market name
    pub market_name: String,

    /// Timestamp when decision was made
    pub timestamp: DateTime<Utc>,

    /// The 18-dim observation vector
    pub observation: Vec<f32>,

    /// Action taken (Hold, BuyYes, BuyNo)
    pub action: u8,

    /// Action name as string
    pub action_name: String,

    /// Model's probability distribution [Hold, BuyYes, BuyNo]
    pub action_probs: [f32; 7],

    /// Model's value estimate for this state
    pub value_estimate: f32,

    /// Entry price (if trade was taken)
    pub entry_price: Option<f64>,

    /// Exit price (when trade closes)
    pub exit_price: Option<f64>,

    /// Reward received (RL reward signal)
    pub reward: Option<f64>,

    /// P&L in dollars
    pub pnl_dollars: Option<f64>,

    /// Trade duration in seconds
    pub duration_secs: Option<i64>,

    /// Episode status
    pub status: EpisodeStatus,
}

impl RlEpisode {
    /// Create a new episode when a decision is made
    pub fn new(
        id: u64,
        market_id: String,
        market_name: String,
        observation: Vec<f32>,
        action: Action,
        action_probs: [f32; 7],
        value_estimate: f32,
        entry_price: Option<f64>,
    ) -> Self {
        let (action_u8, action_name) = match action {
            Action::Hold => (0, "Hold".to_string()),
            Action::BuyYesSmall => (1, "BuyYes_S".to_string()),
            Action::BuyYesMed => (2, "BuyYes_M".to_string()),
            Action::BuyYesLarge => (3, "BuyYes_L".to_string()),
            Action::BuyNoSmall => (4, "BuyNo_S".to_string()),
            Action::BuyNoMed => (5, "BuyNo_M".to_string()),
            Action::BuyNoLarge => (6, "BuyNo_L".to_string()),
        };

        Self {
            id,
            market_id,
            market_name,
            timestamp: Utc::now(),
            observation,
            action: action_u8,
            action_name,
            action_probs,
            value_estimate,
            entry_price,
            exit_price: None,
            reward: None,
            pnl_dollars: None,
            duration_secs: None,
            status: if entry_price.is_some() { EpisodeStatus::Open } else { EpisodeStatus::Closed },
        }
    }

    /// Close the episode with outcome
    pub fn close(&mut self, exit_price: f64, reward: f64, pnl_dollars: f64) {
        self.exit_price = Some(exit_price);
        self.reward = Some(reward);
        self.pnl_dollars = Some(pnl_dollars);
        self.status = EpisodeStatus::Closed;

        // Calculate duration
        let now = Utc::now();
        self.duration_secs = Some((now - self.timestamp).num_seconds());
    }

    /// Mark as expired
    pub fn expire(&mut self) {
        self.status = EpisodeStatus::Expired;
        let now = Utc::now();
        self.duration_secs = Some((now - self.timestamp).num_seconds());
    }
}

// =============================================================================
// RL Metrics - Training statistics
// =============================================================================

/// Comprehensive RL training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlMetrics {
    // =========================================================================
    // Training Status
    // =========================================================================

    /// Model version/checkpoint number
    pub model_version: u32,

    /// Total PPO updates performed
    pub total_updates: u64,

    /// Total experiences collected
    pub total_experiences: u64,

    /// Current experience buffer size
    pub buffer_size: usize,

    /// Experience buffer capacity
    pub buffer_capacity: usize,

    /// Whether RL mode is active
    pub rl_mode_active: bool,

    /// Whether training is enabled
    pub training_enabled: bool,

    // =========================================================================
    // Loss History (last N updates)
    // =========================================================================

    /// Policy loss history
    #[serde(skip)]
    pub policy_loss_history: VecDeque<f32>,

    /// Value loss history
    #[serde(skip)]
    pub value_loss_history: VecDeque<f32>,

    /// Entropy history
    #[serde(skip)]
    pub entropy_history: VecDeque<f32>,

    /// Timestamps for each update (for time-series plots)
    #[serde(skip)]
    pub update_timestamps: VecDeque<DateTime<Utc>>,

    // =========================================================================
    // Reward Statistics
    // =========================================================================

    /// Mean reward (rolling)
    pub reward_mean: f64,

    /// Reward standard deviation (rolling)
    pub reward_std: f64,

    /// Episode reward history
    #[serde(skip)]
    pub episode_rewards: VecDeque<f64>,

    // =========================================================================
    // Action Distribution
    // =========================================================================

    /// Action counts [Hold, BuyYes, BuyNo]
    pub action_counts: [u64; 7],

    // =========================================================================
    // Performance Metrics
    // =========================================================================

    /// Win rate history (rolling windows)
    #[serde(skip)]
    pub win_rate_history: VecDeque<f32>,

    /// Average return history
    #[serde(skip)]
    pub avg_return_history: VecDeque<f64>,

    /// Cumulative P&L
    pub cumulative_pnl: f64,

    /// Total trades taken
    pub total_trades: u64,

    /// Winning trades count
    pub winning_trades: u64,

    // =========================================================================
    // Timestamps
    // =========================================================================

    /// When training started
    #[serde(skip)]
    pub training_started_at: Option<Instant>,

    /// Last update timestamp
    pub last_update_at: Option<DateTime<Utc>>,
}

impl Default for RlMetrics {
    fn default() -> Self {
        Self {
            model_version: 0,
            total_updates: 0,
            total_experiences: 0,
            buffer_size: 0,
            buffer_capacity: 256,
            rl_mode_active: false,
            training_enabled: false,

            policy_loss_history: VecDeque::with_capacity(MAX_HISTORY_SIZE),
            value_loss_history: VecDeque::with_capacity(MAX_HISTORY_SIZE),
            entropy_history: VecDeque::with_capacity(MAX_HISTORY_SIZE),
            update_timestamps: VecDeque::with_capacity(MAX_HISTORY_SIZE),

            reward_mean: 0.0,
            reward_std: 0.0,
            episode_rewards: VecDeque::with_capacity(MAX_HISTORY_SIZE),

            action_counts: [0; 7],

            win_rate_history: VecDeque::with_capacity(MAX_HISTORY_SIZE),
            avg_return_history: VecDeque::with_capacity(MAX_HISTORY_SIZE),
            cumulative_pnl: 0.0,
            total_trades: 0,
            winning_trades: 0,

            training_started_at: None,
            last_update_at: None,
        }
    }
}

impl RlMetrics {
    /// Create new metrics instance
    pub fn new(buffer_capacity: usize) -> Self {
        Self {
            buffer_capacity,
            ..Default::default()
        }
    }

    /// Record a PPO training update
    pub fn record_update(&mut self, policy_loss: f32, value_loss: f32, entropy: f32) {
        // Maintain history size
        if self.policy_loss_history.len() >= MAX_HISTORY_SIZE {
            self.policy_loss_history.pop_front();
            self.value_loss_history.pop_front();
            self.entropy_history.pop_front();
            self.update_timestamps.pop_front();
        }

        self.policy_loss_history.push_back(policy_loss);
        self.value_loss_history.push_back(value_loss);
        self.entropy_history.push_back(entropy);
        self.update_timestamps.push_back(Utc::now());

        self.total_updates += 1;
        self.last_update_at = Some(Utc::now());
    }

    /// Record an action taken
    pub fn record_action(&mut self, action: Action) {
        match action {
            Action::Hold => self.action_counts[0] += 1,
            Action::BuyYesSmall => self.action_counts[1] += 1,
            Action::BuyYesMed => self.action_counts[2] += 1,
            Action::BuyYesLarge => self.action_counts[3] += 1,
            Action::BuyNoSmall => self.action_counts[4] += 1,
            Action::BuyNoMed => self.action_counts[5] += 1,
            Action::BuyNoLarge => self.action_counts[6] += 1,
        }
    }

    /// Record episode reward
    pub fn record_reward(&mut self, reward: f64) {
        if self.episode_rewards.len() >= MAX_HISTORY_SIZE {
            self.episode_rewards.pop_front();
        }
        self.episode_rewards.push_back(reward);
        self.total_experiences += 1;

        // Update rolling statistics
        self.update_reward_stats();
    }

    /// Record trade outcome
    pub fn record_trade_outcome(&mut self, pnl: f64, is_win: bool) {
        self.total_trades += 1;
        self.cumulative_pnl += pnl;

        if is_win {
            self.winning_trades += 1;
        }

        // Update win rate history
        if self.win_rate_history.len() >= MAX_HISTORY_SIZE {
            self.win_rate_history.pop_front();
        }
        self.win_rate_history.push_back(self.current_win_rate());

        // Update average return history
        if self.avg_return_history.len() >= MAX_HISTORY_SIZE {
            self.avg_return_history.pop_front();
        }
        self.avg_return_history.push_back(self.current_avg_return());
    }

    /// Update buffer size
    pub fn set_buffer_size(&mut self, size: usize) {
        self.buffer_size = size;
    }

    /// Set RL mode status
    pub fn set_rl_mode(&mut self, active: bool, training: bool) {
        self.rl_mode_active = active;
        self.training_enabled = training;

        if active && self.training_started_at.is_none() {
            self.training_started_at = Some(Instant::now());
        }
    }

    /// Increment model version
    pub fn increment_model_version(&mut self) {
        self.model_version += 1;
    }

    // =========================================================================
    // Computed Properties
    // =========================================================================

    /// Get current win rate
    pub fn current_win_rate(&self) -> f32 {
        if self.total_trades == 0 {
            return 0.0;
        }
        self.winning_trades as f32 / self.total_trades as f32
    }

    /// Get current average return per trade
    pub fn current_avg_return(&self) -> f64 {
        if self.total_trades == 0 {
            return 0.0;
        }
        self.cumulative_pnl / self.total_trades as f64
    }

    /// Get average policy loss
    pub fn avg_policy_loss(&self) -> f32 {
        if self.policy_loss_history.is_empty() {
            return 0.0;
        }
        self.policy_loss_history.iter().sum::<f32>() / self.policy_loss_history.len() as f32
    }

    /// Get average value loss
    pub fn avg_value_loss(&self) -> f32 {
        if self.value_loss_history.is_empty() {
            return 0.0;
        }
        self.value_loss_history.iter().sum::<f32>() / self.value_loss_history.len() as f32
    }

    /// Get average entropy
    pub fn avg_entropy(&self) -> f32 {
        if self.entropy_history.is_empty() {
            return 0.0;
        }
        self.entropy_history.iter().sum::<f32>() / self.entropy_history.len() as f32
    }

    /// Get uptime in seconds
    pub fn uptime_secs(&self) -> u64 {
        self.training_started_at
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0)
    }

    /// Get action distribution as percentages (7 actions)
    pub fn action_distribution(&self) -> [f32; 7] {
        let total: u64 = self.action_counts.iter().sum();
        if total == 0 {
            return [0.0; 7];
        }
        [
            self.action_counts[0] as f32 / total as f32 * 100.0,  // Hold
            self.action_counts[1] as f32 / total as f32 * 100.0,  // BuyYes_S
            self.action_counts[2] as f32 / total as f32 * 100.0,  // BuyYes_M
            self.action_counts[3] as f32 / total as f32 * 100.0,  // BuyYes_L
            self.action_counts[4] as f32 / total as f32 * 100.0,  // BuyNo_S
            self.action_counts[5] as f32 / total as f32 * 100.0,  // BuyNo_M
            self.action_counts[6] as f32 / total as f32 * 100.0,  // BuyNo_L
        ]
    }

    /// Get buffer fill percentage
    pub fn buffer_fill_pct(&self) -> f32 {
        if self.buffer_capacity == 0 {
            return 0.0;
        }
        self.buffer_size as f32 / self.buffer_capacity as f32 * 100.0
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    fn update_reward_stats(&mut self) {
        if self.episode_rewards.is_empty() {
            self.reward_mean = 0.0;
            self.reward_std = 0.0;
            return;
        }

        let n = self.episode_rewards.len() as f64;
        self.reward_mean = self.episode_rewards.iter().sum::<f64>() / n;

        let variance = self.episode_rewards
            .iter()
            .map(|r| (r - self.reward_mean).powi(2))
            .sum::<f64>() / n;
        self.reward_std = variance.sqrt();
    }
}

// =============================================================================
// Live Inference Data
// =============================================================================

/// Current observation and model output for live display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveInference {
    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Current market being evaluated
    pub market_id: String,

    /// Market name
    pub market_name: String,

    /// Feature names for display
    pub feature_names: Vec<String>,

    /// Current observation values
    pub observation: Vec<f32>,

    /// Model's action probabilities [Hold, BuyYes, BuyNo]
    pub action_probs: [f32; 7],

    /// Value estimate
    pub value_estimate: f32,

    /// Recommended action
    pub recommended_action: String,
}

impl LiveInference {
    /// Create from observation and model output
    pub fn new(
        market_id: String,
        market_name: String,
        observation: Vec<f32>,
        action_probs: [f32; 7],
        value_estimate: f32,
    ) -> Self {
        // Standard feature names for 18-dim observation
        let feature_names = vec![
            "btc_momentum_5m".to_string(),
            "btc_momentum_1m".to_string(),
            "btc_volatility".to_string(),
            "cvd_5m".to_string(),
            "cvd_1m".to_string(),
            "trade_intensity".to_string(),
            "orderbook_imbalance".to_string(),
            "spread_bps".to_string(),
            "poly_yes_ask".to_string(),
            "poly_no_ask".to_string(),
            "poly_yes_size".to_string(),
            "poly_no_size".to_string(),
            "poly_spread".to_string(),
            "poly_mid_price".to_string(),
            "time_to_expiry_hrs".to_string(),
            "market_activity".to_string(),
            "position_size".to_string(),
            "unrealized_pnl".to_string(),
        ];

        // Find recommended action
        let max_idx = action_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let recommended_action = match max_idx {
            0 => "Hold",
            1 => "BuyYes_S",
            2 => "BuyYes_M",
            3 => "BuyYes_L",
            4 => "BuyNo_S",
            5 => "BuyNo_M",
            6 => "BuyNo_L",
            _ => "Unknown",
        }.to_string();

        Self {
            timestamp: Utc::now(),
            market_id,
            market_name,
            feature_names,
            observation,
            action_probs,
            value_estimate,
            recommended_action,
        }
    }
}

// =============================================================================
// Thread-Safe Metrics Collector
// =============================================================================

/// P&L metrics for a single asset (BTC, ETH, SOL, XRP)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssetPnLMetrics {
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
pub struct PerformanceByAssetMetrics {
    pub btc: AssetPnLMetrics,
    pub eth: AssetPnLMetrics,
    pub sol: AssetPnLMetrics,
    pub xrp: AssetPnLMetrics,
    pub total_pnl: f64,
}

impl PerformanceByAssetMetrics {
    pub fn new() -> Self {
        Self {
            btc: AssetPnLMetrics { asset: "BTC".to_string(), ..Default::default() },
            eth: AssetPnLMetrics { asset: "ETH".to_string(), ..Default::default() },
            sol: AssetPnLMetrics { asset: "SOL".to_string(), ..Default::default() },
            xrp: AssetPnLMetrics { asset: "XRP".to_string(), ..Default::default() },
            total_pnl: 0.0,
        }
    }

    /// Record a trade for a specific asset
    pub fn record_trade(&mut self, asset: &str, pnl: f64) {
        let asset_pnl = match asset.to_uppercase().as_str() {
            "BTC" | "BTCUSDT" => &mut self.btc,
            "ETH" | "ETHUSDT" => &mut self.eth,
            "SOL" | "SOLUSDT" => &mut self.sol,
            "XRP" | "XRPUSDT" => &mut self.xrp,
            _ => return,
        };

        asset_pnl.total_pnl += pnl;
        asset_pnl.total_trades += 1;
        if pnl > 0.0 {
            asset_pnl.winning_trades += 1;
        }

        if pnl > asset_pnl.best_trade {
            asset_pnl.best_trade = pnl;
        }
        if pnl < asset_pnl.worst_trade || asset_pnl.total_trades == 1 {
            asset_pnl.worst_trade = pnl;
        }

        if asset_pnl.total_trades > 0 {
            asset_pnl.win_rate = asset_pnl.winning_trades as f32 / asset_pnl.total_trades as f32;
            asset_pnl.avg_return = asset_pnl.total_pnl / asset_pnl.total_trades as f64;
        }

        self.total_pnl = self.btc.total_pnl + self.eth.total_pnl + self.sol.total_pnl + self.xrp.total_pnl;
    }

    /// Extract asset from market_id (e.g., "btc-15m-123" -> "BTC")
    pub fn extract_asset(market_id: &str) -> Option<&'static str> {
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

/// Thread-safe wrapper for RL metrics collection
#[derive(Clone)]
pub struct RlMetricsCollector {
    /// Training metrics
    pub metrics: Arc<RwLock<RlMetrics>>,

    /// Episode log
    pub episodes: Arc<RwLock<VecDeque<RlEpisode>>>,

    /// Live inference state
    pub live_inference: Arc<RwLock<Option<LiveInference>>>,

    /// Episode counter
    episode_counter: Arc<std::sync::atomic::AtomicU64>,

    /// Performance by asset (BTC, ETH, SOL, XRP)
    pub performance_by_asset: Arc<RwLock<PerformanceByAssetMetrics>>,
}

impl Default for RlMetricsCollector {
    fn default() -> Self {
        Self::new(256)
    }
}

impl RlMetricsCollector {
    /// Create new collector with specified buffer capacity
    pub fn new(buffer_capacity: usize) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(RlMetrics::new(buffer_capacity))),
            episodes: Arc::new(RwLock::new(VecDeque::with_capacity(MAX_EPISODES))),
            live_inference: Arc::new(RwLock::new(None)),
            episode_counter: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            performance_by_asset: Arc::new(RwLock::new(PerformanceByAssetMetrics::new())),
        }
    }

    /// Record a PPO update
    pub fn record_update(&self, policy_loss: f32, value_loss: f32, entropy: f32) {
        match self.metrics.write() {
            Ok(mut metrics) => {
                metrics.record_update(policy_loss, value_loss, entropy);
                tracing::debug!("[METRICS] Recorded update: p={:.4} v={:.4} e={:.4}", policy_loss, value_loss, entropy);
            }
            Err(e) => tracing::error!("[METRICS] Failed to record update - lock poisoned: {}", e),
        }
    }

    /// Record an action taken
    pub fn record_action(&self, action: Action) {
        match self.metrics.write() {
            Ok(mut metrics) => {
                metrics.record_action(action);
                tracing::debug!("[METRICS] Recorded action: {:?}", action);
            }
            Err(e) => tracing::error!("[METRICS] Failed to record action - lock poisoned: {}", e),
        }
    }

    /// Record episode reward
    pub fn record_reward(&self, reward: f64) {
        match self.metrics.write() {
            Ok(mut metrics) => {
                metrics.record_reward(reward);
                tracing::debug!("[METRICS] Recorded reward: {:.4}", reward);
            }
            Err(e) => tracing::error!("[METRICS] Failed to record reward - lock poisoned: {}", e),
        }
    }

    /// Record trade outcome
    pub fn record_trade_outcome(&self, pnl: f64, is_win: bool) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.record_trade_outcome(pnl, is_win);
        }
    }

    /// Update buffer size
    pub fn set_buffer_size(&self, size: usize) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.set_buffer_size(size);
        }
    }

    /// Set RL mode status
    pub fn set_rl_mode(&self, active: bool, training: bool) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.set_rl_mode(active, training);
        }
    }

    /// Increment model version
    pub fn increment_model_version(&self) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.increment_model_version();
        }
    }

    /// Record a new episode entry
    pub fn record_episode_entry(
        &self,
        market_id: String,
        market_name: String,
        observation: Vec<f32>,
        action: Action,
        action_probs: [f32; 7],
        value_estimate: f32,
        entry_price: Option<f64>,
    ) -> u64 {
        let id = self.episode_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let episode = RlEpisode::new(
            id,
            market_id,
            market_name,
            observation,
            action,
            action_probs,
            value_estimate,
            entry_price,
        );

        if let Ok(mut episodes) = self.episodes.write() {
            if episodes.len() >= MAX_EPISODES {
                episodes.pop_front();
            }
            episodes.push_back(episode);
        }

        id
    }

    /// Close an episode by ID
    pub fn close_episode(&self, episode_id: u64, exit_price: f64, reward: f64, pnl_dollars: f64) {
        let mut market_id_for_asset: Option<String> = None;

        if let Ok(mut episodes) = self.episodes.write() {
            if let Some(episode) = episodes.iter_mut().find(|e| e.id == episode_id) {
                market_id_for_asset = Some(episode.market_id.clone());
                episode.close(exit_price, reward, pnl_dollars);
            }
        }

        // Also record the trade outcome
        self.record_trade_outcome(pnl_dollars, pnl_dollars > 0.0);
        self.record_reward(reward);

        // Record per-asset P&L
        if let Some(market_id) = market_id_for_asset {
            self.record_asset_pnl(&market_id, pnl_dollars);
        }
    }

    /// Record P&L for a specific asset based on market_id
    pub fn record_asset_pnl(&self, market_id: &str, pnl: f64) {
        if let Some(asset) = PerformanceByAssetMetrics::extract_asset(market_id) {
            if let Ok(mut pba) = self.performance_by_asset.write() {
                pba.record_trade(asset, pnl);
                tracing::debug!("[METRICS] Recorded asset P&L: {} ${:.2} for market {}", asset, pnl, market_id);
            }
        }
    }

    /// Get performance by asset snapshot
    pub fn get_performance_by_asset(&self) -> Option<PerformanceByAssetMetrics> {
        self.performance_by_asset.read().ok().map(|pba| pba.clone())
    }

    /// Update live inference
    pub fn set_live_inference(&self, inference: LiveInference) {
        if let Ok(mut live) = self.live_inference.write() {
            *live = Some(inference);
        }
    }

    /// Clear live inference
    pub fn clear_live_inference(&self) {
        if let Ok(mut live) = self.live_inference.write() {
            *live = None;
        }
    }

    /// Get snapshot of current metrics for API response
    pub fn get_metrics_snapshot(&self) -> Option<RlMetrics> {
        self.metrics.read().ok().map(|m| m.clone())
    }

    /// Get recent episodes for API response
    pub fn get_recent_episodes(&self, limit: usize) -> Vec<RlEpisode> {
        self.episodes
            .read()
            .ok()
            .map(|eps| {
                eps.iter()
                    .rev()
                    .take(limit)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get live inference for API response
    pub fn get_live_inference(&self) -> Option<LiveInference> {
        self.live_inference.read().ok().and_then(|li| li.clone())
    }

    /// Get loss history for charting
    pub fn get_loss_history(&self) -> Option<LossHistory> {
        self.metrics.read().ok().map(|m| LossHistory {
            policy_losses: m.policy_loss_history.iter().copied().collect(),
            value_losses: m.value_loss_history.iter().copied().collect(),
            entropies: m.entropy_history.iter().copied().collect(),
            timestamps: m.update_timestamps.iter().map(|t| t.to_rfc3339()).collect(),
        })
    }
}

/// Loss history for charting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossHistory {
    pub policy_losses: Vec<f32>,
    pub value_losses: Vec<f32>,
    pub entropies: Vec<f32>,
    pub timestamps: Vec<String>,
}

// =============================================================================
// API Response Types
// =============================================================================

/// Status response for /api/rl/status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlStatusResponse {
    pub model_version: u32,
    pub total_updates: u64,
    pub total_experiences: u64,
    pub buffer_size: usize,
    pub buffer_capacity: usize,
    pub buffer_fill_pct: f32,
    pub rl_mode_active: bool,
    pub training_enabled: bool,
    pub uptime_secs: u64,
    pub last_update_at: Option<String>,
}

impl From<&RlMetrics> for RlStatusResponse {
    fn from(m: &RlMetrics) -> Self {
        Self {
            model_version: m.model_version,
            total_updates: m.total_updates,
            total_experiences: m.total_experiences,
            buffer_size: m.buffer_size,
            buffer_capacity: m.buffer_capacity,
            buffer_fill_pct: m.buffer_fill_pct(),
            rl_mode_active: m.rl_mode_active,
            training_enabled: m.training_enabled,
            uptime_secs: m.uptime_secs(),
            last_update_at: m.last_update_at.map(|t| t.to_rfc3339()),
        }
    }
}

/// Performance response for /api/rl/performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlPerformanceResponse {
    pub win_rate: f32,
    pub avg_return: f64,
    pub cumulative_pnl: f64,
    pub total_trades: u64,
    pub winning_trades: u64,
    pub reward_mean: f64,
    pub reward_std: f64,
    pub action_distribution: [f32; 7],
    pub action_counts: [u64; 7],
}

impl From<&RlMetrics> for RlPerformanceResponse {
    fn from(m: &RlMetrics) -> Self {
        Self {
            win_rate: m.current_win_rate(),
            avg_return: m.current_avg_return(),
            cumulative_pnl: m.cumulative_pnl,
            total_trades: m.total_trades,
            winning_trades: m.winning_trades,
            reward_mean: m.reward_mean,
            reward_std: m.reward_std,
            action_distribution: m.action_distribution(),
            action_counts: m.action_counts,
        }
    }
}

/// Losses response for /api/rl/losses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlLossesResponse {
    pub avg_policy_loss: f32,
    pub avg_value_loss: f32,
    pub avg_entropy: f32,
    pub history: LossHistory,
}

// =============================================================================
// WebSocket Message Types
// =============================================================================

/// WebSocket message types for real-time updates
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum RlWsMessage {
    /// Metrics update (sent after each PPO update)
    #[serde(rename = "metrics")]
    Metrics(RlMetricsUpdate),

    /// Episode update (sent on trade entry/exit)
    #[serde(rename = "episode")]
    Episode(RlEpisode),

    /// Live inference update (sent every 1s when active)
    #[serde(rename = "inference")]
    Inference(LiveInference),

    /// Status update
    #[serde(rename = "status")]
    Status(RlStatusResponse),
}

/// Metrics update for WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlMetricsUpdate {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub buffer_size: usize,
    pub total_updates: u64,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_record_update() {
        let mut metrics = RlMetrics::default();

        metrics.record_update(0.5, 0.3, 0.8);
        metrics.record_update(0.4, 0.2, 0.7);

        assert_eq!(metrics.total_updates, 2);
        assert!((metrics.avg_policy_loss() - 0.45).abs() < 0.01);
        assert!((metrics.avg_value_loss() - 0.25).abs() < 0.01);
        assert!((metrics.avg_entropy() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_metrics_action_distribution() {
        let mut metrics = RlMetrics::default();

        metrics.record_action(Action::Hold);
        metrics.record_action(Action::Hold);
        metrics.record_action(Action::BuyYesSmall);
        metrics.record_action(Action::BuyNoLarge);

        let dist = metrics.action_distribution();
        assert!((dist[0] - 50.0).abs() < 0.01); // 50% Hold
        assert!((dist[1] - 25.0).abs() < 0.01); // 25% BuyYesSmall
        assert!((dist[6] - 25.0).abs() < 0.01); // 25% BuyNoLarge
    }

    #[test]
    fn test_collector_thread_safety() {
        let collector = RlMetricsCollector::new(256);

        // Simulate concurrent access
        collector.record_update(0.5, 0.3, 0.8);
        collector.record_action(Action::BuyYesMed);
        collector.set_buffer_size(128);

        let snapshot = collector.get_metrics_snapshot().unwrap();
        assert_eq!(snapshot.total_updates, 1);
        assert_eq!(snapshot.buffer_size, 128);
    }

    #[test]
    fn test_episode_lifecycle() {
        let collector = RlMetricsCollector::new(256);

        let episode_id = collector.record_episode_entry(
            "btc-5m".to_string(),
            "BTC 5-min".to_string(),
            vec![0.0; 18],
            Action::BuyYesLarge,
            [0.1, 0.05, 0.05, 0.6, 0.05, 0.05, 0.1],  // 7 actions
            0.5,
            Some(0.42),
        );

        assert_eq!(episode_id, 0);

        let episodes = collector.get_recent_episodes(10);
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].status, EpisodeStatus::Open);

        collector.close_episode(episode_id, 0.65, 0.8, 12.50);

        let episodes = collector.get_recent_episodes(10);
        assert_eq!(episodes[0].status, EpisodeStatus::Closed);
        assert_eq!(episodes[0].pnl_dollars, Some(12.50));
    }
}
