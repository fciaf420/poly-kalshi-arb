//! Reinforcement Learning module for cross-market state fusion
//!
//! Implements PPO (Proximal Policy Optimization) for trading Polymarket binary options
//! using fused state from Binance futures order flow.
//!
//! # Feature Flags
//! - `rl`: Enable PyTorch-based PPO training (requires libtorch installation)
//!
//! Without the `rl` feature, stub implementations are provided that allow
//! the code to compile but don't perform actual training.

pub mod experience_buffer;
pub mod feature_extractor;
pub mod metrics;
pub mod ppo_agent;
pub mod reward;

// Re-export main types
pub use experience_buffer::{Experience, ExperienceBuffer};
pub use feature_extractor::{
    build_observation, build_observation_normalized, normalize_features,
    binance_metrics_from_json, metrics_from_json,
    calculate_vol_5m, calculate_orderbook_imbalance, calculate_spread_pct,
    BinanceMetrics, PolymarketMetrics, Observation,
};
pub use ppo_agent::{Action, ActorCritic, PpoConfig, PpoTrainer, TrainingMetrics};
pub use reward::{compute_share_reward, compute_spread_adjusted_reward};
pub use metrics::{
    EpisodeStatus, RlEpisode, RlMetrics, RlMetricsCollector,
    LiveInference, LossHistory,
    RlStatusResponse, RlPerformanceResponse, RlLossesResponse,
    RlWsMessage, RlMetricsUpdate,
};
