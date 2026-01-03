//! PPO (Proximal Policy Optimization) Agent
//!
//! Implements actor-critic architecture for trading Polymarket binary options.
//! Uses tch-rs (PyTorch bindings) for neural network operations.
//!
//! This module is only compiled when the "rl" feature is enabled.

use super::feature_extractor::Observation;

/// Trading action enum - 7 actions for conviction-based sizing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Action {
    Hold = 0,
    BuyYesSmall = 1,   // 33% of max position
    BuyYesMed = 2,     // 66% of max position
    BuyYesLarge = 3,   // 100% of max position
    BuyNoSmall = 4,    // 33% of max position
    BuyNoMed = 5,      // 66% of max position
    BuyNoLarge = 6,    // 100% of max position
}

impl Action {
    /// Get the size multiplier for this action (0.0 for Hold)
    pub fn size_multiplier(&self) -> f64 {
        match self {
            Action::Hold => 0.0,
            Action::BuyYesSmall | Action::BuyNoSmall => 0.33,
            Action::BuyYesMed | Action::BuyNoMed => 0.66,
            Action::BuyYesLarge | Action::BuyNoLarge => 1.0,
        }
    }

    /// Check if this is a BuyYes action (any size)
    pub fn is_buy_yes(&self) -> bool {
        matches!(self, Action::BuyYesSmall | Action::BuyYesMed | Action::BuyYesLarge)
    }

    /// Check if this is a BuyNo action (any size)
    pub fn is_buy_no(&self) -> bool {
        matches!(self, Action::BuyNoSmall | Action::BuyNoMed | Action::BuyNoLarge)
    }

    /// Get action name for logging
    pub fn name(&self) -> &'static str {
        match self {
            Action::Hold => "Hold",
            Action::BuyYesSmall => "BuyYes_S",
            Action::BuyYesMed => "BuyYes_M",
            Action::BuyYesLarge => "BuyYes_L",
            Action::BuyNoSmall => "BuyNo_S",
            Action::BuyNoMed => "BuyNo_M",
            Action::BuyNoLarge => "BuyNo_L",
        }
    }
}

impl From<u8> for Action {
    fn from(v: u8) -> Self {
        match v {
            0 => Action::Hold,
            1 => Action::BuyYesSmall,
            2 => Action::BuyYesMed,
            3 => Action::BuyYesLarge,
            4 => Action::BuyNoSmall,
            5 => Action::BuyNoMed,
            6 => Action::BuyNoLarge,
            _ => Action::Hold,
        }
    }
}

impl From<i64> for Action {
    fn from(v: i64) -> Self {
        Action::from(v as u8)
    }
}

/// PPO hyperparameters
#[derive(Debug, Clone)]
pub struct PpoConfig {
    pub lr_actor: f64,
    pub lr_critic: f64,
    pub gamma: f32,           // Discount factor
    pub lambda: f32,          // GAE lambda
    pub clip_epsilon: f32,    // PPO clip parameter
    pub value_coef: f32,      // Value loss coefficient
    pub entropy_coef: f32,    // Entropy bonus coefficient
    pub max_grad_norm: f64,   // Gradient clipping
    pub ppo_epochs: usize,    // Epochs per update
    pub batch_size: usize,    // Minimum experiences before training
    pub history_len: usize,   // Phase 5: Number of past states for temporal encoder
    pub temporal_dim: usize,  // Phase 5: Temporal encoder output dimension
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            lr_actor: 1e-4,
            lr_critic: 3e-4,
            gamma: 0.95,             // Phase 5: 0.95 for 15-min horizon (was 0.99)
            lambda: 0.95,
            clip_epsilon: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.03,      // Phase 5: 0.03 sparse HOLD policy (was 0.10)
            max_grad_norm: 0.5,
            ppo_epochs: 10,
            batch_size: 64,
            history_len: 5,          // Phase 5: 5 past states
            temporal_dim: 32,        // Phase 5: 32-dim temporal features
        }
    }
}

// ============================================================================
// Training Metrics Logger - Track progress and detect readiness
// ============================================================================

/// Training metrics for monitoring PPO progress and detecting readiness
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Recent policy losses (sliding window)
    policy_losses: Vec<f32>,
    /// Recent value losses (sliding window)
    value_losses: Vec<f32>,
    /// Recent entropy values (sliding window)
    entropies: Vec<f32>,
    /// Recent episode rewards
    episode_rewards: Vec<f64>,
    /// Total training updates performed
    pub total_updates: usize,
    /// Total experiences collected
    pub total_experiences: usize,
    /// Window size for moving averages
    window_size: usize,
    /// Thresholds for readiness detection
    min_updates_for_ready: usize,
    policy_loss_threshold: f32,
    win_rate_threshold: f32,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new(100, 50, 0.5, 0.55)
    }
}

impl TrainingMetrics {
    /// Create new training metrics tracker
    ///
    /// # Arguments
    /// * `window_size` - Size of sliding window for moving averages
    /// * `min_updates_for_ready` - Minimum updates before model can be "ready"
    /// * `policy_loss_threshold` - Policy loss must be below this to be ready
    /// * `win_rate_threshold` - Win rate must be above this to be ready
    pub fn new(
        window_size: usize,
        min_updates_for_ready: usize,
        policy_loss_threshold: f32,
        win_rate_threshold: f32,
    ) -> Self {
        Self {
            policy_losses: Vec::with_capacity(window_size),
            value_losses: Vec::with_capacity(window_size),
            entropies: Vec::with_capacity(window_size),
            episode_rewards: Vec::with_capacity(window_size),
            total_updates: 0,
            total_experiences: 0,
            window_size,
            min_updates_for_ready,
            policy_loss_threshold,
            win_rate_threshold,
        }
    }

    /// Record a training update
    pub fn record_update(&mut self, policy_loss: f32, value_loss: f32, entropy: f32) {
        // Maintain window size
        if self.policy_losses.len() >= self.window_size {
            self.policy_losses.remove(0);
            self.value_losses.remove(0);
            self.entropies.remove(0);
        }

        self.policy_losses.push(policy_loss);
        self.value_losses.push(value_loss);
        self.entropies.push(entropy);
        self.total_updates += 1;
    }

    /// Record an episode reward
    pub fn record_reward(&mut self, reward: f64) {
        if self.episode_rewards.len() >= self.window_size {
            self.episode_rewards.remove(0);
        }
        self.episode_rewards.push(reward);
        self.total_experiences += 1;
    }

    /// Get moving average of policy loss
    pub fn avg_policy_loss(&self) -> f32 {
        if self.policy_losses.is_empty() {
            return f32::MAX;
        }
        self.policy_losses.iter().sum::<f32>() / self.policy_losses.len() as f32
    }

    /// Get moving average of value loss
    pub fn avg_value_loss(&self) -> f32 {
        if self.value_losses.is_empty() {
            return f32::MAX;
        }
        self.value_losses.iter().sum::<f32>() / self.value_losses.len() as f32
    }

    /// Get moving average of entropy
    pub fn avg_entropy(&self) -> f32 {
        if self.entropies.is_empty() {
            return 0.0;
        }
        self.entropies.iter().sum::<f32>() / self.entropies.len() as f32
    }

    /// Get win rate (percentage of positive rewards)
    pub fn win_rate(&self) -> f32 {
        if self.episode_rewards.is_empty() {
            return 0.0;
        }
        let wins = self.episode_rewards.iter().filter(|&&r| r > 0.0).count();
        wins as f32 / self.episode_rewards.len() as f32
    }

    /// Get average reward
    pub fn avg_reward(&self) -> f64 {
        if self.episode_rewards.is_empty() {
            return 0.0;
        }
        self.episode_rewards.iter().sum::<f64>() / self.episode_rewards.len() as f64
    }

    /// Check if policy loss is stabilizing (low variance)
    pub fn is_loss_stable(&self) -> bool {
        if self.policy_losses.len() < self.window_size / 2 {
            return false;
        }

        let mean = self.avg_policy_loss();
        let variance = self.policy_losses.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / self.policy_losses.len() as f32;
        let std = variance.sqrt();

        // Coefficient of variation < 0.3 indicates stability
        std / mean.abs().max(0.001) < 0.3
    }

    /// Check if model is ready for live trading
    ///
    /// Criteria:
    /// 1. Minimum number of training updates completed
    /// 2. Policy loss below threshold
    /// 3. Win rate above threshold
    /// 4. Losses are stabilizing (not still decreasing rapidly)
    pub fn is_ready(&self) -> bool {
        if self.total_updates < self.min_updates_for_ready {
            return false;
        }

        let policy_ok = self.avg_policy_loss() < self.policy_loss_threshold;
        let win_rate_ok = self.win_rate() > self.win_rate_threshold;
        let stable = self.is_loss_stable();

        policy_ok && win_rate_ok && stable
    }

    /// Get readiness status as a string
    pub fn readiness_status(&self) -> String {
        let updates_pct = (self.total_updates as f32 / self.min_updates_for_ready as f32 * 100.0).min(100.0);
        let policy_status = if self.avg_policy_loss() < self.policy_loss_threshold { "✓" } else { "✗" };
        let win_status = if self.win_rate() > self.win_rate_threshold { "✓" } else { "✗" };
        let stable_status = if self.is_loss_stable() { "✓" } else { "✗" };

        format!(
            "Updates: {}/{} ({:.0}%) | PL: {:.4} {} | WR: {:.1}% {} | Stable: {} | Ready: {}",
            self.total_updates,
            self.min_updates_for_ready,
            updates_pct,
            self.avg_policy_loss(),
            policy_status,
            self.win_rate() * 100.0,
            win_status,
            stable_status,
            if self.is_ready() { "✅ YES" } else { "❌ NO" }
        )
    }

    /// Get full status report
    pub fn full_report(&self) -> String {
        format!(
            r#"
╔══════════════════════════════════════════════════════════════╗
║                    PPO TRAINING METRICS                      ║
╠══════════════════════════════════════════════════════════════╣
║  Updates: {:>6}  |  Experiences: {:>8}                      ║
║  Policy Loss (avg):  {:>8.4}  (threshold: {:.2})            ║
║  Value Loss (avg):   {:>8.4}                                ║
║  Entropy (avg):      {:>8.4}                                ║
║  Win Rate:           {:>7.1}%  (threshold: {:.0}%)          ║
║  Avg Reward:         {:>8.4}                                ║
║  Loss Stable:        {:>8}                                  ║
╠══════════════════════════════════════════════════════════════╣
║  READY FOR LIVE: {}                                          ║
╚══════════════════════════════════════════════════════════════╝"#,
            self.total_updates,
            self.total_experiences,
            self.avg_policy_loss(),
            self.policy_loss_threshold,
            self.avg_value_loss(),
            self.avg_entropy(),
            self.win_rate() * 100.0,
            self.win_rate_threshold * 100.0,
            self.avg_reward(),
            if self.is_loss_stable() { "YES" } else { "NO" },
            if self.is_ready() { "✅ YES" } else { "❌ NO" }
        )
    }
}

// ============================================================================
// PyTorch-based implementation (requires "rl" feature and libtorch)
// ============================================================================

#[cfg(feature = "rl")]
pub mod torch_impl {
    use super::*;
    use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

    // Phase 5 architecture constants
    const OBS_DIM: i64 = 18;           // Current observation dimension
    const TEMPORAL_DIM: i64 = 32;      // Temporal encoder output
    const COMBINED_DIM: i64 = 50;      // 18 + 32 = 50
    const ACTOR_HIDDEN: i64 = 64;      // Actor hidden size (smaller)
    const CRITIC_HIDDEN: i64 = 96;     // Critic hidden size (larger, asymmetric)
    const HISTORY_LEN: i64 = 5;        // Number of past states
    const NUM_ACTIONS: i64 = 7;        // Hold + 3 sizes × 2 directions

    /// Phase 5: Temporal Encoder
    /// Compresses 5 past states (90-dim) into 32-dim temporal features
    /// Captures momentum, velocity, and trend direction
    pub struct TemporalEncoder {
        fc1: nn::Linear,           // 90 -> 64
        ln1: nn::LayerNorm,
        fc2: nn::Linear,           // 64 -> 32
        ln2: nn::LayerNorm,
    }

    impl TemporalEncoder {
        pub fn new(vs: &nn::Path) -> Self {
            let input_dim = HISTORY_LEN * OBS_DIM;  // 5 * 18 = 90

            Self {
                fc1: nn::linear(vs / "temp_fc1", input_dim, 64, Default::default()),
                ln1: nn::layer_norm(vs / "temp_ln1", vec![64], Default::default()),
                fc2: nn::linear(vs / "temp_fc2", 64, TEMPORAL_DIM, Default::default()),
                ln2: nn::layer_norm(vs / "temp_ln2", vec![TEMPORAL_DIM], Default::default()),
            }
        }

        pub fn forward(&self, history: &Tensor) -> Tensor {
            let x = history.apply(&self.fc1).apply(&self.ln1).tanh();
            x.apply(&self.fc2).apply(&self.ln2).tanh()
        }
    }

    /// Phase 5: Asymmetric Actor-Critic with Temporal Encoders
    ///
    /// Architecture:
    /// - TemporalEncoder: 90 -> 64 -> 32 (captures momentum from 5 past states)
    /// - Combined input: current obs (18) + temporal features (32) = 50
    /// - Actor: 50 -> 64 -> 64 -> 7 (7 actions: Hold + 3 sizes × 2 directions)
    /// - Critic: 50 -> 96 -> 96 -> 1 (larger, for value estimation)
    pub struct ActorCritic {
        // Temporal encoders for history processing (separate for actor/critic)
        actor_temporal_encoder: TemporalEncoder,
        critic_temporal_encoder: TemporalEncoder,

        // Actor network (smaller): 50 -> 64 -> 64 -> 7
        actor_fc1: nn::Linear,
        actor_ln1: nn::LayerNorm,
        actor_fc2: nn::Linear,
        actor_ln2: nn::LayerNorm,
        actor_head: nn::Linear,

        // Critic network (larger, asymmetric): 50 -> 96 -> 96 -> 1
        critic_fc1: nn::Linear,
        critic_ln1: nn::LayerNorm,
        critic_fc2: nn::Linear,
        critic_ln2: nn::LayerNorm,
        critic_head: nn::Linear,

        // Variable store for parameters
        vs: nn::VarStore,
    }

    impl ActorCritic {
        /// Create new Phase 5 actor-critic network with temporal encoder
        pub fn new(device: Device) -> Self {
            let vs = nn::VarStore::new(device);
            let root = vs.root();
            let actor_path = root.set_group(0);
            let critic_path = root.set_group(1);

            // Temporal encoders (separate for actor/critic)
            let actor_temporal_encoder = TemporalEncoder::new(&(&actor_path / "actor_temporal"));
            let critic_temporal_encoder = TemporalEncoder::new(&(&critic_path / "critic_temporal"));

            // Actor network: 50 -> 64 -> 64 -> 7 (7 actions)
            let actor_fc1 = nn::linear(&actor_path / "actor_fc1", COMBINED_DIM, ACTOR_HIDDEN, Default::default());
            let actor_ln1 = nn::layer_norm(&actor_path / "actor_ln1", vec![ACTOR_HIDDEN], Default::default());
            let actor_fc2 = nn::linear(&actor_path / "actor_fc2", ACTOR_HIDDEN, ACTOR_HIDDEN, Default::default());
            let actor_ln2 = nn::layer_norm(&actor_path / "actor_ln2", vec![ACTOR_HIDDEN], Default::default());
            let actor_head = nn::linear(&actor_path / "actor_head", ACTOR_HIDDEN, NUM_ACTIONS, Default::default());

            // Critic network: 50 -> 96 -> 96 -> 1 (asymmetric, larger)
            let critic_fc1 = nn::linear(&critic_path / "critic_fc1", COMBINED_DIM, CRITIC_HIDDEN, Default::default());
            let critic_ln1 = nn::layer_norm(&critic_path / "critic_ln1", vec![CRITIC_HIDDEN], Default::default());
            let critic_fc2 = nn::linear(&critic_path / "critic_fc2", CRITIC_HIDDEN, CRITIC_HIDDEN, Default::default());
            let critic_ln2 = nn::layer_norm(&critic_path / "critic_ln2", vec![CRITIC_HIDDEN], Default::default());
            let critic_head = nn::linear(&critic_path / "critic_head", CRITIC_HIDDEN, 1, Default::default());

            Self {
                actor_temporal_encoder,
                critic_temporal_encoder,
                actor_fc1,
                actor_ln1,
                actor_fc2,
                actor_ln2,
                actor_head,
                critic_fc1,
                critic_ln1,
                critic_fc2,
                critic_ln2,
                critic_head,
                vs,
            }
        }

        /// Forward pass with temporal features
        /// obs: current observation [batch, 18]
        /// history: stacked past observations [batch, 90] (5 * 18)
        pub fn forward_with_history(&self, obs: &Tensor, history: &Tensor) -> (Tensor, Tensor) {
            // Encode temporal features from history (separate for actor/critic)
            let actor_temporal = self.actor_temporal_encoder.forward(history);   // [batch, 32]
            let critic_temporal = self.critic_temporal_encoder.forward(history); // [batch, 32]

            // Concatenate current obs with temporal features
            let actor_combined = Tensor::cat(&[obs, &actor_temporal], 1);   // [batch, 50]
            let critic_combined = Tensor::cat(&[obs, &critic_temporal], 1); // [batch, 50]

            // Actor forward pass
            let actor_x = actor_combined.apply(&self.actor_fc1).apply(&self.actor_ln1).tanh();
            let actor_x = actor_x.apply(&self.actor_fc2).apply(&self.actor_ln2).tanh();
            let logits = actor_x.apply(&self.actor_head);
            let probs = logits.softmax(-1, Kind::Float);

            // Critic forward pass (separate network)
            let critic_x = critic_combined.apply(&self.critic_fc1).apply(&self.critic_ln1).tanh();
            let critic_x = critic_x.apply(&self.critic_fc2).apply(&self.critic_ln2).tanh();
            let value = critic_x.apply(&self.critic_head);

            (probs, value)
        }

        /// Fallback forward pass without history (uses zero temporal features)
        pub fn forward(&self, obs: &Tensor) -> (Tensor, Tensor) {
            let batch_size = obs.size()[0];
            let device = self.vs.device();

            // Create zero history if not provided
            let zero_history = Tensor::zeros(
                &[batch_size, HISTORY_LEN * OBS_DIM],
                (Kind::Float, device)
            );

            self.forward_with_history(obs, &zero_history)
        }

        /// Select action from observation with history
        pub fn select_action_with_history(&self, obs: &Observation, history: &[f32]) -> (Action, f32, f32) {
            let device = self.vs.device();

            let obs_tensor = Tensor::from_slice(obs.as_slice())
                .to_device(device)
                .unsqueeze(0);

            let history_tensor = Tensor::from_slice(history)
                .to_device(device)
                .unsqueeze(0);

            let (probs, value) = self.forward_with_history(&obs_tensor, &history_tensor);

            // Sample from categorical distribution
            let action_tensor = probs.multinomial(1, true);
            let action_idx = action_tensor.int64_value(&[0, 0]);

            // Get log probability
            let log_probs = probs.log();
            let log_prob = log_probs.double_value(&[0, action_idx as i64]);

            // Get value estimate
            let value_est = value.double_value(&[0, 0]);

            (Action::from(action_idx), log_prob as f32, value_est as f32)
        }

        /// Select action from observation (fallback without history)
        pub fn select_action(&self, obs: &Observation) -> (Action, f32, f32) {
            let obs_tensor = Tensor::from_slice(obs.as_slice())
                .to_device(self.vs.device())
                .unsqueeze(0);

            let (probs, value) = self.forward(&obs_tensor);

            // Sample from categorical distribution
            let action_tensor = probs.multinomial(1, true);
            let action_idx = action_tensor.int64_value(&[0, 0]);

            // Get log probability
            let log_probs = probs.log();
            let log_prob = log_probs.double_value(&[0, action_idx as i64]);

            // Get value estimate
            let value_est = value.double_value(&[0, 0]);

            (Action::from(action_idx), log_prob as f32, value_est as f32)
        }

        /// Select action with full action probabilities returned (for dashboard display)
        pub fn select_action_with_probs(&self, obs: &Observation, history: Option<&[f32]>) -> (Action, f32, f32, [f32; 7]) {
            let device = self.vs.device();
            let obs_tensor = Tensor::from_slice(obs.as_slice())
                .to_device(device)
                .unsqueeze(0);

            let (probs, value) = if let Some(h) = history {
                if h.len() == 90 {
                    let history_tensor = Tensor::from_slice(h).to_device(device).unsqueeze(0);
                    self.forward_with_history(&obs_tensor, &history_tensor)
                } else {
                    self.forward(&obs_tensor)
                }
            } else {
                self.forward(&obs_tensor)
            };

            // Sample from categorical distribution
            let action_tensor = probs.multinomial(1, true);
            let action_idx = action_tensor.int64_value(&[0, 0]);

            // Get log probability
            let log_probs = probs.log();
            let log_prob = log_probs.double_value(&[0, action_idx as i64]);

            // Get value estimate
            let value_est = value.double_value(&[0, 0]);

            // Extract all action probabilities
            let mut action_probs = [0.0f32; 7];
            for i in 0..7 {
                action_probs[i] = probs.double_value(&[0, i as i64]) as f32;
            }

            (Action::from(action_idx), log_prob as f32, value_est as f32, action_probs)
        }

        /// Get value estimate for observation with history
        pub fn get_value_with_history(&self, obs: &Observation, history: &[f32]) -> f32 {
            let device = self.vs.device();

            let obs_tensor = Tensor::from_slice(obs.as_slice())
                .to_device(device)
                .unsqueeze(0);

            let history_tensor = Tensor::from_slice(history)
                .to_device(device)
                .unsqueeze(0);

            let (_, value) = self.forward_with_history(&obs_tensor, &history_tensor);
            value.double_value(&[0, 0]) as f32
        }

        /// Get value estimate for observation (fallback without history)
        pub fn get_value(&self, obs: &Observation) -> f32 {
            let obs_tensor = Tensor::from_slice(obs.as_slice())
                .to_device(self.vs.device())
                .unsqueeze(0);

            let (_, value) = self.forward(&obs_tensor);
            value.double_value(&[0, 0]) as f32
        }

        /// Save model weights
        pub fn save(&self, path: &str) -> Result<(), tch::TchError> {
            self.vs.save(path)
        }

        /// Load model weights from PyTorch format
        pub fn load(&mut self, path: &str) -> Result<(), tch::TchError> {
            self.vs.load(path)
        }

        /// Load partial weights from old 3-action model for upgrade to 7-action
        ///
        /// Copies all compatible layers (temporal encoders, hidden layers, critic head)
        /// but leaves actor_head randomly initialized since dimensions changed (3 → 7).
        ///
        /// Use this when upgrading from 3-action to 7-action model to preserve
        /// learned feature representations while relearning the action policy.
        pub fn load_partial_for_upgrade(&mut self, old_path: &str) -> Result<(), tch::TchError> {
            // Load old model into temporary VarStore
            let mut old_vs = nn::VarStore::new(self.vs.device());

            // Create dummy old architecture to populate variable names
            let old_root = old_vs.root();
            let old_actor_path = old_root.set_group(0);
            let old_critic_path = old_root.set_group(1);

            // Temporal encoders (same dimensions)
            let _old_actor_temporal = TemporalEncoder::new(&(&old_actor_path / "actor_temporal"));
            let _old_critic_temporal = TemporalEncoder::new(&(&old_critic_path / "critic_temporal"));

            // Hidden layers (same dimensions)
            let _ = nn::linear(&old_actor_path / "actor_fc1", COMBINED_DIM, ACTOR_HIDDEN, Default::default());
            let _ = nn::layer_norm(&old_actor_path / "actor_ln1", vec![ACTOR_HIDDEN], Default::default());
            let _ = nn::linear(&old_actor_path / "actor_fc2", ACTOR_HIDDEN, ACTOR_HIDDEN, Default::default());
            let _ = nn::layer_norm(&old_actor_path / "actor_ln2", vec![ACTOR_HIDDEN], Default::default());
            let _ = nn::linear(&old_actor_path / "actor_head", ACTOR_HIDDEN, 3, Default::default()); // Old: 3 actions

            let _ = nn::linear(&old_critic_path / "critic_fc1", COMBINED_DIM, CRITIC_HIDDEN, Default::default());
            let _ = nn::layer_norm(&old_critic_path / "critic_ln1", vec![CRITIC_HIDDEN], Default::default());
            let _ = nn::linear(&old_critic_path / "critic_fc2", CRITIC_HIDDEN, CRITIC_HIDDEN, Default::default());
            let _ = nn::layer_norm(&old_critic_path / "critic_ln2", vec![CRITIC_HIDDEN], Default::default());
            let _ = nn::linear(&old_critic_path / "critic_head", CRITIC_HIDDEN, 1, Default::default());

            // Load old weights
            old_vs.load(old_path)?;

            // Layers to copy (all except actor_head which changed dimensions)
            let layers_to_copy = [
                // Actor temporal encoder
                "0.actor_temporal.temp_fc1",
                "0.actor_temporal.temp_ln1",
                "0.actor_temporal.temp_fc2",
                "0.actor_temporal.temp_ln2",
                // Actor hidden layers
                "0.actor_fc1",
                "0.actor_ln1",
                "0.actor_fc2",
                "0.actor_ln2",
                // Critic temporal encoder
                "1.critic_temporal.temp_fc1",
                "1.critic_temporal.temp_ln1",
                "1.critic_temporal.temp_fc2",
                "1.critic_temporal.temp_ln2",
                // Critic layers
                "1.critic_fc1",
                "1.critic_ln1",
                "1.critic_fc2",
                "1.critic_ln2",
                "1.critic_head",  // Critic head unchanged (still 1 output)
            ];

            let old_vars = old_vs.variables();
            let new_vars = self.vs.variables();
            let mut copied_count = 0;

            for layer in layers_to_copy {
                // Copy weight
                let weight_key = format!("{}.weight", layer);
                if let (Some(old_w), Some(new_w)) = (old_vars.get(&weight_key), new_vars.get(&weight_key)) {
                    tch::no_grad(|| {
                        let mut new_w_mut = new_w.shallow_clone();
                        let _ = new_w_mut.copy_(old_w);
                    });
                    copied_count += 1;
                }

                // Copy bias
                let bias_key = format!("{}.bias", layer);
                if let (Some(old_b), Some(new_b)) = (old_vars.get(&bias_key), new_vars.get(&bias_key)) {
                    tch::no_grad(|| {
                        let mut new_b_mut = new_b.shallow_clone();
                        let _ = new_b_mut.copy_(old_b);
                    });
                    copied_count += 1;
                }
            }

            tracing::info!(
                "[RL] Partial transfer complete: copied {} parameters, actor_head reinitialized for {} actions",
                copied_count, NUM_ACTIONS
            );

            Ok(())
        }

        /// Load model weights from Python safetensors format
        ///
        /// Maps Python layer names to Rust layer names:
        /// - actor.temporal_encoder.fc1 → actor_temporal.temp_fc1
        /// - critic.temporal_encoder.fc1 → critic_temporal.temp_fc1
        /// - actor.fc1 → actor_fc1
        /// - critic.fc1 → critic_fc1
        /// etc.
        pub fn load_safetensors(&mut self, path: &str) -> anyhow::Result<()> {
            use safetensors::SafeTensors;
            use std::fs;

            let data = fs::read(path)?;
            let tensors = SafeTensors::deserialize(&data)?;

            // Mapping from Python safetensors keys to Rust tch layer names
            let key_mapping = [
                // Actor temporal encoder
                ("actor.temporal_encoder.fc1.weight", "actor_temporal.temp_fc1.weight"),
                ("actor.temporal_encoder.fc1.bias", "actor_temporal.temp_fc1.bias"),
                ("actor.temporal_encoder.ln1.weight", "actor_temporal.temp_ln1.weight"),
                ("actor.temporal_encoder.ln1.bias", "actor_temporal.temp_ln1.bias"),
                ("actor.temporal_encoder.fc2.weight", "actor_temporal.temp_fc2.weight"),
                ("actor.temporal_encoder.fc2.bias", "actor_temporal.temp_fc2.bias"),
                ("actor.temporal_encoder.ln2.weight", "actor_temporal.temp_ln2.weight"),
                ("actor.temporal_encoder.ln2.bias", "actor_temporal.temp_ln2.bias"),
                // Critic temporal encoder
                ("critic.temporal_encoder.fc1.weight", "critic_temporal.temp_fc1.weight"),
                ("critic.temporal_encoder.fc1.bias", "critic_temporal.temp_fc1.bias"),
                ("critic.temporal_encoder.ln1.weight", "critic_temporal.temp_ln1.weight"),
                ("critic.temporal_encoder.ln1.bias", "critic_temporal.temp_ln1.bias"),
                ("critic.temporal_encoder.fc2.weight", "critic_temporal.temp_fc2.weight"),
                ("critic.temporal_encoder.fc2.bias", "critic_temporal.temp_fc2.bias"),
                ("critic.temporal_encoder.ln2.weight", "critic_temporal.temp_ln2.weight"),
                ("critic.temporal_encoder.ln2.bias", "critic_temporal.temp_ln2.bias"),
                // Actor layers
                ("actor.fc1.weight", "actor_fc1.weight"),
                ("actor.fc1.bias", "actor_fc1.bias"),
                ("actor.ln1.weight", "actor_ln1.weight"),
                ("actor.ln1.bias", "actor_ln1.bias"),
                ("actor.fc2.weight", "actor_fc2.weight"),
                ("actor.fc2.bias", "actor_fc2.bias"),
                ("actor.ln2.weight", "actor_ln2.weight"),
                ("actor.ln2.bias", "actor_ln2.bias"),
                ("actor.fc3.weight", "actor_head.weight"),
                ("actor.fc3.bias", "actor_head.bias"),
                // Critic layers
                ("critic.fc1.weight", "critic_fc1.weight"),
                ("critic.fc1.bias", "critic_fc1.bias"),
                ("critic.ln1.weight", "critic_ln1.weight"),
                ("critic.ln1.bias", "critic_ln1.bias"),
                ("critic.fc2.weight", "critic_fc2.weight"),
                ("critic.fc2.bias", "critic_fc2.bias"),
                ("critic.ln2.weight", "critic_ln2.weight"),
                ("critic.ln2.bias", "critic_ln2.bias"),
                ("critic.fc3.weight", "critic_head.weight"),
                ("critic.fc3.bias", "critic_head.bias"),
            ];

            let device = self.vs.device();
            let mut loaded_count = 0;
            let mut skipped_keys: Vec<String> = Vec::new();

            // Get all variable names from the VarStore
            let variables = self.vs.variables();

            for (py_key, rust_key) in &key_mapping {
                match tensors.tensor(py_key) {
                    Ok(tensor_view) => {
                        let shape: Vec<i64> = tensor_view.shape().iter().map(|&x| x as i64).collect();
                        let data = tensor_view.data();

                        // Convert f32 bytes to tensor
                        let float_data: Vec<f32> = data
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect();

                        let source_tensor = Tensor::from_slice(&float_data)
                            .reshape(&shape)
                            .to_device(device);

                        // Find the target tensor in the VarStore and copy data
                        // We use shallow_clone + copy_ to work around Rust's borrow checker
                        // since PyTorch tensors share underlying storage
                        if let Some(target_tensor) = variables.get(*rust_key) {
                            let mut target_clone = target_tensor.shallow_clone();
                            tch::no_grad(|| {
                                let _ = target_clone.copy_(&source_tensor);
                            });
                            loaded_count += 1;
                        } else {
                            tracing::warn!("[RL] Variable not found in VarStore: {}", rust_key);
                        }
                    }
                    Err(_) => {
                        skipped_keys.push(py_key.to_string());
                    }
                }
            }

            tracing::info!(
                "[RL] Loaded {} weights from safetensors, skipped {} keys",
                loaded_count,
                skipped_keys.len()
            );

            if !skipped_keys.is_empty() {
                tracing::debug!("[RL] Skipped keys: {:?}", skipped_keys);
            }

            Ok(())
        }

        /// Get variable store for optimizer
        pub fn var_store(&self) -> &nn::VarStore {
            &self.vs
        }

        /// Get mutable variable store
        pub fn var_store_mut(&mut self) -> &mut nn::VarStore {
            &mut self.vs
        }
    }

    /// PPO Trainer with Phase 5 architecture
    pub struct PpoTrainer {
        pub model: ActorCritic,
        optimizer: nn::Optimizer,
        config: PpoConfig,
    }

    impl PpoTrainer {
        pub fn new(config: PpoConfig, device: Device) -> Self {
            let model = ActorCritic::new(device);
            let mut optimizer = nn::Adam::default()
                .build(&model.vs, config.lr_actor)
                .expect("Failed to create optimizer");
            // Use a separate learning rate for critic parameters (group 1).
            optimizer.set_lr_group(1, config.lr_critic);

            Self {
                model,
                optimizer,
                config,
            }
        }

        /// Perform PPO update on batch of experiences with history
        pub fn update_with_history(
            &mut self,
            observations: &[&Observation],
            histories: &[Vec<f32>],  // Each history is 90-dim (5 * 18)
            actions: &[u8],
            old_log_probs: &[f32],
            advantages: &[f32],
            returns: &[f32],
        ) -> (f32, f32, f32) {
            let device = self.model.vs.device();
            let batch_size = observations.len();

            // Convert observations to tensor
            let obs_data: Vec<f32> = observations
                .iter()
                .flat_map(|o| o.features.iter().copied())
                .collect();
            let obs_tensor = Tensor::from_slice(&obs_data)
                .reshape(&[batch_size as i64, OBS_DIM])
                .to_device(device);

            // Convert histories to tensor
            let history_data: Vec<f32> = histories.iter().flatten().copied().collect();
            let history_tensor = Tensor::from_slice(&history_data)
                .reshape(&[batch_size as i64, HISTORY_LEN * OBS_DIM])
                .to_device(device);

            let actions_tensor = Tensor::from_slice(actions)
                .to_kind(Kind::Int64)
                .to_device(device);
            let old_log_probs_tensor = Tensor::from_slice(old_log_probs).to_device(device);
            let advantages_tensor = Tensor::from_slice(advantages).to_device(device);
            let returns_tensor = Tensor::from_slice(returns).to_device(device);

            let mut total_policy_loss = 0.0;
            let mut total_value_loss = 0.0;
            let mut total_entropy = 0.0;

            for _ in 0..self.config.ppo_epochs {
                let (probs, values) = self.model.forward_with_history(&obs_tensor, &history_tensor);
                let values = values.squeeze_dim(-1);

                // Get log probs for taken actions
                let log_probs = probs.log();
                let action_log_probs = log_probs
                    .gather(1, &actions_tensor.unsqueeze(-1), false)
                    .squeeze_dim(-1);

                // Compute ratio
                let ratio = (&action_log_probs - &old_log_probs_tensor).exp();

                // Clipped surrogate objective
                let clip_eps = self.config.clip_epsilon as f64;
                let clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
                let surrogate1 = &ratio * &advantages_tensor;
                let surrogate2 = &clipped_ratio * &advantages_tensor;
                let policy_loss = -surrogate1.min_other(&surrogate2).mean(Kind::Float);

                // Value loss
                let value_loss = (&values - &returns_tensor)
                    .pow_tensor_scalar(2)
                    .mean(Kind::Float);

                // Entropy bonus
                let entropy = -(probs.shallow_clone() * log_probs)
                    .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
                    .mean(Kind::Float);

                // Total loss
                let loss = &policy_loss
                    + self.config.value_coef * &value_loss
                    - self.config.entropy_coef * &entropy;

                // Backprop
                self.optimizer.zero_grad();
                loss.backward();

                // Gradient clipping
                let _ = self.model.vs.variables().iter().for_each(|(_, t)| {
                    let _ = t.grad().clamp_(-self.config.max_grad_norm, self.config.max_grad_norm);
                });

                self.optimizer.step();

                total_policy_loss += policy_loss.double_value(&[]) as f32;
                total_value_loss += value_loss.double_value(&[]) as f32;
                total_entropy += entropy.double_value(&[]) as f32;
            }

            let epochs = self.config.ppo_epochs as f32;
            (
                total_policy_loss / epochs,
                total_value_loss / epochs,
                total_entropy / epochs,
            )
        }

        /// Perform PPO update on batch of experiences (fallback without history)
        pub fn update(
            &mut self,
            observations: &[&Observation],
            actions: &[u8],
            old_log_probs: &[f32],
            advantages: &[f32],
            returns: &[f32],
        ) -> (f32, f32, f32) {
            let device = self.model.vs.device();
            let batch_size = observations.len();

            // Convert to tensors
            let obs_data: Vec<f32> = observations
                .iter()
                .flat_map(|o| o.features.iter().copied())
                .collect();
            let obs_tensor = Tensor::from_slice(&obs_data)
                .reshape(&[batch_size as i64, OBS_DIM])
                .to_device(device);

            let actions_tensor = Tensor::from_slice(actions)
                .to_kind(Kind::Int64)
                .to_device(device);
            let old_log_probs_tensor = Tensor::from_slice(old_log_probs).to_device(device);
            let advantages_tensor = Tensor::from_slice(advantages).to_device(device);
            let returns_tensor = Tensor::from_slice(returns).to_device(device);

            let mut total_policy_loss = 0.0;
            let mut total_value_loss = 0.0;
            let mut total_entropy = 0.0;

            for _ in 0..self.config.ppo_epochs {
                let (probs, values) = self.model.forward(&obs_tensor);
                let values = values.squeeze_dim(-1);

                // Get log probs for taken actions
                let log_probs = probs.log();
                let action_log_probs = log_probs
                    .gather(1, &actions_tensor.unsqueeze(-1), false)
                    .squeeze_dim(-1);

                // Compute ratio
                let ratio = (&action_log_probs - &old_log_probs_tensor).exp();

                // Clipped surrogate objective
                let clip_eps = self.config.clip_epsilon as f64;
                let clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
                let surrogate1 = &ratio * &advantages_tensor;
                let surrogate2 = &clipped_ratio * &advantages_tensor;
                let policy_loss = -surrogate1.min_other(&surrogate2).mean(Kind::Float);

                // Value loss
                let value_loss = (&values - &returns_tensor)
                    .pow_tensor_scalar(2)
                    .mean(Kind::Float);

                // Entropy bonus
                let entropy = -(probs.shallow_clone() * log_probs)
                    .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
                    .mean(Kind::Float);

                // Total loss
                let loss = &policy_loss
                    + self.config.value_coef * &value_loss
                    - self.config.entropy_coef * &entropy;

                // Backprop
                self.optimizer.zero_grad();
                loss.backward();

                // Gradient clipping
                let _ = self.model.vs.variables().iter().for_each(|(_, t)| {
                    let _ = t.grad().clamp_(-self.config.max_grad_norm, self.config.max_grad_norm);
                });

                self.optimizer.step();

                total_policy_loss += policy_loss.double_value(&[]) as f32;
                total_value_loss += value_loss.double_value(&[]) as f32;
                total_entropy += entropy.double_value(&[]) as f32;
            }

            let epochs = self.config.ppo_epochs as f32;
            (
                total_policy_loss / epochs,
                total_value_loss / epochs,
                total_entropy / epochs,
            )
        }

        /// Save trainer state
        pub fn save(&self, path: &str) -> Result<(), tch::TchError> {
            self.model.save(path)
        }

        /// Load trainer state
        pub fn load(&mut self, path: &str) -> Result<(), tch::TchError> {
            match self.model.vs.load_partial(path) {
                Ok(missing) => {
                    if !missing.is_empty() {
                        tracing::warn!(
                            "[RL] Partial load: {} missing tensors in {}",
                            missing.len(),
                            path
                        );
                        tracing::debug!("[RL] Missing tensors: {:?}", missing);
                    }
                    Ok(())
                }
                Err(e) => Err(e),
            }
        }
    }
}

// ============================================================================
// Stub implementation (when "rl" feature is disabled)
// ============================================================================

#[cfg(not(feature = "rl"))]
pub mod stub_impl {
    use super::*;

    /// Stub actor-critic (no-op when RL feature disabled)
    pub struct ActorCritic;

    impl ActorCritic {
        pub fn new() -> Self {
            Self
        }

        pub fn select_action(&self, _obs: &Observation) -> (Action, f32, f32) {
            // Random action for testing without RL
            (Action::Hold, 0.0, 0.0)
        }

        pub fn select_action_with_history(
            &self,
            _obs: &Observation,
            _history: &[f32],
        ) -> (Action, f32, f32) {
            (Action::Hold, 0.0, 0.0)
        }

        pub fn select_action_with_probs(
            &self,
            _obs: &Observation,
            _history: Option<&[f32]>,
        ) -> (Action, f32, f32, [f32; 7]) {
            (Action::Hold, 0.0, 0.0, [1.0/7.0; 7])
        }

        pub fn get_value(&self, _obs: &Observation) -> f32 {
            0.0
        }

        pub fn get_value_with_history(&self, _obs: &Observation, _history: &[f32]) -> f32 {
            0.0
        }
    }

    impl Default for ActorCritic {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Stub PPO trainer
    pub struct PpoTrainer {
        pub model: ActorCritic,
    }

    impl PpoTrainer {
        pub fn new(_config: PpoConfig) -> Self {
            eprintln!("WARNING: RL feature not enabled. PPO training disabled.");
            eprintln!("To enable, install libtorch and build with: cargo build --features rl");
            Self {
                model: ActorCritic::new(),
            }
        }

        pub fn update(
            &mut self,
            _observations: &[&Observation],
            _actions: &[u8],
            _old_log_probs: &[f32],
            _advantages: &[f32],
            _returns: &[f32],
        ) -> (f32, f32, f32) {
            (0.0, 0.0, 0.0)
        }

        pub fn update_with_history(
            &mut self,
            _observations: &[&Observation],
            _histories: &[Vec<f32>],
            _actions: &[u8],
            _old_log_probs: &[f32],
            _advantages: &[f32],
            _returns: &[f32],
        ) -> (f32, f32, f32) {
            (0.0, 0.0, 0.0)
        }

        pub fn save(&self, _path: &str) -> Result<(), std::io::Error> {
            // Stub: no-op when RL feature disabled
            Ok(())
        }

        pub fn load(&mut self, _path: &str) -> Result<(), std::io::Error> {
            // Stub: no-op when RL feature disabled
            Ok(())
        }
    }
}

// Re-export the appropriate implementation
#[cfg(feature = "rl")]
pub use torch_impl::{ActorCritic, PpoTrainer};

#[cfg(not(feature = "rl"))]
pub use stub_impl::{ActorCritic, PpoTrainer};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_conversion() {
        assert_eq!(Action::from(0u8), Action::Hold);
        assert_eq!(Action::from(1u8), Action::BuyYesSmall);
        assert_eq!(Action::from(2u8), Action::BuyYesMed);
        assert_eq!(Action::from(3u8), Action::BuyYesLarge);
        assert_eq!(Action::from(4u8), Action::BuyNoSmall);
        assert_eq!(Action::from(5u8), Action::BuyNoMed);
        assert_eq!(Action::from(6u8), Action::BuyNoLarge);
        assert_eq!(Action::from(99u8), Action::Hold); // Invalid -> Hold
    }

    #[test]
    fn test_action_size_multiplier() {
        assert_eq!(Action::Hold.size_multiplier(), 0.0);
        assert!((Action::BuyYesSmall.size_multiplier() - 0.33).abs() < 0.01);
        assert!((Action::BuyYesMed.size_multiplier() - 0.66).abs() < 0.01);
        assert_eq!(Action::BuyYesLarge.size_multiplier(), 1.0);
        assert!((Action::BuyNoSmall.size_multiplier() - 0.33).abs() < 0.01);
        assert!((Action::BuyNoMed.size_multiplier() - 0.66).abs() < 0.01);
        assert_eq!(Action::BuyNoLarge.size_multiplier(), 1.0);
    }

    #[test]
    fn test_action_is_buy() {
        assert!(!Action::Hold.is_buy_yes());
        assert!(!Action::Hold.is_buy_no());

        assert!(Action::BuyYesSmall.is_buy_yes());
        assert!(Action::BuyYesMed.is_buy_yes());
        assert!(Action::BuyYesLarge.is_buy_yes());
        assert!(!Action::BuyYesSmall.is_buy_no());

        assert!(Action::BuyNoSmall.is_buy_no());
        assert!(Action::BuyNoMed.is_buy_no());
        assert!(Action::BuyNoLarge.is_buy_no());
        assert!(!Action::BuyNoSmall.is_buy_yes());
    }

    #[test]
    fn test_config_default() {
        let config = PpoConfig::default();
        assert!((config.lr_actor - 1e-4).abs() < 1e-10);
        assert!((config.lr_critic - 3e-4).abs() < 1e-10);
        assert!((config.gamma - 0.95).abs() < 0.01);  // Phase 5: 0.95
        assert_eq!(config.history_len, 5);  // Phase 5: 5 past states
        assert_eq!(config.temporal_dim, 32);  // Phase 5: 32-dim temporal
    }
}
