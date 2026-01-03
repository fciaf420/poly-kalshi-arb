//! Experience Buffer for PPO Training
//!
//! Stores (observation, action, reward, value) tuples for policy gradient updates.
//! Implements Generalized Advantage Estimation (GAE) for variance reduction.

use super::feature_extractor::Observation;
use std::collections::VecDeque;

/// Single experience tuple
#[derive(Debug, Clone)]
pub struct Experience {
    pub obs: Observation,
    pub next_obs: Observation,
    pub action: u8,      // 0=Hold, 1=BuyYes, 2=BuyNo
    pub log_prob: f32,   // Log probability of action
    pub reward: f32,     // Reward received
    pub value: f32,      // Value estimate at this state
    pub done: bool,      // Episode terminated
    /// Phase 5: Stacked history (90-dim = 5 * 18) for temporal encoder
    pub history: Option<Vec<f32>>,
    /// Phase 5: Next stacked history (90-dim) for bootstrapping
    pub next_history: Option<Vec<f32>>,
}

/// Ring buffer for experience storage
pub struct ExperienceBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ExperienceBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add experience to buffer
    pub fn push(&mut self, exp: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(exp);
    }

    /// Get buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Iterate over experiences
    pub fn iter(&self) -> impl Iterator<Item = &Experience> {
        self.buffer.iter()
    }

    /// Get last experience (for bootstrap value)
    pub fn last(&self) -> Option<&Experience> {
        self.buffer.back()
    }

    /// Compute Generalized Advantage Estimation (GAE)
    ///
    /// GAE provides a trade-off between bias and variance in advantage estimation.
    /// Lambda controls this trade-off: lambda=0 is one-step TD, lambda=1 is MC.
    ///
    /// Returns: Vec of advantages, one per experience
    pub fn compute_gae(&self, gamma: f32, lambda: f32, next_value: f32) -> Vec<f32> {
        if self.buffer.is_empty() {
            return vec![];
        }

        let n = self.buffer.len();
        let mut advantages = vec![0.0f32; n];
        let mut gae = 0.0f32;

        // Iterate backwards for GAE computation
        for i in (0..n).rev() {
            let exp = &self.buffer[i];

            // Get next value (0 if terminal or last experience)
            let next_val = if exp.done {
                0.0
            } else if i + 1 >= n {
                next_value
            } else {
                self.buffer[i + 1].value
            };

            // TD error: delta = r + gamma * V(s') - V(s)
            let not_done = if exp.done { 0.0 } else { 1.0 };
            let delta = exp.reward + gamma * next_val * not_done - exp.value;

            // GAE: A_t = delta_t + gamma * lambda * A_{t+1}
            gae = delta + gamma * lambda * not_done * gae;
            advantages[i] = gae;
        }

        advantages
    }

    /// Compute returns (discounted cumulative rewards)
    pub fn compute_returns(&self, gamma: f32, next_value: f32) -> Vec<f32> {
        if self.buffer.is_empty() {
            return vec![];
        }

        let advantages = self.compute_gae(gamma, 1.0, next_value); // lambda=1 for MC returns

        // Returns = advantages + values
        advantages
            .iter()
            .zip(self.buffer.iter())
            .map(|(adv, exp)| adv + exp.value)
            .collect()
    }

    /// Get batch data for training
    /// Returns (observations, actions, log_probs, advantages, returns)
    pub fn get_batch(
        &self,
        gamma: f32,
        lambda: f32,
        next_value: f32,
    ) -> (Vec<&Observation>, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let advantages = self.compute_gae(gamma, lambda, next_value);

        // Normalize advantages (important for stable training)
        let mean = advantages.iter().sum::<f32>() / advantages.len().max(1) as f32;
        let variance = advantages
            .iter()
            .map(|a| (a - mean).powi(2))
            .sum::<f32>()
            / advantages.len().max(1) as f32;
        let std = (variance + 1e-8).sqrt();
        let normalized_advantages: Vec<f32> = advantages.iter().map(|a| (a - mean) / std).collect();

        let returns: Vec<f32> = advantages
            .iter()
            .zip(self.buffer.iter())
            .map(|(adv, exp)| adv + exp.value)
            .collect();

        let observations: Vec<&Observation> = self.buffer.iter().map(|e| &e.obs).collect();
        let actions: Vec<u8> = self.buffer.iter().map(|e| e.action).collect();
        let log_probs: Vec<f32> = self.buffer.iter().map(|e| e.log_prob).collect();

        (observations, actions, log_probs, normalized_advantages, returns)
    }

    /// Phase 5: Get batch data with history for temporal encoder
    /// Returns (observations, histories, actions, log_probs, advantages, returns)
    pub fn get_batch_with_history(
        &self,
        gamma: f32,
        lambda: f32,
        next_value: f32,
    ) -> (Vec<&Observation>, Vec<Vec<f32>>, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let advantages = self.compute_gae(gamma, lambda, next_value);

        // Normalize advantages (important for stable training)
        let mean = advantages.iter().sum::<f32>() / advantages.len().max(1) as f32;
        let variance = advantages
            .iter()
            .map(|a| (a - mean).powi(2))
            .sum::<f32>()
            / advantages.len().max(1) as f32;
        let std = (variance + 1e-8).sqrt();
        let normalized_advantages: Vec<f32> = advantages.iter().map(|a| (a - mean) / std).collect();

        let returns: Vec<f32> = advantages
            .iter()
            .zip(self.buffer.iter())
            .map(|(adv, exp)| adv + exp.value)
            .collect();

        let observations: Vec<&Observation> = self.buffer.iter().map(|e| &e.obs).collect();

        // Get histories, defaulting to zeros if not present
        let histories: Vec<Vec<f32>> = self.buffer
            .iter()
            .map(|e| e.history.clone().unwrap_or_else(|| vec![0.0; 90]))
            .collect();

        let actions: Vec<u8> = self.buffer.iter().map(|e| e.action).collect();
        let log_probs: Vec<f32> = self.buffer.iter().map(|e| e.log_prob).collect();

        (observations, histories, actions, log_probs, normalized_advantages, returns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exp(reward: f32, value: f32, done: bool) -> Experience {
        Experience {
            obs: Observation::default(),
            next_obs: Observation::default(),
            action: 0,
            log_prob: -0.5,
            reward,
            value,
            done,
            history: None,  // Phase 5: Optional history
            next_history: None,
        }
    }

    #[test]
    fn test_buffer_capacity() {
        let mut buffer = ExperienceBuffer::new(3);
        buffer.push(make_exp(1.0, 0.5, false));
        buffer.push(make_exp(2.0, 0.6, false));
        buffer.push(make_exp(3.0, 0.7, false));
        assert_eq!(buffer.len(), 3);

        buffer.push(make_exp(4.0, 0.8, false));
        assert_eq!(buffer.len(), 3);

        // First experience should be evicted
        let first_reward = buffer.iter().next().unwrap().reward;
        assert_eq!(first_reward, 2.0);
    }

    #[test]
    fn test_gae_computation() {
        let mut buffer = ExperienceBuffer::new(10);
        buffer.push(make_exp(1.0, 0.5, false));
        buffer.push(make_exp(2.0, 0.6, false));
        buffer.push(make_exp(3.0, 0.7, true)); // Terminal

        let advantages = buffer.compute_gae(0.99, 0.95, 0.0);
        assert_eq!(advantages.len(), 3);

        // Last advantage should be: r - V = 3.0 - 0.7 = 2.3
        assert!((advantages[2] - 2.3).abs() < 0.01);
    }
}
