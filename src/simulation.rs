// src/simulation.rs
//! Simulation utilities for realistic RL training
//!
//! Provides:
//! - Book-walking slippage model (walks orderbook depth)
//! - Latency compensation (predictive price adjustment)
//! - Drawdown stops (trailing equity protection)

use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Instant;
use tracing::{info, warn};

// =============================================================================
// BOOK-WALKING SLIPPAGE MODEL
// =============================================================================

/// Orderbook level: (price, size)
/// - price: in dollars (0.0 - 1.0 for prediction markets)
/// - size: number of contracts available at this price
pub type OrderbookLevel = (f64, f64);

/// Result of walking the orderbook
#[derive(Debug, Clone)]
pub struct BookWalkResult {
    /// Average fill price (weighted by size at each level)
    pub avg_price: f64,
    /// Total contracts filled
    pub filled_size: f64,
    /// Number of levels walked
    pub levels_walked: usize,
    /// Slippage from best price in basis points
    pub slippage_bps: f64,
    /// Whether the order was fully filled
    pub fully_filled: bool,
}

/// Walk the orderbook to simulate executing N contracts
///
/// For BUY orders, walks up the ask side (ascending prices)
/// For SELL orders, walks down the bid side (descending prices)
///
/// # Arguments
/// * `orderbook` - Sorted orderbook levels [(price, size), ...]
///   - For buys: asks sorted low to high
///   - For sells: bids sorted high to low
/// * `size` - Number of contracts to execute
/// * `is_buy` - True for buy orders, false for sell orders
///
/// # Returns
/// BookWalkResult with average price, filled size, and slippage
pub fn walk_orderbook(orderbook: &[(f64, f64)], size: f64, is_buy: bool) -> BookWalkResult {
    if orderbook.is_empty() || size <= 0.0 {
        return BookWalkResult {
            avg_price: 0.0,
            filled_size: 0.0,
            levels_walked: 0,
            slippage_bps: 0.0,
            fully_filled: false,
        };
    }

    let best_price = orderbook[0].0;
    let mut remaining = size;
    let mut total_cost = 0.0;
    let mut filled = 0.0;
    let mut levels = 0;

    for &(price, available) in orderbook.iter() {
        if remaining <= 0.0 {
            break;
        }

        let take = remaining.min(available);
        total_cost += take * price;
        filled += take;
        remaining -= take;
        levels += 1;
    }

    let avg_price = if filled > 0.0 { total_cost / filled } else { 0.0 };

    // Calculate slippage from best price
    let slippage_bps = if best_price > 0.0 && avg_price > 0.0 {
        if is_buy {
            // For buys, slippage = how much more we paid than best ask
            (avg_price - best_price) / best_price * 10000.0
        } else {
            // For sells, slippage = how much less we received than best bid
            (best_price - avg_price) / best_price * 10000.0
        }
    } else {
        0.0
    };

    BookWalkResult {
        avg_price,
        filled_size: filled,
        levels_walked: levels,
        slippage_bps,
        fully_filled: remaining <= 0.0,
    }
}

/// Simulate slippage for a given trade size against an orderbook
///
/// This is the main entry point for RL training slippage estimation
///
/// # Arguments
/// * `asks` - Ask levels sorted low to high [(price, size), ...]
/// * `bids` - Bid levels sorted high to low [(price, size), ...]
/// * `size` - Number of contracts
/// * `is_buy` - True for buy orders
///
/// # Returns
/// (expected_fill_price, slippage_bps, fully_filled)
pub fn simulate_slippage(
    asks: &[(f64, f64)],
    bids: &[(f64, f64)],
    size: f64,
    is_buy: bool,
) -> (f64, f64, bool) {
    let result = if is_buy {
        walk_orderbook(asks, size, true)
    } else {
        walk_orderbook(bids, size, false)
    };

    (result.avg_price, result.slippage_bps, result.fully_filled)
}

/// Estimate slippage using a simple model when full orderbook not available
///
/// Uses empirical parameters:
/// - Base slippage: 5 bps per 10 contracts
/// - Depth multiplier: 1.5x per 10 contracts beyond first 10
///
/// # Arguments
/// * `best_price` - Best bid or ask price
/// * `size` - Number of contracts
/// * `is_buy` - True for buy orders
///
/// # Returns
/// Estimated fill price after slippage
pub fn estimate_slippage_simple(best_price: f64, size: f64, is_buy: bool) -> f64 {
    if best_price <= 0.0 || size <= 0.0 {
        return best_price;
    }

    // Base slippage: 5 bps per 10 contracts, with 1.5x multiplier for depth
    let base_bps: f64 = 5.0;
    let depth_multiplier: f64 = 1.5;

    let tiers = (size / 10.0).ceil() as u32;
    let mut total_slippage_bps: f64 = 0.0;

    for tier in 0..tiers {
        let tier_slippage = base_bps * depth_multiplier.powi(tier as i32);
        total_slippage_bps += tier_slippage;
    }

    // Average slippage across tiers
    let avg_slippage_bps = total_slippage_bps / tiers.max(1) as f64;
    let slippage_factor = avg_slippage_bps / 10000.0;

    if is_buy {
        best_price * (1.0 + slippage_factor)
    } else {
        best_price * (1.0 - slippage_factor)
    }
}

// =============================================================================
// LATENCY COMPENSATION
// =============================================================================

/// Latency model for price drift compensation
#[derive(Debug, Clone)]
pub struct LatencyModel {
    /// Expected latency in milliseconds (signal to fill)
    pub expected_latency_ms: f64,
    /// Price drift rate (standard deviation per second)
    pub drift_rate_per_sec: f64,
    /// Adverse selection multiplier (drift tends to be against us)
    pub adverse_selection: f64,
}

impl Default for LatencyModel {
    fn default() -> Self {
        Self {
            expected_latency_ms: 150.0,  // 150ms typical round-trip
            drift_rate_per_sec: 0.01,     // 1% per second (binary options are volatile)
            adverse_selection: 1.5,       // Price tends to drift against us 1.5x
        }
    }
}

impl LatencyModel {
    /// Create a latency model with custom parameters
    pub fn new(expected_latency_ms: f64, drift_rate_per_sec: f64, adverse_selection: f64) -> Self {
        Self {
            expected_latency_ms,
            drift_rate_per_sec,
            adverse_selection,
        }
    }

    /// Compensate price for expected latency
    ///
    /// For buys: expect price to be higher by the time we fill
    /// For sells: expect price to be lower by the time we fill
    ///
    /// # Arguments
    /// * `current_price` - Current observed price
    /// * `is_buy` - True for buy orders
    ///
    /// # Returns
    /// Expected fill price after latency drift
    pub fn compensate_price(&self, current_price: f64, is_buy: bool) -> f64 {
        let latency_sec = self.expected_latency_ms / 1000.0;
        let expected_drift = self.drift_rate_per_sec * latency_sec * self.adverse_selection;

        if is_buy {
            // Price drifts up (against buyer)
            current_price * (1.0 + expected_drift)
        } else {
            // Price drifts down (against seller)
            current_price * (1.0 - expected_drift)
        }
    }

    /// Compensate price with actual measured latency
    pub fn compensate_with_latency(&self, current_price: f64, is_buy: bool, actual_latency_ms: f64) -> f64 {
        let latency_sec = actual_latency_ms / 1000.0;
        let expected_drift = self.drift_rate_per_sec * latency_sec * self.adverse_selection;

        if is_buy {
            current_price * (1.0 + expected_drift)
        } else {
            current_price * (1.0 - expected_drift)
        }
    }

    /// Get confidence interval for fill price
    ///
    /// Returns (lower_bound, expected, upper_bound) at 95% confidence
    pub fn price_confidence_interval(&self, current_price: f64, is_buy: bool) -> (f64, f64, f64) {
        let latency_sec = self.expected_latency_ms / 1000.0;
        let drift = self.drift_rate_per_sec * latency_sec;
        let adverse_drift = drift * self.adverse_selection;

        // 95% CI using 2 standard deviations
        let expected = self.compensate_price(current_price, is_buy);

        if is_buy {
            let lower = current_price * (1.0 + adverse_drift - 2.0 * drift);
            let upper = current_price * (1.0 + adverse_drift + 2.0 * drift);
            (lower, expected, upper)
        } else {
            let lower = current_price * (1.0 - adverse_drift - 2.0 * drift);
            let upper = current_price * (1.0 - adverse_drift + 2.0 * drift);
            (lower, expected, upper)
        }
    }
}

/// Combined slippage and latency adjusted price
pub fn adjust_fill_price(
    current_price: f64,
    orderbook: &[(f64, f64)],
    size: f64,
    is_buy: bool,
    latency_model: &LatencyModel,
) -> f64 {
    // Step 1: Walk orderbook for slippage
    let slippage_price = if orderbook.is_empty() {
        estimate_slippage_simple(current_price, size, is_buy)
    } else {
        let result = walk_orderbook(orderbook, size, is_buy);
        if result.filled_size > 0.0 {
            result.avg_price
        } else {
            estimate_slippage_simple(current_price, size, is_buy)
        }
    };

    // Step 2: Apply latency compensation
    latency_model.compensate_price(slippage_price, is_buy)
}

// =============================================================================
// DRAWDOWN STOPS
// =============================================================================

/// Drawdown tracker for trailing equity protection
pub struct DrawdownTracker {
    /// Peak equity value (in cents to avoid floating point issues)
    peak_equity_cents: AtomicI64,
    /// Drawdown threshold (e.g., 0.10 = 10% drawdown triggers halt)
    threshold: f64,
    /// Whether trading is halted due to drawdown
    halted: std::sync::atomic::AtomicBool,
    /// When drawdown halt was triggered
    halted_at: std::sync::RwLock<Option<Instant>>,
    /// Cooldown period before resuming (seconds)
    cooldown_secs: u64,
}

impl DrawdownTracker {
    /// Create a new drawdown tracker
    ///
    /// # Arguments
    /// * `initial_equity` - Starting equity in dollars
    /// * `threshold` - Drawdown threshold (e.g., 0.10 for 10%)
    /// * `cooldown_secs` - Seconds to wait before allowing trading after halt
    pub fn new(initial_equity: f64, threshold: f64, cooldown_secs: u64) -> Self {
        let peak_cents = (initial_equity * 100.0) as i64;
        info!("[DRAWDOWN] Tracker initialized: equity=${:.2}, threshold={:.1}%, cooldown={}s",
              initial_equity, threshold * 100.0, cooldown_secs);

        Self {
            peak_equity_cents: AtomicI64::new(peak_cents),
            threshold,
            halted: std::sync::atomic::AtomicBool::new(false),
            halted_at: std::sync::RwLock::new(None),
            cooldown_secs,
        }
    }

    /// Create with default parameters (10% drawdown, 5 minute cooldown)
    pub fn default_with_equity(initial_equity: f64) -> Self {
        Self::new(initial_equity, 0.10, 300)
    }

    /// Update equity and check for drawdown
    ///
    /// # Returns
    /// True if trading is allowed, false if halted due to drawdown
    pub fn update_equity(&self, current_equity: f64) -> bool {
        let current_cents = (current_equity * 100.0) as i64;

        // Update peak if we have a new high
        let peak = self.peak_equity_cents.fetch_max(current_cents, Ordering::SeqCst);
        let actual_peak = peak.max(current_cents);

        // Calculate drawdown
        let drawdown = if actual_peak > 0 {
            (actual_peak - current_cents) as f64 / actual_peak as f64
        } else {
            0.0
        };

        // Check if we hit the threshold
        if drawdown >= self.threshold && !self.halted.load(Ordering::SeqCst) {
            warn!("🛑 [DRAWDOWN] HALT: equity=${:.2} peak=${:.2} drawdown={:.1}% >= threshold={:.1}%",
                  current_equity, actual_peak as f64 / 100.0, drawdown * 100.0, self.threshold * 100.0);
            self.halted.store(true, Ordering::SeqCst);
            *self.halted_at.write().unwrap() = Some(Instant::now());
            return false;
        }

        // Check if cooldown has elapsed
        if self.halted.load(Ordering::SeqCst) {
            if let Some(halted_time) = *self.halted_at.read().unwrap() {
                if halted_time.elapsed().as_secs() >= self.cooldown_secs {
                    info!("[DRAWDOWN] Cooldown elapsed, resuming trading");
                    self.halted.store(false, Ordering::SeqCst);
                    *self.halted_at.write().unwrap() = None;
                    // Reset peak to current equity on resume
                    self.peak_equity_cents.store(current_cents, Ordering::SeqCst);
                    return true;
                }
            }
            return false;
        }

        true
    }

    /// Check if trading is currently halted
    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::SeqCst)
    }

    /// Get current drawdown percentage
    pub fn current_drawdown(&self, current_equity: f64) -> f64 {
        let current_cents = (current_equity * 100.0) as i64;
        let peak = self.peak_equity_cents.load(Ordering::SeqCst);

        if peak > 0 {
            (peak - current_cents).max(0) as f64 / peak as f64
        } else {
            0.0
        }
    }

    /// Get peak equity
    pub fn peak_equity(&self) -> f64 {
        self.peak_equity_cents.load(Ordering::SeqCst) as f64 / 100.0
    }

    /// Reset the tracker with new equity
    pub fn reset(&self, new_equity: f64) {
        let new_cents = (new_equity * 100.0) as i64;
        self.peak_equity_cents.store(new_cents, Ordering::SeqCst);
        self.halted.store(false, Ordering::SeqCst);
        *self.halted_at.write().unwrap() = None;
        info!("[DRAWDOWN] Tracker reset: new equity=${:.2}", new_equity);
    }

    /// Get status summary
    pub fn status(&self, current_equity: f64) -> DrawdownStatus {
        DrawdownStatus {
            current_equity,
            peak_equity: self.peak_equity(),
            drawdown_pct: self.current_drawdown(current_equity) * 100.0,
            threshold_pct: self.threshold * 100.0,
            halted: self.is_halted(),
            time_until_resume: if self.is_halted() {
                self.halted_at.read().unwrap()
                    .map(|t| self.cooldown_secs.saturating_sub(t.elapsed().as_secs()))
            } else {
                None
            },
        }
    }
}

/// Drawdown status for reporting
#[derive(Debug, Clone)]
pub struct DrawdownStatus {
    pub current_equity: f64,
    pub peak_equity: f64,
    pub drawdown_pct: f64,
    pub threshold_pct: f64,
    pub halted: bool,
    pub time_until_resume: Option<u64>,
}

impl std::fmt::Display for DrawdownStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.halted {
            let resume = self.time_until_resume.unwrap_or(0);
            write!(f, "🛑 HALTED | equity=${:.2} peak=${:.2} dd={:.1}%/{:.1}% | resume in {}s",
                   self.current_equity, self.peak_equity, self.drawdown_pct,
                   self.threshold_pct, resume)
        } else {
            write!(f, "✅ OK | equity=${:.2} peak=${:.2} dd={:.1}%/{:.1}%",
                   self.current_equity, self.peak_equity, self.drawdown_pct,
                   self.threshold_pct)
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_orderbook_single_level() {
        let asks = vec![(0.50, 100.0)];
        let result = walk_orderbook(&asks, 10.0, true);

        assert_eq!(result.avg_price, 0.50);
        assert_eq!(result.filled_size, 10.0);
        assert_eq!(result.levels_walked, 1);
        assert_eq!(result.slippage_bps, 0.0);
        assert!(result.fully_filled);
    }

    #[test]
    fn test_walk_orderbook_multiple_levels() {
        // Walk through 3 levels with different prices
        let asks = vec![
            (0.50, 10.0),  // 10 @ 50¢
            (0.51, 10.0),  // 10 @ 51¢
            (0.52, 10.0),  // 10 @ 52¢
        ];

        let result = walk_orderbook(&asks, 25.0, true);

        // Should walk all 3 levels: 10@50 + 10@51 + 5@52 = 25 contracts
        assert_eq!(result.filled_size, 25.0);
        assert_eq!(result.levels_walked, 3);
        assert!(result.fully_filled);

        // Average price: (10*0.50 + 10*0.51 + 5*0.52) / 25 = 12.7/25 = 0.508
        assert!((result.avg_price - 0.508).abs() < 0.001);

        // Slippage from 50¢ to 50.8¢ = 1.6% = 160 bps
        assert!((result.slippage_bps - 160.0).abs() < 1.0);
    }

    #[test]
    fn test_walk_orderbook_partial_fill() {
        let asks = vec![(0.50, 10.0)];
        let result = walk_orderbook(&asks, 20.0, true);

        assert_eq!(result.filled_size, 10.0);
        assert!(!result.fully_filled);
    }

    #[test]
    fn test_latency_compensation() {
        let model = LatencyModel::default();

        // Buy at 50¢ - should expect to pay more due to adverse selection
        let compensated = model.compensate_price(0.50, true);
        assert!(compensated > 0.50);

        // Sell at 50¢ - should expect to receive less
        let compensated = model.compensate_price(0.50, false);
        assert!(compensated < 0.50);
    }

    #[test]
    fn test_drawdown_tracker() {
        let tracker = DrawdownTracker::new(1000.0, 0.10, 60);

        // Initial equity - should be allowed
        assert!(tracker.update_equity(1000.0));
        assert!(!tracker.is_halted());

        // New high - should be allowed
        assert!(tracker.update_equity(1100.0));
        assert_eq!(tracker.peak_equity(), 1100.0);

        // Small drawdown (5%) - should be allowed
        assert!(tracker.update_equity(1045.0));
        assert!(!tracker.is_halted());

        // Large drawdown (15%) - should halt
        assert!(!tracker.update_equity(935.0));
        assert!(tracker.is_halted());
    }

    #[test]
    fn test_estimate_slippage_simple() {
        // Small order - minimal slippage
        let fill = estimate_slippage_simple(0.50, 5.0, true);
        assert!(fill > 0.50);
        assert!(fill < 0.51); // Less than 2% slippage for small order

        // Large order - more slippage
        let fill_large = estimate_slippage_simple(0.50, 50.0, true);
        assert!(fill_large > fill); // Larger order = more slippage
    }
}
