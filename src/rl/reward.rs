//! Reward Computation for PPO Training
//!
//! Implements share-based PnL reward from cross-market-state-fusion:
//! pnl = (exit_price - entry_price) × shares
//! where shares = dollars_invested / entry_price
//!
//! This amplifies gains from low-probability entries, matching
//! actual binary market economics.

/// Compute share-based PnL reward
///
/// # Arguments
/// * `entry_price` - Price paid per share (e.g., 0.40 = 40 cents)
/// * `exit_price` - Final settlement price (1.0 if won, 0.0 if lost)
/// * `dollars` - Total dollars invested
///
/// # Returns
/// Share-based PnL in dollars
///
/// # Example
/// ```
/// // Buy YES at 40 cents with $10
/// // shares = 10 / 0.40 = 25 shares
/// // If win: pnl = (1.0 - 0.40) × 25 = $15 profit
/// // If lose: pnl = (0.0 - 0.40) × 25 = -$10 loss
/// let pnl = compute_share_reward(0.40, 1.0, 10.0);
/// assert!((pnl - 15.0).abs() < 0.01);
/// ```
pub fn compute_share_reward(entry_price: f64, exit_price: f64, dollars: f64) -> f64 {
    if entry_price <= 0.0 || dollars <= 0.0 {
        return 0.0;
    }

    let shares = dollars / entry_price;
    let pnl = (exit_price - entry_price) * shares;
    pnl
}

/// Compute reward for a binary market position
///
/// # Arguments
/// * `side` - "YES" or "NO"
/// * `entry_price_cents` - Price in cents (0-100)
/// * `dollars` - Dollars invested
/// * `outcome` - true if YES won, false if NO won
pub fn compute_binary_reward(
    side: &str,
    entry_price_cents: i64,
    dollars: f64,
    outcome: bool,
) -> f64 {
    let entry_price = entry_price_cents as f64 / 100.0;

    // Determine exit price based on side and outcome
    let exit_price = match (side, outcome) {
        ("YES", true) | ("NO", false) => 1.0,  // Won
        ("YES", false) | ("NO", true) => 0.0,  // Lost
        _ => return 0.0,
    };

    compute_share_reward(entry_price, exit_price, dollars)
}

/// Compute normalized reward for RL training
///
/// Scales the raw PnL to a reasonable range for stable gradients.
/// Typical range: [-1, 1] for most trades
pub fn compute_normalized_reward(
    entry_price: f64,
    exit_price: f64,
    dollars: f64,
    scale: f64,
) -> f32 {
    let raw_pnl = compute_share_reward(entry_price, exit_price, dollars);
    // Scale to roughly [-1, 1]
    // Using tanh for bounded output
    (raw_pnl / scale).tanh() as f32
}

/// Compute shaped reward with intermediate signals
///
/// Adds shaping terms to provide learning signal during position holding:
/// - Penalty for unmatched inventory (risk)
/// - Bonus for matched pairs (locked profit)
/// - Small time penalty (encourage action)
pub fn compute_shaped_reward(
    raw_pnl: f64,
    unmatched_inventory: f64,
    matched_pairs: f64,
    time_held_mins: f64,
    scale: f64,
) -> f32 {
    let mut reward = raw_pnl / scale;

    // Inventory penalty: unmatched positions are risky
    reward -= unmatched_inventory * 0.001;

    // Matched bonus: locked profits are good
    reward += matched_pairs * 0.01;

    // Small time penalty: encourage action over inaction
    reward -= time_held_mins * 0.0001;

    reward.clamp(-2.0, 2.0) as f32
}

/// Compute spread-adjusted share-based PnL reward (LIVE MODE)
///
/// Subtracts the actual spread crossing cost from the base PnL to discourage
/// trading when spreads are wide. Uses absolute spread (ask - bid) for
/// accurate live trading cost calculation.
///
/// # Arguments
/// * `entry_price` - Price paid per share (e.g., 0.40 = 40 cents)
/// * `exit_price` - Final settlement price (1.0 if won, 0.0 if lost)
/// * `dollars` - Total dollars invested
/// * `spread` - Absolute spread = ask - bid (e.g., 0.10 for 10 cent spread)
///
/// # Returns
/// Share-based PnL minus spread crossing cost in dollars
///
/// # Example
/// ```
/// // Buy YES at 40 cents with $10, spread is 10 cents (bid=0.35, ask=0.45)
/// // shares = 10 / 0.40 = 25 shares
/// // Base PnL if win: $15
/// // Spread cost: 0.10 * 25 = $2.50
/// // Net reward: $15 - $2.50 = $12.50
/// let pnl = compute_spread_adjusted_reward(0.40, 1.0, 10.0, 0.10);
/// assert!((pnl - 12.50).abs() < 0.01);
/// ```
pub fn compute_spread_adjusted_reward(
    entry_price: f64,
    exit_price: f64,
    dollars: f64,
    spread: f64,  // Absolute spread: ask - bid
) -> f64 {
    let base_pnl = compute_share_reward(entry_price, exit_price, dollars);

    if entry_price <= 0.0 {
        return base_pnl;
    }

    // Spread cost = spread per share * number of shares
    // shares = dollars / entry_price
    // spread_cost = spread * (dollars / entry_price)
    let shares = dollars / entry_price;
    let spread_cost = spread * shares;

    base_pnl - spread_cost
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_share_reward_win() {
        // Buy at 40 cents, win
        let pnl = compute_share_reward(0.40, 1.0, 10.0);
        // shares = 25, pnl = 0.60 * 25 = 15
        assert!((pnl - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_share_reward_loss() {
        // Buy at 40 cents, lose
        let pnl = compute_share_reward(0.40, 0.0, 10.0);
        // shares = 25, pnl = -0.40 * 25 = -10
        assert!((pnl - (-10.0)).abs() < 0.01);
    }

    #[test]
    fn test_share_reward_high_prob() {
        // Buy at 80 cents, win
        let pnl = compute_share_reward(0.80, 1.0, 10.0);
        // shares = 12.5, pnl = 0.20 * 12.5 = 2.5
        assert!((pnl - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_binary_reward() {
        // YES side, YES won
        let pnl = compute_binary_reward("YES", 45, 10.0, true);
        assert!(pnl > 0.0);

        // YES side, NO won (YES lost)
        let pnl = compute_binary_reward("YES", 45, 10.0, false);
        assert!(pnl < 0.0);

        // NO side, NO won
        let pnl = compute_binary_reward("NO", 55, 10.0, false);
        assert!(pnl > 0.0);
    }

    #[test]
    fn test_low_prob_amplification() {
        // Key insight from cross-market-state-fusion:
        // Low probability entries should have amplified returns

        // $10 at 30 cents (low prob)
        let low_prob_win = compute_share_reward(0.30, 1.0, 10.0);
        // shares = 33.33, pnl = 0.70 * 33.33 = 23.33

        // $10 at 70 cents (high prob)
        let high_prob_win = compute_share_reward(0.70, 1.0, 10.0);
        // shares = 14.29, pnl = 0.30 * 14.29 = 4.29

        // Low prob entry should have much higher return
        assert!(low_prob_win > high_prob_win * 4.0);
    }

    #[test]
    fn test_spread_adjusted_reward() {
        // Buy at 40 cents, win, with 10 cent absolute spread (bid=0.35, ask=0.45)
        // Base PnL: shares = 25, pnl = 0.60 * 25 = $15
        // Spread cost: 0.10 * 25 = $2.50
        // Net: $15 - $2.50 = $12.50
        let pnl = compute_spread_adjusted_reward(0.40, 1.0, 10.0, 0.10);
        assert!((pnl - 12.50).abs() < 0.01);
    }

    #[test]
    fn test_spread_adjusted_reward_tight_spread() {
        // Buy at 50 cents, win, with 2 cent spread (tight!)
        // Base PnL: shares = 20, pnl = 0.50 * 20 = $10
        // Spread cost: 0.02 * 20 = $0.40
        // Net: $10 - $0.40 = $9.60
        let pnl = compute_spread_adjusted_reward(0.50, 1.0, 10.0, 0.02);
        assert!((pnl - 9.60).abs() < 0.01);
    }

    #[test]
    fn test_spread_adjusted_reward_wide_spread() {
        // Buy at 50 cents, win, with 20 cent spread (wide!)
        // Base PnL: shares = 20, pnl = 0.50 * 20 = $10
        // Spread cost: 0.20 * 20 = $4.00
        // Net: $10 - $4.00 = $6.00
        let pnl = compute_spread_adjusted_reward(0.50, 1.0, 10.0, 0.20);
        assert!((pnl - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_spread_adjusted_loss() {
        // Buy at 40 cents, lose, with 10 cent spread
        // Base PnL: -$10
        // Spread cost: 0.10 * 25 = $2.50
        // Net: -$10 - $2.50 = -$12.50 (double whammy!)
        let pnl = compute_spread_adjusted_reward(0.40, 0.0, 10.0, 0.10);
        assert!((pnl - (-12.50)).abs() < 0.01);
    }
}
