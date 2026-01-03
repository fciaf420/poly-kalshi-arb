//! Feature Extractor for PPO Agent
//!
//! Builds an 18-dimensional observation vector matching Python cross-market-state-fusion.
//!
//! Feature Layout (matches Python strategies/base.py + run.py):
//! [0-2]   Momentum: returns_1m * 100, returns_5m * 100, returns_10m * 100          (Binance)
//! [3-4]   Orderbook: order_book_imbalance_l1, order_book_imbalance_l5              (Polymarket CLOB)
//! [5-6]   Trade Flow: trade_flow_imbalance, cvd_acceleration                       (Binance)
//! [7]     Spread: spread_pct * 100                                                  (Polymarket CLOB)
//! [8-9]   Microstructure: trade_intensity, large_trade_flag                        (Binance)
//! [10]    Volatility: vol_5m                                                        (Polymarket prob_history)
//! [11]    Volatility: vol_expansion                                                 (Binance)
//! [12-15] Position: has_position, position_side, position_pnl, time_remaining      (Internal)
//! [16-17] Regime: vol_regime, trend_regime                                         (Binance)

use serde::{Deserialize, Serialize};

/// 18-dimensional observation vector for PPO agent
/// Matches Python cross-market-state-fusion layout exactly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub features: [f32; 18],
}

impl Default for Observation {
    fn default() -> Self {
        Self { features: [0.0; 18] }
    }
}

impl Observation {
    /// Convert to tensor-compatible slice
    pub fn as_slice(&self) -> &[f32] {
        &self.features
    }

    /// Get feature by index with name for debugging
    pub fn feature_name(index: usize) -> &'static str {
        match index {
            // Momentum [0-2]
            0 => "returns_1m",
            1 => "returns_5m",
            2 => "returns_10m",
            // Order Flow [3-6]
            3 => "order_book_imbalance_l1",
            4 => "order_book_imbalance_l5",
            5 => "trade_flow_imbalance",
            6 => "cvd_acceleration",
            // Microstructure [7-9]
            7 => "spread_pct",
            8 => "trade_intensity",
            9 => "large_trade_flag",
            // Volatility [10-11]
            10 => "volatility",
            11 => "vol_expansion",
            // Position [12-15]
            12 => "has_position",
            13 => "position_side",
            14 => "position_pnl",
            15 => "time_remaining",
            // Regime [16-17]
            16 => "vol_regime",
            17 => "trend_regime",
            _ => "unknown",
        }
    }

    /// Print all features for debugging
    pub fn debug_print(&self) {
        for (i, &f) in self.features.iter().enumerate() {
            println!("[{:2}] {:25} = {:.4}", i, Self::feature_name(i), f);
        }
    }
}

/// Metrics from Binance futures price feed
/// Features: momentum, trade flow, microstructure (intensity/large_trade), vol_expansion, regimes
#[derive(Debug, Clone, Default)]
pub struct BinanceMetrics {
    // [0-2] Momentum
    pub return_1m: Option<f64>,
    pub return_5m: Option<f64>,
    pub return_10m: Option<f64>,

    // [5-6] Trade flow (from Binance futures)
    pub trade_flow_imbalance: f64,
    pub cvd_acceleration: f64,

    // [8-9] Microstructure (from Binance futures trades)
    pub trade_intensity: f64,
    pub large_trade: bool,

    // [11] Volatility expansion (from Binance)
    pub vol_expansion: f64,

    // [16-17] Regime (derived from Binance)
    pub vol_regime: f64,
    pub trend_regime: f64,
}

/// Metrics from Polymarket CLOB orderbook
/// Features: orderbook imbalance, spread, vol_5m (from prob history)
#[derive(Debug, Clone, Default)]
pub struct PolymarketMetrics {
    // [3-4] Orderbook imbalance (from Polymarket CLOB)
    pub order_book_imbalance_l1: f64,  // (bid_vol - ask_vol) / total at L1
    pub order_book_imbalance_l5: f64,  // Same for top 5 levels

    // [7] Spread (from Polymarket CLOB)
    pub spread_pct: f64,  // (ask - bid) / mid

    // [10] Vol 5m (std of prob_history from Polymarket)
    pub vol_5m: f64,
}

/// Build observation vector matching Python's 18-dim layout exactly
///
/// Python source: cross-market-state-fusion/strategies/base.py lines 84-125
/// Data sources verified from run.py lines 297-352
pub fn build_observation(
    binance: &BinanceMetrics,       // Momentum, trade flow, intensity, large_trade, vol_expansion, regimes
    polymarket: &PolymarketMetrics, // Orderbook imbalance, spread, vol_5m
    has_position: bool,             // Currently holding a position
    position_side: Option<&str>,    // "YES"/"UP" or "NO"/"DOWN"
    position_pnl: f64,              // Unrealized P&L (normalized to ~[-1, 1])
    time_remaining: f64,            // Fraction of market duration left (0-1)
) -> Observation {
    let mut f = [0.0f32; 18];

    // [0-2] Momentum - FROM BINANCE
    // Python: returns_1m * 100, returns_5m * 100, returns_10m * 100
    f[0] = (binance.return_1m.unwrap_or(0.0) * 100.0) as f32;
    f[1] = (binance.return_5m.unwrap_or(0.0) * 100.0) as f32;
    f[2] = (binance.return_10m.unwrap_or(0.0) * 100.0) as f32;

    // [3-4] Orderbook imbalance - FROM POLYMARKET CLOB
    // Python run.py lines 309-319: computed from Polymarket orderbook
    f[3] = polymarket.order_book_imbalance_l1 as f32;
    f[4] = polymarket.order_book_imbalance_l5 as f32;

    // [5-6] Trade flow - FROM BINANCE
    // Python run.py line 335: futures.trade_flow_imbalance
    // Python run.py line 334: cvd acceleration from futures.cvd
    f[5] = binance.trade_flow_imbalance as f32;
    f[6] = binance.cvd_acceleration as f32;

    // [7] Spread - FROM POLYMARKET CLOB
    // Python base.py line 90: spread_pct = self.spread / max(0.01, self.prob)
    f[7] = (polymarket.spread_pct * 100.0) as f32;

    // [8-9] Microstructure - FROM BINANCE
    // Python run.py lines 343-344: from futures.trade_intensity, futures.large_trade_flag
    f[8] = binance.trade_intensity as f32;
    f[9] = if binance.large_trade { 1.0 } else { 0.0 };

    // [10] Vol 5m - FROM POLYMARKET prob_history
    // Python base.py line 87: vol_5m = self._volatility(30) using prob_history
    f[10] = polymarket.vol_5m as f32;

    // [11] Vol expansion - FROM BINANCE
    // Python run.py line 348: futures.vol_ratio - 1.0
    f[11] = binance.vol_expansion as f32;

    // [12-15] Position state - INTERNAL
    f[12] = if has_position { 1.0 } else { 0.0 };
    f[13] = match position_side {
        Some("YES") | Some("UP") => 1.0,
        Some("NO") | Some("DOWN") => -1.0,
        _ => 0.0,
    };
    f[14] = position_pnl as f32;
    f[15] = time_remaining as f32;

    // [16-17] Regime - FROM BINANCE (derived)
    // Python run.py lines 351-352
    f[16] = binance.vol_regime as f32;
    f[17] = binance.trend_regime as f32;

    Observation { features: f }
}

/// Helper to create BinanceMetrics from price_feed's AssetMetrics JSON
/// This is used when receiving metrics over WebSocket
pub fn binance_metrics_from_json(json: &serde_json::Value) -> Option<BinanceMetrics> {
    Some(BinanceMetrics {
        return_1m: json.get("return_1m").and_then(|v| v.as_f64()),
        return_5m: json.get("return_5m").and_then(|v| v.as_f64()),
        return_10m: json.get("return_10m").and_then(|v| v.as_f64()),
        trade_flow_imbalance: json.get("trade_flow_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0),
        cvd_acceleration: json.get("cvd_acceleration").and_then(|v| v.as_f64()).unwrap_or(0.0),
        trade_intensity: json.get("trade_intensity").and_then(|v| v.as_f64()).unwrap_or(0.0),
        large_trade: json.get("large_trade").and_then(|v| v.as_bool()).unwrap_or(false),
        vol_expansion: json.get("vol_expansion").and_then(|v| v.as_f64()).unwrap_or(1.0),
        vol_regime: json.get("vol_regime").and_then(|v| v.as_f64()).unwrap_or(0.0),
        trend_regime: json.get("trend_regime").and_then(|v| v.as_f64()).unwrap_or(0.0),
    })
}

/// Backward compatibility alias
#[deprecated(note = "Use binance_metrics_from_json instead")]
pub fn metrics_from_json(json: &serde_json::Value) -> Option<BinanceMetrics> {
    binance_metrics_from_json(json)
}

/// Calculate vol_5m from probability history (std deviation of last ~30 values)
/// Matches Python base.py line 133-138: _volatility(window=30)
pub fn calculate_vol_5m(prob_history: &[f64], window: usize) -> f64 {
    if prob_history.len() < window {
        return 0.0;
    }
    let recent = &prob_history[prob_history.len().saturating_sub(window)..];
    if recent.is_empty() {
        return 0.0;
    }
    let mean = recent.iter().sum::<f64>() / recent.len() as f64;
    let variance = recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64;
    variance.sqrt()
}

/// Calculate orderbook imbalance from bid/ask volumes
/// Formula: (bid_vol - ask_vol) / (bid_vol + ask_vol)
/// Returns 0.0 if total volume is 0
pub fn calculate_orderbook_imbalance(bid_vol: f64, ask_vol: f64) -> f64 {
    let total = bid_vol + ask_vol;
    if total <= 0.0 {
        return 0.0;
    }
    (bid_vol - ask_vol) / total
}

/// Calculate spread percentage from bid/ask prices
/// Formula: (ask - bid) / mid where mid = (ask + bid) / 2
pub fn calculate_spread_pct(best_bid: f64, best_ask: f64) -> f64 {
    if best_bid <= 0.0 || best_ask <= 0.0 {
        return 0.0;
    }
    let mid = (best_ask + best_bid) / 2.0;
    if mid <= 0.0 {
        return 0.0;
    }
    (best_ask - best_bid) / mid
}

/// Phase 5: Normalize all features to [-1, 1] range with appropriate scaling
///
/// This matches Python cross-market-state-fusion strategies/base.py to_features()
/// Scaling factors are calibrated for stable training across market regimes:
/// - Returns: ×50 (0.01 → 0.5)
/// - CVD acceleration: ×10
/// - Spread: ×20
/// - Trade intensity: /10
/// - Volatility: ×20
/// - PnL: /50 (normalized by $50 range)
pub fn normalize_features(obs: &mut Observation) {
    let f = &mut obs.features;

    // [0-2] Returns: scale by ×50 to amplify small percentage moves
    f[0] = (f[0] * 0.5).clamp(-1.0, 1.0);  // returns_1m (already ×100, so ×0.5 = ×50 effective)
    f[1] = (f[1] * 0.5).clamp(-1.0, 1.0);  // returns_5m
    f[2] = (f[2] * 0.5).clamp(-1.0, 1.0);  // returns_10m

    // [3-4] Orderbook imbalance: already in [-1, 1], just clamp
    f[3] = f[3].clamp(-1.0, 1.0);  // order_book_imbalance_l1
    f[4] = f[4].clamp(-1.0, 1.0);  // order_book_imbalance_l5

    // [5] Trade flow imbalance: already in [-1, 1], just clamp
    f[5] = f[5].clamp(-1.0, 1.0);

    // [6] CVD acceleration: scale by ×10
    f[6] = (f[6] * 10.0).clamp(-1.0, 1.0);

    // [7] Spread: already ×100, scale to reasonable range (×0.2 = effective ×20)
    f[7] = (f[7] * 0.2).clamp(-1.0, 1.0);

    // [8] Trade intensity: /10 to normalize typical values
    f[8] = (f[8] / 10.0).clamp(-1.0, 1.0);

    // [9] Large trade flag: already 0/1, clamp
    f[9] = f[9].clamp(-1.0, 1.0);

    // [10] Vol 5m: scale by ×20
    f[10] = (f[10] * 20.0).clamp(-1.0, 1.0);

    // [11] Vol expansion: clamp (typically around 1.0, deviation is meaningful)
    f[11] = f[11].clamp(-1.0, 1.0);

    // [12] Has position: already 0/1, clamp
    f[12] = f[12].clamp(-1.0, 1.0);

    // [13] Position side: already -1/0/1, clamp
    f[13] = f[13].clamp(-1.0, 1.0);

    // [14] Position PnL: normalize by $50 range
    f[14] = (f[14] / 50.0).clamp(-1.0, 1.0);

    // [15] Time remaining: already 0-1, clamp
    f[15] = f[15].clamp(-1.0, 1.0);

    // [16-17] Regime flags: already -1/0/1, clamp
    f[16] = f[16].clamp(-1.0, 1.0);
    f[17] = f[17].clamp(-1.0, 1.0);
}

/// Build and normalize observation in one step (Phase 5)
pub fn build_observation_normalized(
    binance: &BinanceMetrics,
    polymarket: &PolymarketMetrics,
    has_position: bool,
    position_side: Option<&str>,
    position_pnl: f64,
    time_remaining: f64,
) -> Observation {
    let mut obs = build_observation(binance, polymarket, has_position, position_side, position_pnl, time_remaining);
    normalize_features(&mut obs);
    obs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_default() {
        let obs = Observation::default();
        assert_eq!(obs.features.len(), 18);
        assert!(obs.features.iter().all(|&f| f == 0.0));
    }

    #[test]
    fn test_build_observation_matches_python_layout() {
        let binance = BinanceMetrics {
            return_1m: Some(0.05),     // 0.05% return
            return_5m: Some(-0.10),    // -0.10% return
            return_10m: Some(0.25),    // 0.25% return
            trade_flow_imbalance: -0.1,
            cvd_acceleration: 0.5,
            trade_intensity: 50.0,
            large_trade: true,
            vol_expansion: 1.5,
            vol_regime: 1.0,
            trend_regime: -1.0,
        };

        let polymarket = PolymarketMetrics {
            order_book_imbalance_l1: 0.3,
            order_book_imbalance_l5: 0.2,
            spread_pct: 0.02,    // 2% spread
            vol_5m: 0.15,
        };

        let obs = build_observation(
            &binance,
            &polymarket,
            true,           // has_position
            Some("YES"),    // position_side
            0.05,           // position_pnl
            0.75,           // time_remaining
        );

        // [0-2] Momentum (scaled by 100) - FROM BINANCE
        assert_eq!(obs.features[0], 5.0);      // 0.05 * 100
        assert_eq!(obs.features[1], -10.0);    // -0.10 * 100
        assert_eq!(obs.features[2], 25.0);     // 0.25 * 100

        // [3-4] Orderbook imbalance - FROM POLYMARKET
        assert_eq!(obs.features[3], 0.3);      // order_book_imbalance_l1
        assert_eq!(obs.features[4], 0.2);      // order_book_imbalance_l5

        // [5-6] Trade flow - FROM BINANCE
        assert_eq!(obs.features[5], -0.1);     // trade_flow_imbalance
        assert_eq!(obs.features[6], 0.5);      // cvd_acceleration

        // [7] Spread - FROM POLYMARKET
        assert_eq!(obs.features[7], 2.0);      // spread_pct * 100

        // [8-9] Microstructure - FROM BINANCE
        assert_eq!(obs.features[8], 50.0);     // trade_intensity
        assert_eq!(obs.features[9], 1.0);      // large_trade_flag

        // [10] Vol 5m - FROM POLYMARKET
        assert_eq!(obs.features[10], 0.15);    // vol_5m

        // [11] Vol expansion - FROM BINANCE
        assert_eq!(obs.features[11], 1.5);     // vol_expansion

        // [12-15] Position - INTERNAL
        assert_eq!(obs.features[12], 1.0);     // has_position
        assert_eq!(obs.features[13], 1.0);     // position_side (YES/UP = 1.0)
        assert_eq!(obs.features[14], 0.05);    // position_pnl
        assert_eq!(obs.features[15], 0.75);    // time_remaining

        // [16-17] Regime - FROM BINANCE
        assert_eq!(obs.features[16], 1.0);     // vol_regime
        assert_eq!(obs.features[17], -1.0);    // trend_regime
    }

    #[test]
    fn test_position_side_encoding() {
        let binance = BinanceMetrics::default();
        let polymarket = PolymarketMetrics::default();

        // Test YES/UP side
        let obs = build_observation(&binance, &polymarket, true, Some("YES"), 0.0, 0.0);
        assert_eq!(obs.features[13], 1.0);

        let obs = build_observation(&binance, &polymarket, true, Some("UP"), 0.0, 0.0);
        assert_eq!(obs.features[13], 1.0);

        // Test NO/DOWN side
        let obs = build_observation(&binance, &polymarket, true, Some("NO"), 0.0, 0.0);
        assert_eq!(obs.features[13], -1.0);

        let obs = build_observation(&binance, &polymarket, true, Some("DOWN"), 0.0, 0.0);
        assert_eq!(obs.features[13], -1.0);

        // Test no position
        let obs = build_observation(&binance, &polymarket, false, None, 0.0, 0.0);
        assert_eq!(obs.features[12], 0.0);
        assert_eq!(obs.features[13], 0.0);
    }

    #[test]
    fn test_feature_names() {
        assert_eq!(Observation::feature_name(0), "returns_1m");
        assert_eq!(Observation::feature_name(3), "order_book_imbalance_l1");
        assert_eq!(Observation::feature_name(7), "spread_pct");
        assert_eq!(Observation::feature_name(12), "has_position");
        assert_eq!(Observation::feature_name(16), "vol_regime");
        assert_eq!(Observation::feature_name(99), "unknown");
    }

    #[test]
    fn test_calculate_vol_5m() {
        // Test with insufficient data
        let short_history = vec![0.5, 0.51, 0.49];
        assert_eq!(calculate_vol_5m(&short_history, 30), 0.0);

        // Test with enough data (30 values around 0.5 with small variations)
        let mut history: Vec<f64> = (0..30).map(|i| 0.5 + (i as f64 * 0.001)).collect();
        let vol = calculate_vol_5m(&history, 30);
        assert!(vol > 0.0);  // Should have some volatility
        assert!(vol < 0.1);  // But not too much

        // Test with constant values (zero volatility)
        let constant_history: Vec<f64> = vec![0.5; 30];
        assert_eq!(calculate_vol_5m(&constant_history, 30), 0.0);
    }

    #[test]
    fn test_calculate_orderbook_imbalance() {
        // Balanced book
        assert_eq!(calculate_orderbook_imbalance(100.0, 100.0), 0.0);

        // All bids, no asks
        assert_eq!(calculate_orderbook_imbalance(100.0, 0.0), 1.0);

        // All asks, no bids
        assert_eq!(calculate_orderbook_imbalance(0.0, 100.0), -1.0);

        // 3:1 bid/ask ratio
        let imbalance = calculate_orderbook_imbalance(75.0, 25.0);
        assert!((imbalance - 0.5).abs() < 0.001);

        // Empty book
        assert_eq!(calculate_orderbook_imbalance(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_calculate_spread_pct() {
        // Normal spread (bid=0.48, ask=0.52)
        let spread = calculate_spread_pct(0.48, 0.52);
        assert!((spread - 0.08).abs() < 0.001);  // 4 cent spread / 50 cent mid = 8%

        // Tight spread
        let spread = calculate_spread_pct(0.50, 0.51);
        assert!((spread - 0.0198).abs() < 0.001);  // ~2%

        // Invalid prices
        assert_eq!(calculate_spread_pct(0.0, 0.5), 0.0);
        assert_eq!(calculate_spread_pct(0.5, 0.0), 0.0);
    }
}
