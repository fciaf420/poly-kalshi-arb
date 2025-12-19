// src/paper_trading.rs
//! Paper trading module for simulating trades without real money.
//!
//! Provides a `TradingClient` abstraction that can be either live or paper mode.
//! In paper mode, fills are simulated based on the order price and a configurable
//! slippage model.
//!
//! # Usage in binaries:
//! ```ignore
//! let client = if args.live {
//!     TradingClient::live(shared_client)
//! } else {
//!     TradingClient::paper(1000.0) // $1000 starting balance
//! };
//!
//! // Same interface for both modes
//! let fill = client.buy_fak("token_id", 0.50, 10.0).await?;
//! ```

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use crate::polymarket_clob::{PolyFillAsync, SharedAsyncClient};

/// Paper trading fill result (mirrors PolyFillAsync)
#[derive(Debug, Clone)]
pub struct PaperFill {
    pub order_id: String,
    pub filled_size: f64,
    pub fill_cost: f64,
}

impl From<PaperFill> for PolyFillAsync {
    fn from(pf: PaperFill) -> Self {
        PolyFillAsync {
            order_id: pf.order_id,
            filled_size: pf.filled_size,
            fill_cost: pf.fill_cost,
        }
    }
}

/// A single position in paper trading
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PaperPosition {
    pub token_id: String,
    pub contracts: f64,
    pub cost_basis: f64,
    pub avg_price: f64,
}

impl PaperPosition {
    pub fn add(&mut self, contracts: f64, price: f64) {
        let cost = contracts * price;
        self.cost_basis += cost;
        self.contracts += contracts;
        if self.contracts > 0.0 {
            self.avg_price = self.cost_basis / self.contracts;
        }
    }

    pub fn remove(&mut self, contracts: f64) -> f64 {
        let removed = contracts.min(self.contracts);
        if removed > 0.0 {
            let avg = self.avg_price;
            self.contracts -= removed;
            self.cost_basis = self.contracts * self.avg_price;
            removed * avg
        } else {
            0.0
        }
    }
}

/// A recorded paper trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperTrade {
    pub timestamp: String,
    pub token_id: String,
    pub side: String, // "buy" or "sell"
    pub price: f64,
    pub size: f64,
    pub cost: f64,
    pub order_id: String,
}

/// Paper trading account state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperAccount {
    /// Starting balance
    pub initial_balance: f64,
    /// Current cash balance
    pub balance: f64,
    /// Open positions by token_id
    pub positions: HashMap<String, PaperPosition>,
    /// Trade history
    pub trades: Vec<PaperTrade>,
    /// Realized P&L
    pub realized_pnl: f64,
    /// Order counter for generating order IDs
    #[serde(skip)]
    order_counter: u64,
    /// Slippage in basis points (default 0 = fill at limit price)
    #[serde(default)]
    pub slippage_bps: u16,
    /// Fill rate (0.0-1.0, default 1.0 = always fill fully)
    #[serde(default = "default_fill_rate")]
    pub fill_rate: f64,
}

fn default_fill_rate() -> f64 {
    1.0
}

const PAPER_POSITIONS_FILE: &str = "paper_positions.json";

impl PaperAccount {
    /// Create a new paper account with initial balance
    pub fn new(initial_balance: f64) -> Self {
        Self {
            initial_balance,
            balance: initial_balance,
            positions: HashMap::new(),
            trades: Vec::new(),
            realized_pnl: 0.0,
            order_counter: 0,
            slippage_bps: 0,
            fill_rate: 1.0,
        }
    }

    /// Create with custom slippage
    pub fn with_slippage(mut self, slippage_bps: u16) -> Self {
        self.slippage_bps = slippage_bps;
        self
    }

    /// Create with custom fill rate
    pub fn with_fill_rate(mut self, fill_rate: f64) -> Self {
        self.fill_rate = fill_rate.clamp(0.0, 1.0);
        self
    }

    /// Load from file or create new
    pub fn load_or_new(initial_balance: f64) -> Self {
        Self::load_from(PAPER_POSITIONS_FILE).unwrap_or_else(|| Self::new(initial_balance))
    }

    /// Load from file
    pub fn load_from<P: AsRef<Path>>(path: P) -> Option<Self> {
        std::fs::read_to_string(path.as_ref())
            .ok()
            .and_then(|contents| serde_json::from_str(&contents).ok())
    }

    /// Save to file
    pub fn save(&self) -> Result<()> {
        self.save_to(PAPER_POSITIONS_FILE)
    }

    /// Save to specific path
    pub fn save_to<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Generate unique order ID
    fn next_order_id(&mut self) -> String {
        self.order_counter += 1;
        format!("paper-{}-{}", Utc::now().timestamp_millis(), self.order_counter)
    }

    /// Apply slippage to price (worse price for trader)
    fn apply_slippage(&self, price: f64, is_buy: bool) -> f64 {
        let slip = price * (self.slippage_bps as f64 / 10000.0);
        if is_buy {
            price + slip // Pay more when buying
        } else {
            price - slip // Receive less when selling
        }
    }

    /// Simulate a FAK buy order
    pub fn buy_fak(&mut self, token_id: &str, price: f64, size: f64) -> PaperFill {
        let exec_price = self.apply_slippage(price, true);
        let fill_size = size * self.fill_rate;
        let cost = fill_size * exec_price;

        // Check if we have enough balance
        let actual_fill = if cost <= self.balance {
            fill_size
        } else {
            // Partial fill based on available balance
            (self.balance / exec_price).floor()
        };

        let actual_cost = actual_fill * exec_price;
        let order_id = self.next_order_id();

        if actual_fill > 0.0 {
            self.balance -= actual_cost;

            // Update position
            let position = self.positions
                .entry(token_id.to_string())
                .or_insert_with(|| PaperPosition {
                    token_id: token_id.to_string(),
                    ..Default::default()
                });
            position.add(actual_fill, exec_price);

            // Record trade
            self.trades.push(PaperTrade {
                timestamp: Utc::now().to_rfc3339(),
                token_id: token_id.to_string(),
                side: "buy".to_string(),
                price: exec_price,
                size: actual_fill,
                cost: actual_cost,
                order_id: order_id.clone(),
            });

            // Auto-save
            let _ = self.save();
        }

        PaperFill {
            order_id,
            filled_size: actual_fill,
            fill_cost: actual_cost,
        }
    }

    /// Simulate a FAK sell order
    pub fn sell_fak(&mut self, token_id: &str, price: f64, size: f64) -> PaperFill {
        let exec_price = self.apply_slippage(price, false);
        let fill_size = size * self.fill_rate;
        let order_id = self.next_order_id();

        // Check position
        let position = self.positions.get_mut(token_id);
        let actual_fill = match position {
            Some(pos) if pos.contracts > 0.0 => fill_size.min(pos.contracts),
            _ => 0.0,
        };

        let proceeds = actual_fill * exec_price;

        if actual_fill > 0.0 {
            let position = self.positions.get_mut(token_id).unwrap();
            let cost_basis_sold = position.remove(actual_fill);

            self.balance += proceeds;
            self.realized_pnl += proceeds - cost_basis_sold;

            // Record trade
            self.trades.push(PaperTrade {
                timestamp: Utc::now().to_rfc3339(),
                token_id: token_id.to_string(),
                side: "sell".to_string(),
                price: exec_price,
                size: actual_fill,
                cost: proceeds,
                order_id: order_id.clone(),
            });

            // Auto-save
            let _ = self.save();
        }

        PaperFill {
            order_id,
            filled_size: actual_fill,
            fill_cost: proceeds,
        }
    }

    /// Get current account summary
    pub fn summary(&self) -> PaperSummary {
        let position_value: f64 = self.positions
            .values()
            .map(|p| p.cost_basis)
            .sum();

        let total_contracts: f64 = self.positions
            .values()
            .map(|p| p.contracts)
            .sum();

        PaperSummary {
            initial_balance: self.initial_balance,
            cash_balance: self.balance,
            position_value,
            total_equity: self.balance + position_value,
            realized_pnl: self.realized_pnl,
            unrealized_pnl: 0.0, // Would need current prices
            total_trades: self.trades.len(),
            open_positions: self.positions.values().filter(|p| p.contracts > 0.0).count(),
            total_contracts,
        }
    }

    /// Print summary to log
    pub fn log_summary(&self) {
        let s = self.summary();
        info!("[PAPER] ════════════════════════════════════════════════");
        info!("[PAPER] 📊 PAPER TRADING SUMMARY");
        info!("[PAPER] ════════════════════════════════════════════════");
        info!("[PAPER]   Initial Balance: ${:.2}", s.initial_balance);
        info!("[PAPER]   Cash Balance:    ${:.2}", s.cash_balance);
        info!("[PAPER]   Position Value:  ${:.2}", s.position_value);
        info!("[PAPER]   Total Equity:    ${:.2}", s.total_equity);
        info!("[PAPER]   Realized P&L:    ${:+.2}", s.realized_pnl);
        info!("[PAPER]   Total Trades:    {}", s.total_trades);
        info!("[PAPER]   Open Positions:  {} ({:.0} contracts)", s.open_positions, s.total_contracts);
        info!("[PAPER] ════════════════════════════════════════════════");
    }
}

/// Summary of paper account state
#[derive(Debug, Clone)]
pub struct PaperSummary {
    pub initial_balance: f64,
    pub cash_balance: f64,
    pub position_value: f64,
    pub total_equity: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_trades: usize,
    pub open_positions: usize,
    pub total_contracts: f64,
}

/// Shared paper account for concurrent access
pub type SharedPaperAccount = Arc<RwLock<PaperAccount>>;

/// Create a shared paper account
pub fn create_paper_account(initial_balance: f64) -> SharedPaperAccount {
    Arc::new(RwLock::new(PaperAccount::load_or_new(initial_balance)))
}

// ============================================================================
// TradingClient - Unified interface for live vs paper trading
// ============================================================================

/// Trading client that abstracts live vs paper execution
#[derive(Clone)]
pub enum TradingClient {
    /// Live trading with real money
    Live(Arc<SharedAsyncClient>),
    /// Paper trading with simulated fills
    Paper(SharedPaperAccount),
}

impl TradingClient {
    /// Create a live trading client
    pub fn live(client: Arc<SharedAsyncClient>) -> Self {
        TradingClient::Live(client)
    }

    /// Create a paper trading client with initial balance
    pub fn paper(initial_balance: f64) -> Self {
        TradingClient::Paper(create_paper_account(initial_balance))
    }

    /// Create a paper trading client with custom settings
    pub fn paper_with_settings(initial_balance: f64, slippage_bps: u16, fill_rate: f64) -> Self {
        let account = PaperAccount::new(initial_balance)
            .with_slippage(slippage_bps)
            .with_fill_rate(fill_rate);
        TradingClient::Paper(Arc::new(RwLock::new(account)))
    }

    /// Check if this is paper trading mode
    pub fn is_paper(&self) -> bool {
        matches!(self, TradingClient::Paper(_))
    }

    /// Execute a FAK buy order
    pub async fn buy_fak(&self, token_id: &str, price: f64, size: f64) -> Result<PolyFillAsync> {
        match self {
            TradingClient::Live(client) => client.buy_fak(token_id, price, size).await,
            TradingClient::Paper(account) => {
                let fill = account.write().await.buy_fak(token_id, price, size);
                Ok(fill.into())
            }
        }
    }

    /// Execute a FAK sell order
    pub async fn sell_fak(&self, token_id: &str, price: f64, size: f64) -> Result<PolyFillAsync> {
        match self {
            TradingClient::Live(client) => client.sell_fak(token_id, price, size).await,
            TradingClient::Paper(account) => {
                let fill = account.write().await.sell_fak(token_id, price, size);
                Ok(fill.into())
            }
        }
    }

    /// Get paper account summary (only for paper mode)
    pub async fn paper_summary(&self) -> Option<PaperSummary> {
        match self {
            TradingClient::Paper(account) => Some(account.read().await.summary()),
            TradingClient::Live(_) => None,
        }
    }

    /// Log paper account summary (no-op for live mode)
    pub async fn log_summary(&self) {
        if let TradingClient::Paper(account) = self {
            account.read().await.log_summary();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paper_buy() {
        let mut account = PaperAccount::new(1000.0);
        let fill = account.buy_fak("token1", 0.50, 10.0);

        assert_eq!(fill.filled_size, 10.0);
        assert!((fill.fill_cost - 5.0).abs() < 0.001);
        assert!((account.balance - 995.0).abs() < 0.001);

        let pos = account.positions.get("token1").unwrap();
        assert_eq!(pos.contracts, 10.0);
        assert!((pos.avg_price - 0.50).abs() < 0.001);
    }

    #[test]
    fn test_paper_sell() {
        let mut account = PaperAccount::new(1000.0);
        account.buy_fak("token1", 0.50, 10.0); // Buy at 50¢

        let fill = account.sell_fak("token1", 0.60, 5.0); // Sell at 60¢

        assert_eq!(fill.filled_size, 5.0);
        assert!((fill.fill_cost - 3.0).abs() < 0.001);
        assert!((account.realized_pnl - 0.50).abs() < 0.001); // $0.50 profit
    }

    #[test]
    fn test_insufficient_balance() {
        let mut account = PaperAccount::new(10.0);
        let fill = account.buy_fak("token1", 0.50, 100.0); // Try to buy $50 with $10

        assert_eq!(fill.filled_size, 20.0); // Can only buy 20 contracts
        assert!((fill.fill_cost - 10.0).abs() < 0.001);
        assert!((account.balance - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_slippage() {
        let mut account = PaperAccount::new(1000.0).with_slippage(100); // 1% slippage
        let fill = account.buy_fak("token1", 0.50, 10.0);

        // Should pay 0.505 per contract (0.5% worse)
        assert!((fill.fill_cost - 5.05).abs() < 0.001);
    }

    #[test]
    fn test_partial_fill_rate() {
        let mut account = PaperAccount::new(1000.0).with_fill_rate(0.5);
        let fill = account.buy_fak("token1", 0.50, 10.0);

        assert_eq!(fill.filled_size, 5.0); // Only 50% fills
    }
}
