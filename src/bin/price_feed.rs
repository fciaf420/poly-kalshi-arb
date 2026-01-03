//! Shared Price Feed Service with Order Flow Metrics
//!
//! Connects to Binance Futures and broadcasts prices + order flow metrics to local clients.
//! Other binaries connect to ws://127.0.0.1:9999 to receive updates.
//!
//! Order Flow Metrics (matches Python cross-market-state-fusion):
//! - CVD (Cumulative Volume Delta): buy volume - sell volume
//! - CVD Acceleration: rate of change of CVD
//! - Trade Intensity: trades per second
//! - Trade Flow Imbalance: (buy_vol - sell_vol) / total over window
//! - Orderbook Imbalance L1/L5: bid vs ask size imbalance
//! - Momentum: 1m, 5m, 10m returns
//! - Volatility: rolling std dev of 1-min returns
//! - Vol Expansion: current vol / average vol
//! - Vol/Trend Regime: market state classification

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, RwLock};
use tokio_tungstenite::{accept_async, connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

// Binance FUTURES WebSocket with depth stream for orderbook
const BINANCE_WS_URL: &str = "wss://fstream.binance.com/stream?streams=\
btcusdt@aggTrade/ethusdt@aggTrade/solusdt@aggTrade/xrpusdt@aggTrade/\
btcusdt@depth5@100ms/ethusdt@depth5@100ms/solusdt@depth5@100ms/xrpusdt@depth5@100ms";
const LOCAL_WS_PORT: u16 = 9999;

// Order flow computation parameters
const CVD_RESET_INTERVAL_SECS: u64 = 300; // Reset CVD every 5 minutes
const TRADE_INTENSITY_WINDOW_SECS: u64 = 5; // Window for trades/sec calculation
const TRADE_FLOW_WINDOW_SECS: u64 = 30; // Window for trade flow imbalance
const MOMENTUM_HISTORY_SIZE: usize = 600; // 10 minutes of 1-second samples
const VOLATILITY_WINDOW: usize = 30; // 30 samples for volatility
const CVD_HISTORY_SIZE: usize = 60; // 1 minute of CVD samples for acceleration
const VOL_HISTORY_SIZE: usize = 60; // 1 minute of vol samples for expansion

/// Per-asset order flow state
#[derive(Debug, Clone)]
struct AssetState {
    // Current price and orderbook
    price: Option<f64>,
    best_bid: f64,
    best_ask: f64,
    bid_qty_l1: f64,
    ask_qty_l1: f64,
    bid_qty_l5: f64,
    ask_qty_l5: f64,

    // Order flow metrics
    cvd: f64,                          // Cumulative Volume Delta (buy - sell)
    cvd_last_reset: Option<Instant>,
    cvd_history: VecDeque<(Instant, f64)>, // For CVD acceleration
    trade_count_window: VecDeque<Instant>, // Recent trades for intensity calc

    // Trade flow tracking (buy vs sell volume)
    trade_flow_history: VecDeque<(Instant, f64)>, // (time, signed_volume)

    // Momentum tracking (1-second samples)
    price_history: VecDeque<(Instant, f64)>,

    // Large trade detection
    last_large_trade: Option<Instant>,
    large_trade_threshold: f64,

    // Volatility tracking
    vol_history: VecDeque<f64>, // Recent volatility values for expansion calc
}

impl Default for AssetState {
    fn default() -> Self {
        Self {
            price: None,
            best_bid: 0.0,
            best_ask: 0.0,
            bid_qty_l1: 0.0,
            ask_qty_l1: 0.0,
            bid_qty_l5: 0.0,
            ask_qty_l5: 0.0,
            cvd: 0.0,
            cvd_last_reset: None,
            cvd_history: VecDeque::with_capacity(CVD_HISTORY_SIZE),
            trade_count_window: VecDeque::new(),
            trade_flow_history: VecDeque::new(),
            price_history: VecDeque::with_capacity(MOMENTUM_HISTORY_SIZE),
            last_large_trade: None,
            large_trade_threshold: 0.0,
            vol_history: VecDeque::with_capacity(VOL_HISTORY_SIZE),
        }
    }
}

impl AssetState {
    fn new(large_trade_threshold: f64) -> Self {
        Self {
            large_trade_threshold,
            ..Default::default()
        }
    }

    /// Update with new trade data
    fn update_trade(&mut self, price: f64, quantity: f64, is_buyer_maker: bool) {
        let now = Instant::now();

        // Update price
        self.price = Some(price);

        // Update CVD (positive = buying pressure, negative = selling)
        // is_buyer_maker=true means the trade was a SELL (maker was buyer, taker sold)
        let signed_volume = if is_buyer_maker { -quantity } else { quantity };
        let dollar_volume = signed_volume * price;
        self.cvd += dollar_volume;

        // Track CVD history for acceleration
        let should_sample_cvd = self.cvd_history.back()
            .map(|(t, _)| now.duration_since(*t).as_millis() >= 1000)
            .unwrap_or(true);
        if should_sample_cvd {
            self.cvd_history.push_back((now, self.cvd));
            while self.cvd_history.len() > CVD_HISTORY_SIZE {
                self.cvd_history.pop_front();
            }
        }

        // Reset CVD periodically
        if let Some(last_reset) = self.cvd_last_reset {
            if now.duration_since(last_reset).as_secs() > CVD_RESET_INTERVAL_SECS {
                self.cvd = dollar_volume;
                self.cvd_last_reset = Some(now);
                self.cvd_history.clear();
            }
        } else {
            self.cvd_last_reset = Some(now);
        }

        // Track trade flow for imbalance calculation
        self.trade_flow_history.push_back((now, dollar_volume));
        // Clean old entries
        while let Some((t, _)) = self.trade_flow_history.front() {
            if now.duration_since(*t).as_secs() > TRADE_FLOW_WINDOW_SECS {
                self.trade_flow_history.pop_front();
            } else {
                break;
            }
        }

        // Track trade for intensity calculation
        self.trade_count_window.push_back(now);
        // Clean old trades from window
        while let Some(front) = self.trade_count_window.front() {
            if now.duration_since(*front).as_secs() > TRADE_INTENSITY_WINDOW_SECS {
                self.trade_count_window.pop_front();
            } else {
                break;
            }
        }

        // Track price history for momentum (sample every ~1 second)
        let should_sample = self.price_history.back()
            .map(|(t, _)| now.duration_since(*t).as_millis() >= 1000)
            .unwrap_or(true);

        if should_sample {
            self.price_history.push_back((now, price));
            while self.price_history.len() > MOMENTUM_HISTORY_SIZE {
                self.price_history.pop_front();
            }

            // Also track volatility history
            if let Some(vol) = self.volatility() {
                self.vol_history.push_back(vol);
                while self.vol_history.len() > VOL_HISTORY_SIZE {
                    self.vol_history.pop_front();
                }
            }
        }

        // Detect large trades
        let trade_value = quantity * price;
        if trade_value >= self.large_trade_threshold {
            self.last_large_trade = Some(now);
        }
    }

    /// Update orderbook from depth stream
    fn update_orderbook(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        // L1 (top of book)
        if let Some((price, qty)) = bids.first() {
            self.best_bid = *price;
            self.bid_qty_l1 = *qty;
        }
        if let Some((price, qty)) = asks.first() {
            self.best_ask = *price;
            self.ask_qty_l1 = *qty;
        }

        // L5 (sum of top 5 levels)
        self.bid_qty_l5 = bids.iter().take(5).map(|(_, q)| q).sum();
        self.ask_qty_l5 = asks.iter().take(5).map(|(_, q)| q).sum();
    }

    /// Get spread percentage
    fn spread_pct(&self) -> Option<f64> {
        let mid = (self.best_bid + self.best_ask) / 2.0;
        if mid > 0.0 && self.best_ask > self.best_bid {
            Some((self.best_ask - self.best_bid) / mid * 100.0)
        } else {
            None
        }
    }

    /// Get orderbook imbalance at L1: (bid - ask) / (bid + ask)
    fn orderbook_imbalance_l1(&self) -> f64 {
        let total = self.bid_qty_l1 + self.ask_qty_l1;
        if total > 0.0 {
            (self.bid_qty_l1 - self.ask_qty_l1) / total
        } else {
            0.0
        }
    }

    /// Get orderbook imbalance at L5: (bid - ask) / (bid + ask)
    fn orderbook_imbalance_l5(&self) -> f64 {
        let total = self.bid_qty_l5 + self.ask_qty_l5;
        if total > 0.0 {
            (self.bid_qty_l5 - self.ask_qty_l5) / total
        } else {
            0.0
        }
    }

    /// Get trade flow imbalance: (buy_vol - sell_vol) / total over window
    fn trade_flow_imbalance(&self) -> f64 {
        let (buy, sell) = self.trade_flow_history.iter()
            .fold((0.0, 0.0), |(b, s), (_, vol)| {
                if *vol > 0.0 {
                    (b + vol.abs(), s)
                } else {
                    (b, s + vol.abs())
                }
            });
        let total = buy + sell;
        if total > 0.0 {
            (buy - sell) / total
        } else {
            0.0
        }
    }

    /// Get CVD acceleration (rate of change of CVD)
    fn cvd_acceleration(&self) -> f64 {
        if self.cvd_history.len() < 2 {
            return 0.0;
        }

        // Compare CVD from 5 seconds ago to now
        let now = Instant::now();
        let mut cvd_old = self.cvd_history.front().map(|(_, c)| *c).unwrap_or(0.0);
        let cvd_new = self.cvd_history.back().map(|(_, c)| *c).unwrap_or(0.0);

        // Find CVD from ~5 seconds ago
        for (t, c) in self.cvd_history.iter().rev() {
            if now.duration_since(*t).as_secs() >= 5 {
                cvd_old = *c;
                break;
            }
        }

        // Rate of change (normalized by typical CVD magnitude)
        let delta = cvd_new - cvd_old;
        // Normalize to roughly [-1, 1] assuming typical CVD deltas are ~$1M
        (delta / 1_000_000.0).clamp(-1.0, 1.0)
    }

    /// Get trade intensity (trades per second)
    fn trade_intensity(&self) -> f64 {
        let count = self.trade_count_window.len() as f64;
        count / TRADE_INTENSITY_WINDOW_SECS as f64
    }

    /// Get momentum (return over N seconds)
    fn momentum(&self, seconds: u64) -> Option<f64> {
        let now = Instant::now();
        let current_price = self.price?;

        // Find price from N seconds ago
        for (t, p) in self.price_history.iter().rev() {
            let age = now.duration_since(*t).as_secs();
            if age >= seconds {
                return Some((current_price - p) / p * 100.0); // Percentage return
            }
        }
        None
    }

    /// Get realized volatility (std dev of 1-minute returns)
    fn volatility(&self) -> Option<f64> {
        if self.price_history.len() < VOLATILITY_WINDOW + 1 {
            return None;
        }

        // Calculate returns over the window
        let mut returns = Vec::new();
        let history: Vec<_> = self.price_history.iter().collect();

        for i in 1..history.len().min(VOLATILITY_WINDOW + 1) {
            let prev = history[history.len() - 1 - i].1;
            let curr = history[history.len() - i].1;
            let ret = (curr - prev) / prev;
            returns.push(ret);
        }

        if returns.is_empty() {
            return None;
        }

        // Calculate std dev
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        Some(variance.sqrt() * 100.0) // Percentage
    }

    /// Get volatility expansion (current vol / average vol)
    fn vol_expansion(&self) -> f64 {
        if self.vol_history.len() < 2 {
            return 1.0;
        }

        let current_vol = self.vol_history.back().copied().unwrap_or(0.0);
        let avg_vol = self.vol_history.iter().sum::<f64>() / self.vol_history.len() as f64;

        if avg_vol > 0.0 {
            (current_vol / avg_vol).clamp(0.0, 5.0) // Cap at 5x
        } else {
            1.0
        }
    }

    /// Get volatility regime: 1.0 = high vol, 0.0 = normal, -1.0 = low vol
    fn vol_regime(&self) -> f64 {
        let expansion = self.vol_expansion();
        if expansion > 1.5 {
            1.0 // High volatility
        } else if expansion < 0.7 {
            -1.0 // Low volatility
        } else {
            0.0 // Normal
        }
    }

    /// Get trend regime: 1.0 = trending up, -1.0 = trending down, 0.0 = ranging
    fn trend_regime(&self) -> f64 {
        // Use 5-minute momentum to determine trend
        let mom_5m = self.momentum(300).unwrap_or(0.0);
        let mom_1m = self.momentum(60).unwrap_or(0.0);

        // Consistent direction = trending
        if mom_5m > 0.2 && mom_1m > 0.0 {
            1.0 // Uptrend
        } else if mom_5m < -0.2 && mom_1m < 0.0 {
            -1.0 // Downtrend
        } else {
            0.0 // Ranging
        }
    }

    /// Check if large trade occurred recently
    fn has_recent_large_trade(&self, within_secs: u64) -> bool {
        self.last_large_trade
            .map(|t| t.elapsed().as_secs() < within_secs)
            .unwrap_or(false)
    }
}

/// Full state for all assets
#[derive(Debug)]
struct FullState {
    btc: AssetState,
    eth: AssetState,
    sol: AssetState,
    xrp: AssetState,
}

impl FullState {
    fn new() -> Self {
        Self {
            btc: AssetState::new(100_000.0),  // $100k for BTC large trade
            eth: AssetState::new(50_000.0),   // $50k for ETH
            sol: AssetState::new(10_000.0),   // $10k for SOL
            xrp: AssetState::new(10_000.0),   // $10k for XRP
        }
    }
}

/// Broadcast message with prices and order flow metrics
/// Matches Python's MarketState features for RL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceUpdate {
    // Prices
    pub btc_price: Option<f64>,
    pub eth_price: Option<f64>,
    pub sol_price: Option<f64>,
    pub xrp_price: Option<f64>,
    pub timestamp: i64,

    // Order flow metrics (per asset)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub btc_metrics: Option<AssetMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eth_metrics: Option<AssetMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sol_metrics: Option<AssetMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xrp_metrics: Option<AssetMetrics>,
}

/// Asset metrics matching Python's MarketState features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetMetrics {
    // Basic
    pub cvd: f64,              // Cumulative Volume Delta (dollar value)
    pub trade_intensity: f64,  // Trades per second
    pub return_1m: Option<f64>,  // 1-minute return %
    pub return_5m: Option<f64>,  // 5-minute return %
    pub return_10m: Option<f64>, // 10-minute return %
    pub volatility: Option<f64>, // Realized volatility %
    pub large_trade: bool,       // Large trade in last 10 seconds

    // NEW: Order flow features (matches Python)
    pub order_book_imbalance_l1: f64,  // Top of book imbalance
    pub order_book_imbalance_l5: f64,  // Depth imbalance (top 5 levels)
    pub trade_flow_imbalance: f64,     // Buy vs sell pressure [-1, 1]
    pub cvd_acceleration: f64,         // Rate of change of CVD [-1, 1]

    // NEW: Microstructure
    pub spread_pct: Option<f64>,  // Spread as percentage

    // NEW: Volatility features
    pub vol_expansion: f64,  // Current vol / average vol
    pub vol_regime: f64,     // 1.0 = high, 0.0 = normal, -1.0 = low
    pub trend_regime: f64,   // 1.0 = up, 0.0 = ranging, -1.0 = down
}

impl AssetMetrics {
    fn from_state(state: &AssetState) -> Self {
        Self {
            cvd: state.cvd,
            trade_intensity: state.trade_intensity(),
            return_1m: state.momentum(60),
            return_5m: state.momentum(300),
            return_10m: state.momentum(600),
            volatility: state.volatility(),
            large_trade: state.has_recent_large_trade(10),
            order_book_imbalance_l1: state.orderbook_imbalance_l1(),
            order_book_imbalance_l5: state.orderbook_imbalance_l5(),
            trade_flow_imbalance: state.trade_flow_imbalance(),
            cvd_acceleration: state.cvd_acceleration(),
            spread_pct: state.spread_pct(),
            vol_expansion: state.vol_expansion(),
            vol_regime: state.vol_regime(),
            trend_regime: state.trend_regime(),
        }
    }
}

impl Default for PriceUpdate {
    fn default() -> Self {
        Self {
            btc_price: None,
            eth_price: None,
            sol_price: None,
            xrp_price: None,
            timestamp: 0,
            btc_metrics: None,
            eth_metrics: None,
            sol_metrics: None,
            xrp_metrics: None,
        }
    }
}

/// Binance combined stream message (generic for different stream types)
#[derive(Debug, Deserialize)]
struct BinanceStreamMessage {
    stream: String,
    data: serde_json::Value,
}

/// Binance aggTrade data
#[derive(Debug, Deserialize)]
struct BinanceAggTrade {
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "q")]
    quantity: String,
    #[serde(rename = "m")]
    is_buyer_maker: bool, // true = sell (maker was buyer), false = buy
}

/// Binance depth update (partial book)
#[derive(Debug, Deserialize)]
struct BinanceDepth {
    #[serde(rename = "b")]
    bids: Vec<(String, String)>, // [[price, qty], ...]
    #[serde(rename = "a")]
    asks: Vec<(String, String)>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("price_feed=info")
        .with_target(true)
        .init();

    info!("================================================================================");
    info!("   SHARED PRICE FEED SERVICE (Binance FUTURES + Full Order Flow)");
    info!("================================================================================");
    info!("Local WebSocket server: ws://127.0.0.1:{}", LOCAL_WS_PORT);
    info!("Source: Binance Futures WSS (fstream.binance.com)");
    info!("Streams: aggTrade + depth5@100ms for BTC, ETH, SOL, XRP");
    info!("");
    info!("Metrics (matches Python cross-market-state-fusion):");
    info!("  - Momentum: 1m/5m/10m returns");
    info!("  - Order Flow: CVD, CVD acceleration, trade flow imbalance");
    info!("  - Orderbook: L1/L5 imbalance, spread");
    info!("  - Volatility: realized vol, vol expansion, vol regime");
    info!("  - Trend: trend regime");
    info!("================================================================================");

    // Broadcast channel for price updates
    let (tx, _) = broadcast::channel::<String>(100);
    let tx = Arc::new(tx);

    // Full state with order flow tracking
    let full_state = Arc::new(RwLock::new(FullState::new()));

    // Simple prices for quick client updates
    let prices = Arc::new(RwLock::new(PriceUpdate::default()));

    // Start local WebSocket server for clients
    let tx_clone = tx.clone();
    let prices_clone = prices.clone();
    tokio::spawn(async move {
        run_local_server(tx_clone, prices_clone).await;
    });

    // Connect to Binance and broadcast prices
    loop {
        if let Err(e) = run_binance_feed(full_state.clone(), prices.clone(), tx.clone()).await {
            error!("[BINANCE] Connection error: {}", e);
        }
        warn!("[BINANCE] Reconnecting in 5 seconds...");
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }
}

async fn run_local_server(tx: Arc<broadcast::Sender<String>>, prices: Arc<RwLock<PriceUpdate>>) {
    let listener = TcpListener::bind(format!("127.0.0.1:{}", LOCAL_WS_PORT))
        .await
        .expect("Failed to bind local WebSocket server");

    info!("[SERVER] Listening on ws://127.0.0.1:{}", LOCAL_WS_PORT);

    while let Ok((stream, addr)) = listener.accept().await {
        info!("[SERVER] New client connected: {}", addr);
        let rx = tx.subscribe();
        let prices_clone = prices.clone();
        tokio::spawn(handle_client(stream, rx, prices_clone));
    }
}

async fn handle_client(
    stream: TcpStream,
    mut rx: broadcast::Receiver<String>,
    prices: Arc<RwLock<PriceUpdate>>,
) {
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            error!("[SERVER] WebSocket handshake failed: {}", e);
            return;
        }
    };

    let (mut write, mut read) = ws_stream.split();

    // Send current prices immediately on connect
    {
        let state = prices.read().await;
        if let Ok(json) = serde_json::to_string(&*state) {
            let _ = write.send(Message::Text(json)).await;
        }
    }

    // Spawn task to forward broadcasts to this client
    let write = Arc::new(tokio::sync::Mutex::new(write));
    let write_clone = write.clone();

    let forward_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            let mut w = write_clone.lock().await;
            if w.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming messages (pings, etc)
    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Ping(data)) => {
                let mut w = write.lock().await;
                let _ = w.send(Message::Pong(data)).await;
            }
            Ok(Message::Close(_)) => break,
            Err(_) => break,
            _ => {}
        }
    }

    forward_task.abort();
    info!("[SERVER] Client disconnected");
}

async fn run_binance_feed(
    full_state: Arc<RwLock<FullState>>,
    prices: Arc<RwLock<PriceUpdate>>,
    tx: Arc<broadcast::Sender<String>>,
) -> anyhow::Result<()> {
    info!("[BINANCE] Connecting to WebSocket...");

    let (ws_stream, _) = connect_async(BINANCE_WS_URL).await?;
    let (write, mut read) = ws_stream.split();

    info!("[BINANCE FUTURES] Connected - streaming aggTrades + depth5 for BTC, ETH, SOL, XRP");

    // Keepalive ping every 30 seconds
    let write = Arc::new(tokio::sync::Mutex::new(write));
    let write_clone = write.clone();
    let ping_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            let mut w = write_clone.lock().await;
            if w.send(Message::Ping(vec![1, 2, 3, 4])).await.is_err() {
                break;
            }
        }
    });

    // Broadcast metrics every second (not every trade)
    let full_state_clone = full_state.clone();
    let prices_clone = prices.clone();
    let tx_clone = tx.clone();
    let metrics_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
        loop {
            interval.tick().await;
            let state = full_state_clone.read().await;
            let now = chrono::Utc::now().timestamp_millis();

            let update = PriceUpdate {
                btc_price: state.btc.price,
                eth_price: state.eth.price,
                sol_price: state.sol.price,
                xrp_price: state.xrp.price,
                timestamp: now,
                btc_metrics: state.btc.price.map(|_| AssetMetrics::from_state(&state.btc)),
                eth_metrics: state.eth.price.map(|_| AssetMetrics::from_state(&state.eth)),
                sol_metrics: state.sol.price.map(|_| AssetMetrics::from_state(&state.sol)),
                xrp_metrics: state.xrp.price.map(|_| AssetMetrics::from_state(&state.xrp)),
            };

            // Update shared prices state
            {
                let mut p = prices_clone.write().await;
                *p = update.clone();
            }

            // Broadcast to clients
            if let Ok(json) = serde_json::to_string(&update) {
                let _ = tx_clone.send(json);
            }
        }
    });

    let mut trade_count: u64 = 0;
    let mut depth_count: u64 = 0;
    let mut last_log = Instant::now();

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if let Ok(stream_msg) = serde_json::from_str::<BinanceStreamMessage>(&text) {
                    let stream = stream_msg.stream.as_str();

                    // Handle aggTrade streams
                    if stream.ends_with("@aggTrade") {
                        if let Ok(trade) = serde_json::from_value::<BinanceAggTrade>(stream_msg.data) {
                            let price: f64 = match trade.price.parse() {
                                Ok(p) => p,
                                Err(_) => continue,
                            };
                            let quantity: f64 = match trade.quantity.parse() {
                                Ok(q) => q,
                                Err(_) => continue,
                            };

                            let mut state = full_state.write().await;

                            match stream {
                                "btcusdt@aggTrade" => state.btc.update_trade(price, quantity, trade.is_buyer_maker),
                                "ethusdt@aggTrade" => state.eth.update_trade(price, quantity, trade.is_buyer_maker),
                                "solusdt@aggTrade" => state.sol.update_trade(price, quantity, trade.is_buyer_maker),
                                "xrpusdt@aggTrade" => state.xrp.update_trade(price, quantity, trade.is_buyer_maker),
                                _ => {}
                            }

                            trade_count += 1;
                        }
                    }
                    // Handle depth streams
                    else if stream.contains("@depth") {
                        if let Ok(depth) = serde_json::from_value::<BinanceDepth>(stream_msg.data) {
                            // Parse bids and asks
                            let bids: Vec<(f64, f64)> = depth.bids.iter()
                                .filter_map(|(p, q)| {
                                    Some((p.parse().ok()?, q.parse().ok()?))
                                })
                                .collect();
                            let asks: Vec<(f64, f64)> = depth.asks.iter()
                                .filter_map(|(p, q)| {
                                    Some((p.parse().ok()?, q.parse().ok()?))
                                })
                                .collect();

                            let mut state = full_state.write().await;

                            if stream.starts_with("btcusdt") {
                                state.btc.update_orderbook(&bids, &asks);
                            } else if stream.starts_with("ethusdt") {
                                state.eth.update_orderbook(&bids, &asks);
                            } else if stream.starts_with("solusdt") {
                                state.sol.update_orderbook(&bids, &asks);
                            } else if stream.starts_with("xrpusdt") {
                                state.xrp.update_orderbook(&bids, &asks);
                            }

                            depth_count += 1;
                        }
                    }

                    // Log summary every 10 seconds
                    if last_log.elapsed().as_secs() >= 10 {
                        let state = full_state.read().await;
                        let btc_cvd = state.btc.cvd / 1_000_000.0; // $M
                        let btc_intensity = state.btc.trade_intensity();
                        let btc_imb = state.btc.orderbook_imbalance_l1();
                        let btc_flow = state.btc.trade_flow_imbalance();

                        info!(
                            "[FLOW] {} trades, {} depth | BTC: CVD=${:.2}M, {:.1}t/s, OB_imb={:+.2}, flow={:+.2}",
                            trade_count, depth_count, btc_cvd, btc_intensity, btc_imb, btc_flow
                        );

                        if let Some(btc_ret) = state.btc.momentum(60) {
                            let vol_exp = state.btc.vol_expansion();
                            let vol_reg = state.btc.vol_regime();
                            let trend_reg = state.btc.trend_regime();
                            debug!(
                                "[BTC] 1m: {:+.3}%, vol_exp: {:.2}x, vol_regime: {:.0}, trend_regime: {:.0}",
                                btc_ret, vol_exp, vol_reg, trend_reg
                            );
                        }

                        trade_count = 0;
                        depth_count = 0;
                        last_log = Instant::now();
                    }
                }
            }
            Ok(Message::Ping(data)) => {
                let mut w = write.lock().await;
                let _ = w.send(Message::Pong(data)).await;
            }
            Ok(Message::Pong(_)) => {}
            Ok(Message::Close(_)) => {
                warn!("[BINANCE] Connection closed by server");
                break;
            }
            Err(e) => {
                error!("[BINANCE] WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    ping_task.abort();
    metrics_task.abort();
    Ok(())
}
