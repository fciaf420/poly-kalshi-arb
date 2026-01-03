//! Compare latency between direct Binance and Polymarket RTDS (which relays Binance)
//!
//! Usage:
//!   cargo run --release --bin compare_price_feeds
//!
//! Shows how much latency RTDS adds on top of direct Binance connection

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn, error};

const BINANCE_WS_URL: &str = "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade/solusdt@trade/xrpusdt@trade";
const RTDS_WS_URL: &str = "wss://ws-live-data.polymarket.com";

#[derive(Debug, Clone)]
struct PriceUpdate {
    source: &'static str,
    symbol: String,
    price: f64,
    source_ts_ms: u64,
    local_ts_ms: u64,
}

#[derive(Debug, Default)]
struct LatencyStats {
    binance_updates: u64,
    rtds_updates: u64,
    binance_first: u64,
    rtds_first: u64,
    total_comparisons: u64,
    avg_delta_ms: f64,
    deltas: Vec<i64>,
    last_binance_price: f64,
    last_rtds_price: f64,
}

// Binance trade message
#[derive(Debug, Deserialize)]
struct BinanceStreamWrapper {
    stream: String,
    data: BinanceTrade,
}

#[derive(Debug, Deserialize)]
struct BinanceTrade {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "T")]
    trade_time: u64,
}

// RTDS message types
#[derive(Debug, Deserialize)]
struct RtdsMessage {
    #[serde(rename = "type")]
    msg_type: Option<String>,
    topic: Option<String>,
    data: Option<RtdsData>,
}

#[derive(Debug, Deserialize)]
struct RtdsData {
    symbol: Option<String>,
    price: Option<f64>,
    timestamp: Option<u64>,
}

type StatsState = Arc<RwLock<HashMap<String, LatencyStats>>>;
type LastUpdate = Arc<RwLock<HashMap<String, (u64, u64)>>>; // symbol -> (binance_ts, rtds_ts)

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn normalize_symbol(s: &str) -> String {
    s.to_uppercase().replace("USDT", "USD")
}

async fn run_binance_feed(stats: StatsState, last_update: LastUpdate) -> Result<()> {
    info!("[BINANCE] Connecting to direct feed...");

    let (ws_stream, _) = connect_async(BINANCE_WS_URL).await?;
    let (mut write, mut read) = ws_stream.split();

    info!("[BINANCE] Connected - streaming BTC, ETH, SOL, XRP trades");

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                let local_ts = now_ms();

                if let Ok(wrapper) = serde_json::from_str::<BinanceStreamWrapper>(&text) {
                    let symbol = normalize_symbol(&wrapper.data.symbol);
                    let price: f64 = wrapper.data.price.parse().unwrap_or(0.0);
                    let source_ts = wrapper.data.trade_time;

                    // Update stats
                    let mut stats_guard = stats.write().await;
                    let entry = stats_guard.entry(symbol.clone()).or_default();
                    entry.binance_updates += 1;
                    entry.last_binance_price = price;

                    // Check for RTDS comparison
                    let mut last_guard = last_update.write().await;
                    let times = last_guard.entry(symbol.clone()).or_insert((0, 0));
                    times.0 = local_ts;

                    // If RTDS updated within 500ms, compare
                    if times.1 > 0 && local_ts.saturating_sub(times.1) < 500 {
                        let delta = local_ts as i64 - times.1 as i64;

                        if delta < 0 {
                            entry.binance_first += 1;
                        } else {
                            entry.rtds_first += 1;
                        }

                        entry.total_comparisons += 1;
                        entry.deltas.push(delta);
                        if entry.deltas.len() > 1000 {
                            entry.deltas.remove(0);
                        }

                        let sum: i64 = entry.deltas.iter().sum();
                        entry.avg_delta_ms = sum as f64 / entry.deltas.len() as f64;

                        let winner = if delta < 0 { "BINANCE" } else { "RTDS" };
                        info!(
                            "[{}] BINANCE ${:.2} | delta {}ms ({} first) | B:{} R:{} | avg {:.1}ms",
                            symbol, price, delta.abs(), winner,
                            entry.binance_first, entry.rtds_first, entry.avg_delta_ms
                        );
                    }
                }
            }
            Ok(Message::Ping(data)) => {
                let _ = write.send(Message::Pong(data)).await;
            }
            Err(e) => {
                error!("[BINANCE] Error: {}", e);
                break;
            }
            _ => {}
        }
    }

    Ok(())
}

async fn run_rtds_feed(stats: StatsState, last_update: LastUpdate) -> Result<()> {
    info!("[RTDS] Connecting to Polymarket RTDS...");

    let (ws_stream, _) = connect_async(RTDS_WS_URL).await?;
    let (mut write, mut read) = ws_stream.split();

    // Subscribe to crypto prices (Binance source)
    let sub = r#"{"type":"subscribe","topic":"crypto_prices","filters":"btcusdt,ethusdt,solusdt,xrpusdt"}"#;
    write.send(Message::Text(sub.to_string())).await?;
    info!("[RTDS] Subscribed to crypto_prices");

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                let local_ts = now_ms();

                if let Ok(m) = serde_json::from_str::<RtdsMessage>(&text) {
                    if let Some(data) = m.data {
                        if let (Some(sym), Some(price)) = (data.symbol, data.price) {
                            let symbol = normalize_symbol(&sym);

                            // Update stats
                            let mut stats_guard = stats.write().await;
                            let entry = stats_guard.entry(symbol.clone()).or_default();
                            entry.rtds_updates += 1;
                            entry.last_rtds_price = price;

                            // Check for Binance comparison
                            let mut last_guard = last_update.write().await;
                            let times = last_guard.entry(symbol.clone()).or_insert((0, 0));
                            times.1 = local_ts;

                            // If Binance updated within 500ms, compare
                            if times.0 > 0 && local_ts.saturating_sub(times.0) < 500 {
                                let delta = times.0 as i64 - local_ts as i64;

                                if delta < 0 {
                                    entry.binance_first += 1;
                                } else {
                                    entry.rtds_first += 1;
                                }

                                entry.total_comparisons += 1;
                                entry.deltas.push(delta);
                                if entry.deltas.len() > 1000 {
                                    entry.deltas.remove(0);
                                }

                                let sum: i64 = entry.deltas.iter().sum();
                                entry.avg_delta_ms = sum as f64 / entry.deltas.len() as f64;

                                let winner = if delta < 0 { "BINANCE" } else { "RTDS" };
                                info!(
                                    "[{}] RTDS    ${:.2} | delta {}ms ({} first) | B:{} R:{} | avg {:.1}ms",
                                    symbol, price, delta.abs(), winner,
                                    entry.binance_first, entry.rtds_first, entry.avg_delta_ms
                                );
                            }
                        }
                    }
                } else {
                    // Log unknown messages for debugging
                    if !text.contains("connected") && !text.contains("subscribed") {
                        warn!("[RTDS] Unknown: {}", &text[..text.len().min(100)]);
                    } else {
                        info!("[RTDS] {}", text);
                    }
                }
            }
            Ok(Message::Ping(data)) => {
                let _ = write.send(Message::Pong(data)).await;
            }
            Err(e) => {
                error!("[RTDS] Error: {}", e);
                break;
            }
            _ => {}
        }
    }

    Ok(())
}

async fn print_summary(stats: StatsState) {
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;

        let stats_guard = stats.read().await;

        println!("\n============ LATENCY COMPARISON: BINANCE vs RTDS ============\n");
        println!("{:<10} {:>14} {:>14} {:>10} {:>12} {:>12}",
                 "SYMBOL", "BINANCE_WINS", "RTDS_WINS", "AVG_Δms", "BIN_UPD", "RTDS_UPD");
        println!("{}", "-".repeat(76));

        let mut total_binance = 0u64;
        let mut total_rtds = 0u64;
        let mut overall_delta = 0.0f64;
        let mut count = 0;

        for (symbol, s) in stats_guard.iter() {
            let total = s.binance_first + s.rtds_first;
            if total > 0 {
                let bin_pct = s.binance_first as f64 / total as f64 * 100.0;
                let rtds_pct = s.rtds_first as f64 / total as f64 * 100.0;

                println!(
                    "{:<10} {:>10} ({:>3.0}%) {:>10} ({:>3.0}%) {:>+10.1} {:>12} {:>12}",
                    symbol,
                    s.binance_first, bin_pct,
                    s.rtds_first, rtds_pct,
                    s.avg_delta_ms,
                    s.binance_updates,
                    s.rtds_updates
                );

                total_binance += s.binance_first;
                total_rtds += s.rtds_first;
                overall_delta += s.avg_delta_ms;
                count += 1;
            }
        }

        if count > 0 {
            println!("{}", "-".repeat(76));
            let total = total_binance + total_rtds;
            let bin_pct = total_binance as f64 / total as f64 * 100.0;
            let rtds_pct = total_rtds as f64 / total as f64 * 100.0;
            println!(
                "{:<10} {:>10} ({:>3.0}%) {:>10} ({:>3.0}%) {:>+10.1}",
                "TOTAL",
                total_binance, bin_pct,
                total_rtds, rtds_pct,
                overall_delta / count as f64
            );
        }

        println!("\nNegative avg Δ = Binance faster (RTDS adds latency)");
        println!("Positive avg Δ = RTDS faster (unlikely, would indicate clock skew)");
        println!("================================================================\n");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  BINANCE vs RTDS Latency Comparison                          ║");
    println!("║  Direct Binance feed vs Polymarket RTDS (Binance relay)      ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Binance: {}     ║", BINANCE_WS_URL);
    println!("║  RTDS:    {}                       ║", RTDS_WS_URL);
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let stats: StatsState = Arc::new(RwLock::new(HashMap::new()));
    let last_update: LastUpdate = Arc::new(RwLock::new(HashMap::new()));

    // Spawn Binance feed
    let binance_stats = stats.clone();
    let binance_last = last_update.clone();
    tokio::spawn(async move {
        loop {
            if let Err(e) = run_binance_feed(binance_stats.clone(), binance_last.clone()).await {
                error!("[BINANCE] Feed error: {}", e);
            }
            warn!("[BINANCE] Reconnecting in 3s...");
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    });

    // Spawn RTDS feed
    let rtds_stats = stats.clone();
    let rtds_last = last_update.clone();
    tokio::spawn(async move {
        loop {
            if let Err(e) = run_rtds_feed(rtds_stats.clone(), rtds_last.clone()).await {
                error!("[RTDS] Feed error: {}", e);
            }
            warn!("[RTDS] Reconnecting in 3s...");
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    });

    // Print summary every 30s
    print_summary(stats).await;

    Ok(())
}
