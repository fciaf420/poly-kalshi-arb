//! Polymarket Ping Pong Bot
//!
//! STRATEGY:
//! - Buy when price drops to a low threshold (e.g., 30¢)
//! - When price reverts above a higher threshold (e.g., 50¢), look for arb
//! - If YES + NO asks < $1, buy both sides for guaranteed profit
//!
//! Usage:
//!   cargo run --release --bin poly_ping_pong -- --buy-at 30 --arb-above 50 --max-arb-cost 99
//!
//! Environment:
//!   POLY_PRIVATE_KEY - Your Polymarket/Polygon wallet private key
//!   POLY_FUNDER - Your funder address (proxy wallet)
//!   POLYGON_API_KEY - Polygon.io API key for price feed (optional)

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

use arb_bot::polymarket_clob::{PolymarketAsyncClient, PreparedCreds, SharedAsyncClient};

/// Polymarket Ping Pong Bot
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Symbol to trade: BTC, ETH, SOL, XRP (default: all)
    #[arg(long, default_value = "all")]
    sym: String,

    /// Buy threshold in cents - buy when price drops to this level (e.g., 30 means buy at 30c or below)
    #[arg(short, long, default_value_t = 30)]
    threshold: i64,

    /// Max total cost in cents to buy both sides for arb (default: 97 = 3c profit)
    #[arg(long, default_value_t = 97)]
    max_arb_cost: i64,

    /// Number of contracts to buy per trade
    #[arg(short, long, default_value_t = 50.0)]
    contracts: f64,

    /// Live trading mode (default is dry run)
    #[arg(short, long, default_value_t = false)]
    live: bool,

    /// Minimum minutes remaining to trade (default: 1)
    #[arg(long, default_value_t = 1)]
    min_minutes: i64,

    /// Maximum minutes remaining to trade (default: 14)
    #[arg(long, default_value_t = 14)]
    max_minutes: i64,

    /// Maximum market age in minutes (default: 5 = only trade first 5 mins)
    #[arg(long, default_value_t = 5.0)]
    max_age: f64,
}

const POLYMARKET_WS_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const GAMMA_API_BASE: &str = "https://gamma-api.polymarket.com";
const LOCAL_PRICE_SERVER: &str = "ws://127.0.0.1:9999";

/// Price feed state
#[derive(Debug, Default)]
struct PriceState {
    btc_price: Option<f64>,
    eth_price: Option<f64>,
    sol_price: Option<f64>,
    xrp_price: Option<f64>,
    last_update: Option<std::time::Instant>,
}

#[derive(Deserialize, Debug)]
struct LocalPriceUpdate {
    btc_price: Option<f64>,
    eth_price: Option<f64>,
    sol_price: Option<f64>,
    xrp_price: Option<f64>,
}

/// Market state
#[derive(Debug, Clone)]
struct Market {
    condition_id: String,
    question: String,
    yes_token: String,
    no_token: String,
    asset: String,
    end_time: Option<chrono::DateTime<chrono::Utc>>,  // Store actual end time, not minutes
    // Orderbook
    yes_ask: Option<i64>,
    yes_bid: Option<i64>,
    no_ask: Option<i64>,
    no_bid: Option<i64>,
    yes_ask_size: f64,
    no_ask_size: f64,
}

impl Market {
    /// Get minutes remaining until expiry (recalculated each call)
    fn minutes_remaining(&self) -> Option<f64> {
        self.end_time.map(|end| {
            let now = Utc::now();
            let diff = end.signed_duration_since(now);
            diff.num_seconds() as f64 / 60.0
        })
    }
}

/// Position tracking
#[derive(Debug, Clone, Default)]
struct Position {
    yes_qty: f64,
    no_qty: f64,
    yes_cost: f64,
    no_cost: f64,
    // Track if we've bought during the crash
    bought_crash: bool,
}

impl Position {
    fn matched(&self) -> f64 {
        self.yes_qty.min(self.no_qty)
    }
    fn total_cost(&self) -> f64 {
        self.yes_cost + self.no_cost
    }
    fn locked_profit(&self) -> f64 {
        // Matched pairs pay $1
        self.matched() - self.total_cost()
    }
}

/// Global state
struct State {
    markets: HashMap<String, Market>,
    positions: HashMap<String, Position>,
    prices: PriceState,
}

impl State {
    fn new() -> Self {
        Self {
            markets: HashMap::new(),
            positions: HashMap::new(),
            prices: PriceState::default(),
        }
    }
}

/// Connect to local price server
async fn run_price_feed(state: Arc<RwLock<State>>) {
    loop {
        info!("[PRICES] Connecting to local price server {}...", LOCAL_PRICE_SERVER);

        match connect_async(LOCAL_PRICE_SERVER).await {
            Ok((ws, _)) => {
                info!("[PRICES] Connected to local price server");
                let (mut write, mut read) = ws.split();

                while let Some(msg) = read.next().await {
                    match msg {
                        Ok(Message::Text(text)) => {
                            if let Ok(update) = serde_json::from_str::<LocalPriceUpdate>(&text) {
                                let mut s = state.write().await;
                                if let Some(btc) = update.btc_price {
                                    s.prices.btc_price = Some(btc);
                                }
                                if let Some(eth) = update.eth_price {
                                    s.prices.eth_price = Some(eth);
                                }
                                if let Some(sol) = update.sol_price {
                                    s.prices.sol_price = Some(sol);
                                }
                                if let Some(xrp) = update.xrp_price {
                                    s.prices.xrp_price = Some(xrp);
                                }
                                s.prices.last_update = Some(std::time::Instant::now());
                            }
                        }
                        Ok(Message::Ping(data)) => {
                            let _ = write.send(Message::Pong(data)).await;
                        }
                        Ok(Message::Close(_)) | Err(_) => break,
                        _ => {}
                    }
                }
            }
            Err(e) => {
                warn!("[PRICES] Failed to connect: {}", e);
            }
        }

        warn!("[PRICES] Disconnected, reconnecting in 3s...");
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}

// === Gamma API for market discovery ===

#[derive(Debug, Deserialize)]
struct GammaSeries {
    events: Option<Vec<GammaEvent>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GammaEvent {
    slug: Option<String>,
    title: Option<String>,
    closed: Option<bool>,
    #[serde(rename = "endDate")]
    end_date: Option<String>,
    #[serde(rename = "enableOrderBook")]
    enable_order_book: Option<bool>,
}

/// Crypto series slugs for 15-minute markets
const POLY_SERIES_SLUGS: &[(&str, &str)] = &[
    ("btc-up-or-down-15m", "BTC"),
    ("eth-up-or-down-15m", "ETH"),
    ("sol-up-or-down-15m", "SOL"),
    ("xrp-up-or-down-15m", "XRP"),
];

/// Discover crypto markets on Polymarket
async fn discover_markets(market_filter: Option<&str>) -> Result<Vec<Market>> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    let mut markets = Vec::new();

    let series_to_check: Vec<(&str, &str)> = if let Some(filter) = market_filter {
        let filter_lower = filter.to_lowercase();
        POLY_SERIES_SLUGS
            .iter()
            .filter(|(slug, asset)| {
                slug.contains(&filter_lower) || asset.to_lowercase().contains(&filter_lower)
            })
            .copied()
            .collect()
    } else {
        POLY_SERIES_SLUGS.to_vec()
    };

    for (series_slug, asset) in series_to_check {
        let url = format!("{}/series?slug={}", GAMMA_API_BASE, series_slug);

        let resp = client
            .get(&url)
            .header("User-Agent", "poly_ping_pong/1.0")
            .send()
            .await?;

        if !resp.status().is_success() {
            warn!(
                "[DISCOVER] Failed to fetch series '{}': {}",
                series_slug,
                resp.status()
            );
            continue;
        }

        let series_list: Vec<GammaSeries> = resp.json().await?;
        let Some(series) = series_list.first() else {
            continue;
        };
        let Some(events) = &series.events else {
            continue;
        };

        let event_slugs: Vec<String> = events
            .iter()
            .filter(|e| e.closed != Some(true) && e.enable_order_book == Some(true))
            .filter_map(|e| e.slug.clone())
            .take(20)
            .collect();

        for event_slug in event_slugs {
            let event_url = format!("{}/events?slug={}", GAMMA_API_BASE, event_slug);
            let resp = match client
                .get(&event_url)
                .header("User-Agent", "poly_ping_pong/1.0")
                .send()
                .await
            {
                Ok(r) => r,
                Err(_) => continue,
            };

            let event_details: Vec<serde_json::Value> = match resp.json().await {
                Ok(ed) => ed,
                Err(_) => continue,
            };

            let Some(ed) = event_details.first() else {
                continue;
            };
            let Some(mkts) = ed.get("markets").and_then(|m| m.as_array()) else {
                continue;
            };

            for mkt in mkts {
                let condition_id = mkt
                    .get("conditionId")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let clob_tokens_str = mkt
                    .get("clobTokenIds")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let question = mkt
                    .get("question")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| event_slug.clone());
                let end_date_str = mkt
                    .get("endDate")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let Some(cid) = condition_id else {
                    continue;
                };
                let Some(cts) = clob_tokens_str else {
                    continue;
                };

                let token_ids: Vec<String> = serde_json::from_str(&cts).unwrap_or_default();
                if token_ids.len() < 2 {
                    continue;
                }

                let end_time = end_date_str.as_ref().and_then(|d| parse_end_time(d));

                if end_time.is_none() {
                    continue;
                }

                markets.push(Market {
                    condition_id: cid,
                    question,
                    yes_token: token_ids[0].clone(),
                    no_token: token_ids[1].clone(),
                    asset: asset.to_string(),
                    end_time,
                    yes_ask: None,
                    yes_bid: None,
                    no_ask: None,
                    no_bid: None,
                    yes_ask_size: 0.0,
                    no_ask_size: 0.0,
                });
            }
        }
    }

    let mut seen = std::collections::HashSet::new();
    markets.retain(|m| seen.insert(m.condition_id.clone()));

    markets.sort_by(|a, b| {
        a.end_time
            .cmp(&b.end_time)
    });

    Ok(markets)
}

fn parse_end_time(end_date: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    let dt = chrono::DateTime::parse_from_rfc3339(end_date).ok()?;
    let utc_dt = dt.with_timezone(&chrono::Utc);
    let now = Utc::now();
    if utc_dt > now {
        Some(utc_dt)
    } else {
        None
    }
}

// === Polymarket WebSocket ===

#[derive(Deserialize, Debug)]
struct BookSnapshot {
    asset_id: String,
    bids: Vec<PriceLevel>,
    asks: Vec<PriceLevel>,
}

#[derive(Deserialize, Debug)]
struct PriceLevel {
    price: String,
    size: String,
}

#[derive(Serialize)]
struct SubscribeCmd {
    assets_ids: Vec<String>,
    #[serde(rename = "type")]
    sub_type: &'static str,
}

fn parse_price_cents(s: &str) -> i64 {
    s.parse::<f64>()
        .map(|p| (p * 100.0).round() as i64)
        .unwrap_or(0)
}

fn parse_size(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or(0.0)
}

// === Main ===

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("poly_ping_pong=info".parse().unwrap())
                .add_directive("arb_bot=info".parse().unwrap()),
        )
        .init();

    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("POLYMARKET PING PONG BOT");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("Symbol: {}  |  Mode: {}  |  Contracts: {:.0}",
             args.sym.to_uppercase(),
             if args.live { "LIVE" } else { "DRY RUN" },
             args.contracts);
    println!("Threshold: {}c / {}c  |  Max arb: {}c  |  Time: {}m-{}m  |  Max age: {:.0}m",
             args.threshold, 100 - args.threshold, args.max_arb_cost,
             args.min_minutes, args.max_minutes, args.max_age);
    println!("═══════════════════════════════════════════════════════════════════════");

    dotenvy::dotenv().ok();
    let private_key = std::env::var("POLY_PRIVATE_KEY").context("POLY_PRIVATE_KEY not set")?;
    let funder = std::env::var("POLY_FUNDER").context("POLY_FUNDER not set")?;

    let poly_client = PolymarketAsyncClient::new("https://clob.polymarket.com", 137, &private_key, &funder)?;

    println!("Deriving API credentials...");
    let api_creds = poly_client.derive_api_key(0).await?;
    let prepared_creds = PreparedCreds::from_api_creds(&api_creds)?;
    let shared_client = Arc::new(SharedAsyncClient::new(poly_client, prepared_creds, 137));

    if args.live {
        println!("*** LIVE MODE - Real money! ***");
        tokio::time::sleep(Duration::from_secs(3)).await;
    }

    // Filter by symbol
    let symbol_filter: Option<String> = if args.sym.to_lowercase() == "all" {
        None
    } else {
        Some(args.sym.to_lowercase())
    };

    println!("Discovering markets...");
    let discovered = discover_markets(symbol_filter.as_deref()).await?;
    println!("Found {} markets", discovered.len());

    for m in &discovered {
        println!("  {} | {:.1}m left | {}",
                 m.asset,
                 m.minutes_remaining().unwrap_or(0.0),
                 &m.question[..m.question.len().min(50)]);
    }

    if discovered.is_empty() {
        println!("No markets found!");
        return Ok(());
    }

    let state = Arc::new(RwLock::new({
        let mut s = State::new();
        for m in discovered {
            let id = m.condition_id.clone();
            s.positions.insert(id.clone(), Position::default());
            s.markets.insert(id, m);
        }
        s
    }));

    // Start price feed
    let state_price = state.clone();
    tokio::spawn(async move {
        run_price_feed(state_price).await;
    });

    let tokens: Vec<String> = {
        let s = state.read().await;
        s.markets
            .values()
            .flat_map(|m| vec![m.yes_token.clone(), m.no_token.clone()])
            .collect()
    };

    let threshold = args.threshold;
    let max_arb_cost = args.max_arb_cost;
    let contracts = args.contracts;
    let dry_run = !args.live;
    let min_minutes = args.min_minutes as f64;
    let max_minutes = args.max_minutes as f64;
    let max_age = args.max_age;

    loop {
        info!("[WS] Connecting to Polymarket WebSocket {}...", POLYMARKET_WS_URL);

        let (ws, _) = match connect_async(POLYMARKET_WS_URL).await {
            Ok(ws) => ws,
            Err(e) => {
                error!("[WS] Connect failed: {}", e);
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        let (mut write, mut read) = ws.split();

        let sub = SubscribeCmd {
            assets_ids: tokens.clone(),
            sub_type: "market",
        };
        let _ = write.send(Message::Text(serde_json::to_string(&sub)?)).await;
        info!("[WS] Connected! Subscribed to {} tokens across {} markets", tokens.len(), tokens.len() / 2);
        println!("══════════════════════════════════════════════════════════════════════════════════════════════════════");
        println!("Sym=Asset | age=market age | left=time to expiry | YES bid/ask | NO bid/ask | sp=spread | liq=liquidity");
        println!("══════════════════════════════════════════════════════════════════════════════════════════════════════");

        let mut ping_interval = tokio::time::interval(Duration::from_secs(30));
        let mut status_interval = tokio::time::interval(Duration::from_secs(1));

        loop {
            tokio::select! {
                _ = ping_interval.tick() => {
                    if let Err(e) = write.send(Message::Ping(vec![])).await {
                        error!("[WS] Ping failed: {}", e);
                        break;
                    }
                }

                _ = status_interval.tick() => {
                    let s = state.read().await;
                    let now = Utc::now().format("%H:%M:%S");

                    // Price feed staleness
                    let price_age_secs = s.prices.last_update
                        .map(|t| t.elapsed().as_secs())
                        .unwrap_or(999);
                    let price_status = if price_age_secs < 5 { "●" } else if price_age_secs < 30 { "○" } else { "✗" };

                    for (id, market) in &s.markets {
                        let pos = s.positions.get(id).cloned().unwrap_or_default();
                        let expiry = market.minutes_remaining().unwrap_or(0.0);
                        let market_age = 15.0 - expiry; // How many minutes into the market

                        if expiry < min_minutes || expiry > max_minutes {
                            continue;
                        }
                        if market_age > max_age {
                            continue; // Market too old
                        }

                        // Get spot price for this asset
                        let spot = match market.asset.as_str() {
                            "BTC" => s.prices.btc_price.map(|p| format!("${:.0}", p)),
                            "ETH" => s.prices.eth_price.map(|p| format!("${:.0}", p)),
                            "SOL" => s.prices.sol_price.map(|p| format!("${:.2}", p)),
                            "XRP" => s.prices.xrp_price.map(|p| format!("${:.4}", p)),
                            _ => None,
                        }.unwrap_or_else(|| "-".to_string());

                        let yes_ask = market.yes_ask.unwrap_or(100);
                        let no_ask = market.no_ask.unwrap_or(100);
                        let yes_bid = market.yes_bid.unwrap_or(0);
                        let no_bid = market.no_bid.unwrap_or(0);
                        let combined = yes_ask + no_ask;

                        // Spreads
                        let yes_spread = yes_ask - yes_bid;
                        let no_spread = no_ask - no_bid;

                        // Arb analysis
                        let arb_profit = if combined < 100 { 100 - combined } else { 0 };
                        let arb_flag = if combined <= max_arb_cost {
                            format!("*ARB +{}c*", arb_profit)
                        } else {
                            format!("gap={}c", combined - 100)
                        };

                        // Distance from threshold
                        let yes_to_thresh = yes_ask - threshold;
                        let no_to_thresh = no_ask - threshold;
                        let thresh_flag = if yes_to_thresh <= 5 || no_to_thresh <= 5 {
                            format!("NEAR(Y{}c N{}c)", yes_to_thresh, no_to_thresh)
                        } else {
                            String::new()
                        };

                        // Position P&L (if any position)
                        let pnl_str = if pos.yes_qty > 0.0 || pos.no_qty > 0.0 {
                            let matched = pos.matched();
                            let unrealized = matched * 1.0 - pos.total_cost();
                            format!(" | PnL=${:.2} (Y={:.0}@{:.0}c N={:.0}@{:.0}c)",
                                unrealized,
                                pos.yes_qty,
                                if pos.yes_qty > 0.0 { pos.yes_cost / pos.yes_qty * 100.0 } else { 0.0 },
                                pos.no_qty,
                                if pos.no_qty > 0.0 { pos.no_cost / pos.no_qty * 100.0 } else { 0.0 }
                            )
                        } else {
                            String::new()
                        };

                        // Liquidity
                        let liq_str = format!("liq Y={:.0} N={:.0}", market.yes_ask_size, market.no_ask_size);

                        // Mid price implied probability
                        let yes_mid = (yes_bid + yes_ask) as f64 / 2.0;
                        let implied_prob = format!("prob={:.0}%", yes_mid);

                        println!(
                            "{} {} {} spot={} | age={:.1}m left={:.1}m | Y {}/{}c (sp{}c) N {}/{}c (sp{}c) | sum={}c {} | {} | {} {} {}{}",
                            now,
                            price_status,
                            market.asset,
                            spot,
                            market_age,
                            expiry,
                            yes_bid, yes_ask, yes_spread,
                            no_bid, no_ask, no_spread,
                            combined,
                            arb_flag,
                            implied_prob,
                            liq_str,
                            thresh_flag,
                            if pos.bought_crash { "[BOUGHT]" } else { "" },
                            pnl_str
                        );
                    }
                }

                msg = read.next() => {
                    let Some(msg) = msg else { break; };

                    match msg {
                        Ok(Message::Text(text)) => {
                            if let Ok(books) = serde_json::from_str::<Vec<BookSnapshot>>(&text) {
                                for book in books {
                                    let mut s = state.write().await;

                                    let market_id = s
                                        .markets
                                        .iter()
                                        .find(|(_, m)| {
                                            m.yes_token == book.asset_id || m.no_token == book.asset_id
                                        })
                                        .map(|(id, _)| id.clone());

                                    let Some(market_id) = market_id else { continue };

                                    let best_ask = book
                                        .asks
                                        .iter()
                                        .filter_map(|l| {
                                            let price = parse_price_cents(&l.price);
                                            if price > 0 {
                                                Some((price, parse_size(&l.size)))
                                            } else {
                                                None
                                            }
                                        })
                                        .min_by_key(|(p, _)| *p);

                                    let best_bid = book
                                        .bids
                                        .iter()
                                        .filter_map(|l| {
                                            let price = parse_price_cents(&l.price);
                                            if price > 0 {
                                                Some((price, parse_size(&l.size)))
                                            } else {
                                                None
                                            }
                                        })
                                        .max_by_key(|(p, _)| *p);

                                    let Some(market) = s.markets.get_mut(&market_id) else {
                                        continue;
                                    };

                                    let is_yes = book.asset_id == market.yes_token;

                                    if is_yes {
                                        market.yes_ask = best_ask.map(|(p, _)| p);
                                        market.yes_bid = best_bid.map(|(p, _)| p);
                                        market.yes_ask_size = best_ask.map(|(_, s)| s).unwrap_or(0.0);
                                    } else {
                                        market.no_ask = best_ask.map(|(p, _)| p);
                                        market.no_bid = best_bid.map(|(p, _)| p);
                                        market.no_ask_size = best_ask.map(|(_, s)| s).unwrap_or(0.0);
                                    }

                                    // Extract values
                                    let asset = market.asset.clone();
                                    let mins = market.minutes_remaining().unwrap_or(0.0);
                                    let yes_ask_price = market.yes_ask;
                                    let no_ask_price = market.no_ask;
                                    let yes_token = market.yes_token.clone();
                                    let no_token = market.no_token.clone();
                                    let _question = market.question.clone();

                                    // Get underlying price for this asset
                                    let underlying_price = match asset.as_str() {
                                        "BTC" => s.prices.btc_price,
                                        "ETH" => s.prices.eth_price,
                                        "SOL" => s.prices.sol_price,
                                        "XRP" => s.prices.xrp_price,
                                        _ => None,
                                    };

                                    let market_age = 15.0 - mins; // How many minutes into the market
                                    if mins < min_minutes || mins > max_minutes {
                                        continue;
                                    }
                                    if market_age > max_age {
                                        continue; // Market too old, skip trading
                                    }

                                    let yes_ask = yes_ask_price.unwrap_or(100);
                                    let no_ask = no_ask_price.unwrap_or(100);

                                    // Get position
                                    let pos = s.positions.get(&market_id).cloned().unwrap_or_default();
                                    let market_id_clone = market_id.clone();
                                    drop(s);

                                    // Format underlying price
                                    let spot_str = underlying_price
                                        .map(|p| format!("${:.2}", p))
                                        .unwrap_or_else(|| "-".to_string());
                                    let now = Utc::now().format("%H:%M:%S");

                                    // === STRATEGY 1: Buy on crash ===
                                    // Buy YES if it drops to threshold threshold
                                    if yes_ask <= threshold && yes_ask > 0 {
                                        if dry_run {
                                            info!("[DRY] {} {} YES @{}c <= {}c | age={:.0}m spot={} | Would BUY {:.0} contracts",
                                                  now, asset, yes_ask, threshold, market_age, spot_str, contracts);
                                        } else {
                                            let price = yes_ask as f64 / 100.0;

                                            warn!(
                                                "[TRADE] BUY {:.0} YES @{}c | {}",
                                                contracts, yes_ask, asset
                                            );

                                            match shared_client.buy_fak(&yes_token, price, contracts).await {
                                                Ok(fill) => {
                                                    if fill.filled_size > 0.0 {
                                                        warn!(
                                                            "[TRADE] Filled {:.2} @${:.2}",
                                                            fill.filled_size, fill.fill_cost
                                                        );
                                                        let mut s = state.write().await;
                                                        if let Some(pos) = s.positions.get_mut(&market_id_clone) {
                                                            pos.yes_qty += fill.filled_size;
                                                            pos.yes_cost += fill.fill_cost;
                                                            pos.bought_crash = true;
                                                        }
                                                    }
                                                }
                                                Err(e) => error!("[TRADE] YES buy failed: {}", e),
                                            }
                                        }
                                    }

                                    // Buy NO if it drops to threshold
                                    if no_ask <= threshold && no_ask > 0 {
                                        if dry_run {
                                            info!("[DRY] {} {} NO @{}c <= {}c | age={:.0}m spot={} | Would BUY {:.0} contracts",
                                                  now, asset, no_ask, threshold, market_age, spot_str, contracts);
                                        } else {
                                            let price = no_ask as f64 / 100.0;

                                            warn!(
                                                "[TRADE] BUY {:.0} NO @{}c | {}",
                                                contracts, no_ask, asset
                                            );

                                            match shared_client.buy_fak(&no_token, price, contracts).await {
                                                Ok(fill) => {
                                                    if fill.filled_size > 0.0 {
                                                        warn!(
                                                            "[TRADE] Filled {:.2} @${:.2}",
                                                            fill.filled_size, fill.fill_cost
                                                        );
                                                        let mut s = state.write().await;
                                                        if let Some(pos) = s.positions.get_mut(&market_id_clone) {
                                                            pos.no_qty += fill.filled_size;
                                                            pos.no_cost += fill.fill_cost;
                                                            pos.bought_crash = true;
                                                        }
                                                    }
                                                }
                                                Err(e) => error!("[TRADE] NO buy failed: {}", e),
                                            }
                                        }
                                    }

                                    // === STRATEGY 2: Arb when both sides cheap ===
                                    // If YES + NO <= max_arb_cost, buy both for guaranteed profit
                                    let combined = yes_ask + no_ask;
                                    if combined <= max_arb_cost && yes_ask > 0 && no_ask > 0 {
                                        let profit = 100 - combined;

                                        if dry_run {
                                            info!("[DRY] {} {} ARB Y={}c + N={}c = {}c | age={:.0}m spot={} | {:.0} contracts | profit {}c",
                                                  now, asset, yes_ask, no_ask, combined, market_age, spot_str, contracts, profit);
                                        } else {
                                            // Buy both sides
                                            let yes_price = yes_ask as f64 / 100.0;
                                            let no_price = no_ask as f64 / 100.0;

                                            warn!(
                                                "[TRADE] ARB BUY {:.0} YES @{}c + NO @{}c | profit {}c",
                                                contracts, yes_ask, no_ask, profit
                                            );

                                            // Buy YES
                                            match shared_client.buy_fak(&yes_token, yes_price, contracts).await
                                            {
                                                Ok(fill) => {
                                                    if fill.filled_size > 0.0 {
                                                        warn!(
                                                            "[TRADE] YES Filled {:.2} @${:.2}",
                                                            fill.filled_size, fill.fill_cost
                                                        );
                                                        let mut s = state.write().await;
                                                        if let Some(pos) = s.positions.get_mut(&market_id_clone)
                                                        {
                                                            pos.yes_qty += fill.filled_size;
                                                            pos.yes_cost += fill.fill_cost;
                                                        }
                                                    }
                                                }
                                                Err(e) => error!("[TRADE] YES arb buy failed: {}", e),
                                            }

                                            // Buy NO
                                            match shared_client.buy_fak(&no_token, no_price, contracts).await {
                                                Ok(fill) => {
                                                    if fill.filled_size > 0.0 {
                                                        warn!(
                                                            "[TRADE] NO Filled {:.2} @${:.2}",
                                                            fill.filled_size, fill.fill_cost
                                                        );
                                                        let mut s = state.write().await;
                                                        if let Some(pos) = s.positions.get_mut(&market_id_clone)
                                                        {
                                                            pos.no_qty += fill.filled_size;
                                                            pos.no_cost += fill.fill_cost;
                                                        }
                                                    }
                                                }
                                                Err(e) => error!("[TRADE] NO arb buy failed: {}", e),
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Ok(Message::Ping(data)) => {
                            let _ = write.send(Message::Pong(data)).await;
                        }
                        Ok(Message::Close(_)) | Err(_) => break,
                        _ => {}
                    }
                }
            }
        }

        println!("Disconnected, reconnecting in 5s...");
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}
