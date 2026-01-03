// Fetch positions from Kalshi/Polymarket and create positions.json
use arb_bot::kalshi::{KalshiApiClient, KalshiConfig};
use arb_bot::polymarket_clob::{PolymarketAsyncClient, PreparedCreds, SharedAsyncClient};
use arb_bot::position_tracker::*;
use arb_bot::types::MarketPair;
use std::collections::HashMap;
use serde::{Deserialize};

const POLYGON_CHAIN_ID: u64 = 137;
const DISCOVERY_CACHE_PATH: &str = ".discovery_cache.json";

#[derive(Deserialize)]
struct DiscoveryCache {
    pairs: Vec<MarketPair>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    println!("\n🔄 RECOVERING YOUR POSITIONS FROM EXCHANGES\n");

    // 0. Load discovery cache to get market pairings
    println!("0️⃣ Loading market pairings from discovery cache...");
    let cache_json = std::fs::read_to_string(DISCOVERY_CACHE_PATH)?;
    let cache: DiscoveryCache = serde_json::from_str(&cache_json)?;

    // Build lookup maps: Kalshi ticker → MarketPair, Poly token → MarketPair
    let mut kalshi_to_pair: HashMap<String, &MarketPair> = HashMap::new();
    let mut poly_token_to_pair: HashMap<String, (&MarketPair, &str)> = HashMap::new(); // (pair, "yes" or "no")

    for pair in &cache.pairs {
        kalshi_to_pair.insert(pair.kalshi_market_ticker.to_string(), pair);
        poly_token_to_pair.insert(pair.poly_yes_token.to_string(), (pair, "yes"));
        poly_token_to_pair.insert(pair.poly_no_token.to_string(), (pair, "no"));
    }

    println!("   Loaded {} market pairs", cache.pairs.len());

    // 1. Fetch from Kalshi
    println!("\n1️⃣ Fetching Kalshi positions...");
    let kalshi_config = KalshiConfig::from_env()?;
    let kalshi = KalshiApiClient::new(kalshi_config);

    let kalshi_positions = kalshi.get_positions().await?;
    println!("   Found {} Kalshi positions:\n", kalshi_positions.len());

    for (i, pos) in kalshi_positions.iter().enumerate() {
        let cost = pos.market_exposure as f64 / 100.0;  // cents to dollars
        let pnl = pos.realized_pnl as f64 / 100.0;
        let fees = pos.fees_paid as f64 / 100.0;

        println!("   {}. {} - {} contracts", i+1, pos.ticker, pos.position);
        println!("      Cost: ${:.2} | Realized P&L: ${:.2} | Fees: ${:.2}", cost, pnl, fees);
    }

    // 2. Fetch from Polymarket
    println!("\n2️⃣ Fetching Polymarket positions...");
    let poly_private_key = std::env::var("POLY_PRIVATE_KEY")?;
    let poly_funder = std::env::var("POLY_FUNDER")?;

    let poly_async_client = PolymarketAsyncClient::new(
        "https://clob.polymarket.com",
        POLYGON_CHAIN_ID,
        &poly_private_key,
        &poly_funder,
    )?;
    let api_creds = poly_async_client.derive_api_key(0).await?;
    let prepared_creds = PreparedCreds::from_api_creds(&api_creds)?;
    let poly = SharedAsyncClient::new(poly_async_client, prepared_creds, POLYGON_CHAIN_ID);

    let poly_positions = poly.get_positions().await?;
    println!("   Found {} Polymarket positions:\n", poly_positions.len());

    for (i, pos) in poly_positions.iter().enumerate() {
        println!("   {}. {} ({}) - {:.1} contracts @ ${:.2}",
                 i+1, pos.title, pos.outcome, pos.size, pos.avg_price);
        println!("      Initial: ${:.2} | Current: ${:.2} | P&L: ${:.2}",
                 pos.initial_value, pos.current_value, pos.cash_pnl);
    }

    // 3. Create position tracker with PAIRED positions
    println!("\n3️⃣ Matching positions to arbitrage pairs...");
    let mut tracker = PositionTracker::new();
    let mut matched_kalshi = 0;
    let mut matched_poly = 0;
    let mut unmatched_kalshi = Vec::new();
    let mut unmatched_poly = Vec::new();

    // Process Kalshi positions
    for pos in &kalshi_positions {
        if let Some(pair) = kalshi_to_pair.get(&pos.ticker) {
            let is_yes = pos.position > 0;
            let contracts = pos.position.abs() as f64;
            let avg_price = if contracts > 0.0 {
                (pos.market_exposure as f64 / 100.0) / contracts
            } else {
                0.0
            };

            let fill = FillRecord {
                market_id: pair.pair_id.to_string(),  // Use pair_id!
                description: pair.description.to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                platform: "kalshi".to_string(),
                side: if is_yes { "yes" } else { "no" }.to_string(),
                contracts,
                price: avg_price,
                fees: pos.fees_paid as f64 / 100.0,
                order_id: format!("recovered-kalshi-{}", pos.ticker),
            };

            tracker.record_fill_internal(&fill);
            matched_kalshi += 1;
            println!("   ✓ Matched Kalshi: {} → {}", pos.ticker, pair.pair_id);
        } else {
            unmatched_kalshi.push(&pos.ticker);
            println!("   ✗ Unmatched Kalshi: {} (no pairing in discovery cache)", pos.ticker);
        }
    }

    // Process Polymarket positions
    for pos in &poly_positions {
        if let Some((pair, side)) = poly_token_to_pair.get(&pos.asset) {
            let fill = FillRecord {
                market_id: pair.pair_id.to_string(),  // Use pair_id!
                description: pair.description.to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                platform: "polymarket".to_string(),
                side: side.to_string(),
                contracts: pos.size,
                price: pos.avg_price,
                fees: 0.0,
                order_id: format!("recovered-poly-{}", pos.asset),
            };

            tracker.record_fill_internal(&fill);
            matched_poly += 1;
            println!("   ✓ Matched Poly: {} ({}) → {}", pos.title, pos.outcome, pair.pair_id);
        } else {
            unmatched_poly.push((&pos.title, &pos.asset));
            println!("   ✗ Unmatched Poly: {} token={} (no pairing)", pos.title, pos.asset);
        }
    }

    // 4. Save
    println!("\n4️⃣ Saving to positions.json...");
    tracker.save()?;

    let summary = tracker.summary();
    println!("\n📊 RECOVERY SUMMARY");
    println!("   Matched Kalshi positions: {}/{}", matched_kalshi, kalshi_positions.len());
    println!("   Matched Poly positions: {}/{}", matched_poly, poly_positions.len());
    println!("   ✅ Created {} paired positions", summary.open_positions);
    println!("   Total contracts: {}", summary.total_contracts);
    println!("   Total cost: ${:.2}", summary.total_cost_basis);

    if !unmatched_kalshi.is_empty() || !unmatched_poly.is_empty() {
        println!("\n⚠️  WARNING: Some positions couldn't be matched:");
        if !unmatched_kalshi.is_empty() {
            println!("   Unmatched Kalshi: {:?}", unmatched_kalshi);
        }
        if !unmatched_poly.is_empty() {
            println!("   Unmatched Poly: {:?}", unmatched_poly.iter().map(|(t, _)| t).collect::<Vec<_>>());
        }
        println!("   These positions were likely from old markets no longer in the cache.");
    }

    println!("\n🎉 RECOVERY COMPLETE!");
    println!("\nNext steps:");
    println!("   1. Restart your bot: cargo run --release");
    println!("   2. Check dashboard: http://localhost:3000");
    println!("   3. Positions should now show as proper arbitrage pairs!");

    Ok(())
}
