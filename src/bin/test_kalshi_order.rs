// Test Kalshi order placement and fill_count parsing
use anyhow::Result;
use arb_bot::kalshi::{KalshiConfig, KalshiApiClient};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    println!("Loading Kalshi config...");
    let config = KalshiConfig::from_env()?;
    let client = KalshiApiClient::new(config);

    // Get balance first
    let balance = client.get_balance().await?;
    println!("Kalshi balance: ${:.2}", balance);

    if balance < 1.0 {
        println!("Need at least $1 to test. Exiting.");
        return Ok(());
    }

    // Find an active market to test with
    // Use a safe market - pick something with low price
    let test_ticker = "KXNBAGAME-25DEC18NYKIND-IND"; // Example - adjust as needed

    println!("\nPlacing test IOC order on {} ...", test_ticker);
    println!("  Side: yes, Price: 1 cent, Count: 1 contract");
    println!("  (This should NOT fill - price too low)");

    // Place a 1 cent order that won't fill (just to test parsing)
    match client.buy_ioc(test_ticker, "yes", 1, 1).await {
        Ok(resp) => {
            println!("\n✅ Order response received!");
            println!("  order_id: {}", resp.order.order_id);
            println!("  status: {}", resp.order.status);
            println!("  fill_count: {:?}", resp.order.fill_count);
            println!("  filled_count(): {}", resp.order.filled_count());
            println!("  remaining_count: {:?}", resp.order.remaining_count);
            println!("  taker_fill_cost: {:?}", resp.order.taker_fill_cost);
            println!("  maker_fill_cost: {:?}", resp.order.maker_fill_cost);
        }
        Err(e) => {
            println!("\n❌ Order failed: {}", e);
        }
    }

    // Now try a real order that should fill
    println!("\n\nPlacing REAL test order (50 cents, should fill)...");
    match client.buy_ioc(test_ticker, "yes", 50, 2).await {
        Ok(resp) => {
            println!("\n✅ Order response received!");
            println!("  order_id: {}", resp.order.order_id);
            println!("  status: {}", resp.order.status);
            println!("  fill_count: {:?}", resp.order.fill_count);
            println!("  filled_count(): {}", resp.order.filled_count());
            println!("  remaining_count: {:?}", resp.order.remaining_count);
            println!("  taker_fill_cost: {:?}", resp.order.taker_fill_cost);
            println!("  maker_fill_cost: {:?}", resp.order.maker_fill_cost);
        }
        Err(e) => {
            println!("\n❌ Order failed: {}", e);
        }
    }

    // Check balance after
    let balance_after = client.get_balance().await?;
    println!("\nKalshi balance after: ${:.2}", balance_after);
    println!("Spent: ${:.2}", balance - balance_after);

    Ok(())
}
