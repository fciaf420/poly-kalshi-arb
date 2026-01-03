//! Simple Polymarket credential checker
//! Tests that POLY_PRIVATE_KEY and POLY_FUNDER are correctly configured.

use anyhow::Result;
use arb_bot::polymarket_clob::{PolymarketAsyncClient, PreparedCreds, SharedAsyncClient};

const POLY_CLOB_HOST: &str = "https://clob.polymarket.com";
const POLYGON_CHAIN_ID: u64 = 137;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("════════════════════════════════════════════════════════════════════════");
    println!("🔐 POLYMARKET CREDENTIAL CHECK");
    println!("════════════════════════════════════════════════════════════════════════");

    let poly_private_key = std::env::var("POLY_PRIVATE_KEY")
        .expect("POLY_PRIVATE_KEY must be set in .env");
    let poly_funder = std::env::var("POLY_FUNDER")
        .expect("POLY_FUNDER must be set in .env");

    println!("✓ POLY_PRIVATE_KEY: {}...{}", &poly_private_key[..6], &poly_private_key[poly_private_key.len()-4..]);
    println!("✓ POLY_FUNDER: {}", poly_funder);

    println!("\n[1/3] Creating client...");
    let client = PolymarketAsyncClient::new(
        POLY_CLOB_HOST,
        POLYGON_CHAIN_ID,
        &poly_private_key,
        &poly_funder,
    )?;
    println!("  ✓ Client created");
    println!("  ✓ Wallet address (EOA): {}", client.wallet_address());
    println!("  ✓ Funder address: {}", client.funder());

    println!("\n[2/3] Deriving API credentials from Polymarket...");
    let api_creds = client.derive_api_key(0).await?;
    println!("  ✓ API Key: {}...", &api_creds.api_key[..12]);
    println!("  ✓ API credentials derived successfully!");

    println!("\n[3/3] Creating trading client...");
    let prepared_creds = PreparedCreds::from_api_creds(&api_creds)?;
    let _shared = SharedAsyncClient::new(client, prepared_creds, POLYGON_CHAIN_ID);
    println!("  ✓ Trading client ready!");

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("✅ ALL CHECKS PASSED - Your Polymarket credentials are configured correctly!");
    println!("   You can now run with --live flag");
    println!("════════════════════════════════════════════════════════════════════════");

    Ok(())
}
