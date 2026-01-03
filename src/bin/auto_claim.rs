//! Auto-Claim for Polymarket Winning Positions
//!
//! Fetches redeemable positions from Polymarket Data API,
//! filters for winners (curPrice === 1), and claims them
//! via the CTF contract through a Gnosis Safe.
//!
//! Usage:
//!   cargo run --release --bin auto_claim              # Dry run (default)
//!   cargo run --release --bin auto_claim -- --live    # Execute claims

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use ethers::prelude::*;
use ethers::providers::{Http, Provider};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, Bytes, H256, U256};
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// CONTRACT ABIS (from TypeScript)
// ============================================================================

abigen!(
    CTF,
    r#"[
        function redeemPositions(address collateralToken, bytes32 parentCollectionId, bytes32 conditionId, uint256[] calldata indexSets) external
    ]"#
);

abigen!(
    GnosisSafe,
    r#"[
        function execTransaction(address to, uint256 value, bytes calldata data, uint8 operation, uint256 safeTxGas, uint256 baseGas, uint256 gasPrice, address gasToken, address refundReceiver, bytes memory signatures) external payable returns (bool success)
        function nonce() external view returns (uint256)
    ]"#
);

abigen!(
    ERC20,
    r#"[
        function balanceOf(address owner) external view returns (uint256)
    ]"#
);

// ============================================================================
// CONSTANTS (from TypeScript)
// ============================================================================

const CTF_ADDRESS: &str = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045";
const USDC_ADDRESS: &str = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174";
const DATA_API_URL: &str = "https://data-api.polymarket.com";
const POLYGON_GAS_STATION_URL: &str = "https://gasstation.polygon.technology/v2";
const DEFAULT_RPC_URL: &str = "https://polygon-rpc.com";
const POLYGON_CHAIN_ID: u64 = 137;

// ============================================================================
// CLI ARGUMENTS
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "auto_claim", about = "Auto-claim winning Polymarket positions")]
struct Args {
    /// Execute live claims (default is dry-run)
    #[arg(long, default_value_t = false)]
    live: bool,

    /// Run continuously, checking every N minutes (e.g., --loop 60 for hourly)
    #[arg(long, value_name = "MINUTES")]
    r#loop: Option<u64>,

    /// Polygon RPC URL (overrides POLYGON_RPC_URL env var)
    #[arg(long)]
    rpc_url: Option<String>,
}

// ============================================================================
// DATA STRUCTURES (from TypeScript)
// ============================================================================

/// Position from Polymarket Data API
#[derive(Debug, Deserialize)]
struct RedeemablePosition {
    asset: String,
    #[serde(rename = "conditionId")]
    condition_id: String,
    size: f64,
    #[serde(rename = "avgPrice", default)]
    avg_price: f64,
    #[serde(rename = "curPrice", default)]
    cur_price: f64,
    #[serde(default)]
    outcome: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    redeemable: bool,
}

/// Wallet balance info
#[derive(Debug)]
struct WalletBalance {
    usdc: f64,
    matic: f64,
}

/// Result of a claim attempt
#[derive(Debug)]
struct ClaimResult {
    condition_id: String,
    title: String,
    outcome: String,
    size: f64,
    tx_hash: Option<String>,
    error: Option<String>,
}

/// Gas station response (from TypeScript)
#[derive(Debug, Deserialize, Default)]
struct GasStationResponse {
    #[serde(default)]
    fast: GasPrice,
    #[serde(default)]
    standard: GasPrice,
}

#[derive(Debug, Deserialize, Default)]
struct GasPrice {
    #[serde(rename = "maxPriorityFee", default)]
    max_priority_fee: f64,
    #[serde(rename = "maxFee", default)]
    max_fee: f64,
}

// ============================================================================
// AUTO-CLAIM CLIENT
// ============================================================================

struct AutoClaimClient {
    provider: Arc<SignerMiddleware<Provider<Http>, LocalWallet>>,
    safe_address: Address,
    signer_address: Address,
    http_client: reqwest::Client,
}

impl AutoClaimClient {
    fn new(rpc_url: &str, private_key: &str, safe_address_str: &str) -> Result<Self> {
        let provider = Provider::<Http>::try_from(rpc_url)
            .context("Failed to create provider")?;

        let wallet: LocalWallet = private_key
            .parse::<LocalWallet>()
            .context("Failed to parse private key")?
            .with_chain_id(POLYGON_CHAIN_ID);

        let signer_address = wallet.address();
        let provider = Arc::new(SignerMiddleware::new(provider, wallet));

        let safe_address: Address = safe_address_str
            .parse()
            .context("Failed to parse safe address")?;

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;

        Ok(Self {
            provider,
            safe_address,
            signer_address,
            http_client,
        })
    }

    /// Get gas prices from Polygon Gas Station (from TypeScript)
    async fn get_polygon_gas_prices(&self) -> Result<(U256, U256)> {
        let resp = self.http_client
            .get(POLYGON_GAS_STATION_URL)
            .send()
            .await;

        let gas_prices = match resp {
            Ok(r) if r.status().is_success() => {
                r.json::<GasStationResponse>().await.unwrap_or_default()
            }
            _ => {
                println!("  Warning: Gas station unavailable, using defaults");
                GasStationResponse {
                    fast: GasPrice { max_priority_fee: 60.0, max_fee: 150.0 },
                    ..Default::default()
                }
            }
        };

        // Convert gwei to wei
        let max_priority_fee = U256::from((gas_prices.fast.max_priority_fee * 1e9) as u64);
        let max_fee = U256::from((gas_prices.fast.max_fee * 1e9) as u64);

        Ok((max_fee, max_priority_fee))
    }

    /// Get wallet balances (from TypeScript)
    async fn get_wallet_balances(&self) -> Result<WalletBalance> {
        // POL balance from signer (pays gas)
        let matic_balance = self.provider
            .get_balance(self.signer_address, None)
            .await
            .unwrap_or_default();
        let matic = matic_balance.as_u128() as f64 / 1e18;

        // USDC balance from Safe
        let usdc_address: Address = USDC_ADDRESS.parse()?;
        let usdc_contract = ERC20::new(usdc_address, self.provider.clone());
        let usdc_balance = usdc_contract
            .balance_of(self.safe_address)
            .call()
            .await
            .unwrap_or_default();
        let usdc = usdc_balance.as_u128() as f64 / 1e6;

        Ok(WalletBalance {
            usdc: (usdc * 100.0).round() / 100.0,
            matic: (matic * 10000.0).round() / 10000.0,
        })
    }

    /// Fetch redeemable winning positions (from TypeScript)
    async fn fetch_redeemable_winners(&self) -> Result<Vec<RedeemablePosition>> {
        let url = format!(
            "{}/positions?user={:?}&redeemable=true&sizeThreshold=0&limit=100",
            DATA_API_URL,
            self.safe_address
        );

        let resp = self.http_client
            .get(&url)
            .header("User-Agent", "auto_claim/1.0")
            .send()
            .await
            .context("Failed to fetch positions from Data API")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Data API error {}: {}", status, body));
        }

        let positions: Vec<RedeemablePosition> = resp
            .json()
            .await
            .context("Failed to parse positions response")?;

        // Filter for winning positions (curPrice === 1)
        let winners: Vec<RedeemablePosition> = positions
            .into_iter()
            .filter(|p| (p.cur_price - 1.0).abs() < 0.001 && p.size > 0.0)
            .collect();

        Ok(winners)
    }

    /// Build Safe signature for 1-of-1 owner (from TypeScript)
    fn build_safe_signature(&self) -> Bytes {
        // From TypeScript:
        // r: 12 zero bytes + 20-byte owner address
        // s: 32 zero bytes
        // v: 0x01 (contract signature type)
        let mut sig = vec![0u8; 65];
        sig[12..32].copy_from_slice(self.signer_address.as_bytes());
        sig[64] = 0x01;
        Bytes::from(sig)
    }

    /// Execute a claim for a single position
    async fn execute_claim(&self, position: &RedeemablePosition) -> ClaimResult {
        let mut result = ClaimResult {
            condition_id: position.condition_id.clone(),
            title: position.title.clone(),
            outcome: position.outcome.clone(),
            size: position.size,
            tx_hash: None,
            error: None,
        };

        // Parse condition ID
        let condition_id_str = if position.condition_id.starts_with("0x") {
            position.condition_id.clone()
        } else {
            format!("0x{}", position.condition_id)
        };

        let condition_id_bytes: [u8; 32] = match hex::decode(condition_id_str.trim_start_matches("0x")) {
            Ok(bytes) if bytes.len() == 32 => bytes.try_into().unwrap(),
            Ok(bytes) => {
                result.error = Some(format!("Invalid conditionId length: {}", bytes.len()));
                return result;
            }
            Err(e) => {
                result.error = Some(format!("Failed to decode conditionId: {}", e));
                return result;
            }
        };

        // Build CTF redeemPositions calldata
        let ctf_address: Address = CTF_ADDRESS.parse().unwrap();
        let usdc_address: Address = USDC_ADDRESS.parse().unwrap();
        let ctf = CTF::new(ctf_address, self.provider.clone());

        let index_sets = vec![U256::from(1), U256::from(2)];
        let redeem_call = ctf.redeem_positions(
            usdc_address,
            H256::zero().into(),
            condition_id_bytes,
            index_sets,
        );

        let redeem_data = match redeem_call.calldata() {
            Some(data) => data,
            None => {
                result.error = Some("Failed to encode redeemPositions calldata".to_string());
                return result;
            }
        };

        // Get gas prices
        let (max_fee, max_priority_fee) = match self.get_polygon_gas_prices().await {
            Ok(prices) => prices,
            Err(e) => {
                result.error = Some(format!("Failed to get gas prices: {}", e));
                return result;
            }
        };

        println!("  Gas: priority={} gwei, max={} gwei",
            max_priority_fee.as_u64() / 1_000_000_000,
            max_fee.as_u64() / 1_000_000_000
        );

        // Build Safe signature
        let signature = self.build_safe_signature();

        // Execute through Safe
        let safe = GnosisSafe::new(self.safe_address, self.provider.clone());

        let tx = safe.exec_transaction(
            ctf_address,
            U256::zero(),
            redeem_data,
            0, // operation = CALL
            U256::zero(),
            U256::zero(),
            U256::zero(),
            Address::zero(),
            Address::zero(),
            signature,
        )
        .gas(500_000u64)
        .gas_price(max_fee);

        println!("  Submitting transaction...");

        match tx.send().await {
            Ok(pending_tx) => {
                let tx_hash = pending_tx.tx_hash();
                println!("  TX submitted: {:?}", tx_hash);
                result.tx_hash = Some(format!("{:?}", tx_hash));

                // Wait for confirmation
                match pending_tx.await {
                    Ok(Some(receipt)) => {
                        if receipt.status == Some(U64::from(1)) {
                            println!("  SUCCESS! Gas used: {:?}", receipt.gas_used);
                        } else {
                            result.error = Some("Transaction reverted".to_string());
                            println!("  REVERTED");
                        }
                    }
                    Ok(None) => {
                        result.error = Some("Transaction dropped".to_string());
                        println!("  Transaction dropped from mempool");
                    }
                    Err(e) => {
                        // TX was submitted but confirmation failed - still count as submitted
                        println!("  Warning: Confirmation check failed: {}", e);
                    }
                }
            }
            Err(e) => {
                let err_str = e.to_string();
                // Check if already claimed (from TypeScript error handling)
                if err_str.contains("revert") || err_str.contains("execution reverted") {
                    result.error = Some("Position may already be claimed".to_string());
                    println!("  Warning: Position may already be claimed");
                } else {
                    result.error = Some(format!("TX failed: {}", err_str));
                    println!("  ERROR: {}", err_str);
                }
            }
        }

        result
    }
}

// ============================================================================
// MAIN
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let args = Args::parse();

    // Load credentials once
    let private_key = std::env::var("POLY_PRIVATE_KEY")
        .context("POLY_PRIVATE_KEY not set in .env")?;
    let safe_address = std::env::var("POLY_FUNDER")
        .context("POLY_FUNDER not set in .env")?;
    let rpc_url = args.rpc_url.clone()
        .or_else(|| std::env::var("POLYGON_RPC_URL").ok())
        .unwrap_or_else(|| DEFAULT_RPC_URL.to_string());

    // Create client once
    let client = AutoClaimClient::new(&rpc_url, &private_key, &safe_address)?;

    let loop_minutes = args.r#loop;
    let mut iteration = 0u64;

    loop {
        iteration += 1;

        println!("========================================");
        if loop_minutes.is_some() {
            println!("POLYMARKET AUTO-CLAIM (Run #{})", iteration);
        } else {
            println!("POLYMARKET AUTO-CLAIM");
        }
        println!("========================================");
        println!("Safe address: {}", safe_address);
        println!("RPC: {}", rpc_url);
        println!("Mode: {}", if args.live { "LIVE" } else { "DRY RUN" });
        if let Some(mins) = loop_minutes {
            println!("Loop: Every {} minutes", mins);
        }
        println!();
        println!("Signer address: {:?}", client.signer_address);

        // Get wallet balances
        println!("\nChecking balances...");
        match client.get_wallet_balances().await {
            Ok(balance) => {
                println!("  USDC (Safe): ${:.2}", balance.usdc);
                println!("  POL (Signer): {:.4}", balance.matic);
                if balance.matic < 0.01 {
                    println!("\n  WARNING: Low POL balance for gas!");
                }
            }
            Err(e) => {
                println!("  Error fetching balances: {}", e);
            }
        }

        // Fetch redeemable positions
        println!("\nFetching redeemable positions...");
        let positions = match client.fetch_redeemable_winners().await {
            Ok(p) => p,
            Err(e) => {
                println!("  Error fetching positions: {}", e);
                if let Some(mins) = loop_minutes {
                    println!("\nWaiting {} minutes before next check...", mins);
                    tokio::time::sleep(Duration::from_secs(mins * 60)).await;
                    continue;
                } else {
                    return Err(e);
                }
            }
        };

        if positions.is_empty() {
            println!("  No winning positions to claim.");

            if let Some(mins) = loop_minutes {
                println!("\n========================================");
                println!("Waiting {} minutes before next check...", mins);
                println!("========================================");
                tokio::time::sleep(Duration::from_secs(mins * 60)).await;
                continue;
            } else {
                println!("\n========================================");
                return Ok(());
            }
        }

        let total_value: f64 = positions.iter().map(|p| p.size).sum();
        println!("  Found {} winning position(s) worth ${:.2}", positions.len(), total_value);
        println!();

        for (i, pos) in positions.iter().enumerate() {
            println!("  {}. {} ({})", i + 1,
                if pos.title.is_empty() { &pos.asset } else { &pos.title },
                pos.outcome
            );
            println!("     Size: ${:.2} | ConditionID: {}...",
                pos.size,
                &pos.condition_id[..16.min(pos.condition_id.len())]
            );
        }

        if !args.live {
            println!("\n========================================");
            println!("DRY RUN - No transactions executed");
            println!("Run with --live to claim positions");
            println!("========================================");

            if let Some(mins) = loop_minutes {
                println!("\nWaiting {} minutes before next check...", mins);
                tokio::time::sleep(Duration::from_secs(mins * 60)).await;
                continue;
            } else {
                return Ok(());
            }
        }

        // Execute claims
        println!("\n========================================");
        println!("EXECUTING CLAIMS");
        println!("========================================");

        let mut claim_results: Vec<ClaimResult> = vec![];

        for (i, position) in positions.iter().enumerate() {
            println!("\n[{}/{}] Claiming: {}",
                i + 1,
                positions.len(),
                if position.title.is_empty() { &position.asset } else { &position.title }
            );
            println!("  Outcome: {} | Size: ${:.2}", position.outcome, position.size);

            let result = client.execute_claim(position).await;

            if let Some(ref tx_hash) = result.tx_hash {
                println!("  View: https://polygonscan.com/tx/{}", tx_hash);
            }

            claim_results.push(result);

            // Delay between claims
            if i < positions.len() - 1 {
                println!("  Waiting 2s before next claim...");
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }

        // Summary
        let success_count = claim_results.iter().filter(|r| r.tx_hash.is_some() && r.error.is_none()).count();
        let submitted_count = claim_results.iter().filter(|r| r.tx_hash.is_some()).count();

        println!("\n========================================");
        println!("SUMMARY");
        println!("========================================");
        println!("Submitted: {}/{}", submitted_count, claim_results.len());
        println!("Successful: {}/{}", success_count, claim_results.len());

        // Show any errors
        for result in &claim_results {
            if let Some(ref error) = result.error {
                println!("  {} - Error: {}",
                    &result.condition_id[..16.min(result.condition_id.len())],
                    error
                );
            }
        }

        // Refresh balance
        println!("\nRefreshing balances...");
        if let Ok(updated_balance) = client.get_wallet_balances().await {
            println!("  USDC (Safe): ${:.2}", updated_balance.usdc);
            println!("  POL (Signer): {:.4}", updated_balance.matic);
        }

        println!("\n========================================");

        // Loop or exit
        if let Some(mins) = loop_minutes {
            println!("\nWaiting {} minutes before next check...", mins);
            println!("========================================\n");
            tokio::time::sleep(Duration::from_secs(mins * 60)).await;
        } else {
            break;
        }
    }

    Ok(())
}
