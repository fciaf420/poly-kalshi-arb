# Run arb bot with logging to dated file
# Log levels: error, warn, info (default), debug, trace
# Set RUST_LOG=debug for more details, RUST_LOG=warn for minimal output

$date = Get-Date -Format "yyyy-MM-dd"
$logFile = "logs\bot_$date.log"

# Default to info level (shows: startup, heartbeat, arb opportunities, trades, errors)
# Override with: $env:RUST_LOG = "debug" before running
if (-not $env:RUST_LOG) {
    $env:RUST_LOG = "arb_bot=info"
}

Write-Host "Starting bot (log level: $env:RUST_LOG), logging to $logFile"
dotenvx run -- cargo run --release 2>&1 | Tee-Object -FilePath $logFile -Append
