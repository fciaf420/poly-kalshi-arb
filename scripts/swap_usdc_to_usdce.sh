#!/bin/bash

# Swap native USDC to USDC.e on Polygon via Uniswap V3
# Usage: ./swap_usdc_to_usdce.sh [amount_usdc]
# If no amount specified, swaps entire USDC balance

set -e

# Load environment
source /home/ubuntu/poly-kalshi-arb/.env

# Contract addresses
USDC="0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
USDC_E="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
UNISWAP_ROUTER="0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"
RPC="https://polygon-rpc.com"
CAST=~/.foundry/bin/cast

# Get current balances
echo "Checking balances..."
USDC_BALANCE=$($CAST call $USDC "balanceOf(address)(uint256)" $POLY_WALLET --rpc-url $RPC | head -1 | awk '{print $1}')
USDC_E_BALANCE=$($CAST call $USDC_E "balanceOf(address)(uint256)" $POLY_WALLET --rpc-url $RPC | head -1 | awk '{print $1}')

echo "Native USDC: $(echo "scale=6; $USDC_BALANCE / 1000000" | bc)"
echo "USDC.e:      $(echo "scale=6; $USDC_E_BALANCE / 1000000" | bc)"

# Determine amount to swap
if [ -n "$1" ]; then
    AMOUNT=$(echo "$1 * 1000000" | bc | cut -d'.' -f1)
else
    AMOUNT=$USDC_BALANCE
fi

if [ "$AMOUNT" -eq 0 ] || [ "$AMOUNT" = "0" ]; then
    echo "No USDC to swap."
    exit 0
fi

AMOUNT_HUMAN=$(echo "scale=6; $AMOUNT / 1000000" | bc)
echo ""
echo "Swapping $AMOUNT_HUMAN USDC to USDC.e..."

# Check allowance
ALLOWANCE=$($CAST call $USDC "allowance(address,address)(uint256)" $POLY_WALLET $UNISWAP_ROUTER --rpc-url $RPC | head -1 | awk '{print $1}')

if [ "$ALLOWANCE" -lt "$AMOUNT" ]; then
    echo "Approving Uniswap router..."
    $CAST send $USDC "approve(address,uint256)" $UNISWAP_ROUTER 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff \
        --private-key $POLY_PRIVATE_KEY --rpc-url $RPC > /dev/null
    echo "Approved."
fi

# Calculate minimum output (99% - 1% slippage)
MIN_OUT=$(echo "$AMOUNT * 99 / 100" | bc)

# Execute swap
echo "Executing swap..."
TX=$($CAST send $UNISWAP_ROUTER \
    "exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))" \
    "($USDC,$USDC_E,100,$POLY_WALLET,$AMOUNT,$MIN_OUT,0)" \
    --private-key $POLY_PRIVATE_KEY --rpc-url $RPC --json | jq -r '.transactionHash')

echo "Tx: $TX"

# Show new balances
echo ""
echo "New balances:"
USDC_BALANCE=$($CAST call $USDC "balanceOf(address)(uint256)" $POLY_WALLET --rpc-url $RPC | head -1 | awk '{print $1}')
USDC_E_BALANCE=$($CAST call $USDC_E "balanceOf(address)(uint256)" $POLY_WALLET --rpc-url $RPC | head -1 | awk '{print $1}')

echo "Native USDC: $(echo "scale=6; $USDC_BALANCE / 1000000" | bc)"
echo "USDC.e:      $(echo "scale=6; $USDC_E_BALANCE / 1000000" | bc)"
echo ""
echo "Done!"
