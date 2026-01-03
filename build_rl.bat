@echo off
set LIBTORCH=C:\libtorch
set PATH=%PATH%;C:\libtorch\lib
cargo build --release --features rl --bin poly_atm_sniper
