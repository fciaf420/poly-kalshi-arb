$env:LIBTORCH = "C:\libtorch"
$env:Path = $env:Path + ";C:\libtorch\lib"
Write-Host "LIBTORCH = $env:LIBTORCH"
cargo build --release --features rl --bin poly_atm_sniper
