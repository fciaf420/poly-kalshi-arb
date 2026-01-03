$env:PATH = "C:\libtorch\lib;" + $env:PATH
& "$PSScriptRoot\target\debug\poly_atm_sniper.exe" @args
