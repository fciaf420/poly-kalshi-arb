@echo off
set PATH=C:\libtorch\lib;%PATH%
cd /d "C:\Users\fciaf\OneDrive\Desktop\Projects\poly_kalshi arb"
echo Starting RL test with GPU...
echo GPU Memory before:
nvidia-smi --query-gpu=memory.used --format=csv,noheader
dotenvx run -- target\release\poly_atm_sniper.exe --rl-mode --rl-safetensors models\rl_model.safetensors 2>&1
echo Exit code: %ERRORLEVEL%
echo GPU Memory after:
nvidia-smi --query-gpu=memory.used --format=csv,noheader
