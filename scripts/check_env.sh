#!/bin/bash
# Verify Podium environment before training

set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; ERRORS=$((ERRORS+1)); }
warn() { echo -e "${YELLOW}!${NC} $1"; }

ERRORS=0
echo "Podium Environment Check"
echo "========================"

# Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
    pass "Python $PYTHON_VERSION"
else
    fail "Python $PYTHON_VERSION (need 3.11+)"
fi

# CUDA
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    pass "CUDA $CUDA_VERSION | $GPU_COUNT GPU(s) detected"
    if [ "$GPU_COUNT" -lt 18 ]; then
        warn "Only $GPU_COUNT GPUs found (18 recommended for full training)"
    fi
else
    fail "nvidia-smi not found — no CUDA GPUs detected"
fi

# PyTorch + CUDA
python3 -c "
import torch
gpus = torch.cuda.device_count()
if gpus > 0:
    print(f'\033[0;32m✓\033[0m PyTorch {torch.__version__} | {gpus} CUDA device(s)')
else:
    print(f'\033[0;31m✗\033[0m PyTorch found but no CUDA devices')
    exit(1)
" || ERRORS=$((ERRORS+1))

# Required packages
PACKAGES=("transformers" "peft" "trl" "deepspeed" "accelerate" "vllm" "kaggle" "chromadb" "docker" "fastapi")
for pkg in "${PACKAGES[@]}"; do
    python3 -c "import $pkg" 2>/dev/null && pass "$pkg installed" || fail "$pkg NOT installed"
done

# Environment variables
ENV_VARS=("KAGGLE_USERNAME" "KAGGLE_KEY" "VLLM_API_KEY")
for var in "${ENV_VARS[@]}"; do
    if [ -n "${!var}" ]; then
        pass "$var set"
    else
        fail "$var not set (required)"
    fi
done

OPTIONAL_VARS=("WANDB_API_KEY" "HF_TOKEN" "ANTHROPIC_API_KEY")
for var in "${OPTIONAL_VARS[@]}"; do
    if [ -n "${!var}" ]; then
        pass "$var set"
    else
        warn "$var not set (optional)"
    fi
done

# Docker (for CV-RL execution harness)
if command -v docker &> /dev/null; then
    docker info &>/dev/null && pass "Docker daemon running" || fail "Docker daemon not running"
    docker image inspect podium-execution:latest &>/dev/null \
        && pass "podium-execution Docker image built" \
        || warn "podium-execution image not built (run: docker build -f deploy/Dockerfile.execution -t podium-execution .)"
else
    fail "Docker not installed (required for CV-RL execution harness)"
fi

# Data directories
mkdir -p data/raw data/synthesized data/rl data/dpo data/competition_memory
pass "Data directories created"

echo ""
echo "========================"
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}All checks passed. Ready to run Podium pipeline.${NC}"
    echo "  Run: python pipeline.py --list  (to see all stages)"
    echo "  Run: python pipeline.py         (to start full pipeline)"
else
    echo -e "${RED}$ERRORS check(s) failed. Fix errors before running pipeline.${NC}"
    exit 1
fi
