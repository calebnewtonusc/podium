#!/usr/bin/env bash
# Start 4 vLLM instances for synthesis (GPUs 0-15, leaving 16-17 for training)
# Each instance uses 4 GPUs in tensor-parallel mode for Qwen2.5-72B-Instruct.
# Ports 8001-8004. API key set via VLLM_API_KEY env var.

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
API_KEY="${VLLM_API_KEY:-synthesis}"
LOG_DIR="${LOG_DIR:-./logs}"
HEALTH_TIMEOUT="${VLLM_HEALTH_TIMEOUT:-300}"

mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "[$(date '+%H:%M:%S')]  $1"; }
pass() { echo -e "[$(date '+%H:%M:%S')]  ${GREEN}[UP]${NC}    $1"; }
fail() { echo -e "[$(date '+%H:%M:%S')]  ${RED}[FAIL]${NC}  $1"; }
info() { echo -e "[$(date '+%H:%M:%S')]  ${YELLOW}[INFO]${NC}  $1"; }

echo ""
echo "Podium vLLM Synthesis Servers"
echo "=============================="
info "Model: $MODEL"
info "API key: ${API_KEY:0:8}..."
echo ""

# Start 4 vLLM instances for synthesis (GPUs 0-15, leaving 16-17 for training)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL" \
    --tensor-parallel-size 4 \
    --port 8001 \
    --api-key "$API_KEY" \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --disable-log-requests \
    > "$LOG_DIR/vllm_8001.log" 2>&1 &
echo $! > "$LOG_DIR/vllm_8001.pid"

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve "$MODEL" \
    --tensor-parallel-size 4 \
    --port 8002 \
    --api-key "$API_KEY" \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --disable-log-requests \
    > "$LOG_DIR/vllm_8002.log" 2>&1 &
echo $! > "$LOG_DIR/vllm_8002.pid"

CUDA_VISIBLE_DEVICES=8,9,10,11 vllm serve "$MODEL" \
    --tensor-parallel-size 4 \
    --port 8003 \
    --api-key "$API_KEY" \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --disable-log-requests \
    > "$LOG_DIR/vllm_8003.log" 2>&1 &
echo $! > "$LOG_DIR/vllm_8003.pid"

CUDA_VISIBLE_DEVICES=12,13,14,15 vllm serve "$MODEL" \
    --tensor-parallel-size 4 \
    --port 8004 \
    --api-key "$API_KEY" \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --disable-log-requests \
    > "$LOG_DIR/vllm_8004.log" 2>&1 &
echo $! > "$LOG_DIR/vllm_8004.pid"

# GPUs 16-17 reserved for training / evaluation inference

echo ""
info "Waiting for vLLM instances to start... (72B model takes 3-8 minutes to load)"
echo ""

wait_healthy() {
    local port="$1"
    local elapsed=0
    while [ "$elapsed" -lt "$HEALTH_TIMEOUT" ]; do
        if curl -sf "http://localhost:${port}/v1/models" \
            -H "Authorization: Bearer $API_KEY" > /dev/null 2>&1; then
            pass "vLLM port $port is UP"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $((elapsed % 60)) -eq 0 ]; then
            info "Port $port still loading... (${elapsed}s elapsed)"
        fi
    done
    fail "Port $port did not start within ${HEALTH_TIMEOUT}s -- check $LOG_DIR/vllm_${port}.log"
    return 1
}

ALL_OK=true
for port in 8001 8002 8003 8004; do
    wait_healthy "$port" || ALL_OK=false
done

echo ""
echo "=============================="
if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}All 4 vLLM instances are LIVE on ports 8001-8004${NC}"
    echo ""
    echo "  GPUs 0-3:   port 8001"
    echo "  GPUs 4-7:   port 8002"
    echo "  GPUs 8-11:  port 8003"
    echo "  GPUs 12-15: port 8004"
    echo "  GPUs 16-17: reserved for training"
    echo ""
    echo "  Next: export VLLM_SYNTHESIS_URL=http://localhost:8001/v1"
    echo "        python synthesis/synthesize_bulk.py"
else
    echo -e "${RED}One or more vLLM servers failed to start. Check logs in $LOG_DIR/${NC}"
    exit 1
fi
