#!/bin/bash
# Full Podium pipeline: data → training → evaluation
# ~76 hours total. Run in tmux or screen.

set -e

echo "================================================"
echo "  Podium: Full Training Pipeline"
echo "  Estimated: ~76 hours on 18× A6000 + Azure"
echo "================================================"

# Load environment
if [ -f .env ]; then
	set -a && source .env && set +a
	echo "✓ Environment loaded from .env"
else
	echo "✗ .env not found. Copy .env.example and fill in values."
	exit 1
fi

# Check environment
bash scripts/check_env.sh

echo ""
echo "Starting pipeline at $(date)"
echo ""

# Run full pipeline via pipeline.py (handles logging and error recovery)
python pipeline.py "$@"

echo ""
echo "================================================"
echo "  Pipeline complete at $(date)"
echo "  Model: ./checkpoints/dpo"
echo "  Results: ./results/podium_bench_results.json"
echo "  Deploy: docker compose -f deploy/docker-compose.yml up -d"
echo "================================================"
