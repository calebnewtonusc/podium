# GPU Cluster Setup: 18× NVIDIA A6000

## Hardware Configuration

| Component | Spec |
|-----------|------|
| GPUs | 18× NVIDIA A6000 (48GB VRAM each) |
| Total VRAM | 864GB |
| RAM | 512GB+ (required for ZeRO-3 CPU offload) |
| Storage | 10TB+ NVMe (training data + checkpoints) |
| Network | InfiniBand or 100GbE (for gradient sync) |
| Azure burst | 25k credits for synthesis scale-out |

---

## Environment Setup

```bash
# Python 3.11+ required
conda create -n podium python=3.11
conda activate podium

# CUDA 12.1+ required
nvidia-smi  # Verify A6000s visible

# Install dependencies
pip install -r requirements.txt

# Verify all 18 GPUs
python -c "import torch; print(torch.cuda.device_count())"  # Should print 18

# Set environment
cp .env.example .env
# Edit .env with your API keys
```

---

## DeepSpeed ZeRO-3 Configuration

18 GPUs with ZeRO-3 + CPU offload enables training Qwen2.5-7B with full precision optimizer states.

```json
// training/configs/deepspeed_zero3.json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": 2,
  "bf16": {
    "enabled": true
  }
}
```

---

## Synthesis Servers (Azure Burst)

During data synthesis (Stage 0), we run 4 vLLM synthesis servers on Azure using the 25k credits. Each server runs Qwen2.5-72B for high-quality pair generation.

```bash
# Provision Azure VMs (4× Standard_NC96ads_A100_v4 = 4× A100 each)
az vm create \
  --resource-group podium-synthesis \
  --name synthesis-server-{1..4} \
  --image Ubuntu2204 \
  --size Standard_NC96ads_A100_v4 \
  --generate-ssh-keys

# On each synthesis server:
bash scripts/start_vllm.sh --model Qwen/Qwen2.5-72B-Instruct --port 8000
```

**Azure cost estimate**:
- Standard_NC96ads_A100_v4: ~$20/hr
- 4 servers × 50 hours synthesis = ~$4,000 of 25k credits
- Remaining 21k credits for parallel evaluation and burst training runs

---

## Training Launch

### Stage 1: SFT (~6 hours)
```bash
torchrun --nproc_per_node=18 \
  training/train.py \
  --config training/configs/sft_config.yaml \
  --deepspeed training/configs/deepspeed_zero3.json
```

### Stage 2: CV-RL / GRPO (~4 hours)
```bash
torchrun --nproc_per_node=18 \
  training/train_rl.py \
  --config training/configs/rl_config.yaml \
  --deepspeed training/configs/deepspeed_zero3.json \
  --execution_harness docker  # requires Docker daemon
```

### Stage 3: DPO (~2 hours)
```bash
torchrun --nproc_per_node=18 \
  training/train_dpo.py \
  --config training/configs/dpo_config.yaml \
  --deepspeed training/configs/deepspeed_zero3.json
```

---

## GPU Memory Budget

| Stage | Per-GPU VRAM | Total VRAM |
|-------|-------------|------------|
| SFT (LoRA rank 64) | ~20GB | 360GB |
| CV-RL (GRPO, smaller batch) | ~30GB | 540GB |
| DPO (paired inputs) | ~35GB | 630GB |
| Inference (4-bit GPTQ) | ~6GB | 6GB (1 GPU) |
| Synthesis (Qwen2.5-72B, 4 GPUs) | ~44GB | 176GB |

---

## Docker Execution Harness (CV-RL)

The Stage 2 RL reward signal requires executing generated code in isolated Docker containers. Each worker gets one container.

```bash
# Build execution container
docker build -f deploy/Dockerfile.execution -t podium-execution .

# Test execution harness
python validation/validate_cv.py --test_run

# The harness runs 18 parallel execution workers (one per GPU worker)
# Each worker: generate code → docker exec → cv score → reward
```

### Execution Container Spec
```dockerfile
FROM python:3.11-slim
RUN pip install scikit-learn lightgbm xgboost catboost pandas numpy scipy
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install timm transformers datasets
# CPU-only for fast execution validation (no GPU needed for CV scoring)
WORKDIR /workspace
COPY deploy/execution_runner.py .
```

---

## Monitoring

```bash
# WandB training dashboard (requires WANDB_API_KEY in .env)
# Tracks: loss curves, CV score distributions, reward signals, medal rates

# GPU utilization
watch -n1 nvidia-smi

# Training progress
tail -f logs/training.log
```

---

## Full Pipeline Time Estimate

| Stage | Wall Time | Compute |
|-------|-----------|---------|
| Data discovery | 4 hours | CPU |
| Synthesis (4 Azure servers) | 50 hours | 4× Azure A100 |
| Data validation | 6 hours | 4× A6000 |
| SFT training | 6 hours | 18× A6000 |
| CV-RL training | 4 hours | 18× A6000 |
| DPO training | 2 hours | 18× A6000 |
| PodiumBench eval | 4 hours | 4× A6000 |
| **Total** | **~76 hours** | |
