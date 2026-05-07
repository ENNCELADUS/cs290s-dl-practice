#!/bin/bash
# Usage:
#   sbatch scripts/run_kv_cache_benchmark.sh large
#   sbatch scripts/run_kv_cache_benchmark.sh medium
#   sbatch scripts/run_kv_cache_benchmark.sh configs/experiments/kv_cache/large.yaml
#   sbatch scripts/run_kv_cache_benchmark.sh configs/experiments/kv_cache/medium.yaml
#SBATCH -J cs290s-kv-cache
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIATITANRTX:1
#SBATCH --exclude=ai_gpu26,ai_gpu28
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

RUN_TARGET="${1:-large}"
case "${RUN_TARGET}" in
    large)
        BENCHMARK_CONFIG="configs/experiments/kv_cache/large.yaml"
        ;;
    medium)
        BENCHMARK_CONFIG="configs/experiments/kv_cache/medium.yaml"
        ;;
    *.yaml|*.yml)
        BENCHMARK_CONFIG="${RUN_TARGET}"
        ;;
    *)
        echo "Usage: sbatch scripts/run_kv_cache_benchmark.sh {large|medium|path/to/config.yaml}" >&2
        exit 2
        ;;
esac

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

detect_visible_gpu_ids() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python - <<'PY'
import os

visible = [item for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if item.strip()]
print(",".join(visible))
PY
        return
    fi

    if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
        echo "${SLURM_JOB_GPUS}"
        return
    fi

    if [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
        echo "${SLURM_STEP_GPUS}"
        return
    fi

    if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
        python - <<PY
print(",".join(str(index) for index in range(${SLURM_GPUS_ON_NODE})))
PY
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -
        return
    fi

    python - <<'PY'
import torch

print(",".join(str(index) for index in range(torch.cuda.device_count())))
PY
}

GPU_IDS="$(detect_visible_gpu_ids | tr -d '[:space:]')"

filter_healthy_gpu_ids() {
    local raw_ids="$1"
    local healthy_ids=()
    local gpu_id

    IFS=',' read -ra gpu_id_array <<< "${raw_ids}"
    for gpu_id in "${gpu_id_array[@]}"; do
        if [[ -z "${gpu_id}" ]]; then
            continue
        fi

        if command -v nvidia-smi >/dev/null 2>&1; then
            if ! timeout 15s nvidia-smi -i "${gpu_id}" --query-gpu=index,name,memory.total \
                --format=csv,noheader >/dev/null 2>&1; then
                echo "Skipping GPU ${gpu_id}: nvidia-smi health check failed." >&2
                continue
            fi
        fi

        if ! timeout 30s env CUDA_VISIBLE_DEVICES="${gpu_id}" python - <<'PY' >/dev/null 2>&1
import torch

if torch.cuda.device_count() != 1:
    raise SystemExit(1)
_ = torch.empty(1, device="cuda")
torch.cuda.synchronize()
PY
        then
            echo "Skipping GPU ${gpu_id}: torch CUDA health check failed." >&2
            continue
        fi

        healthy_ids+=("${gpu_id}")
    done

    local IFS=','
    echo "${healthy_ids[*]}"
}

if [[ "${PROJECT2_DISABLE_GPU_HEALTH_FILTER:-0}" != "1" ]]; then
    GPU_IDS="$(filter_healthy_gpu_ids "${GPU_IDS}")"
fi

if [[ -n "${PROJECT2_MAX_GPUS:-}" ]]; then
    GPU_IDS="$(GPU_IDS="${GPU_IDS}" PROJECT2_MAX_GPUS="${PROJECT2_MAX_GPUS}" python - <<'PY'
import os

gpu_ids = [item for item in os.environ["GPU_IDS"].split(",") if item]
max_gpus = int(os.environ["PROJECT2_MAX_GPUS"])
print(",".join(gpu_ids[:max_gpus]))
PY
)"
fi

GPU_IDS="$(GPU_IDS="${GPU_IDS}" python - <<'PY'
import os

gpu_ids = [item for item in os.environ["GPU_IDS"].split(",") if item]
print(",".join(gpu_ids[:1]))
PY
)"

NUM_GPUS="$(GPU_IDS="${GPU_IDS}" python - <<'PY'
import os

print(len([item for item in os.environ["GPU_IDS"].split(",") if item]))
PY
)"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "No CUDA GPUs detected for this job." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Launching KV cache benchmark with config ${BENCHMARK_CONFIG}."

python benchmark_kv_cache.py --config "${BENCHMARK_CONFIG}"
