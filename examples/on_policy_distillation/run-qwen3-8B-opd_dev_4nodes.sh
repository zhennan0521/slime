#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd_dev_4nodes.sh
# 3 nodes train/rollout (1 actor + 2 rollout) + 1 node teacher (external)

set -ex

# ============================================================
# Load .env (SLIME_DIR, DATASETS_DIR, MODELS_DIR, WANDB_*, etc.)
# ============================================================
if [ -f .env ]; then
    set -a; source .env; set +a
else
    echo "ERROR: .env not found. Run this script from repo root."; exit 1
fi

# ============================================================
# Paths & Names (derived from .env)
# ============================================================
TIME=$(date +%Y%m%d%H%M%S)

TEACHER_IP="6.179.173.207"
TEACHER_PORT=8000
TEACHER_MODEL_NAME="Qwen3-32B"

STU_MODEL_PATH="${MODELS_DIR}/Qwen3-8B-Base-sft-dolci-think/iter_0005375-hf/"
STU_MODEL_PATH_MEG="${MODELS_DIR}/Qwen3-8B-Base-sft-dolci-think/iter_0005375_torch_dist/"
OUTPUT_DIR="${SLIME_DIR}/outputs/opd_qwen3_8b_sft_dolci_think_${TEACHER_MODEL_NAME}_${TIME}"
RUN_LOG_DIR="${SLIME_DIR}/logs/run"

source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"

# ============================================================
# Teacher model server health check
# Start it manually on the teacher node before running this script:
#   python3 -m sglang.launch_server \
#       --model-path ${MODELS_DIR}/Qwen3-32B \
#       --host 0.0.0.0 --port 8000 --tp 2 \
#       --chunked-prefill-size 4096 --mem-fraction-static 0.8
# ============================================================
echo "Waiting for external teacher model server at $TEACHER_IP:$TEACHER_PORT ..."
until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
    echo "  still waiting..."
    sleep 5
done
curl http://$TEACHER_IP:$TEACHER_PORT/get_model_info
echo "Teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT."

# ============================================================
# NVLink detection
# ============================================================
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export PYTHONBUFFERED=16

# ============================================================
# Wandb (unset proxy first — wandb server is internal)
# ============================================================
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export WANDB_API_KEY WANDB_ENTITY
export WANDB_PROJECT=slime-rl
wandb login --relogin --host=${WANDB_HOST} ${WANDB_API_KEY}

# ============================================================
# Training args
# ============================================================
CKPT_ARGS=(
   --hf-checkpoint $STU_MODEL_PATH
   --ref-load $STU_MODEL_PATH_MEG
   --save $OUTPUT_DIR
   --save-interval 64
)

ROLLOUT_ARGS=(
   --prompt-data ${DATASETS_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 30000
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
)

RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
   --custom-rollout-log-function-path slime.utils.opd_log.log_rollout_data
   --custom-eval-rollout-log-function-path slime.utils.opd_log.log_eval_rollout_data
)

EVAL_ARGS=(
   --eval-interval 64
   --eval-prompt-data aime ${DATASETS_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 30000
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 30000
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-opd
   --wandb-group zhennan-slime-opd-qwen3-8b-sft-dolci-think-teacher-${TEACHER_MODEL_NAME}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ============================================================
# Launch
# ============================================================
CURRENT_DIR=$(pwd)
RUNTIME_ENV_JSON="{
  \"working_dir\": \"${CURRENT_DIR}\",
  \"excludes\": [
    \".git\",
    \"*.pyc\",
    \"__pycache__\",
    \"data\",
    \"models\",
    \"*.pt\",
    \"*.bin\",
    \"*.safetensors\"
  ],
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${CURRENT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_IB_TIMEOUT\": \"22\",
    \"NCCL_DEBUG\": \"WARN\"
  }
}"

mkdir -p $RUN_LOG_DIR
RUN_LOG_FILE="$RUN_LOG_DIR/run_qwen3_8b_sft_dolci_think_${TEACHER_MODEL_NAME}_${TIME}.log"

ray job submit  \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 16 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]} 2>&1 | tee $RUN_LOG_FILE 
