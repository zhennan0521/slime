#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd.sh

set -ex

TIME=$(date +%Y%m%d%H%M%S)

# ============================================================
# Teacher model server config (run on a separate 5th node)
# Start it manually on the teacher node before running this script:
#   python3 -m sglang.launch_server \
#       --model-path /path/to/Qwen3-32B \
#       --host 0.0.0.0 --port 13141 --tp 2 \
#       --chunked-prefill-size 4096 --mem-fraction-static 0.8
# ============================================================
TEACHER_IP="<FILL_IN_TEACHER_NODE_IP>"
TEACHER_PORT=13141
TEACHER_MODEL_NAME="Qwen3-32B"

echo "Waiting for external teacher model server at $TEACHER_IP:$TEACHER_PORT ..."
until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
    echo "  still waiting..."
    sleep 5
done
curl http://$TEACHER_IP:$TEACHER_PORT/get_model_info
echo "Teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT."


export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"


STU_MODEL_PATH="/jpfs-5p/chenyanxu.9/model/Qwen3-8B-Base-sft-dolci-think/iter_0005375-hf/"
STU_MODEL_PATH_MEG="/jpfs-5p/chenyanxu.9/model/Qwen3-8B-Base-sft-dolci-think/iter_0005375_torch_dist/"

SLIME_DIR="/jpfs-5p/shenzhennan/slime"
OUTPUT_DIR="${SLIME_DIR}/outputs/opd_qwen3_8b_sft_dolci_think_${TEACHER_MODEL_NAME}_${TIME}"
source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"


CKPT_ARGS=(
   --hf-checkpoint $STU_MODEL_PATH
   --ref-load $STU_MODEL_PATH_MEG
   --save $OUTPUT_DIR
   --save-interval 64
)

DATA_DIR="/jpfs/shenzhennan.1/datasets"

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
)

RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
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

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
export WANDB_API_KEY=local-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 # your_wandb_key
export WANDB_ENTITY=automl # your_wandb_entity
export WANDB_PROJECT=slime-rl
wandb login --relogin --host=http://11.71.1.153:8080 ${WANDB_API_KEY}

WANDB_ARGS=(
   # 取消注释以启用 wandb
   --use-wandb
   --wandb-project slime-opd
   --wandb-group zhennan-slime-opd-qwen3-8b-sft-dolci-think-teacher-${TEACHER_MODEL_NAME}
)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.4
)


MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)


CURRENT_DIR=$(pwd)
# launch the master node of ray in container
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
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"


# launch the master node of ray in container
# export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
RUN_LOG_DIR="/jpfs-5p/shenzhennan/slime/logs/run/"
mkdir -p $RUN_LOG_DIR
RUN_LOG_FILE="$RUN_LOG_DIR/run_qwen3_8b_sft_dolci_think_${TEACHER_MODEL_NAME}_${TIME}.log"


ray job submit  \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 24 \
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
   ${RM_ARGS[@]} 2>&1 | tee $RUN_LOG_FILE &



####clear after training
# pkill -9 sglang
# sleep 3
# ray stop --force
# pkill -9 ray
# pkill -9 python
# sleep 3
# pkill -9 ray
# pkill -9 python