#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOSTFILE="${HOSTFILE:?Set HOSTFILE to a DeepSpeed hostfile with '<host> slots=<gpus>' entries}"
SSH_USER="${SSH_USER:-root}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
CONFIG_NAME="${CONFIG_NAME:-robotwin_train}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-config/deepspeed/zero2.json}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_INIT_WORKERS="${NUM_INIT_WORKERS:-4}"
MULTINODE_LAUNCHER="${MULTINODE_LAUNCHER:-ssh}"

if [[ ! -f "${HOSTFILE}" ]]; then
  echo "Hostfile not found: ${HOSTFILE}" >&2
  exit 1
fi
if [[ "${MULTINODE_LAUNCHER}" == "pdsh" ]] && ! command -v pdsh >/dev/null 2>&1; then
  echo "pdsh is required on the launch node" >&2
  exit 1
fi
if [[ "${MULTINODE_LAUNCHER}" != "ssh" && "${MULTINODE_LAUNCHER}" != "pdsh" ]]; then
  echo "MULTINODE_LAUNCHER must be ssh or pdsh" >&2
  exit 1
fi

mapfile -t HOSTS < <(awk 'NF && $1 !~ /^#/ {print $1}' "${HOSTFILE}")
if (( ${#HOSTS[@]} == 0 )); then
  echo "Hostfile has no active hosts: ${HOSTFILE}" >&2
  exit 1
fi

NNODES="${NNODES:-${#HOSTS[@]}}"
if (( NNODES < 1 || NNODES > ${#HOSTS[@]} )); then
  echo "NNODES=${NNODES} is outside hostfile range 1..${#HOSTS[@]}" >&2
  exit 1
fi
ACTIVE_HOSTS=("${HOSTS[@]:0:${NNODES}}")
MASTER_ADDR="${MASTER_ADDR:-${ACTIVE_HOSTS[0]}}"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-net0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-1800}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Checking ${NNODES} hosts from ${HOSTFILE}"
for host in "${ACTIVE_HOSTS[@]}"; do
  ssh \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" \
    "${SSH_USER}@${host}" \
    "cd '${ROOT_DIR}' && command -v deepspeed >/dev/null && test \"\$(python -c 'import torch; print(torch.cuda.device_count())')\" -ge '${GPUS_PER_NODE}'"
  echo "  ${host}: ready"
done

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "PRECHECK_ONLY=1; skipping launch"
  exit 0
fi

ARGS=(
  --config-name "${CONFIG_NAME}"
  --distributed-backend deepspeed
  --deepspeed-config "${DEEPSPEED_CONFIG}"
  --num-workers "${NUM_WORKERS}"
  --num-init-workers "${NUM_INIT_WORKERS}"
)
if [[ -n "${SAVE_ROOT:-}" ]]; then
  ARGS+=(--save-root "${SAVE_ROOT}")
fi
if [[ -n "${DATASET_PATH:-}" ]]; then
  ARGS+=(--dataset-path "${DATASET_PATH}")
fi
if [[ -n "${MODEL_PATH:-}" ]]; then
  ARGS+=(--model-path "${MODEL_PATH}")
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
  ARGS+=(--resume-from "${RESUME_FROM}")
fi
if [[ "${DISABLE_WANDB:-0}" == "1" ]]; then
  ARGS+=(--disable-wandb)
fi
ARGS+=("$@")

cd "${ROOT_DIR}"
if [[ "${MULTINODE_LAUNCHER}" == "pdsh" ]]; then
  deepspeed \
    --hostfile "${HOSTFILE}" \
    --launcher pdsh \
    --num_nodes "${NNODES}" \
    --num_gpus "${GPUS_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    --module wan_va.train \
    "${ARGS[@]}"
  exit 0
fi

VISIBLE_DEVICES="$(seq -s, 0 $((GPUS_PER_NODE - 1)))"
printf -v ROOT_Q '%q' "${ROOT_DIR}"
printf -v ARGS_Q ' %q' "${ARGS[@]}"
printf -v VISIBLE_Q '%q' "${VISIBLE_DEVICES}"
printf -v IFNAME_Q '%q' "${NCCL_SOCKET_IFNAME}"
printf -v IB_DISABLE_Q '%q' "${NCCL_IB_DISABLE}"
printf -v NCCL_DEBUG_Q '%q' "${NCCL_DEBUG}"
printf -v DIST_TIMEOUT_Q '%q' "${TORCH_DISTRIBUTED_TIMEOUT}"
printf -v ALLOC_CONF_Q '%q' "${PYTORCH_CUDA_ALLOC_CONF}"

PIDS=()
for node_rank in "${!ACTIVE_HOSTS[@]}"; do
  host="${ACTIVE_HOSTS[${node_rank}]}"
  remote_command="cd ${ROOT_Q} && \
CUDA_VISIBLE_DEVICES=${VISIBLE_Q} \
NCCL_SOCKET_IFNAME=${IFNAME_Q} \
NCCL_IB_DISABLE=${IB_DISABLE_Q} \
NCCL_DEBUG=${NCCL_DEBUG_Q} \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
TORCH_NCCL_BLOCKING_WAIT=1 \
TORCH_DISTRIBUTED_TIMEOUT=${DIST_TIMEOUT_Q} \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=${ALLOC_CONF_Q} \
python -m torch.distributed.run \
--nnodes=${NNODES} \
--nproc-per-node=${GPUS_PER_NODE} \
--node-rank=${node_rank} \
--master-addr=${MASTER_ADDR} \
--master-port=${MASTER_PORT} \
-m wan_va.train${ARGS_Q}"
  ssh \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" \
    "${SSH_USER}@${host}" \
    "${remote_command}" &
  PIDS+=("$!")
done

cleanup_children() {
  kill "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup_children INT TERM
for _ in "${PIDS[@]}"; do
  if ! wait -n; then
    cleanup_children
    wait || true
    exit 1
  fi
done
trap - INT TERM
