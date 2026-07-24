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
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_LOCAL_RANK="${LOG_LOCAL_RANK:-0}"
REMOTE_NOFILE_LIMIT="${REMOTE_NOFILE_LIMIT:-65536}"
REMOTE_KILL_GRACE_SECONDS="${REMOTE_KILL_GRACE_SECONDS:-5}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-$$}"

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, digits, dot, underscore, and dash" >&2
  exit 1
fi
if [[ ! "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS_PER_NODE must be a positive integer" >&2
  exit 1
fi
if [[ ! "${REMOTE_NOFILE_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REMOTE_NOFILE_LIMIT must be a positive integer" >&2
  exit 1
fi

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

mapfile -t HOST_ENTRIES < <(awk 'NF && $1 !~ /^#/ {print $1, $2}' "${HOSTFILE}")
if (( ${#HOST_ENTRIES[@]} == 0 )); then
  echo "Hostfile has no active hosts: ${HOSTFILE}" >&2
  exit 1
fi

HOSTS=()
declare -A SEEN_HOSTS=()
for entry in "${HOST_ENTRIES[@]}"; do
  read -r host slots_field <<<"${entry}"
  if [[ -n "${SEEN_HOSTS[${host}]:-}" ]]; then
    echo "Duplicate host in ${HOSTFILE}: ${host}" >&2
    exit 1
  fi
  if [[ ! "${slots_field}" =~ ^slots=([1-9][0-9]*)$ ]]; then
    echo "Invalid hostfile entry for ${host}: expected slots=<positive integer>" >&2
    exit 1
  fi
  if (( BASH_REMATCH[1] != GPUS_PER_NODE )); then
    echo "Host ${host} declares ${BASH_REMATCH[1]} slots; expected ${GPUS_PER_NODE}" >&2
    exit 1
  fi
  SEEN_HOSTS[${host}]=1
  HOSTS+=("${host}")
done

NNODES="${NNODES:-${#HOSTS[@]}}"
if (( NNODES < 1 || NNODES > ${#HOSTS[@]} )); then
  echo "NNODES=${NNODES} is outside hostfile range 1..${#HOSTS[@]}" >&2
  exit 1
fi
ACTIVE_HOSTS=("${HOSTS[@]:0:${NNODES}}")
MASTER_ADDR="${MASTER_ADDR:-${ACTIVE_HOSTS[0]}}"
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
LOG_ROOT="${LOG_ROOT:-${SAVE_ROOT:-${ROOT_DIR}/train_out}/launcher_logs/${RUN_ID}}"
PID_ROOT="${LOG_ROOT}/pids"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-net0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-1800}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
printf -v PYTHON_Q '%q' "${PYTHON_BIN}"

echo "Checking ${NNODES} hosts (${WORLD_SIZE} ranks) from ${HOSTFILE}"
PRECHECK_PIDS=()
for host in "${ACTIVE_HOSTS[@]}"; do
  (
    ssh \
      -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" \
      "${SSH_USER}@${host}" \
      "cd '${ROOT_DIR}' && ip link show '${NCCL_SOCKET_IFNAME}' >/dev/null && test \"\$(ulimit -Hn)\" -ge '${REMOTE_NOFILE_LIMIT}' && ${PYTHON_Q} -c 'import deepspeed' && test \"\$(${PYTHON_Q} -c 'import torch; print(torch.cuda.device_count())')\" -ge '${GPUS_PER_NODE}'"
    echo "  ${host}: ready"
  ) &
  PRECHECK_PIDS+=("$!")
done
precheck_failed=0
for index in "${!PRECHECK_PIDS[@]}"; do
  if ! wait "${PRECHECK_PIDS[${index}]}"; then
    echo "  ${ACTIVE_HOSTS[${index}]}: precheck failed" >&2
    precheck_failed=1
  fi
done
if (( precheck_failed )); then
  exit 1
fi

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
if [[ -n "${EMPTY_EMB_PATH:-}" ]]; then
  ARGS+=(--empty-emb-path "${EMPTY_EMB_PATH}")
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
  "${PYTHON_BIN}" -m deepspeed.launcher.runner \
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
printf -v ASYNC_ERROR_Q '%q' "${TORCH_NCCL_ASYNC_ERROR_HANDLING}"
printf -v BLOCKING_WAIT_Q '%q' "${TORCH_NCCL_BLOCKING_WAIT}"
printf -v OMP_THREADS_Q '%q' "${OMP_NUM_THREADS}"
printf -v LOG_LOCAL_RANK_Q '%q' "${LOG_LOCAL_RANK}"

mkdir -p "${LOG_ROOT}" "${PID_ROOT}"

PIDS=()
declare -A PID_TO_HOST=()
declare -A PID_TO_RANK=()

cleanup_remote() {
  set +e
  echo "Stopping remote process groups for run ${RUN_ID}" >&2
  local cleanup_pids=()
  local node_rank host pid_file pid_file_q
  for node_rank in "${!ACTIVE_HOSTS[@]}"; do
    host="${ACTIVE_HOSTS[${node_rank}]}"
    pid_file="${PID_ROOT}/node_${node_rank}.pid"
    printf -v pid_file_q '%q' "${pid_file}"
    ssh \
      -o BatchMode=yes \
      -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" \
      "${SSH_USER}@${host}" \
      "pid_file=${pid_file_q}; if [[ -f \${pid_file} ]]; then pgid=\$(cat \${pid_file}); if [[ \${pgid} =~ ^[0-9]+$ ]] && (( pgid > 1 )); then kill -TERM -- -\${pgid} 2>/dev/null || true; fi; fi" &
    cleanup_pids+=("$!")
  done
  for pid in "${cleanup_pids[@]}"; do
    wait "${pid}" || true
  done
  sleep "${REMOTE_KILL_GRACE_SECONDS}"
  cleanup_pids=()
  for node_rank in "${!ACTIVE_HOSTS[@]}"; do
    host="${ACTIVE_HOSTS[${node_rank}]}"
    pid_file="${PID_ROOT}/node_${node_rank}.pid"
    printf -v pid_file_q '%q' "${pid_file}"
    ssh \
      -o BatchMode=yes \
      -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" \
      "${SSH_USER}@${host}" \
      "pid_file=${pid_file_q}; if [[ -f \${pid_file} ]]; then pgid=\$(cat \${pid_file}); if [[ \${pgid} =~ ^[0-9]+$ ]] && (( pgid > 1 )); then kill -KILL -- -\${pgid} 2>/dev/null || true; fi; rm -f \${pid_file}; fi" &
    cleanup_pids+=("$!")
  done
  for pid in "${cleanup_pids[@]}"; do
    wait "${pid}" || true
  done
  kill "${PIDS[@]}" 2>/dev/null || true
  for pid in "${PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  set -e
}

cleanup_required=1
on_exit() {
  local status=$?
  trap - EXIT HUP INT TERM
  if (( cleanup_required && status != 0 )); then
    cleanup_remote
  fi
  exit "${status}"
}
trap on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

for node_rank in "${!ACTIVE_HOSTS[@]}"; do
  host="${ACTIVE_HOSTS[${node_rank}]}"
  node_log_dir="${LOG_ROOT}/node_${node_rank}"
  pid_file="${PID_ROOT}/node_${node_rank}.pid"
  printf -v NODE_LOG_Q '%q' "${node_log_dir}"
  printf -v PID_FILE_Q '%q' "${pid_file}"
  inner_command="echo \$\$ > ${PID_FILE_Q} && exec ${PYTHON_Q} -m torch.distributed.run \
--nnodes=${NNODES} \
--nproc-per-node=${GPUS_PER_NODE} \
--node-rank=${node_rank} \
--master-addr=${MASTER_ADDR} \
--master-port=${MASTER_PORT} \
--log-dir=${NODE_LOG_Q} \
--redirects=3 \
--tee=3 \
--local-ranks-filter=${LOG_LOCAL_RANK_Q} \
-m wan_va.train${ARGS_Q}"
  printf -v INNER_COMMAND_Q '%q' "${inner_command}"
  remote_command="cd ${ROOT_Q} && \
mkdir -p ${NODE_LOG_Q} && \
ulimit -n ${REMOTE_NOFILE_LIMIT} && \
CUDA_VISIBLE_DEVICES=${VISIBLE_Q} \
NCCL_SOCKET_IFNAME=${IFNAME_Q} \
NCCL_IB_DISABLE=${IB_DISABLE_Q} \
NCCL_DEBUG=${NCCL_DEBUG_Q} \
TORCH_NCCL_ASYNC_ERROR_HANDLING=${ASYNC_ERROR_Q} \
TORCH_NCCL_BLOCKING_WAIT=${BLOCKING_WAIT_Q} \
TORCH_DISTRIBUTED_TIMEOUT=${DIST_TIMEOUT_Q} \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=${ALLOC_CONF_Q} \
OMP_NUM_THREADS=${OMP_THREADS_Q} \
setsid --wait bash -c ${INNER_COMMAND_Q}"
  ssh \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" \
    "${SSH_USER}@${host}" \
    "${remote_command}" &
  PIDS+=("$!")
  PID_TO_HOST[$!]="${host}"
  PID_TO_RANK[$!]="${node_rank}"
done

remaining=${#PIDS[@]}
while (( remaining > 0 )); do
  finished_pid=""
  if wait -n -p finished_pid; then
    :
  else
    status=$?
    echo "Node ${PID_TO_RANK[${finished_pid}]:-?} (${PID_TO_HOST[${finished_pid}]:-unknown}) failed with status ${status}" >&2
    exit "${status}"
  fi
  remaining=$((remaining - 1))
done

cleanup_required=0
rm -f "${PID_ROOT}"/node_*.pid
trap - EXIT HUP INT TERM
echo "All ${NNODES} nodes completed successfully. Logs: ${LOG_ROOT}"
