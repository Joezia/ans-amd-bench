#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 PARTITION[,PARTITION...] [REQUIRED_GPUS]" >&2
  exit 2
fi

candidate_csv="$1"
required_gpus="${2:-0}"
if [[ ! "${required_gpus}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: REQUIRED_GPUS must be a non-negative integer" >&2
  exit 2
fi

tres_count() {
  local tres="$1"
  local key="$2"
  local item
  IFS=',' read -r -a items <<< "${tres}"
  for item in "${items[@]}"; do
    if [[ "${item%%=*}" == "${key}" ]]; then
      echo "${item#*=}"
      return
    fi
  done
  echo 0
}

state_rank() {
  case "$1" in
    idle*) echo 0 ;;
    mix*) echo 1 ;;
    alloc*) echo 2 ;;
    comp*) echo 3 ;;
    *) echo 4 ;;
  esac
}

IFS=',' read -r -a raw_candidates <<< "${candidate_csv}"
declare -a up_partitions=()
declare -a immediate_partitions=()
best_score=999999

for raw_partition in "${raw_candidates[@]}"; do
  partition="${raw_partition//[[:space:]]/}"
  [[ -n "${partition}" ]] || continue

  partition_info="$(scontrol show partition "${partition}" -o 2>/dev/null || true)"
  if [[ -z "${partition_info}" || "${partition_info}" != *"State=UP"* ]]; then
    echo "partition-selector: skip ${partition}: missing or not UP" >&2
    continue
  fi
  partition_score=999999
  partition_free=-1
  exclusive_incompatible=0
  while IFS='|' read -r node compact_state; do
    [[ -n "${node}" ]] || continue
    node_info="$(scontrol show node "${node}" -o 2>/dev/null || true)"
    [[ -n "${node_info}" ]] || continue
    cfg_tres="$(sed -n 's/.*CfgTRES=\([^ ]*\).*/\1/p' <<< "${node_info}")"
    alloc_tres="$(sed -n 's/.*AllocTRES=\([^ ]*\).*/\1/p' <<< "${node_info}")"

    if (( required_gpus > 0 )); then
      configured="$(tres_count "${cfg_tres}" "gres/gpu")"
      allocated="$(tres_count "${alloc_tres}" "gres/gpu")"
      if [[ "${partition_info}" == *"OverSubscribe=EXCLUSIVE"* ]] &&
         (( configured > required_gpus )); then
        exclusive_incompatible=1
      fi
    else
      configured="$(tres_count "${cfg_tres}" "cpu")"
      allocated="$(tres_count "${alloc_tres}" "cpu")"
    fi
    free=$((configured - allocated))
    (( free < 0 )) && free=0

    rank="$(state_rank "${compact_state}")"
    score=$((rank * 1000 - free))
    if (( free >= required_gpus && free > 0 && score < partition_score )); then
      partition_score="${score}"
      partition_free="${free}"
    fi
  done < <(sinfo -h -N -p "${partition}" -o '%N|%t')

  if (( exclusive_incompatible != 0 )); then
    echo "partition-selector: skip ${partition}: exclusive nodes exceed requested GPU count" >&2
    continue
  fi
  up_partitions+=("${partition}")

  if (( partition_score < 999999 )); then
    echo "partition-selector: ${partition} immediately eligible; free_resource=${partition_free} score=${partition_score}" >&2
    if (( partition_score < best_score )); then
      best_score="${partition_score}"
      immediate_partitions=("${partition}")
    elif (( partition_score == best_score )); then
      immediate_partitions+=("${partition}")
    fi
  else
    echo "partition-selector: ${partition} is UP but has no immediately free requested resource" >&2
  fi
done

if (( ${#immediate_partitions[@]} > 0 )); then
  printf '%s\n' "${immediate_partitions[@]}" | shuf -n 1
  exit 0
fi
if (( ${#up_partitions[@]} > 0 )); then
  echo "partition-selector: no immediate match; choosing an UP partition for queued scheduling" >&2
  printf '%s\n' "${up_partitions[@]}" | shuf -n 1
  exit 0
fi

echo "ERROR: no UP partition found in: ${candidate_csv}" >&2
exit 1
