#!/usr/bin/env bash
set -euo pipefail

bench_dir="${ANS_BENCH_ROOT:-${SLURM_SUBMIT_DIR:?Submit from ans-amd-bench}}"
cd "${bench_dir}"
r2_mi350="${1:?Usage: job_lds_s1_followup.sh R2_MI350 R3_MI300 R3_MI350}"
r3_mi300="${2:?Usage: job_lds_s1_followup.sh R2_MI350 R3_MI300 R3_MI350}"
r3_mi350="${3:?Usage: job_lds_s1_followup.sh R2_MI350 R3_MI300 R3_MI350}"
output="${bench_dir}/results/profiling/lds-s1-followup-${SLURM_JOB_ID}"
mkdir -p "${output}"

jobs=("${r2_mi350}" "${r3_mi300}" "${r3_mi350}")
machines=("mi350x-r2" "mi300x-r3" "mi350x-r3")
jobs_csv="${r2_mi350},${r3_mi300},${r3_mi350}"
sacct -n -X -P -j "${jobs_csv}" \
  --format=JobIDRaw,State,ExitCode,Elapsed,NodeList >"${output}/sacct.tsv"
printf 'job_id\tstage\tstate\texit_code\tanalysis\tprofile_dir\n' \
  >"${output}/handoff.tsv"

status_for() {
  local job_id="$1"
  awk -F'|' -v wanted="${job_id}" '$1 == wanted { print $2 "|" $3; exit }' \
    "${output}/sacct.tsv"
}

for index in 0 1 2; do
  job_id="${jobs[${index}]}"
  stage="${machines[${index}]}"
  status="$(status_for "${job_id}")"
  state="${status%%|*}"
  exit_code="${status#*|}"
  if [[ -z "${status}" ]]; then
    state=UNKNOWN
    exit_code=UNKNOWN
  fi
  analysis=not_applicable
  profile_dir=""

  if [[ "${stage}" == mi300x-r3 || "${stage}" == mi350x-r3 ]]; then
    machine="${stage%-r3}"
    profile_dir="${bench_dir}/results/profiling/lds-r3-glm-${machine}-${job_id}/rocprof"
    analysis=skipped_job_not_successful
    if [[ "${state}" == COMPLETED && "${exit_code}" == 0:0 ]]; then
      if [[ -d "${profile_dir}/counters/0" ]] && command -v rocprof-compute >/dev/null 2>&1; then
        analysis=failed
        if bash scripts/analyze_lds_profile.sh "${profile_dir}" \
            >"${profile_dir}/analysis-launch.log" 2>&1; then
          analysis=available
        else
          analysis="failed_rc_$?"
        fi
      else
        analysis=no_counter_data_or_rocprof_compute_unavailable
      fi
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${stage}" "${state}" "${exit_code}" "${analysis}" "${profile_dir}" \
    >>"${output}/handoff.tsv"
done

printf 'S1 queue follow-up complete: %s\n' "${output}"
