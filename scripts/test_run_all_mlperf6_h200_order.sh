#!/usr/bin/env bash
#
# Regression test for scripts/run_all_mlperf6_h200.sh sequencing, reuse, and
# failure behavior. It is hermetic: no real Docker, downloads, benchmark runs,
# or report generation happen. A fake R2 downloader is served over curl's
# file:// transport, and the benchmark run / report / preflight commands are
# replaced with stubs on PATH or inside a throwaway fake repo tree.
#
# Skips cleanly on non-Linux / no-bash hosts (the production script refuses to
# run anywhere but the Linux benchmark server), so this is meant to be executed
# on that server.

set -euo pipefail

SKIP_CODE=0

if ! command -v bash > /dev/null 2>&1; then
  echo "SKIP: bash not available"
  exit "${SKIP_CODE}"
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "SKIP: run_all_mlperf6_h200.sh only runs on Linux; this test mirrors that guard (got $(uname -s))."
  exit "${SKIP_CODE}"
fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_SCRIPT="${THIS_DIR}/run_all_mlperf6_h200.sh"

if [[ ! -f "${REAL_SCRIPT}" ]]; then
  echo "FAIL: cannot find script under test: ${REAL_SCRIPT}" >&2
  exit 1
fi

PASS_COUNT=0
fail() {
  echo "FAIL: $*" >&2
  exit 1
}
ok() {
  PASS_COUNT=$((PASS_COUNT + 1))
  echo "  ok: $*"
}

# ---------------------------------------------------------------------------
# Static, source-level evidence: the production script never deletes data.
# Strip comments first so a "no rm -rf" comment can't trip the check; we only
# care about executable code paths.
# ---------------------------------------------------------------------------
REAL_CODE="$(sed 's/#.*//' "${REAL_SCRIPT}")"
if printf '%s\n' "${REAL_CODE}" | grep -qE 'rm[[:space:]]+-rf'; then
  fail "production script contains an 'rm -rf' code path"
fi
if printf '%s\n' "${REAL_CODE}" | grep -qE 'rsync[^\n]*--delete'; then
  fail "production script contains an 'rsync --delete' code path"
fi
ok "no 'rm -rf' or 'rsync --delete' in production script code"

# ---------------------------------------------------------------------------
# Scaffolding shared by all scenarios.
# ---------------------------------------------------------------------------
WORKDIR="$(mktemp -d)"
cleanup() { rm -rf -- "${WORKDIR}"; }
trap cleanup EXIT

FAKE_BIN="${WORKDIR}/bin"
mkdir -p "${FAKE_BIN}"

# Stub the preflight commands that may be absent on a CI box. require_cmd only
# needs them to resolve on PATH; their behavior is irrelevant.
for cmd in docker git python3; do
  cat > "${FAKE_BIN}/${cmd}" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB
  chmod +x "${FAKE_BIN}/${cmd}"
done

# Fake R2 downloader, served to the script via `curl -fsSL file://...`.
# Invoked as: <downloader> -d <dirname> <uri>
# Creates the destination directory so the caller's `touch <marker>` succeeds.
# Fails for a single URI named in FAKE_FAIL_URI to exercise download failure.
FAKE_DOWNLOADER="${WORKDIR}/fake-r2-downloader.sh"
cat > "${FAKE_DOWNLOADER}" <<'DL'
#!/usr/bin/env bash
set -euo pipefail
dir=""
uri=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d) dir="$2"; shift 2 ;;
    *)  uri="$1"; shift ;;
  esac
done
if [[ -n "${FAKE_FAIL_URI:-}" && "${uri}" == "${FAKE_FAIL_URI}" ]]; then
  echo "fake downloader: forced failure for ${uri}" >&2
  exit 1
fi
mkdir -p "${dir}"
DL
chmod +x "${FAKE_DOWNLOADER}"

# Build a throwaway fake repo tree so REPO_ROOT (derived from the script's own
# location) points at stub sibling scripts instead of the real ones.
make_fake_repo() {
  local root="$1"
  mkdir -p "${root}/scripts" "${root}/configs"
  cp "${REAL_SCRIPT}" "${root}/scripts/run_all_mlperf6_h200.sh"
  chmod +x "${root}/scripts/run_all_mlperf6_h200.sh"

  # Stub benchmark runner: records nothing, just succeeds.
  cat > "${root}/scripts/run_mlperf6_h200.sh" <<'RUN'
#!/usr/bin/env bash
echo "stub run_mlperf6_h200 $*"
exit 0
RUN
  chmod +x "${root}/scripts/run_mlperf6_h200.sh"

  # Stub report generator (generate_report calls python3 on it; python3 is also
  # stubbed, but keep the file present for realism).
  cat > "${root}/scripts/report_mlperf6_results.py" <<'PY'
print("stub report")
PY
}

# Write a minimal env file. All roots live inside the scenario dir.
write_env() {
  local env_file="$1"
  local data_root="$2"
  local results_root="$3"
  local work_root="$4"
  cat > "${env_file}" <<ENV
export MLPERF_WORK_ROOT="${work_root}"
export MLPERF_DATA_ROOT="${data_root}"
export MLPERF_RESULTS_ROOT="${results_root}"
export MLPERF_UPSTREAM_DIR="${work_root}/mlcommons-training"
export MLPERF_R2_DOWNLOADER_URL="file://${FAKE_DOWNLOADER}"
export MLPERF_ORCH_ROOT="${results_root}/orchestration"
export MLPERF_PIPELINE_STATUS_PATH="${results_root}/orchestration/pipeline-status.tsv"
export MLPERF_FINAL_REPORT_PATH="${results_root}/final-report.md"

export MLPERF_LLAMA31_PREPROCESSED_PATH="${data_root}/llama31/preprocessed_c4"
export MLPERF_LLAMA31_TOKENIZER_PATH="${data_root}/llama31/tokenizer"
export MLPERF_LLAMA31_DATASET_URI="uri://llama31-c4"
export MLPERF_LLAMA31_TOKENIZER_URI="uri://llama31-tokenizer"

export MLPERF_LLAMA2_MODE="skip"
export MLPERF_LLAMA2_DATASET_PATH="${data_root}/llama2/dataset"
export MLPERF_LLAMA2_MODEL_ROOT="${data_root}/llama2/models"
export MLPERF_LLAMA2_LOCAL_DATASET_SUBDIR="scrolls"
export MLPERF_LLAMA2_LOCAL_MODEL_SUBDIR="model"
export MLPERF_LLAMA2_RCLONE_CONFIG=""

export MLPERF_GPT_OSS_DATA_PATH="${data_root}/gpt_oss_20b/data"
export MLPERF_GPT_OSS_MODEL_PATH="${data_root}/gpt_oss_20b/model"
export MLPERF_GPT_OSS_DATASET_URI="uri://llama31-c4"
export MLPERF_GPT_OSS_TOKENIZER_SOURCE="${data_root}/llama31/tokenizer"

export MLPERF_FLUX_DATASET_PATH="${data_root}/flux"
export MLPERF_FLUX_CC12M_PREPROCESSED_URI="uri://flux-cc12m"
export MLPERF_FLUX_COCO_PREPROCESSED_URI="uri://flux-coco"
export MLPERF_FLUX_EMPTY_ENCODINGS_URI="uri://flux-empty"
ENV
}

# Seed a completed llama31 dataset + tokenizer so reuse/sync paths are exercised
# without a real download.
seed_llama31() {
  local data_root="$1"
  mkdir -p "${data_root}/llama31/preprocessed_c4" "${data_root}/llama31/tokenizer"
  touch "${data_root}/llama31/preprocessed_c4/.mlperf-download-complete"
  touch "${data_root}/llama31/preprocessed_c4/shard0.bin"
  touch "${data_root}/llama31/tokenizer/.mlperf-download-complete"
  echo "tokenizer-bytes" > "${data_root}/llama31/tokenizer/tokenizer.model"
}

# First status-file line number for a benchmark+stage pair (exact match on the
# first two tab-separated fields). Prints empty if absent.
line_of() {
  local status_file="$1" benchmark="$2" stage="$3"
  awk -F '\t' -v b="${benchmark}" -v s="${stage}" \
    'NR>1 && $1==b && $2==s { print NR; exit }' "${status_file}"
}

status_field() {
  local status_file="$1" benchmark="$2" stage="$3"
  awk -F '\t' -v b="${benchmark}" -v s="${stage}" \
    'NR>1 && $1==b && $2==s { print $3; exit }' "${status_file}"
}

run_pipeline() {
  # run_pipeline <scenario_root> <extra-args...> ; relies on env exported by caller.
  local root="$1"; shift
  PATH="${FAKE_BIN}:${PATH}" bash "${root}/scripts/run_all_mlperf6_h200.sh" \
    --env-file "${root}/configs/h200.env" "$@"
}

# ===========================================================================
echo "Scenario 1: happy-path ordering (llama31 then gpt_oss20b)"
# ===========================================================================
S1="${WORKDIR}/s1"
mkdir -p "${S1}"
make_fake_repo "${S1}"
DATA1="${S1}/data"; RES1="${S1}/results"; WORK1="${S1}/work"
write_env "${S1}/configs/h200.env" "${DATA1}" "${RES1}" "${WORK1}"
seed_llama31 "${DATA1}"
# Pre-existing file in the gpt_oss model dir must survive the additive sync.
mkdir -p "${DATA1}/gpt_oss_20b/model"
echo "keep" > "${DATA1}/gpt_oss_20b/model/pre_existing.txt"

set +e
run_pipeline "${S1}" --skip-bootstrap --benchmarks llama31,gpt_oss20b > "${S1}/out.log" 2>&1
rc=$?
set -e
[[ ${rc} -eq 0 ]] || { cat "${S1}/out.log"; fail "scenario 1 pipeline exited ${rc}"; }

STATUS1="${RES1}/orchestration/pipeline-status.tsv"
[[ -f "${STATUS1}" ]] || fail "scenario 1 status file missing"

l_ds="$(line_of "${STATUS1}" llama31 download-dataset)"
l_tk="$(line_of "${STATUS1}" llama31 download-tokenizer)"
l_l31_run="$(line_of "${STATUS1}" llama31 run)"
g_ds="$(line_of "${STATUS1}" gpt_oss20b download-dataset)"
g_tk="$(line_of "${STATUS1}" gpt_oss20b tokenizer)"
g_run="$(line_of "${STATUS1}" gpt_oss20b run)"

for v in "${l_ds}" "${l_tk}" "${l_l31_run}" "${g_ds}" "${g_tk}" "${g_run}"; do
  [[ -n "${v}" ]] || { cat "${STATUS1}"; fail "scenario 1 missing an expected status stage"; }
done

(( l_ds < l_l31_run )) || fail "llama31 download-dataset must precede llama31 run"
(( l_tk < l_l31_run )) || fail "llama31 download-tokenizer must precede llama31 run"
(( l_l31_run < g_ds )) || fail "llama31 run must precede gpt_oss20b dataset work"
(( l_l31_run < g_tk )) || fail "llama31 run must precede gpt_oss20b tokenizer work"
(( g_ds < g_run )) || fail "gpt_oss20b dataset must precede gpt_oss20b run"
(( g_tk < g_run )) || fail "gpt_oss20b tokenizer must precede gpt_oss20b run"
ok "ordering: llama31 download -> run -> gpt_oss20b dataset/tokenizer -> run"

# Dataset reuse symlink created, tokenizer synced non-destructively.
[[ -L "${DATA1}/gpt_oss_20b/data" ]] || fail "gpt_oss20b dataset should be a reuse symlink"
[[ "$(status_field "${STATUS1}" gpt_oss20b download-dataset)" == "skipped" ]] \
  || fail "gpt_oss20b dataset reuse should record 'skipped'"
[[ -f "${DATA1}/gpt_oss_20b/model/pre_existing.txt" ]] \
  || fail "pre-existing tokenizer dest file was destroyed by sync"
[[ -f "${DATA1}/gpt_oss_20b/model/tokenizer.model" ]] \
  || fail "tokenizer file was not synced into the gpt_oss model dir"
ok "reuse symlink created; tokenizer sync additive (kept pre-existing file)"

# ===========================================================================
echo "Scenario 2: download failure skips that run, continues to next benchmark"
# ===========================================================================
S2="${WORKDIR}/s2"
mkdir -p "${S2}"
make_fake_repo "${S2}"
DATA2="${S2}/data"; RES2="${S2}/results"; WORK2="${S2}/work"
write_env "${S2}/configs/h200.env" "${DATA2}" "${RES2}" "${WORK2}"
# Do NOT seed llama31, so it must download; force that download to fail.
export FAKE_FAIL_URI="uri://llama31-c4"

set +e
run_pipeline "${S2}" --skip-bootstrap --benchmarks llama31,flux > "${S2}/out.log" 2>&1
rc=$?
set -e
unset FAKE_FAIL_URI
# Non fail-fast: overall pipeline keeps going; exit code may be non-zero, but the
# behavior we assert is in the status file.
STATUS2="${RES2}/orchestration/pipeline-status.tsv"
[[ -f "${STATUS2}" ]] || fail "scenario 2 status file missing"

[[ "$(status_field "${STATUS2}" llama31 download-dataset)" == "failed" ]] \
  || { cat "${STATUS2}"; fail "scenario 2 llama31 download-dataset should be failed"; }
[[ "$(status_field "${STATUS2}" llama31 run)" == "skipped" ]] \
  || { cat "${STATUS2}"; fail "scenario 2 llama31 run should be skipped after download failure"; }
# Pipeline continued to flux.
[[ -n "$(line_of "${STATUS2}" flux download-cc12m-preprocessed)" ]] \
  || { cat "${STATUS2}"; fail "scenario 2 should continue to flux after llama31 failure"; }
ok "download failure -> run skipped -> continues to flux (no --fail-fast)"

# ===========================================================================
echo "Scenario 3: --fail-fast stops before the failing benchmark's run"
# ===========================================================================
S3="${WORKDIR}/s3"
mkdir -p "${S3}"
make_fake_repo "${S3}"
DATA3="${S3}/data"; RES3="${S3}/results"; WORK3="${S3}/work"
write_env "${S3}/configs/h200.env" "${DATA3}" "${RES3}" "${WORK3}"
export FAKE_FAIL_URI="uri://llama31-c4"

set +e
run_pipeline "${S3}" --skip-bootstrap --fail-fast --benchmarks llama31,flux > "${S3}/out.log" 2>&1
rc=$?
set -e
unset FAKE_FAIL_URI
[[ ${rc} -ne 0 ]] || fail "scenario 3 --fail-fast should exit non-zero"
STATUS3="${RES3}/orchestration/pipeline-status.tsv"
[[ "$(status_field "${STATUS3}" llama31 download-dataset)" == "failed" ]] \
  || { cat "${STATUS3}"; fail "scenario 3 llama31 download should be failed"; }
[[ -z "$(line_of "${STATUS3}" llama31 run)" ]] \
  || { cat "${STATUS3}"; fail "scenario 3 should stop before recording any llama31 run"; }
[[ -z "$(line_of "${STATUS3}" flux download-cc12m-preprocessed)" ]] \
  || { cat "${STATUS3}"; fail "scenario 3 --fail-fast should not reach flux"; }
ok "--fail-fast halts before failed benchmark run and never reaches flux"

# ===========================================================================
echo "Scenario 4: invalid gpt_oss20b reuse symlink fails closed"
# ===========================================================================
S4="${WORKDIR}/s4"
mkdir -p "${S4}"
make_fake_repo "${S4}"
DATA4="${S4}/data"; RES4="${S4}/results"; WORK4="${S4}/work"
write_env "${S4}/configs/h200.env" "${DATA4}" "${RES4}" "${WORK4}"
seed_llama31 "${DATA4}"
# Plant a stale symlink pointing somewhere other than the llama31 corpus.
mkdir -p "${DATA4}/gpt_oss_20b" "${DATA4}/bogus_target"
ln -sfn "${DATA4}/bogus_target" "${DATA4}/gpt_oss_20b/data"

set +e
run_pipeline "${S4}" --skip-bootstrap --benchmarks gpt_oss20b > "${S4}/out.log" 2>&1
rc=$?
set -e
STATUS4="${RES4}/orchestration/pipeline-status.tsv"
[[ "$(status_field "${STATUS4}" gpt_oss20b download-dataset)" == "failed" ]] \
  || { cat "${STATUS4}"; fail "scenario 4 invalid symlink should record failed download-dataset"; }
[[ "$(status_field "${STATUS4}" gpt_oss20b run)" == "skipped" ]] \
  || { cat "${STATUS4}"; fail "scenario 4 gpt_oss20b run should be skipped after invalid symlink"; }
# The stale symlink must be left untouched (non-destructive).
[[ -L "${DATA4}/gpt_oss_20b/data" ]] || fail "scenario 4 stale symlink should not be deleted"
ok "invalid reuse symlink fails closed and is left in place"

echo
echo "ALL ${PASS_COUNT} CHECKS PASSED"
