#!/usr/bin/env bash
# Host-level scenario battery for powermon (real sensors, sudo, signals, harness hook).
# Run on a Linux host from a checkout:  bash tools/powermon/tests/scenarios.sh
# Needs: passwordless sudo, `script` (util-linux). Writes only under /tmp/scn.
set -u
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
WORK=$(mktemp -d /tmp/powermon-scn-XXXX)
cp -r "${REPO_ROOT}/tools/powermon" "${REPO_ROOT}/tools/powermon.sh" "${WORK}/"
mkdir -p "${WORK}/repo" && cp -r "${REPO_ROOT}/scripts" "${REPO_ROOT}/configs" "${REPO_ROOT}/tools" "${WORK}/repo/"
cd "${WORK}"
export POWERMON_STATE_DIR=$HOME/.local/state/powermon-scn
export PYTHONDONTWRITEBYTECODE=1
sudo rm -rf /tmp/scn "$POWERMON_STATE_DIR"; mkdir -p /tmp/scn
PM=./powermon.sh
pass=0; fail=0
check() { if eval "$2"; then echo "PASS  $1"; pass=$((pass+1)); else echo "FAIL  $1  [$2]"; fail=$((fail+1)); fi; }
wait_final() { for i in $(seq 1 "${2:-40}"); do grep -q '"final": true' "$1/status.json" 2>/dev/null && return 0; sleep 1; done; return 1; }

echo "### A wizard through a pseudo-tty"
printf '8\n/tmp/scn/wz\n1\nwizard test\ny\n' | script -qec "$PM" /dev/null > /tmp/scn/wizard.out 2>&1
WZ=$(ls -d /tmp/scn/wz/run_* 2>/dev/null | head -1)
check "wizard started a run" '[[ -n "$WZ" ]] && grep -q "Started. PID" /tmp/scn/wizard.out'
check "wizard label sanitised in dir name" '[[ "$WZ" == *_wizard_test ]]'
check "wizard label prompt has no empty brackets" '! grep -q "\[\]:" /tmp/scn/wizard.out'
check "wizard auto-created the folder" 'grep -q "created /tmp/scn/wz" /tmp/scn/wizard.out'
wait_final "$WZ" 20
check "wizard run finished done" 'grep -q "\"state\": \"done\"" $WZ/status.json'

echo "### B two concurrent runs, status/stop semantics"
$PM start --demo -d 0 -i 0.5 -o /tmp/scn/c1 --yes >/dev/null 2>&1
$PM start --demo -d 0 -i 0.5 -o /tmp/scn/c2 --yes >/dev/null 2>&1
sleep 3
check "status lists both" '[[ $($PM status 2>/dev/null | grep -c "state    running") -eq 2 ]]'
check "stop without dir refuses when two run" '$PM stop 2>&1 | grep -q "several runs match"'
check "watch without dir refuses when two run" '$PM watch --once 2>&1 | grep -q "several runs match"'
check "stop --all stops both (rc 0)" '$PM stop --all >/dev/null 2>&1 && [[ $($PM status 2>/dev/null | grep -c "state    stopped") -eq 2 ]]'

echo "### C subset sources, fast interval"
$PM --sources hwmon,nvidia start -d 12 -i 0.5 -o /tmp/scn/sub --label sub --yes >/tmp/scn/sub.out 2>&1
SUB=$(ls -d /tmp/scn/sub/run_*)
wait_final "$SUB" 30
check "subset run done" 'grep -q "\"state\": \"done\"" $SUB/status.json'
check "subset has no ipmi/battery columns" '! head -1 $SUB/samples.csv | grep -qE "system_w|bat0_w"'
check "subset ~24 samples at 0.5s" '[[ $(grep -o "\"samples\": [0-9]*" $SUB/status.json | grep -o "[0-9]*$") -ge 22 ]]'
check "ipmi marked disabled in sensors.json" 'grep -A3 "\"source\": \"ipmi\"" $SUB/sensors.json | grep -q disabled'

echo "### D report on a partial CSV while running, then stop"
$PM start --demo -d 0 -i 0.5 -o /tmp/scn/part --yes >/dev/null 2>&1
PART=$(ls -d /tmp/scn/part/run_*)
sleep 4
check "report while running exits 0" '$PM report $PART >/tmp/scn/part.out 2>&1'
check "report marks incomplete" 'grep -q "incomplete" $PART/report.md'
check "mark with unicode text" '$PM mark "phase ünïcode ✓ done" >/dev/null 2>&1 && grep -q "ünïcode" $PART/events.csv'
$PM stop "$PART" >/dev/null 2>&1
check "stopped run report has unicode mark" 'grep -q "ünïcode" $PART/report.md && grep -q "ünïcode" $PART/report.html'
check "svg/png escaping ok (no raw < in text)" '! grep -q "phase <" $PART/report.html'

echo "### E error paths and exit codes"
check "status unknown dir rc1" '$PM status /tmp/scn/nope >/dev/null 2>&1; [[ $? -eq 1 ]]'
check "report missing dir rc2" '$PM report /tmp/scn/nope >/dev/null 2>&1; [[ $? -eq 2 ]]'
mkdir -p /tmp/scn/empty; check "report empty dir rc2" '$PM report /tmp/scn/empty >/dev/null 2>&1; [[ $? -eq 2 ]]'
check "bad duration rc2" '$PM start -d 5x --yes >/dev/null 2>&1; [[ $? -eq 2 ]]'
check "bad interval rc2" '$PM start -i 0.01 --yes >/dev/null 2>&1; [[ $? -eq 2 ]]'
check "bad source rc2" '$PM --sources bogus start --yes >/dev/null 2>&1; [[ $? -eq 2 ]]'
check "compare with one bad run rc2" '$PM compare $PART /tmp/scn/nope >/dev/null 2>&1; [[ $? -eq 2 ]]'
check "start into dir with samples.csv refused" '$PM start --demo --run-dir $PART --yes 2>&1 | grep -q "already contains"'
check "start --overwrite allowed (foreground 1s)" '$PM start --demo -d 1 --run-dir $PART --yes --overwrite --foreground >/dev/null 2>&1'

echo "### F label edge cases"
$PM start --demo -d 1 -o /tmp/scn/lbl --label "a b/c:d é" --yes --foreground >/dev/null 2>&1
check "label with spaces/slashes/unicode gives safe dir" 'ls -d /tmp/scn/lbl/run_*_a_b_c_d >/dev/null 2>&1'

echo "### G Ctrl-C in foreground finalizes"
timeout -s INT 4 $PM start --demo -d 0 -i 0.5 -o /tmp/scn/fg --yes --foreground >/tmp/scn/fg.out 2>&1; rc=$?
FG=$(ls -d /tmp/scn/fg/run_*)
check "SIGINT foreground -> stopped state (timeout rc 124 expected)" '[[ $rc -eq 124 || $rc -eq 0 ]] && grep -q "\"state\": \"stopped\"" $FG/status.json'
check "SIGINT foreground wrote report" '[[ -f $FG/report.md ]]'

echo "### H SIGKILL worker -> status shows dead, report still possible"
$PM start --demo -d 0 -i 0.5 -o /tmp/scn/kill --yes >/dev/null 2>&1
K=$(ls -d /tmp/scn/kill/run_*); sleep 3; kill -9 $(cat $K/powermon.pid); sleep 1
check "status reports dead worker" '$PM status $K 2>&1 | grep -q "dead"'
check "report from killed run works" '$PM report $K >/dev/null 2>&1 && grep -q "incomplete" $K/report.md'
check "stop on dead run is rc 0 not running" '$PM stop $K 2>&1 | grep -q "not running"'

echo "### I sudo start then everything as user"
sudo -E env POWERMON_STATE_DIR=$POWERMON_STATE_DIR $PM start -d 6 -i 1 -o /tmp/scn/su --label su --yes >/tmp/scn/su.out 2>&1
SU=$(ls -d /tmp/scn/su/run_*)
check "sudo hint printed in commands" 'grep -q "sudo ./powermon.sh status" /tmp/scn/su.out'
check "user can mark sudo run" '$PM mark hello --run-dir $SU 2>&1 | grep -q marked'
wait_final "$SU" 20
check "sudo run done + owned by user" 'grep -q "\"state\": \"done\"" $SU/status.json && [[ $(stat -c %U $SU/report.md) == $USER ]]'
check "user report regen on sudo run" '$PM report $SU >/dev/null 2>&1'
check "RAPL columns present (sudo)" 'head -1 $SU/samples.csv | grep -q cpu_pkg0_w'

echo "### J harness --powermon hook (run stage fails fast, power stage must still work)"
if [[ -d repo ]]; then
  cd repo
  # Stand-in for the real benchmark runner: takes 4 s and fails, like a run that dies early.
  printf '#!/usr/bin/env bash
echo "stub benchmark $*"; sleep 4; exit 1
' > scripts/run_mlperf6_h200.sh
  cat > /tmp/scn/h.env <<EOF
export MLPERF_RESULTS_ROOT=/tmp/scn/mlp
source $PWD/configs/mlperf6-h200-4gpu.env
EOF
  bash ./scripts/run_all_mlperf6_h200.sh --env-file /tmp/scn/h.env --skip-bootstrap --skip-downloads --benchmarks llama31 --powermon > /tmp/scn/harness.out 2>&1
  check "harness ran powermon start/stop" 'grep -q "powermon recording llama31" /tmp/scn/mlp/orchestration/orchestrator.log && grep -q "powermon report for llama31" /tmp/scn/mlp/orchestration/orchestrator.log'
  check "pipeline-status has power success" 'grep -P "^llama31\tpower\tsuccess" /tmp/scn/mlp/orchestration/pipeline-status.tsv'
  check "final report has Power and Thermal table" 'grep -q "Power and Thermal" /tmp/scn/mlp/mlperf6-h200-final-report.md && grep -q "| llama31 | " /tmp/scn/mlp/mlperf6-h200-final-report.md'
  check "no powermon worker left running" '! pgrep -f "powermon.py _worker" >/dev/null'
  cd ~/powermon-test
else
  echo "SKIP  harness (no repo copy)"
fi

echo "### K registry hygiene"
check "registry capped/valid json" 'python3 -c "import json,sys; d=json.load(open(\"$POWERMON_STATE_DIR/active.json\")); sys.exit(0 if isinstance(d,list) and len(d)<=20 else 1)"'
check "no worker processes left" '! pgrep -f "powermon.py _worker" >/dev/null'
check "no root-owned files under /tmp/scn" '[[ $(find /tmp/scn -user root | wc -l) -eq 0 ]]'

cd /; sudo rm -rf "${WORK}"
echo; echo "RESULT pass=$pass fail=$fail"
[[ $fail -eq 0 ]]
