#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${1:-${REPO_ROOT}/artifacts/system}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: collect_system_config.sh must be run on a Linux server." >&2
  exit 1
fi

sanitize_label() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-'
}

HOST_LABEL_RAW="${HOSTNAME_OVERRIDE:-$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown-host)}"
HOST_LABEL="$(sanitize_label "${HOST_LABEL_RAW}")"
TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
SNAPSHOT_DIR="${OUTPUT_ROOT}/${HOST_LABEL}-${TIMESTAMP_UTC}"
RAW_DIR="${SNAPSHOT_DIR}/raw"
MANIFEST_FILE="${SNAPSHOT_DIR}/manifest.tsv"
WARNINGS_FILE="${SNAPSHOT_DIR}/warnings.txt"
SUMMARY_FILE="${SNAPSHOT_DIR}/summary.md"

mkdir -p "${RAW_DIR}"

printf 'name\tstatus\texit_code\tcommand_path\tcommand\n' > "${MANIFEST_FILE}"
: > "${WARNINGS_FILE}"

append_warning() {
  printf '%s\n' "$1" >> "${WARNINGS_FILE}"
}

capture_cmd() {
  local slug="$1"
  shift

  local command_name="$1"
  shift

  local command_path
  local output_file="${RAW_DIR}/${slug}.txt"

  if ! command_path="$(command -v "${command_name}" 2>/dev/null)"; then
    printf '%s\tmissing\t127\t-\t%s\n' "${slug}" "${command_name} $*" >> "${MANIFEST_FILE}"
    append_warning "Missing optional command: ${command_name}"
    return 0
  fi

  if "${command_path}" "$@" > "${output_file}" 2>&1; then
    printf '%s\tcaptured\t0\t%s\t%s\n' "${slug}" "${command_path}" "${command_name} $*" >> "${MANIFEST_FILE}"
    return 0
  fi

  local exit_code=$?
  printf '%s\tfailed\t%s\t%s\t%s\n' "${slug}" "${exit_code}" "${command_path}" "${command_name} $*" >> "${MANIFEST_FILE}"
  append_warning "Command failed (${exit_code}): ${command_name} $*"
  return 0
}

capture_shell() {
  local slug="$1"
  shift

  local command_text="$1"
  local output_file="${RAW_DIR}/${slug}.txt"
  local bash_path

  bash_path="$(command -v bash)"

  if "${bash_path}" -lc "${command_text}" > "${output_file}" 2>&1; then
    printf '%s\tcaptured\t0\t%s\t%s\n' "${slug}" "${bash_path}" "${command_text}" >> "${MANIFEST_FILE}"
    return 0
  fi

  local exit_code=$?
  printf '%s\tfailed\t%s\t%s\t%s\n' "${slug}" "${exit_code}" "${bash_path}" "${command_text}" >> "${MANIFEST_FILE}"
  append_warning "Shell command failed (${exit_code}): ${command_text}"
  return 0
}

capture_cmd "collector-date" date date -u
capture_cmd "os-release" cat /etc/os-release
capture_cmd "kernel-uname" uname -a
capture_cmd "hostnamectl" hostnamectl status
capture_cmd "lscpu" lscpu
capture_cmd "cpuinfo" cat /proc/cpuinfo
capture_cmd "numactl-hardware" numactl --hardware
capture_cmd "meminfo" cat /proc/meminfo
capture_cmd "free" free -h
capture_cmd "lsblk" lsblk -e7 -o NAME,KNAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINT,ROTA
capture_cmd "findmnt" findmnt -D
capture_cmd "df" df -hT
capture_cmd "mount" mount
capture_cmd "lspci" lspci -nn
capture_cmd "lsmod" lsmod
capture_cmd "nvidia-smi" nvidia-smi
capture_cmd "nvidia-smi-list" nvidia-smi -L
capture_cmd "nvidia-smi-query" nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit,pcie.link.gen.current,pcie.link.width.current,mig.mode.current --format=csv,noheader
capture_cmd "nvidia-topology" nvidia-smi topo -m
capture_cmd "nvidia-smi-q" nvidia-smi -q
capture_cmd "modinfo-nvidia" modinfo nvidia
capture_cmd "nvcc-version" nvcc --version
capture_cmd "docker-version" docker version
capture_cmd "podman-version" podman version
capture_cmd "python3-version" python3 --version
capture_cmd "gcc-version" gcc --version
capture_cmd "cmake-version" cmake --version
capture_cmd "git-version" git --version
capture_cmd "mpirun-version" mpirun --version
capture_cmd "ompi-info" ompi_info --version
capture_cmd "ibv-devinfo" ibv_devinfo
capture_cmd "ibstat" ibstat
capture_cmd "ofed-info" ofed_info -s
capture_cmd "rdma-link" rdma link

capture_shell "network-interfaces" "if [[ -d /sys/class/net ]]; then for iface in \$(ls /sys/class/net); do echo \"### \${iface}\"; if command -v ethtool >/dev/null 2>&1; then ethtool -i \"\${iface}\" 2>&1 || true; ethtool \"\${iface}\" 2>&1 | grep -E 'Speed:|Duplex:|Port:' || true; else echo 'ethtool missing'; fi; echo; done; else echo '/sys/class/net not found'; fi"
capture_shell "infiniband-sysfs" "if [[ -d /sys/class/infiniband ]]; then ls -R /sys/class/infiniband; else echo '/sys/class/infiniband not present'; fi"
capture_shell "env-allowlist" "for var in CUDA_HOME CUDA_PATH CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_NET_PLUGIN UCX_TLS UCX_NET_DEVICES UCX_IB_GPU_DIRECT_RDMA CONDA_DEFAULT_ENV CONDA_PREFIX VIRTUAL_ENV PYTHONPATH; do if [[ -n \${!var-} ]]; then printf '%s=%s\n' \"\${var}\" \"\${!var}\"; else printf '%s=<unset>\n' \"\${var}\"; fi; done"

captured_count="$(awk 'NR > 1 && $2 == "captured" { count += 1 } END { print count + 0 }' "${MANIFEST_FILE}")"
missing_count="$(awk 'NR > 1 && $2 == "missing" { count += 1 } END { print count + 0 }' "${MANIFEST_FILE}")"
failed_count="$(awk 'NR > 1 && $2 == "failed" { count += 1 } END { print count + 0 }' "${MANIFEST_FILE}")"

cat > "${SUMMARY_FILE}" <<EOF
# System Snapshot

- Host label: ${HOST_LABEL}
- Created (UTC): ${TIMESTAMP_UTC}
- Snapshot directory: ${SNAPSHOT_DIR}
- Captured commands: ${captured_count}
- Missing optional commands: ${missing_count}
- Failed commands: ${failed_count}

## Review before publishing

Review the raw files in this directory before committing or pushing them to a
public repository. This collector avoids full environment dumps and IP address
collection, but hardware topology, host labels, package versions, and driver
details may still be sensitive in your environment.

## Files

- \`manifest.tsv\`: command inventory with status, exit code, and path
- \`warnings.txt\`: missing tools and command failures
- \`raw/\`: captured command outputs
EOF

printf '%s\n' "${SNAPSHOT_DIR}"

