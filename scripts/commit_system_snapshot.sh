#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COLLECTOR="${SCRIPT_DIR}/collect_system_config.sh"
OUTPUT_ROOT="${1:-${REPO_ROOT}/artifacts/system}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: commit_system_snapshot.sh must be run on a Linux server." >&2
  exit 1
fi

if ! git -C "${REPO_ROOT}" rev-parse --show-toplevel > /dev/null 2>&1; then
  echo "ERROR: initialize this directory as a git repository before using the commit helper." >&2
  exit 1
fi

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "ERROR: repository must be clean before collecting and committing a snapshot." >&2
  echo "Commit or stash existing changes first, then re-run this helper." >&2
  exit 1
fi

SNAPSHOT_DIR="$("${COLLECTOR}" "${OUTPUT_ROOT}")"
RELATIVE_SNAPSHOT_DIR="${SNAPSHOT_DIR#"${REPO_ROOT}/"}"

if [[ "${RELATIVE_SNAPSHOT_DIR}" == "${SNAPSHOT_DIR}" ]]; then
  echo "ERROR: snapshot path is outside the repository root: ${SNAPSHOT_DIR}" >&2
  exit 1
fi

echo "Created snapshot: ${RELATIVE_SNAPSHOT_DIR}"
git -C "${REPO_ROOT}" add -- "${RELATIVE_SNAPSHOT_DIR}"

if git -C "${REPO_ROOT}" diff --cached --quiet -- "${RELATIVE_SNAPSHOT_DIR}"; then
  echo "ERROR: no staged snapshot changes were detected." >&2
  exit 1
fi

echo "Staged files:"
git -C "${REPO_ROOT}" diff --cached --name-status -- "${RELATIVE_SNAPSHOT_DIR}"

COMMIT_MESSAGE="Add system snapshot ${RELATIVE_SNAPSHOT_DIR##*/}"
git -C "${REPO_ROOT}" commit -m "${COMMIT_MESSAGE}"

echo "Created local commit: ${COMMIT_MESSAGE}"
echo "Review the commit before pushing to any remote."
