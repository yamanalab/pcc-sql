#!/usr/bin/env bash
# Fetch Spider-ACL (Klisura et al., 2026) and convert it into data_spider_acl/.
# Requires Spider 1.0 from setup_spider.sh for the SQLite DBs.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${PROJECT_ROOT}/data_spider_acl"
SPIDER_DB="${PROJECT_ROOT}/spider_data/spider_data/database"
REPO_URL="https://github.com/klisura-code/LLM-Access-Control-Datasets.git"
REPO_COMMIT="9179b2ec6d84df56803cbbeb2650cd86b7c1b56f"

[[ -f "${OUT_DIR}/dev.json" && -f "${OUT_DIR}/tables.json" ]] && exit 0
[[ -d "${SPIDER_DB}" ]] || { echo "Run setup_spider.sh first." >&2; exit 1; }

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

git clone -q "${REPO_URL}" "${TMP_DIR}/repo"
git -C "${TMP_DIR}/repo" checkout -q "${REPO_COMMIT}"

cd "${PROJECT_ROOT}"
python3 -m benchmark.spider_acl_adapter \
    --input-zip "${TMP_DIR}/repo/data/spider/spider-access-control.zip" \
    --output-dir "${OUT_DIR}"
ln -sfn "${SPIDER_DB}" "${OUT_DIR}/database"
