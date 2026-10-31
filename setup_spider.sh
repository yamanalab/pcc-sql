#!/usr/bin/env bash
# Fetch Spider 1.0 (Yu et al. 2018, CC BY-SA 4.0) into spider_data/.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_DIR="${PROJECT_ROOT}/spider_data/spider_data"
GDRIVE_ID="1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"

[[ -f "${SPIDER_DIR}/tables.json" && -d "${SPIDER_DIR}/database" ]] && exit 0

command -v gdown >/dev/null 2>&1 || python3 -m pip install --user gdown

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "${TMP_DIR}/spider.zip"

mkdir -p "${PROJECT_ROOT}/spider_data"
unzip -q "${TMP_DIR}/spider.zip" \
    "spider_data/tables.json" \
    "spider_data/dev.json" \
    "spider_data/train_spider.json" \
    "spider_data/README.txt" \
    "spider_data/database/*" \
    -d "${PROJECT_ROOT}/spider_data"
