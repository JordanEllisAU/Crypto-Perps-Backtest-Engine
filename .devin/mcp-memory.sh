#!/usr/bin/env bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MEMORY_FILE_PATH="$REPO_ROOT/runtime/memory/session.jsonl"
mkdir -p "$(dirname "$MEMORY_FILE_PATH")"
exec npx -y @modelcontextprotocol/server-memory
