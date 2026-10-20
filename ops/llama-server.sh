#!/usr/bin/env bash
# Starts llama-server with the local model, with the flags validated by the evals.
# It is the launcher for development and measurements; as a service, deploy.sh --model installs
# the daily-llm unit with the same flags.
#
#     ./ops/llama-server.sh [port [llama-server flags...]]   (default 8081, localhost only)
#
# The paths come from LLAMA_SERVER, LLAMA_MODEL and LLAMA_DRAFT; the defaults are llama-server
# on the PATH and the E4B files that ./ops/get-model.sh downloads into models/.
# LLAMA_THREADS sets the CPU threads (default 4).
#
# --reasoning off is NOT enough: Gemma 4 E4B still reasons on some turns and the reasoning
# tokens use up the generation cap, leaving the answer empty. --reasoning-budget 0 zeroes it.
# A single slot (-np 1) and an 8192 context (LLAMA_CTX): the prompt is 4,000-7,000 tokens
# depending on the examples, and the prefix cache only pays off with one request at a time.
#
# MTP speculative decoding: if the drafter is present, +50% generation speed with the same
# answers. ggml-org publishes it in the same repository as the model
# (mtp-gemma-4-E4B-it-Q4_0.gguf). Disable it with LLAMA_DRAFT= (empty). The default draft
# length (3) is the best one measured: with 5 the turn gets slower.
set -euo pipefail

MODELS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
LLAMA="${LLAMA_SERVER:-llama-server}"
MODEL="${LLAMA_MODEL:-$MODELS/gemma-4-E4B-it-Q4_0.gguf}"
DRAFT="${LLAMA_DRAFT-$MODELS/mtp-gemma-4-E4B-it-Q4_0.gguf}"
CTX="${LLAMA_CTX:-8192}"
THREADS="${LLAMA_THREADS:-4}"
PORT="${1:-8081}"
shift || true  # the other arguments go to llama-server (for example --spec-draft-n-max 4)

SPECULATIVE=()
if [ -n "$DRAFT" ] && [ -f "$DRAFT" ]; then
    SPECULATIVE=(--spec-type draft-mtp --spec-draft-model "$DRAFT")
fi

exec "$LLAMA" -m "$MODEL" -t "$THREADS" -c "$CTX" -np 1 \
    --reasoning off --reasoning-budget 0 \
    --host 127.0.0.1 --port "$PORT" "${SPECULATIVE[@]}" "$@"
