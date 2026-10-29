#!/usr/bin/env bash
# Starts llama-server with the local model, with the flags validated by the evals.
# It is the launcher for development and measurements; as a service, deploy.sh --model installs
# the daily-llm unit with the same flags.
#
#     LLAMA_MODEL=path/to/model.gguf ./ops/llama-server.sh [port [llama-server flags...]]
#
# The model is the GGUF fine-tuned with training/finetune-colab.ipynb (LLAMA_MODEL, required);
# llama-server comes from LLAMA_SERVER (default: the one on the PATH). The default port is 8081,
# localhost only. LLAMA_THREADS sets the CPU threads (default: all of them).
#
# --reasoning off is NOT enough: some turns still reason and the reasoning tokens use up the
# generation cap, leaving the answer empty. --reasoning-budget 0 zeroes it.
# A single slot (-np 1) and an 8192 context (LLAMA_CTX): the prefix cache only pays off with one
# request at a time. -cms 0 (no minimum spacing between context checkpoints) and 96-token
# batches are the flags the model was measured with.
set -euo pipefail

LLAMA="${LLAMA_SERVER:-llama-server}"
MODEL="${LLAMA_MODEL:?set LLAMA_MODEL to the model GGUF}"
CTX="${LLAMA_CTX:-8192}"
THREADS="${LLAMA_THREADS:-$(nproc)}"
PORT="${1:-8081}"
shift || true  # the other arguments go to llama-server

exec "$LLAMA" -m "$MODEL" -t "$THREADS" -c "$CTX" -np 1 -cms 0 -b 96 -ub 96 \
    --reasoning off --reasoning-budget 0 \
    --host 127.0.0.1 --port "$PORT" "$@"
