#!/usr/bin/env bash
# Downloads a Gemma 4 model and its MTP drafter from Hugging Face and checks their SHA-256.
#
#     ./ops/get-model.sh [e4b|e2b] [folder]   (default: e4b, into models/)
#
# E4B (about 4.6 GB) is the more accurate model; E2B (about 2.9 GB) is about twice as fast and
# needs DAILY_LLM_PROMPT_STYLE=turns. The drafter (about 60 MB) speeds generation up by
# about half with the same answers. The files are the ones the evals were measured with: the
# revision is pinned and every file is checked. An interrupted download resumes.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-e4b}"
DEST="${2:-$REPO_DIR/models}"

case "$VARIANT" in
    e4b)
        REPO=ggml-org/gemma-4-E4B-it-GGUF
        REVISION=a69f018d7678a2b9f5e705f8903d6fb29b9b25dc
        MODEL=gemma-4-E4B-it-Q4_0.gguf
        MODEL_SHA=a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2
        DRAFT=mtp-gemma-4-E4B-it-Q4_0.gguf
        DRAFT_SHA=d19a11e4fd5019967d9296f77f4d3e139c8903909f70c3a9cf009c9af4b9b88d
        ;;
    e2b)
        REPO=ggml-org/gemma-4-E2B-it-GGUF
        REVISION=6938c58e698502037401a5f19f6905f2fb2f7dee
        MODEL=gemma-4-E2B-it-Q4_0.gguf
        MODEL_SHA=8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52
        DRAFT=mtp-gemma-4-E2B-it-Q4_0.gguf
        DRAFT_SHA=718d3a44057924d5840c65a0aeedf7929366adeeda92a2006955ccabaf645ef4
        ;;
    *)
        echo "Usage: $0 [e4b|e2b] [folder]" >&2
        exit 2
        ;;
esac

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}

fetch() {
    local name="$1" expected="$2" path="$DEST/$1"
    if [[ -f "$path" && "$(sha256 "$path")" == "$expected" ]]; then
        echo "==> $name already present"
        return
    fi
    echo "==> Downloading $name"
    curl --fail --location --continue-at - --output "$path.part" \
        "https://huggingface.co/$REPO/resolve/$REVISION/$name"
    if [[ "$(sha256 "$path.part")" != "$expected" ]]; then
        rm -f "$path.part"
        echo "    $name: wrong checksum, the file was deleted. Run the script again." >&2
        exit 1
    fi
    mv "$path.part" "$path"
}

mkdir -p "$DEST"
fetch "$MODEL" "$MODEL_SHA"
fetch "$DRAFT" "$DRAFT_SHA"

echo
echo "Done. Gemma 4 is released under the Apache 2.0 license."
echo "Start the model server with:"
echo "    LLAMA_MODEL=$DEST/$MODEL LLAMA_DRAFT=$DEST/$DRAFT ./ops/llama-server.sh"
if [[ "$VARIANT" == "e2b" ]]; then
    echo "and add DAILY_LLM_PROMPT_STYLE=turns to the bot's configuration."
fi
