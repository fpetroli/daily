#!/usr/bin/env bash
# Installs or updates daily on a Linux server. Run it with sudo, as a regular user, from the
# repository root:
#
#     sudo ./ops/deploy.sh                    # the bot only
#     sudo ./ops/deploy.sh --model e4b        # the bot and the model server (or e2b)
#
# With --model it also installs the model server as the daily-llm service: it copies
# llama-server from a llama.cpp build (by default ../llama.cpp/build/bin next to the
# repository, or the one on the PATH; --llama-server PATH to choose), downloads and checks the
# model and sets the bot's prompt style for it.
#
# It is idempotent: the first run creates users, environment and configuration; later runs
# update the code and restart the services. The bot starts once its token is in the
# configuration.
set -euo pipefail

usage() {
    echo "Usage: sudo $0 [--model e4b|e2b] [--llama-server PATH]" >&2
    exit 2
}

if [[ $EUID -ne 0 ]]; then
    echo "Root privileges are needed: sudo $0 $*" >&2
    exit 1
fi
BUILDER="${SUDO_USER:-}"
if [[ -z "$BUILDER" || "$BUILDER" == "root" ]]; then
    echo "Run it with sudo from a regular user: the package is built without root." >&2
    exit 1
fi

MODEL=""
LLAMA_SERVER_BIN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 || usage ;;
        --llama-server) LLAMA_SERVER_BIN="${2:-}"; shift 2 || usage ;;
        *) usage ;;
    esac
done
[[ -z "$MODEL" || "$MODEL" == e4b || "$MODEL" == e2b ]] || usage

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE=daily.service
SERVICE_USER=daily
APP_DIR=/opt/daily
ENV_DIR=/etc/daily
ENV_FILE="$ENV_DIR/daily.env"
DATA_DIR=/var/lib/daily
LLM_SERVICE=daily-llm.service
LLM_USER=daily-llm
LLM_DIR=/opt/daily-llm
LLM_PORT=8081

if [[ -n "$MODEL" ]]; then
    # Found before anything is installed: a missing llama-server stops the script early.
    if [[ -z "$LLAMA_SERVER_BIN" ]]; then
        for candidate in "$REPO_DIR/../llama.cpp/build/bin/llama-server" \
            "$(sudo -u "$BUILDER" bash -lc 'command -v llama-server' 2>/dev/null || true)"; do
            if [[ -n "$candidate" && -x "$candidate" ]]; then
                LLAMA_SERVER_BIN="$candidate"
                break
            fi
        done
    fi
    if [[ -z "$LLAMA_SERVER_BIN" || ! -x "$LLAMA_SERVER_BIN" ]]; then
        echo "llama-server not found: build llama.cpp (see the README) or pass" >&2
        echo "--llama-server /path/to/llama.cpp/build/bin/llama-server" >&2
        exit 1
    fi
fi

# Sets KEY=VALUE in the configuration, replacing the line if the key is already there.
set_env() {
    if grep -q "^$1=" "$ENV_FILE"; then
        sed -i "s|^$1=.*|$1=$2|" "$ENV_FILE"
    else
        echo "$1=$2" >> "$ENV_FILE"
    fi
}

echo "==> Service user"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    useradd --system --user-group --no-create-home --home-dir "$DATA_DIR" \
        --shell /sbin/nologin "$SERVICE_USER"
fi

echo "==> Package (built as $BUILDER)"
if [[ ! -x "$REPO_DIR/.venv/bin/pip" ]]; then
    sudo -u "$BUILDER" python3.12 -m venv "$REPO_DIR/.venv"
fi
WHEEL_DIR="$(mktemp -d)"
trap 'rm -rf "$WHEEL_DIR"' EXIT
chown "$BUILDER" "$WHEEL_DIR"
sudo -u "$BUILDER" "$REPO_DIR/.venv/bin/pip" wheel --quiet --no-deps \
    --wheel-dir "$WHEEL_DIR" "$REPO_DIR"

echo "==> Virtual environment in $APP_DIR (owned by root: the service cannot change its own code)"
mkdir -p "$APP_DIR"
[[ -x "$APP_DIR/venv/bin/python" ]] || python3.12 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --quiet --requirement "$REPO_DIR/requirements.lock"
"$APP_DIR/venv/bin/pip" install --quiet --no-deps --force-reinstall "$WHEEL_DIR"/*.whl

echo "==> Configuration in $ENV_DIR"
install -d -m 750 -o root -g "$SERVICE_USER" "$ENV_DIR"
if [[ ! -f "$ENV_FILE" ]]; then
    (
        umask 077
        cat > "$ENV_FILE" <<ENV
DAILY_DATA_DIR=$DATA_DIR
AGNO_TELEMETRY=false
# Token of the bot, from @BotFather.
DAILY_TELEGRAM_BOT_TOKEN=
# Telegram user ids allowed to use the bot, comma-separated (the bot tells you yours).
DAILY_TELEGRAM_ALLOWED_USERS=
# The user's time zone, for example Europe/Rome (default: the server's).
DAILY_TIMEZONE=$(timedatectl show --property=Timezone --value 2>/dev/null || true)
ENV
    )
    chown root:"$SERVICE_USER" "$ENV_FILE"
    chmod 640 "$ENV_FILE"
    echo "    created $ENV_FILE"
else
    echo "    $ENV_FILE already exists: left untouched"
fi

if [[ -n "$MODEL" ]]; then
    VARIANT="${MODEL^^}"
    MODEL_FILE="$LLM_DIR/models/gemma-4-$VARIANT-it-Q4_0.gguf"
    DRAFT_FILE="$LLM_DIR/models/mtp-gemma-4-$VARIANT-it-Q4_0.gguf"
    echo "==> Model server ($VARIANT) in $LLM_DIR, from $LLAMA_SERVER_BIN"
    if ! id -u "$LLM_USER" >/dev/null 2>&1; then
        useradd --system --user-group --no-create-home --home-dir "$LLM_DIR" \
            --shell /sbin/nologin "$LLM_USER"
    fi
    # The binary and its libraries are copied: the service never runs code from a home folder.
    install -d -m 755 "$LLM_DIR/bin" "$LLM_DIR/models"
    LLAMA_BUILD="$(dirname "$(readlink -f "$LLAMA_SERVER_BIN")")"
    cp -f "$LLAMA_BUILD/llama-server" "$LLAMA_BUILD"/*.so* "$LLM_DIR/bin/"
    "$REPO_DIR/ops/get-model.sh" "$MODEL" "$LLM_DIR/models" | grep '^==>' || true
    [[ -f "$MODEL_FILE" && -f "$DRAFT_FILE" ]] || { echo "    model download failed" >&2; exit 1; }
    chmod -R a+rX "$LLM_DIR"
    ! command -v restorecon >/dev/null || restorecon -R "$LLM_DIR"

    cat > "/etc/systemd/system/$LLM_SERVICE" <<UNIT
[Unit]
Description=daily - local model (llama-server, Gemma 4 $VARIANT)
Documentation=https://github.com/fpetroli/daily
After=network.target

[Service]
Type=simple
User=$LLM_USER
Group=$LLM_USER
# The flags validated by the evals: see ops/llama-server.sh.
Environment=LD_LIBRARY_PATH=$LLM_DIR/bin
ExecStart=$LLM_DIR/bin/llama-server -m $MODEL_FILE \\
    --spec-type draft-mtp --spec-draft-model $DRAFT_FILE \\
    -t $(nproc) -c 8192 -np 1 --reasoning off --reasoning-budget 0 \\
    --host 127.0.0.1 --port $LLM_PORT
Restart=on-failure
RestartSec=5

NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
PrivateDevices=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes
RestrictNamespaces=yes
LockPersonality=yes
CapabilityBoundingSet=
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX

[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload
    systemctl enable "$LLM_SERVICE" >/dev/null
    systemctl restart "$LLM_SERVICE"
    echo -n "    waiting for the model to load"
    for _ in $(seq 1 60); do
        if curl -fsS "http://127.0.0.1:$LLM_PORT/health" >/dev/null 2>&1; then
            break
        fi
        echo -n "."
        sleep 2
    done
    echo
    if ! curl -fsS "http://127.0.0.1:$LLM_PORT/health" >/dev/null 2>&1; then
        echo "    The model server does not answer: journalctl -u $LLM_SERVICE -n 50" >&2
        exit 1
    fi
    set_env DAILY_LLM_BASE_URL "http://127.0.0.1:$LLM_PORT/v1"
    set_env DAILY_LLM_PROMPT_STYLE "$([[ "$MODEL" == e2b ]] && echo turns || echo classic)"
    echo "    OK: $LLM_SERVICE answers; the bot's configuration points to it"
fi

echo "==> systemd service"
install -m 644 "$REPO_DIR/ops/$SERVICE" "/etc/systemd/system/$SERVICE"
systemctl daemon-reload
systemctl enable "$SERVICE" >/dev/null

if ! grep -q '^DAILY_TELEGRAM_BOT_TOKEN=..*' "$ENV_FILE"; then
    systemctl stop "$SERVICE" 2>/dev/null || true
    echo
    echo "Installed. Now write the bot token in the configuration and start the service:"
    echo "    sudoedit $ENV_FILE"
    echo "    sudo systemctl start $SERVICE"
    exit 0
fi
systemctl restart "$SERVICE"

echo "==> Check"
sleep 5
if systemctl is-active --quiet "$SERVICE"; then
    echo "    OK: $SERVICE is running. Write to the bot on Telegram."
    exit 0
fi
echo "    The service is not running: journalctl -u $SERVICE -n 50" >&2
exit 1
