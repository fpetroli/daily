# daily

A personal scheduling assistant on **Telegram**, driven by a **small language model running
on your own server**: no cloud AI, no schedule data leaving the machine. You write to your
bot in natural language ("move the gym to tomorrow at 6 pm"); a local LLM turns the message
into a structured tool call; deterministic code validates it, executes it and answers with a
fixed, self-contained message.

Scheduling is the use case chosen to prove the engine. The interesting parts are the
engine itself, the evaluation harness and the operations recipe:

- **The LLM is a parser, not an actor.** The model only produces one JSON tool call per turn,
  constrained by a GBNF grammar generated from the tool schema: it cannot invent a tool or a
  field, and the code never reinterprets a value. It never writes text for the user and never
  writes data directly.
- **Everything else is exact code**: placement by rules ("the shower after the gym"),
  conflict detection, cascades on derived events (always proposed, never automatic), habit
  detection with thresholds, one-level undo.
- **Measured, not guessed.** An eval harness checks the *final state* of the schedule over a
  committed set of cases (dev and held-out splits, multi-turn conversations, temporal
  expressions), so two different but equivalent tool calls both count as right.
- **Self-hosted on a CPU-only VM.** llama.cpp with Qwen3.5-0.8B fine-tuned on examples
  labelled by construction (Q4_0, about 500 MB), prompt-cache-aware prompt layout, a hardened
  systemd unit. The bot polls Telegram: nothing listens on a public port, so it also runs at
  home behind a router.

## How a turn works

```
Telegram  ◀──long polling──  bot  ──▶  llama-server (local model)
                              │
                              ├─ 1. parse:   message + recent history → one tool call (grammar-constrained JSON)
                              ├─ 2. execute: tool call → services (rules, conflicts, cascades, habits) → typed outcome
                              └─ 3. respond: outcome → fixed template (+ quick replies as buttons)
                              │
                              └─ SQLite (schedule, rules, usual times, habit patterns, turn log)
```

- The three steps are an [Agno](https://github.com/agno-agi/agno) workflow; no exception
  crosses a step, and a model that is down only degrades the chat: the commands (`/today`,
  `/tomorrow`, `/rules`, `/undo`, `/settings`) never go through the model.
- The prompt is laid out for llama.cpp's prompt cache: fixed instructions and tools first,
  then a 15-day calendar the model *looks up* instead of computing dates, then the current
  time, the last 3 exchanges and the message.
- Every write gets an **Undo** button (one level, never through the model), and questions get
  their answers as buttons ("Yes"/"No", "Before"/"After", which event).
- Every turn is logged (`turn_log`) and `/report <what went wrong>` flags the last answer:
  flagged turns are the raw material for new eval cases.

## Results so far

With **Qwen3.5-0.8B** fine-tuned on generated examples (Q4_0) on a 2-core Arm VM (Ampere,
no GPU):

| Eval set | Correct final state |
|---|---|
| dev (180 cases) | 136 (75.6%) |
| holdout (36 cases) | 29 (80.6%) |

Unwanted writes (something extra in the schedule or among the rules): 32 of 216 cases.
Latency per turn with a warm cache and a changing clock, as in production: p50 2.0 s, p95
3.3 s. The model server uses about 1.3 GB of RAM.

The target is 90% on dev, 85% on holdout and at most 3% unwanted writes: the turn log and
`/report` collect the real sentences the model gets wrong, which become new training
examples and eval cases.

## Repository layout

| Path | What it is |
|---|---|
| `src/daily/` | The Python 3.12 package: Telegram bot, Agno workflow, services, SQLite storage, eval harness (see [Development](#development)) |
| `tests/` | The tests, one folder per package |
| `evals/cases/` | The committed eval cases (YAML, one file per section) |
| `ops/` | Deploy script, systemd unit and model server launcher |
| `training/` | The Colab notebook that fine-tunes the model |

## Quick start

You need a Linux or macOS machine with Python 3.12, `git`, `cmake` and a C++ compiler, about
2 GB of free disk, 2 GB of RAM, a Google account (to train the model on Colab) and a Telegram
account. All commands run from the repository root unless said otherwise.

1. **Build the model server**, [llama.cpp](https://github.com/ggml-org/llama.cpp):
   ```
   git clone https://github.com/ggml-org/llama.cpp ../llama.cpp
   cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release
   cmake --build ../llama.cpp/build -j --target llama-server
   ```
2. **Train the model**: open `training/finetune-colab.ipynb` on Google Colab (a free T4 is
   enough, see [Training the model](#training-the-model)) and download the GGUF it saves to
   your Google Drive, for example into `models/`.
3. **Start the model server** and leave it running (it listens on localhost only):
   ```
   LLAMA_SERVER=../llama.cpp/build/bin/llama-server LLAMA_MODEL=models/your-model.gguf \
       ./ops/llama-server.sh
   ```
4. **Create your bot** and get its token: see [Setting up the bot](#setting-up-the-bot).
5. **Install the bot** in another terminal:
   ```
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.lock && .venv/bin/pip install -e .
   ```
   and write its configuration in `.env` (the allowed users stay empty for now):
   ```
   DAILY_TELEGRAM_BOT_TOKEN=123456789:your-token
   DAILY_TELEGRAM_ALLOWED_USERS=
   DAILY_TIMEZONE=Europe/Rome
   DAILY_DATA_DIR=data
   ```
6. **Start the bot**:
   ```
   .venv/bin/python -m daily
   ```
   It logs `Connected to Telegram as @your_bot` and starts preparing the model's prompt, until
   `The model is ready.` Messages sent meanwhile are answered when it is done.
7. **Let yourself in** with your Telegram user id (see [Setting up the bot](#setting-up-the-bot)),
   then write "gym tomorrow at 6 pm for an hour". `/help` lists the commands.

You can also chat from the terminal, without Telegram:
`.venv/bin/python -m daily.cli chat --llm-url http://127.0.0.1:8081/v1`.

No GPU or strong machine? `DAILY_LLM_BASE_URL` can point to any OpenAI-compatible
`llama-server` endpoint, for example one running on another computer. The bot relies on
llama.cpp's `grammar` and `cache_prompt` request fields, so other OpenAI-compatible servers
will not work as they are.

## Setting up the bot

The bot needs two things from Telegram: **its token**, which lets the program act as the
bot, and **your user id**, which tells the bot who it may talk to.

**The token.** In Telegram, open [@BotFather](https://t.me/BotFather) and send `/newbot`. It
asks for a display name (for example "daily") and a username that ends in `bot` (for
example `my_daily_bot`), then answers with a token like `123456789:AAH…`. Anyone with the
token controls the bot, so treat it as a password: it goes only in the configuration file,
never in the repository or in a chat. If it leaks, send `/revoke` to @BotFather, pick the bot
and put the new token in the configuration: the bot keeps its name and its chats.

**Your user id.** It is a number, not your @username. The bot tells you: start it with
`DAILY_TELEGRAM_ALLOWED_USERS` empty, open `t.me/<your bot's username>` and send anything.
The bot is private, so it only answers
`This bot is private. Your Telegram user id is 123456789: add it to DAILY_TELEGRAM_ALLOWED_USERS to use it.`
Put that number in the configuration and restart the bot. To let other people in, add their
ids separated by commas (`123456789,987654321`): they all share the same schedule. The bot
ignores group chats.

**Where the settings go.** The same `DAILY_` settings live in one of two files, depending on
how the bot runs:

| How it runs | File | Apply a change |
|---|---|---|
| By hand, `python -m daily` from the repository root | `.env` (ignored by git) | Ctrl-C and start it again |
| As a service, after `deploy.sh` | `/etc/daily/daily.env` (readable only by root and the service): `sudoedit /etc/daily/daily.env` | `sudo systemctl restart daily` |

| Setting | Required | Meaning |
|---|---|---|
| `DAILY_TELEGRAM_BOT_TOKEN` | yes | The token from @BotFather |
| `DAILY_TELEGRAM_ALLOWED_USERS` | yes, to be served | User ids allowed to use the bot, comma-separated |
| `DAILY_TIMEZONE` | no | Your time zone, for example `Europe/Rome` (default: the server's). Turns are dated in it |
| `DAILY_DATA_DIR` | no | Folder of the SQLite database (default `data`) |
| `DAILY_LLM_BASE_URL` | no | The model server (default `http://127.0.0.1:8081/v1`) |
| `DAILY_LLM_TIMEOUT_SECONDS` | no | Longest wait for the model (default 300 s: it must cover the cold prompt) |
| `DAILY_LLM_MAX_TOKENS` | no | Cap on the tokens generated for one tool call (default 160) |

**If something is off**, the log says so (`journalctl -u daily -f` for the service):

- `Telegram refused the bot token`: the token is wrong or was revoked.
- `Telegram refused the polling (getUpdates: 409 Conflict…)`: another program is using the
  same token, for example a second copy of the bot. Only one may run per token.
- `Prompt warm-up failed: the model does not answer`, or the bot replies "The model is not
  answering": the model server is not running or `DAILY_LLM_BASE_URL` points elsewhere.

## Deployment

To keep it running on a Linux server with systemd, build llama.cpp (step 1 of the quick
start) and, from the repository root:

```
sudo ./ops/deploy.sh --model models/your-model.gguf
sudoedit /etc/daily/daily.env                         # token, user ids, time zone
sudo systemctl start daily
```

`deploy.sh` sets up two services, both started at boot and listening on no public port:

- **`daily-llm`**, the model server: it copies `llama-server` and its libraries from your
  llama.cpp build (`../llama.cpp/build/bin` next to the repository, or the one on the PATH;
  `--llama-server PATH` to choose) and the model into `/opt/daily-llm`, and runs it as its
  own `daily-llm` user on `127.0.0.1:8081`.
- **`daily`**, the bot: dedicated `daily` user, read-only code in `/opt/daily`, private data
  folder in `/var/lib/daily`. It creates `/etc/daily/daily.env` on the first run (later runs
  leave your settings alone) and starts the bot only once the token is there.

Run the same command again after pulling a new version. Without `--model` only the bot is
installed or updated, for example when the model server runs elsewhere
(`DAILY_LLM_BASE_URL`). The logs are `journalctl -u daily -f` and `journalctl -u daily-llm -f`.

## Development

| Package | Role |
|---|---|
| `daily/telegram` | The bot: long polling, allowlist, inline buttons, commands |
| `daily/agent` | The turn workflow, tool call schema, prompt, adapters to the services, reply templates |
| `daily/llm` | The `llama-server` client and the GBNF grammar derived from the tool call schema |
| `daily/services` | Exact schedule operations with typed outcomes, called by the model's tools; one-level undo |
| `daily/domain` | Pure domain logic: models, time bands, ISO time arithmetic |
| `daily/storage` | SQLite connection, numbered migrations applied on startup, one repository per table |
| `daily/evals` | Eval runner, report and latency measurement |
| `daily/datagen` | Training examples labelled by construction, for fine-tuning |

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/ruff check .
```

The tests use temporary folders and in-memory databases, and never call a real model or
Telegram. To try the whole turn from the terminal, without Telegram, with a hand-written JSON
tool call or with the real model:

```
.venv/bin/python -m daily.cli chat
.venv/bin/python -m daily.cli chat --llm-url http://127.0.0.1:8081/v1
```

Use a separate bot (another token) for development if one already runs as a service: only one
program may poll each token.

### Evals

The cases live in `evals/cases/`, one YAML file per section, with `dev` and `holdout` splits.
A case is a starting database, one or more messages and the expected final state: the eval
checks what is left in the schedule, not the exact tool call.

```
.venv/bin/python -m daily.evals run --parser oracle --split dev
```

The oracle parser returns the expected tool call of every turn: it must score 100%, which
proves the runner and the cases consistent. With the real model (start
`./ops/llama-server.sh` first; the command warms the prompt up before measuring):

```
.venv/bin/python -m daily.evals run --parser llama --split dev --llm-timeout 300 --llm-max-tokens 200
```

Latency of a turn as in production (a different time on every request; the eval uses a fixed
time and underestimates it):

```
.venv/bin/python -m daily.evals latency --llm-timeout 300
```

The `holdout` split is never used to tune the prompt.

### Training the model

`daily.datagen` writes training examples labelled by construction: it picks the values first
(title, day, time, duration), says them in one of many ways and builds the expected tool call
from the same values. Each line is a chat in the prompt shape the bot sends (instructions and
calendar, no examples) with the tool call as the answer. Messages
equal to an eval case are dropped and the titles share nothing with the cases.

`--dialogues` adds conversations (an answer to the bot's question, a yes or no to a
proposal, a pick among several events): they run through the real agent with the planned
calls, so the history the model sees holds the bot's real replies, and a conversation whose
replies are not the expected ones is dropped.

```
.venv/bin/python -m daily.datagen --count 500 --dialogues 250 --out data/datagen/train.jsonl
```

`training/finetune-colab.ipynb` trains Qwen3.5-0.8B on these examples on Google Colab (LoRA,
a T4 is enough): it generates them from this repository, trains, and saves a `Q4_0` GGUF to
Google Drive. Changing the prompt shape (`daily/agent/prompts.py`) means training the model
again.

## Security model

The project is built for **a single user** on their own server.

- The bot only serves the Telegram user ids in `DAILY_TELEGRAM_ALLOWED_USERS`, and only in
  private chats; anybody else just gets their own id back. Several allowed ids share the same
  schedule.
- Nothing listens on the network: the bot polls Telegram over HTTPS and calls the model on
  localhost. Messages and replies do travel through Telegram's servers, like any Telegram
  chat; the schedule itself, the model and the logs stay on your machine.
- The model has no network access and no direct access to the data: it only proposes a tool
  call that the code validates.
- The bot token is a secret: it lives in the configuration file (readable only by root and
  the service) and is never logged.

If you find a security problem, please open an issue without including exploit details, and
ask for a private channel.

## Limitations

- Single user, English only, Telegram only.
- The bot writes first only around an event (a reminder before it, the day's schedule and a
  countdown when it starts); no sync with other calendars.
- The eval cases are written by hand: real sentences are messier, and the turn log exists to
  catch the gap.

## License

[MIT](LICENSE). The Qwen3.5 base model is distributed separately under its own license
(Apache 2.0).
