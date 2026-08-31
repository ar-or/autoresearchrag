#!/bin/bash
# codex_loop.sh — Run codex in a loop, auto-continuing until the agent says "all done"
#
# Uses `codex exec --json` to stream events in real-time.
# Uses `codex exec resume --last` to preserve full conversation context.
#
# Usage:
#   ./scripts/codex_loop.sh "Evaluate hypothesis H1 first, then proceed to the rest"
#   ./scripts/codex_loop.sh  # uses default prompt
#   MAX_ITERATIONS=100 ./scripts/codex_loop.sh "Start evaluating"

set -uo pipefail

CONTINUE_MSG="Proceed in evaluating all hypotheses until the last one is done. When all done, respond with 'all done'"
DONE_MARKER="all done"
MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
INITIAL_PROMPT="${1:-$CONTINUE_MSG}"
LAST_MSG="/tmp/codex_last_msg_$$.txt"
ERRLOG="/tmp/codex_loop_errors_$$.log"
ITERATION=0

cleanup() { rm -f "$LAST_MSG"; }
abort()  { echo ""; err "Interrupted"; exit 130; }
trap abort INT TERM
trap cleanup EXIT

# Control messages with background color
info()  { printf '\033[44;97m %s \033[0m\n' "$*"; }  # white on blue
ok()    { printf '\033[42;97m %s \033[0m\n' "$*"; }   # white on green
err()   { printf '\033[41;97m %s \033[0m\n' "$*"; }   # white on red

# Stream codex --json events in real-time, showing agent messages and commands.
# Captures the last agent message to LAST_MSG for the done check.
stream_and_capture() {
    > "$LAST_MSG"
    while IFS= read -r line; do
        type=$(echo "$line" | jq -r '.type // empty' 2>/dev/null)
        if [ "$type" = "item.completed" ]; then
            item_type=$(echo "$line" | jq -r '.item.type // empty' 2>/dev/null)
            if [ "$item_type" = "agent_message" ]; then
                text=$(echo "$line" | jq -r '.item.text // empty' 2>/dev/null)
                echo "$text"
                echo "$text" > "$LAST_MSG"
            fi
        fi
    done
}

run_codex() {
    local rc=0
    "$@" 2> >(tee -a "$ERRLOG" >&2) | stream_and_capture || rc=$?
    if [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then
        return "$rc"
    fi
    if [ "$rc" -ne 0 ]; then
        err "codex exited with code $rc"
        err "stderr tail:"
        tail -20 "$ERRLOG" >&2
    fi
    return "$rc"
}

run_or_exit() {
    local rc=0
    run_codex "$@" || rc=$?
    if [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then
        echo ""
        err "Interrupted"
        exit 130
    fi
    if [ "$rc" -ne 0 ]; then
        exit "$rc"
    fi
}

info "stderr → $ERRLOG"
info "codex loop (max $MAX_ITERATIONS iterations)"
info "Iteration $((++ITERATION)) (initial)"

run_or_exit codex exec --json -s danger-full-access "$INITIAL_PROMPT"

while true; do
    if [ -f "$LAST_MSG" ] && grep -qi "$DONE_MARKER" "$LAST_MSG"; then
        echo ""
        ok "Done after $ITERATION iteration(s)"
        break
    fi

    if [ "$ITERATION" -ge "$MAX_ITERATIONS" ]; then
        echo ""
        err "Hit max iterations ($MAX_ITERATIONS)"
        exit 1
    fi

    echo ""
    info "Iteration $((++ITERATION)) (resume)"
    run_or_exit codex exec resume --last --json --dangerously-bypass-approvals-and-sandbox -o "$LAST_MSG" "$CONTINUE_MSG"
done
