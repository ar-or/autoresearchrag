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

set -euo pipefail

CONTINUE_MSG="Proceed in evaluating all hypotheses until the last one is done. When all done, respond with 'all done'"
DONE_MARKER="all done"
MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
INITIAL_PROMPT="${1:-$CONTINUE_MSG}"
LAST_MSG="/tmp/codex_last_msg_$$.txt"
ERRLOG="/tmp/codex_loop_errors_$$.log"
ITERATION=0

cleanup() { rm -f "$LAST_MSG"; }
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
            elif [ "$item_type" = "command_execution" ]; then
                cmd=$(echo "$line" | jq -r '.item.command // empty' 2>/dev/null)
                output=$(echo "$line" | jq -r '.item.aggregated_output // empty' 2>/dev/null)
                exit_code=$(echo "$line" | jq -r '.item.exit_code // empty' 2>/dev/null)
                echo "  > $cmd"
                [ -n "$output" ] && echo "$output" | head -20
                [ "$exit_code" != "0" ] && [ "$exit_code" != "null" ] && echo "  [exit $exit_code]"
            fi
        elif [ "$type" = "item.started" ]; then
            item_type=$(echo "$line" | jq -r '.item.type // empty' 2>/dev/null)
            if [ "$item_type" = "command_execution" ]; then
                cmd=$(echo "$line" | jq -r '.item.command // empty' 2>/dev/null)
                echo "  [running] $cmd"
            fi
        fi
    done
}

run_codex() {
    "$@" 2>>"$ERRLOG" | stream_and_capture
}

info "stderr → $ERRLOG"
info "codex loop (max $MAX_ITERATIONS iterations)"
info "Iteration $((++ITERATION)) (initial)"

run_codex codex exec --json -s danger-full-access "$INITIAL_PROMPT"

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
    run_codex codex exec resume --last --json -o "$LAST_MSG" "$CONTINUE_MSG"
done
