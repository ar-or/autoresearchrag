#!/bin/bash
# codex_loop.sh — Run codex in a loop, auto-continuing until the agent says "all done"
#
# Uses `codex exec resume --last` so every iteration has the full conversation context.
# Uses `codex exec` (non-interactive) so it exits after each turn.
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

echo "stderr redirected to $ERRLOG"

echo "=== codex loop (max $MAX_ITERATIONS iterations) ==="
echo "--- Iteration $((++ITERATION)) (initial) ---"

codex exec -s danger-full-access -o "$LAST_MSG" "$INITIAL_PROMPT" 2>>"$ERRLOG"

while true; do
    if [ -f "$LAST_MSG" ] && grep -qi "$DONE_MARKER" "$LAST_MSG"; then
        echo ""
        echo "=== Done after $ITERATION iteration(s) ==="
        break
    fi

    if [ "$ITERATION" -ge "$MAX_ITERATIONS" ]; then
        echo ""
        echo "=== Hit max iterations ($MAX_ITERATIONS) ==="
        exit 1
    fi

    echo ""
    echo "--- Iteration $((++ITERATION)) (resume) ---"
    codex exec resume --last -o "$LAST_MSG" "$CONTINUE_MSG" 2>>"$ERRLOG"
done
