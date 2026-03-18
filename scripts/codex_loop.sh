#!/bin/bash
# codex_loop.sh — Run codex in a loop, auto-continuing until the agent says "all done"
#
# Uses `codex resume --last` so every iteration has the full conversation context.
# Uses --no-alt-screen so output stays in your terminal scrollback.
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
ITERATION=0

# Find the latest session JSONL
latest_session_file() {
    find ~/.codex/sessions -name '*.jsonl' -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-
}

# Check if the session transcript contains the done marker
check_done() {
    local f="$1"
    [ -f "$f" ] || return 1
    tail -50 "$f" | grep -qi "$DONE_MARKER"
}

echo "=== codex loop (max $MAX_ITERATIONS iterations) ==="
echo "--- Iteration $((++ITERATION)) (initial) ---"

codex --no-alt-screen -s danger-full-access -a never "$INITIAL_PROMPT"

while true; do
    if check_done "$(latest_session_file)"; then
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
    codex resume --last --no-alt-screen -s danger-full-access -a never "$CONTINUE_MSG"
done
