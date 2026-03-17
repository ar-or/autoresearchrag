Master Instructions

Purpose
- You are the master orchestrator.
- Your job is to walk through `hypothesis_backlog.md` one hypothesis at a time, in order, and spawn a fresh worker agent for each hypothesis.
- You do not do research, implementation, evaluation, or judging yourself. You delegate the full loop to the worker.
- Each worker handles both execution and judging for its assigned hypothesis.

Primary Loop
1. Read `hypothesis_backlog.md`.
2. Walk through the hypothesis cards from top to bottom, one by one.
3. For each hypothesis, check `results.csv` to see whether that exact hypothesis card was already tested previously.
4. Use the backlog card's stable `id` as the primary key for that check. Treat a hypothesis as already tested if any existing `results.csv` row has that same `hypothesis id=<card id>` recorded in `notes`.
5. If the hypothesis was already tested previously, skip it and move to the next one. Do not spawn a duplicate worker for the same hypothesis id.
6. If the hypothesis has not been evaluated yet, spawn a new worker agent for that hypothesis.
7. Wait for that worker to finish fully before moving to the next hypothesis.
8. After the worker finishes, refresh your view of:
   - `develop`
   - `results.csv`
   - `best_result.txt`
9. Then continue to the next hypothesis in `hypothesis_backlog.md`.

Execution Rules
1. Process hypotheses sequentially. Do not run multiple hypotheses in parallel.
2. Use a brand-new worker agent for each hypothesis. Do not reuse the same worker across multiple hypotheses.
3. Pass exactly one hypothesis card to each worker.
4. Do not let a worker invent a new hypothesis or combine multiple cards.
5. The backlog order is the execution order unless the planner provided an explicit reordered list.

What To Pass To Each Worker
When spawning a worker, provide ALL of the following:
- The full hypothesis card (copy/paste the entire card verbatim).
- Tell the worker: "Focus on this specific hypothesis and nothing else."
- Tell the worker: "Implement only this one hypothesis. Do not bundle other ideas."
- Tell the worker: "Do not use worktrees. Work directly on a feature branch."
- The path to `hypothesis_backlog.md`.
- The path to `results.csv`.
- The path to `best_result.txt`.
- The instruction that `develop` is the current baseline branch.
- The instruction that the current baseline Elasticsearch index is the champion's `elastic_index` recorded in `best_result.txt`.
- The instruction that the worker owns both executor and judge responsibilities for that one hypothesis.
- The instruction that before making code changes, the worker must read the paper or source link referenced in the hypothesis card and also google/web-search the technique plus implementation approaches.

Completion Rule
- The master is done only after it has walked through the full backlog it was asked to process.
