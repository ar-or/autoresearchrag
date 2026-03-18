Runner Instructions

Purpose
- You are the runner/orchestrator.
- Your job is to walk through `hypothesis_backlog.md` one hypothesis at a time, in order, and spawn a fresh sub-agent for each hypothesis.
- The runner does not do research, implementation, evaluation, or judging itself. The runner delegates that full loop to the sub-agent.
- Keep executor and judge together inside the same sub-agent.

Primary Loop
1. Read `hypothesis_backlog.md`.
2. Walk through the hypothesis cards from top to bottom, one by one.
3. For each hypothesis, check `results.csv` to see whether that exact hypothesis card was already tested previously.
4. Use the backlog card's stable `id` as the primary key for that check. Treat a hypothesis as already tested if any existing `results.csv` row has that same `hypothesis id=<card id>` recorded in `notes`.
5. If the hypothesis was already tested previously, skip it and move to the next one. Do not spawn a duplicate sub-agent run for the same hypothesis id.
6. If the hypothesis has not been evaluated yet, follow the execution mission instructions on implementation.
8. After the sub-agent finishes, refresh your view of:
   - `develop`
   - `results.csv`
   - `best_result.txt`
9. Then continue to the next hypothesis in `hypothesis_backlog.md`.


Executioner Mission
- switch to a new branch (see branch setup)
- Execute exactly one hypothesis.
- Read the linked paper/source for that hypothesis card before implementing it.
- Google/web-search how the technique is typically implemented before choosing the minimal repo-specific change.
- Implement the hypothesis.
- Evaluate it on HotpotQA with `N=30`.
- Judge it against the current champion in `best_result.txt`.
- Record the result in `results.csv`.
- If the hypothesis wins, update `best_result.txt` and merge the code into `develop`.
- If it loses, do not merge the code into `develop`.


Research Rules
- Before writing code, read the paper or source link referenced in the hypothesis card's implementation sketch. If the card has multiple relevant source links, read the ones needed to understand the specific technique being tested.
- Before writing code, google/web-search the technique name plus implementation terms such as `implementation`, `code`, `repo`, `pseudocode`, `retrieval`, `chunking`, or other technique-specific keywords as needed.
- Prefer primary sources first: the paper, official code repository, official project docs, or author-provided implementation notes. Use secondary sources only to clarify practical implementation details.
- Do not implement from memory alone when the hypothesis cites an external technique.
- If the paper/source cannot be accessed or the implementation remains too unclear after research, stop and record the run as rejected or blocked rather than inventing an ungrounded implementation.

Scope and Constraints
- One hypothesis per run. Do not bundle multiple unrelated ideas into one experiment.
- Default editable files are `agent.py` and `scripts/ingest.py`.
- Do not change evaluator scoring logic.
- Do not change the embedding model.
- Do not remove, bypass, or replace the sqlite embedding cache.
- Prefer deterministic retrieval/indexing changes over adding extra LLM calls, unless the hypothesis explicitly requires the extra call.

Branch and Index Setup
1. Start from the latest `develop`.
2. Create a branch named `feature_<hypothesis_slug>`.
3. Read the current champion from `best_result.txt` and treat its `elastic_index` value as the baseline Elasticsearch index for this experiment. If `best_result.txt` is uninitialized or lacks `elastic_index`, fall back to `mtrag`.
4. Set environment variables:
   - `N=30`
5. Keep the hypothesis slug short, lowercase, and stable so the branch name and index name match.

Implementation Rules
0. Before coding, complete the required research pass: read the cited paper/source and google/web-search practical implementation approaches for the technique.
1. Implement the minimum code change needed to test the hypothesis.
2. Keep the change attributable. If you catch yourself changing chunking, retrieval, and prompt behavior together, stop and split the work into separate hypotheses.
3. If the hypothesis requires scope outside the allowed files, reject it as out of scope instead of widening scope ad hoc.
4. Do not modify `results.csv` rows that already exist.

Ingestion Rules
1. If ingestion, chunking, embedding, or document serialization changed in any way:
   - Clone the current champion Elasticsearch index from `best_result.txt` into `feature_<hypothesis_slug>`. If no champion index is recorded yet, clone `mtrag`.
   - Set environment variable `ES_INDEX=feature_<hypothesis_slug>`
   - delete all documents in the newly created `ES_INDEX` where `collection_name=mycollection1`
   - reingest HotpotQA with `uv run python evaluators/hotpotqa/ingest_hotpotqa.py`
2. Unless ingestion, chunking, embedding, or document serialization changed, do not create a new Elasticsearch index. Instead, set `ES_INDEX` to the champion's `elastic_index` from `best_result.txt` and operate on the documents that already exist there.
3. If chunk count explodes or collapses unexpectedly after an ingestion change, stop and record the run as rejected with the reason in `notes`.

Evaluation Protocol
1. Run exactly one HotpotQA evaluation with `N=30`.
2. Do not run MT-RAG or any other evaluator unless the planner explicitly asked for it.
3. Do not save any output artifacts to disk. Only record the HotpotQA metrics in `results.csv`.

Recording Rules
1. Append one row to `results.csv` for every evaluated hypothesis, including rejected runs.
2. Always record:
   - `hypothesis_name`
   - `git_commit`
   - `time`
   - `elastic_index`
   - `retrieval_k`
   - `agent_mode`
   - `hotpot_n`
   - `hotpot_parallelism`
   - `hotpot_ok`
   - `hotpot_answer_em`
   - `hotpot_answer_f1`
   - `hotpot_sp_em`
   - `hotpot_sp_f1`
   - `hotpot_joint_em`
   - `hotpot_joint_f1`
   - `hotpot_wall_clock_s`
   - token and cost columns
   - `hotpot_artifact_path` (leave blank — artifacts are no longer saved)
   - `notes`
3. Leave benchmark columns blank only if the benchmark could not run (e.g. a failure before reaching it).
4. In `notes`, include:
   - hypothesis id
   - changed files
   - `reingest=yes` or `reingest=no`
   - one-sentence summary of what changed
   - final judge decision

Judge Rules
1. Compare the experiment to the current champion in `best_result.txt` on HotpotQA answer_f1 (primary metric) and joint_f1 (tiebreaker).
2. Possible decisions are:
   - `rejected`
   - `promoted`
3. Reject the experiment if any of the following is true:
   - the benchmark run failed
   - the change cannot be attributed to a single hypothesis
   - cost or latency increased materially without meaningful accuracy gain
4. Promotion rule:
   - If `challenger_hotpot_answer_f1 > champion_hotpot_answer_f1`: **promote**.
   - If `challenger_hotpot_answer_f1 < champion_hotpot_answer_f1`: **reject**.
   - If `challenger_hotpot_answer_f1 == champion_hotpot_answer_f1` (tie): promote only if `challenger_hotpot_joint_f1 > champion_hotpot_joint_f1`. Otherwise **reject**.
5. Additionally, promote only if the win is explainable:
   - the result is explainable from the hypothesis
   - the cost increase is acceptable for the gain

Merge Rules
1. Merge the new `results.csv` row back into `develop` for every completed experiment.
2. If the experiment is rejected, do not merge code into `develop`.
3. If the experiment is promoted:
   - update `best_result.txt`
   - record the promoted run's `elastic_index` in `best_result.txt` so the next experiment inherits the winning index as its default `ES_INDEX`
   - merge the experiment branch into `develop`
   - use the promoted code as the new baseline for the next experiment

Champion File
- `best_result.txt` is the single source of truth for the current champion.
- `best_result.txt` must include the champion's `elastic_index`, and that value is the default `ES_INDEX` baseline for the next experiment.
- `best_result.txt` must always include the champion's HotpotQA metrics.
- If `best_result.txt` is uninitialized, the first successful `N=30` HotpotQA run accepted on `develop` becomes the initial champion.

Cleanup
1. After recording results, clean up rejected branches and if created a new es index - also delete it.
2. Do not save artifacts to disk.
3. If `develop` changed materially during the experiment, restart the hypothesis from the updated `develop`.

Runner Completion Rule
- The runner is done only after it has walked through the full backlog it was asked to process.

Per-Hypothesis Completion Rule
- A sub-agent is done only when all of the following are true:
  - exactly one hypothesis was executed
  - code changes are committed on the experiment branch
  - HotpotQA evaluation completed or failed cleanly
  - a row was appended to `results.csv`
  - a judge decision was made
  - `best_result.txt` was updated only if the hypothesis was promoted
