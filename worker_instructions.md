Worker Instructions

Purpose
- You are a worker agent responsible for executing exactly one hypothesis.
- You handle the full cycle: research, implementation, evaluation, and judging.
- You must focus solely on the specific hypothesis card provided to you by the master.
- Do not invent new hypotheses, combine multiple cards, or deviate from the assigned hypothesis.
- Do not use worktrees. Work directly on a feature branch.

Research Rules
- Before writing code, read the paper or source link referenced in the hypothesis card's implementation sketch. If the card has multiple relevant source links, read the ones needed to understand the specific technique being tested.
- Before writing code, google/web-search the technique name plus implementation terms such as `implementation`, `code`, `repo`, `pseudocode`, `retrieval`, `chunking`, or other technique-specific keywords as needed.
- Prefer primary sources first: the paper, official code repository, official project docs, or author-provided implementation notes. Use secondary sources only to clarify practical implementation details.
- Do not implement from memory alone when the hypothesis cites an external technique.
- If the paper/source cannot be accessed or the implementation remains too unclear after research, stop and record the run as rejected or blocked rather than inventing an ungrounded implementation.

Scope and Constraints
- One hypothesis per run. Do not bundle multiple unrelated ideas into one experiment.
- Default editable files are `agent.py` and `scripts/ingest.py`.
- If the hypothesis is specifically about FinQA document rendering, `evaluators/finqa/ingest_finqa.py` may also be changed.
- Do not change evaluator scoring logic.
- Do not change the embedding model.
- Do not remove, bypass, or replace the sqlite embedding cache.
- Prefer deterministic retrieval/indexing changes over adding extra LLM calls, unless the hypothesis explicitly requires the extra call.

Branch and Index Setup
1. Start from the latest `develop`.
2. Create a branch named `feature_<hypothesis_slug>`.
3. Do not use worktrees. Check out the branch directly.
4. Read the current champion from `best_result.txt` and treat its `elastic_index` value as the baseline Elasticsearch index for this experiment. If `best_result.txt` is uninitialized or lacks `elastic_index`, fall back to `mtrag`.
5. Set environment variables:
   - `N=30`
6. Keep the hypothesis slug short, lowercase, and stable so the branch name, index name, and artifact path all match.

Implementation Rules
0. Before coding, complete the required research pass: read the cited paper/source and google/web-search practical implementation approaches for the technique.
1. Implement the minimum code change needed to test the hypothesis.
2. Keep the change attributable. If you catch yourself changing chunking, retrieval, and prompt behavior together, stop and split the work into separate hypotheses.
3. If the hypothesis requires fine tuning or reinforcemnt learning - reject it. Otherwise try to implement it within the allowed files. Including creating new tools, new ingesting strategies, etc.
   If you think you need to reject a hypothesis because it's implementation out of scope, confirm with the user first.
4. Do not modify `results.csv` rows that already exist.

Ingestion Rules
1. If ingestion, chunking, embedding, or document serialization changed in any way:
   - Clone the current champion Elasticsearch index from `best_result.txt` into `feature_<hypothesis_slug>`. If no champion index is recorded yet, clone `mtrag`.
   - Set environment variable `ES_INDEX=feature_<hypothesis_slug>`
   - Delete all documents in the newly created `ES_INDEX` where `collection_name=mycollection1`
   - Reingest FinQA with `uv run python evaluators/finqa/ingest_finqa.py`
   - Reingest HotpotQA with `uv run python evaluators/hotpotqa/ingest_hotpotqa.py`
2. Unless ingestion, chunking, embedding, or document serialization changed, do not create a new Elasticsearch index. Instead, set `ES_INDEX` to the champion's `elastic_index` from `best_result.txt` and operate on the documents that already exist there.
3. If chunk count explodes or collapses unexpectedly after an ingestion change, stop and record the run as rejected with the reason in `notes`.

Evaluation Protocol
1. Run exactly one FinQA evaluation with `N=30`.
2. Run exactly one HotpotQA evaluation with `N=30`.
3. Both benchmarks are always required. Do not skip either one.
4. Do not run MT-RAG or any other evaluator unless the planner explicitly asked for it.
5. Save the FinQA output artifact under `artifacts/<hypothesis_slug>/` and record the artifact path in `results.csv`.
6. Save the HotpotQA output artifact under `artifacts/<hypothesis_slug>/` and record the HotpotQA metrics and artifact path in `results.csv`.

Recording Rules
1. Append one row to `results.csv` for every evaluated hypothesis, including rejected runs.
2. Always record:
   - `hypothesis_name`
   - `git_commit`
   - `time`
   - `elastic_index`
   - `retrieval_k`
   - `agent_mode`
   - `finqa_n`
   - `finqa_parallelism`
   - `finqa_ok`
   - `finqa_accuracy`
   - `finqa_correct`
   - `finqa_wall_clock_s`
   - token and cost columns
   - `finqa_artifact_path`
   - `notes`
3. Always also record HotpotQA results:
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
   - `hotpot_artifact_path`
4. Leave unrelated benchmark columns blank only if that benchmark could not run (e.g. a failure before reaching it).
5. In `notes`, include:
   - hypothesis id
   - changed files
   - `reingest=yes` or `reingest=no`
   - one-sentence summary of what changed
   - final judge decision

Judge Rules
1. Compare the experiment to the current champion in `best_result.txt` on both FinQA accuracy and HotpotQA answer_f1.
2. Possible decisions are:
   - `rejected`
   - `promoted`
3. Reject the experiment if any of the following is true:
   - either benchmark run failed
   - the change cannot be attributed to a single hypothesis
   - cost or latency increased materially without meaningful accuracy gain
4. Compute the net improvement across both benchmarks:
   - `finqa_delta = challenger_finqa_accuracy − champion_finqa_accuracy`
   - `hotpot_delta = challenger_hotpot_answer_f1 − champion_hotpot_answer_f1`
5. Promotion rule based on net improvement:
   - If both deltas are >= 0 and at least one is > 0: **promote**.
   - If both deltas are < 0: **reject**.
   - If the deltas have mixed signs: promote only if the magnitude of the improvement on one benchmark strictly exceeds the magnitude of the decline on the other (i.e. `abs(positive_delta) > abs(negative_delta)`). Otherwise **reject**.
   - If both deltas are exactly 0: **reject** (pure tie).
6. Additionally, promote only if the win is explainable:
   - the result is explainable from the hypothesis
   - the cost increase is acceptable for the gain

Merge Rules
1. Merge the new `results.csv` row back into `develop` for every completed experiment.
2. If the experiment is rejected, do not merge code into `develop`.
3. If the experiment is promoted:
   - Update `best_result.txt`
   - Record the promoted run's `elastic_index` in `best_result.txt` so the next experiment inherits the winning index as its default `ES_INDEX`
   - Merge the experiment branch into `develop`
   - Use the promoted code as the new baseline for the next experiment

Champion File
- `best_result.txt` is the single source of truth for the current champion.
- `best_result.txt` must include the champion's `elastic_index`, and that value is the default `ES_INDEX` baseline for the next experiment.
- `best_result.txt` must always include the champion's HotpotQA metrics (since both benchmarks are always run).
- If `best_result.txt` is uninitialized, the first successful `N=30` FinQA run accepted on `develop` becomes the initial champion.

Cleanup
1. After recording results, clean up rejected branches and if created a new ES index, also delete it.
2. Keep accepted artifacts long enough to audit the promotion decision.
3. If `develop` changed materially during the experiment, restart the hypothesis from the updated `develop`.

Completion Rule
- The worker is done only when all of the following are true:
  - exactly one hypothesis was executed
  - code changes are committed on the experiment branch
  - FinQA evaluation completed or failed cleanly
  - HotpotQA evaluation completed or failed cleanly
  - a row was appended to `results.csv`
  - a judge decision was made
  - `best_result.txt` was updated only if the hypothesis was promoted
