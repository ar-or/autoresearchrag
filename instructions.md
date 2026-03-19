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
   - `results.csv`
   - `best_result.txt`
9. Then continue to the next hypothesis in `hypothesis_backlog.md`.


Executioner Mission
- Create an experiment folder (see experiment setup).
- Execute exactly one hypothesis.
- Read the linked paper/source for that hypothesis card before implementing it.
- Google/web-search how the technique is typically implemented before choosing the minimal repo-specific change.
- Implement the hypothesis.
- Evaluate it on HotpotQA with `N=30`.
- Judge it against the current champion in `best_result.txt`.
- Record the result in `results.csv`.
- If the hypothesis wins, update `best_result.txt` and promote the code to root.
- If it loses, keep the experiment folder as a record but do not promote.


Research Rules
- Before writing code, read the paper or source link referenced in the hypothesis card's implementation sketch. If the card has multiple relevant source links, read the ones needed to understand the specific technique being tested.
- Before writing code, google/web-search the technique name plus implementation terms such as `implementation`, `code`, `repo`, `pseudocode`, `retrieval`, `chunking`, or other technique-specific keywords as needed.
- Prefer primary sources first: the paper, official code repository, official project docs, or author-provided implementation notes. Use secondary sources only to clarify practical implementation details.
- Do not implement from memory alone when the hypothesis cites an external technique.
- If the paper/source cannot be accessed or the implementation remains too unclear after research, stop and record the run as rejected or blocked rather than inventing an ungrounded implementation.

Scope and Constraints
- One hypothesis per run. Do not bundle multiple unrelated ideas into one experiment.
- Default editable files are `experiments/h<id>/agent.py` and optionally `experiments/h<id>/ingest.py`.
- Do not change evaluator scoring logic.
- Do not change the embedding model.
- Do not remove, bypass, or replace the sqlite embedding cache.
- Prefer deterministic retrieval/indexing changes over adding extra LLM calls, unless the hypothesis explicitly requires the extra call.

File Layout
- `agent_base.py` — shared types and session management. Exports: `RetrievedContext`, `ChatMessage`, `TokenUsage`, `SendMessageResult`, `Session`, `RetrievalTraceStep`, `create_session`, `get_session`, `_add_message`, `_sessions`.
- `agent.py` — the current champion agent. Imports all types from `agent_base`. This is the baseline that gets overwritten when an experiment is promoted.
- `experiments/TEMPLATE/` — starter files for new experiments. Contains `agent.py` (minimal agent that re-exports the champion) and `ingest.py` (skeleton for custom ingestion).
- `experiments/h<id>/` — one folder per experiment. Each must contain `__init__.py` and `agent.py`. May optionally contain `ingest.py`.
- `scripts/ingest.py` — shared ingestion pipeline (chunking, embedding, Elasticsearch indexing). Custom ingest scripts import from here.
- `evaluators/agent_client.py` — loads the agent module at runtime via `importlib.import_module(os.environ["AGENT_MODULE"])`. The loaded module must export `create_session() -> str` and `send_message(session_id, message) -> SendMessageResult`.

Experiment Setup
1. Stay on `develop`. All experiments run on `develop` — no feature branches.
2. Create an experiment folder:
   ```bash
   mkdir -p experiments/h<id>
   touch experiments/h<id>/__init__.py
   cp agent.py experiments/h<id>/agent.py
   ```
   Reference `experiments/TEMPLATE/` for a minimal starting point.
3. Read the current champion from `best_result.txt` and treat its `elastic_index` value as the baseline Elasticsearch index for this experiment. If `best_result.txt` is uninitialized or lacks `elastic_index`, fall back to `mtrag`.
4. Set environment variables:
   - `N=30`
   - `AGENT_MODULE=experiments.h<id>.agent` — the evaluator calls `importlib.import_module()` with this value, so it must be a valid Python dotted module path.
5. Keep the hypothesis id short, lowercase, and numeric (matching the backlog card id).

Implementation Rules
0. Before coding, complete the required research pass: read the cited paper/source and google/web-search practical implementation approaches for the technique.
1. Edit `experiments/h<id>/agent.py` with the minimum code change needed to test the hypothesis.
2. The experiment `agent.py` must export two functions: `create_session() -> str` and `send_message(session_id, message) -> SendMessageResult`. These are the only entry points the evaluator calls.
3. Import shared types from `agent_base` — do not duplicate dataclass definitions.
4. Keep the change attributable. If you catch yourself changing chunking, retrieval, and prompt behavior together, stop and split the work into separate hypotheses.
5. If the hypothesis requires scope outside the allowed files, reject it as out of scope instead of widening scope ad hoc.
6. Do not modify `results.csv` rows that already exist.

Ingestion Rules
1. If ingestion, chunking, embedding, or document serialization changed in any way:
   - Clone the current champion Elasticsearch index from `best_result.txt` into `h<id>`. If no champion index is recorded yet, clone `mtrag`.
   - Set environment variable `ES_INDEX=h<id>`
   - Delete all documents in the newly created `ES_INDEX` where `collection_name=mycollection1`
   - If the experiment has a custom ingest script at `experiments/h<id>/ingest.py`, run it:
     ```bash
     ES_INDEX=h<id> uv run python experiments/h<id>/ingest.py
     ```
   - Otherwise reingest HotpotQA with the default script:
     ```bash
     ES_INDEX=h<id> uv run python evaluators/hotpotqa/ingest_hotpotqa.py
     ```
2. Unless ingestion, chunking, embedding, or document serialization changed, do not create a new Elasticsearch index. Instead, set `ES_INDEX` to the champion's `elastic_index` from `best_result.txt` and operate on the documents that already exist there.
3. If chunk count explodes or collapses unexpectedly after an ingestion change, stop and record the run as rejected with the reason in `notes`.

Writing a Custom Ingest Script
- Copy the template: `cp experiments/TEMPLATE/ingest.py experiments/h<id>/ingest.py`
- The script must resolve the project root and add it to `sys.path` so it can import from `scripts.ingest`:
  ```python
  PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
  sys.path.insert(0, str(PROJECT_ROOT))
  from scripts.ingest import ensure_ingest_ready, prepare_document, ingest_prepared_items
  ```
- Call `ensure_ingest_ready()` before indexing.
- Use `prepare_document()` or `prepare_document_chunks()` to build documents, then `ingest_prepared_items()` to embed and index them. See `evaluators/hotpotqa/ingest_hotpotqa.py` for a complete working example.

Evaluation Protocol
1. Run exactly one HotpotQA evaluation with `N=30`:
   ```bash
   AGENT_MODULE=experiments.h<id>.agent N=30 uv run python evaluators/hotpotqa/evaluate.py
   ```
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

Promotion Rules
1. If the experiment is promoted:
   - Copy the winning experiment agent to root: `cp experiments/h<id>/agent.py agent.py`
   - Update `best_result.txt` with the promoted run's metrics and `elastic_index`
   - Commit all changes on `develop`
2. If the experiment is rejected:
   - Do not copy the experiment agent to root.
   - The experiment folder stays as a record.
3. Always commit the new `results.csv` row on `develop` for every completed experiment.

Champion File
- `best_result.txt` is the single source of truth for the current champion.
- `best_result.txt` must include the champion's `elastic_index`, and that value is the default `ES_INDEX` baseline for the next experiment.
- `best_result.txt` must always include the champion's HotpotQA metrics.
- If `best_result.txt` is uninitialized, the first successful `N=30` HotpotQA run accepted on `develop` becomes the initial champion.

Cleanup
1. After recording results, if a rejected experiment created a new ES index, delete it.
2. Do not save artifacts to disk.
3. Keep experiment folders — they serve as the historical record (replacing old branches).

Runner Completion Rule
- The runner is done only after it has walked through the full backlog it was asked to process.

Per-Hypothesis Completion Rule
- A sub-agent is done only when all of the following are true:
  - exactly one hypothesis was executed
  - code changes are in `experiments/h<id>/agent.py`
  - HotpotQA evaluation completed or failed cleanly
  - a row was appended to `results.csv`
  - a judge decision was made
  - `best_result.txt` was updated only if the hypothesis was promoted
  - if promoted, the winning agent.py was copied to root
