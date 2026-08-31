# Hypothesis Backlog

This backlog is for researcher and planner handoff. Executor consumes one card at a time from this file or from a planner-generated copy of the same schema.

Card schema
- `id`: stable hypothesis id
- `name`: short label
- `hypothesis`: the claim being tested
- `implementation sketch`: the minimum change that tests the claim
- `expected upside`: why this might win
- `main risk`: how it can fail
- `judge focus`: what the judge should look at beyond raw accuracy

Repository note
- Current default retrieval is not BM25-only. `agent.py` runs dense vector search plus Elasticsearch text search, fuses them with reciprocal-rank fusion, adds a page-to-child retrieval path, and sentence-reranks the merged contexts. The lexical text leg is effectively BM25 because the Elasticsearch text fields use default similarity and do not override it.


## H01: Hybrid Vector + Text Retrieval

Hypothesis
- Combining dense vector retrieval with lexical text retrieval and fusing the rankings will improve FinQA recall for exact metric names, entity names, and numeric cues.

Implementation sketch
- Run both vector search and text search for the same query.
- Fuse the results with a simple reciprocal-rank or score-normalized merge.
- Deduplicate by chunk id before returning the final top `k`.

Expected upside
- The current code already fuses dense and lexical retrieval, but only with a generic RRF merge. This hypothesis tests whether a more intentionally tuned hybrid can beat the current default on exact-match and numeric-heavy questions.

Main risk
- Lexical hits can add noise and crowd out semantically relevant chunks.

Judge focus
- Check whether accuracy gain comes with a big token-cost increase.
- Inspect whether returned contexts become noisier even when accuracy improves.

## H02: Document-Diversified Top-K

Hypothesis
- Diversifying the final retrieval set by `document_id` or title will reduce redundant chunks from the same report and expose more useful evidence to the model.

Implementation sketch
- Retrieve a larger candidate pool, for example `2k` or `3k`.
- Re-rank or filter so the final `k` does not over-concentrate on one document.
- Allow a small cap per document instead of a hard one-document-only rule.

Expected upside
- The current retriever can return multiple near-duplicate chunks from one source. FinQA often benefits more from broader evidence coverage than from repeated adjacent text.

Main risk
- Some questions really do need multiple nearby chunks from one report, so too much diversification can hurt.

Judge focus
- Look for reduced duplication in contexts.
- Reject if diversification lowers accuracy on numerically dense questions that need local continuity.

## H03: Table-Preserving FinQA Ingestion

Hypothesis
- Preserving table headers and row structure more explicitly during FinQA ingestion will improve retrieval of the exact row and column needed for numerical reasoning.

Implementation sketch
- Serialize each table row with repeated header names and clearer separators.
- Add section markers such as `REPORT_TEXT_BEFORE_TABLE`, `TABLE_ROWS`, and `REPORT_TEXT_AFTER_TABLE`.
- Keep rows intact during chunking whenever possible.

Expected upside
- The current ingestion flattens tables into plain text. Better structure should make both lexical and semantic retrieval more precise for financial metrics.

Main risk
- Over-structuring can inflate chunk count, duplicate headers too much, and drown out the narrative text.

Judge focus
- Verify chunk count does not blow up excessively.
- Check whether gains come from better retrieval rather than prompt-format luck.

## H04: Structure-Aware Chunk Boundaries

Hypothesis
- Chunking on paragraph, newline, or table-row boundaries before falling back to fixed-width character windows will reduce broken evidence spans and improve retrieval quality.

Implementation sketch
- Split on larger structural boundaries first.
- Pack units up to the chunk-size budget.
- Fall back to current sliding-window chunking only when a unit is too large.

Expected upside
- The current `chunk_text` slices by character count only. That often splits rows, sentences, and labels at arbitrary points.

Main risk
- Variable chunk lengths can hurt embedding consistency or create very uneven recall behavior.

Judge focus
- Compare chunk distribution and whether retrieval becomes more stable on table-heavy reports.

## H05: Adaptive Retrieval for Numeric Queries

Hypothesis
- Numeric or table-oriented questions benefit from a slightly different retrieval policy, such as larger candidate pools or mandatory hybrid retrieval, while simpler questions do not need the extra work.

Implementation sketch
- Add a lightweight heuristic that detects numeric questions from tokens like `percentage`, `increase`, `decrease`, `how much`, `ratio`, years, or currency markers.
- For matched queries, use a larger candidate pool or hybrid retrieval path.
- Keep the default path unchanged for everything else.

Expected upside
- This targets extra recall where FinQA is hardest without paying the same latency and token cost on every query.

Main risk
- Heuristics can misclassify questions and create inconsistent behavior.

Judge focus
- Require that any accuracy gain is not erased by a large latency or token increase.

## H06: Neighbor Chunk Expansion

Hypothesis
- After finding a strong chunk, fetching one adjacent chunk on either side from the same document will restore local context that was lost during chunking.

Implementation sketch
- Retrieve top hits normally.
- For the strongest one or two hits, fetch neighboring `chunk_index` values from the same `document_id`.
- Deduplicate before building the final context block.

Expected upside
- Financial evidence is often split across neighboring chunks, especially around tables and explanatory paragraphs.

Main risk
- Neighbor expansion can waste context budget on irrelevant text or duplicate evidence already implied by the hit.

Judge focus
- Ensure the final context block contains more unique evidence, not just more text.

## H07: Context Packing and Dedup

Hypothesis
- Deduplicating repeated snippets and packing contexts with clearer source labels will let the model see more unique evidence within the same prompt budget.

Implementation sketch
- Remove near-duplicate chunks before prompt assembly.
- Trim low-information repeated prefixes.
- Format each context with a compact source label and keep the useful body text.

Expected upside
- The current augmented prompt can spend tokens on repeated titles and overlapping chunk text. Better packing should increase evidence density.

Main risk
- Over-trimming can remove small numeric details that matter to FinQA scoring.

Judge focus
- Check prompt token counts and accuracy together. A small accuracy gain with lower cost is still a good outcome here.

## Research-Sourced Additions

### DRAG and Retrieval-Control Papers

## H08: Retrieval Debate Reranking

Hypothesis
- Running a lightweight proponent-versus-opponent debate over the first retrieval pool will demote misleading chunks and improve evidence quality on ambiguity-heavy questions.

Implementation sketch
- Retrieve a wider candidate pool, then ask one pass to argue which chunks support the task and another to challenge weak or misleading chunks.
- Use the resulting debate notes as a reranking signal before final context packing.
- Paper: [Removal of Hallucination on Hallucination: Debate-Augmented RAG](https://arxiv.org/abs/2505.18581)

Expected upside
- This directly targets "hallucination on hallucination" by stress-testing evidence before it reaches the answer step.

Main risk
- Extra reasoning may add latency and amplify prompt noise if the debate itself is low quality.

Judge focus
- Check whether retrieval precision improves enough to justify the extra call budget.

## H09: Judge-Based Evidence Selection

Hypothesis
- A separate judge pass that selects only the most defensible retrieved chunks will outperform naive top-`k` truncation when retrieval quality is uneven.

Implementation sketch
- After initial retrieval, ask a judge prompt to keep only chunks that are both relevant and mutually consistent.
- Use the judge output as a filter rather than as a new answer generator.
- Paper: [Removal of Hallucination on Hallucination: Debate-Augmented RAG](https://arxiv.org/abs/2505.18581)

Expected upside
- A dedicated selection step may cut contradictory or off-topic chunks before they contaminate the answer prompt.

Main risk
- Over-filtering can drop the one chunk that contains the decisive number or entity mention.

Judge focus
- Look for better factual grounding without a large recall collapse.

## H10: Asymmetric Generation Debate

Hypothesis
- Splitting generation into asymmetric roles, where one pass answers from current evidence and another explicitly looks for failure cases, will reduce overconfident wrong answers.

Implementation sketch
- Keep the normal answer draft, but add a challenger pass that only searches for contradictions, missing assumptions, or unsupported claims in that draft.
- Resolve with a short final synthesis prompt that can revise or abstain.
- Paper: [Removal of Hallucination on Hallucination: Debate-Augmented RAG](https://arxiv.org/abs/2505.18581)

Expected upside
- This may improve answer faithfulness even when retrieval is only partially correct.

Main risk
- The challenge pass can become generic and add cost without finding real errors.

Judge focus
- Inspect whether wrong answers become more cautious or better cited rather than merely longer.

### Query Performance Prediction

## H11: QPP-Gated Evidence Acceptance

Hypothesis
- Estimating retrieval quality before injecting retrieved context into the reasoning chain will reduce bad iterations in agentic RAG.

Implementation sketch
- Compute a post-retrieval QPP score for each generated sub-query.
- If the score is low, either retry with a rewritten query, switch retriever settings, or skip context injection for that iteration.
- Paper: [Am I on the Right Track? What Can Predicted Query Performance Tell Us about the Search Behaviour of Agentic RAG](https://arxiv.org/html/2507.10411v1#S3.SS3)

Expected upside
- The paper's core claim is that query quality correlates with answer quality, so filtering low-value retrievals may shorten harmful loops.

Main risk
- A conservative gate can suppress retrieval exactly when the model needs it most.

Judge focus
- Compare both accuracy and average iterations, not accuracy alone.

## H12: NQC Score Dispersion

Hypothesis
- Normalized query commitment (NQC) can serve as a cheap retrieval-confidence signal for deciding whether a generated sub-query is worth trusting.

Implementation sketch
- Compute NQC from the top retrieval score distribution for each generated query.
- Use the score to trigger a retry, fallback, or retrieval abstention threshold.
- Paper: [Am I on the Right Track? What Can Predicted Query Performance Tell Us about the Search Behaviour of Agentic RAG](https://arxiv.org/html/2507.10411v1#S4.SS3)

Expected upside
- NQC is cheap and retriever-agnostic enough to test without redesigning the stack.

Main risk
- Score dispersion can be unstable across retrievers and index settings.

Judge focus
- Verify that the threshold transfers across question types instead of only helping a narrow slice.

## H13: Max-Score Retrieval Gate

Hypothesis
- For neural retrieval setups, the top document score alone may be a sufficiently strong gate for accepting or retrying a generated sub-query.

Implementation sketch
- Record the highest retrieval score for each sub-query.
- If it falls below a tuned threshold, rewrite or skip the retrieval result for that step.
- Paper: [Am I on the Right Track? What Can Predicted Query Performance Tell Us about the Search Behaviour of Agentic RAG](https://arxiv.org/html/2507.10411v1#S4.SS3)

Expected upside
- This is simpler than full score-distribution methods and easy to deploy in fast loops.

Main risk
- One strong but wrong lexical hit can produce false confidence.

Judge focus
- Check whether the simplicity win survives on multi-hop and compositional questions.

## H14: A-Pair-Ratio Coherence Gate

Hypothesis
- Measuring whether top retrieved documents are semantically more coherent than tail documents can identify good versus noisy dense-retrieval results.

Implementation sketch
- For dense retrieval only, compute the A-Pair-Ratio coherence score over the retrieved set.
- Use it as a gate for whether to trust the current retrieval result or ask for a reformulated query.
- Paper: [Am I on the Right Track? What Can Predicted Query Performance Tell Us about the Search Behaviour of Agentic RAG](https://arxiv.org/html/2507.10411v1#S4.SS3)

Expected upside
- Coherence is closer to "useful evidence cluster" than raw top-score confidence.

Main risk
- A coherent cluster of irrelevant documents can still pass the gate.

Judge focus
- Inspect failure cases where coherent but off-target evidence slips through.

## H15: Dense-QPP Geometry Gate

Hypothesis
- The geometric spread of query-plus-document embeddings can flag weak dense-retrieval results before they pollute the reasoning trace.

Implementation sketch
- Compute the Dense-QPP hypercube-style geometry score over the query and top dense hits.
- Use it to choose between accepting the result, broadening retrieval, or reformulating the query.
- Paper: [Am I on the Right Track? What Can Predicted Query Performance Tell Us about the Search Behaviour of Agentic RAG](https://arxiv.org/html/2507.10411v1#S4.SS3)

Expected upside
- This may catch cases where the retrieved set is too diffuse to support a precise answer.

Main risk
- The signal depends on embedding-space behavior and may be brittle with the current encoder.

Judge focus
- Require gains that hold across multiple question forms, not just entity lookup.

### Reasoning-Augmented Retrieval

## H16: Few-Shot Chain-of-Thought

Hypothesis
- Adding explicit few-shot chain-of-thought demonstrations before answer generation will help the model use retrieved evidence more faithfully on multi-step questions.

Implementation sketch
- Keep retrieval unchanged and only add concise reasoning exemplars that show evidence-grounded intermediate steps.
- Use short demonstrations to avoid crowding out retrieved context.
- Paper: [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

Expected upside
- Better intermediate reasoning can increase the value extracted from the same retrieved evidence.

Main risk
- Long exemplars may steal tokens from evidence and hurt rather than help.

Judge focus
- Check whether gains come from better reasoning, not just longer outputs.

## H17: Draft-as-Query Iter-RetGen

Hypothesis
- Using the model's current draft answer as the next retrieval query will surface missing evidence that the original question did not state explicitly.

Implementation sketch
- Generate a first-pass response from the question.
- Use that draft, or a compressed version of it, as the next retrieval query and regenerate with the expanded evidence set.
- Paper: [Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy](https://arxiv.org/abs/2305.15294)

Expected upside
- The draft exposes latent information needs that are often absent from the user question alone.

Main risk
- Early wrong drafts can anchor retrieval toward the wrong part of the corpus.

Judge focus
- Look for gains on multi-hop questions without a spike in self-reinforcing errors.

## H18: Whole-Context Iterative Refresh

Hypothesis
- Rebuilding the full evidence bundle after each retrieval-generation round will work better than only appending local snippets when the question evolves over multiple hops.

Implementation sketch
- After each iteration, recompute the full retrieved context using both the original question and the latest generated draft.
- Replace the answer context wholesale instead of endlessly appending more evidence.
- Paper: [Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy](https://arxiv.org/abs/2305.15294)

Expected upside
- This can keep the prompt coherent and reduce stale or contradictory evidence accumulation.

Main risk
- Full refreshes may discard a previously useful chunk unless dedup and carry-forward logic are careful.

Judge focus
- Compare evidence coherence and token efficiency, not just raw correctness.

## H19: CoT-Sentence Retrieval

Hypothesis
- Using the latest chain-of-thought sentence as the next retrieval query will improve recall for multi-hop questions over question-only retrieval.

Implementation sketch
- Generate one reasoning sentence at a time.
- Use the latest sentence as the sub-query for the next retrieval step, then continue reasoning with the expanded evidence set.
- Paper: [Interleaving Retrieval with Chain-of-Thought Reasoning](https://arxiv.org/abs/2212.10509)

Expected upside
- Each reasoning step can focus retrieval on the current unresolved hop.

Main risk
- Hallucinated intermediate reasoning can steer retrieval away from the answer.

Judge focus
- Inspect whether retrieval recall improves before trusting downstream answer gains.

## H20: Final Read over Accumulated Evidence

Hypothesis
- After iterative retrieval, a final answer pass over the union of collected evidence will outperform answering directly from the last retrieval step alone.

Implementation sketch
- Keep all retrieved paragraphs across IRCoT-style steps.
- Run a clean final answer prompt over the accumulated evidence after the loop ends.
- Paper: [Interleaving Retrieval with Chain-of-Thought Reasoning](https://arxiv.org/abs/2212.10509)

Expected upside
- The final answer step can integrate all hops without forcing the last retrieval call to carry the entire burden.

Main risk
- The accumulated pool can become too noisy without strong deduplication.

Judge focus
- Check whether final-pass gains survive once prompt length is controlled.

## H21: Forward-Looking Sentence Search

Hypothesis
- Predicting the next sentence before writing it and using that prediction as a retrieval query will improve long-form grounded generation.

Implementation sketch
- Before committing the next sentence, ask the model for a one-sentence forecast of what it expects to say next.
- Use that forecast as the retrieval query, then regenerate the sentence with the retrieved evidence.
- Paper: [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983)

Expected upside
- This lets retrieval follow the model's future information needs instead of staying anchored to the original question only.

Main risk
- Weak forecasts can produce irrelevant retrieval and add an unstable extra step.

Judge focus
- Check whether future-looking retrieval helps enough to offset the extra generation turn.

## H22: Low-Confidence Regeneration

Hypothesis
- Regenerating only when the next sentence contains low-confidence tokens will beat always-retrieve and never-retrieve baselines on cost-adjusted quality.

Implementation sketch
- Monitor token confidence or a cheap uncertainty proxy while drafting the next sentence.
- Trigger retrieval-plus-regeneration only when confidence drops below a threshold.
- Paper: [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983)

Expected upside
- This makes active retrieval selective instead of paying the same cost on every step.

Main risk
- Confidence estimates may be noisy and miss precisely the hard factual cases.

Judge focus
- Require evidence of better accuracy-per-token, not just better raw quality.

## H23: Uncertainty-Triggered Search

Hypothesis
- Explicitly detecting uncertainty markers in the reasoning trace and turning them into search opportunities will improve difficult reasoning questions.

Implementation sketch
- Add a lightweight detector for uncertainty phrases or confidence dips in the reasoning stream.
- When triggered, pause reasoning, issue a search query, and continue with the retrieved evidence.
- Paper: [Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366)

Expected upside
- This borrows Search-o1's central intuition without needing a full new reasoning model.

Main risk
- Keyword-style uncertainty detection can be brittle and easy to game.

Judge focus
- Compare failure modes on hard questions where the baseline currently guesses from parametric memory.

## H24: Reason-in-Documents Compression

Hypothesis
- A dedicated "reason in documents" pass that compresses retrieved pages into task-relevant evidence will outperform directly stuffing raw retrieval into the main reasoning prompt.

Implementation sketch
- Insert a document-analysis pass between retrieval and answering.
- Ask it to extract only the facts that move the current reasoning step forward, then pass the compressed result onward.
- Paper: [Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366)

Expected upside
- This should reduce noise from long retrieved pages while preserving the facts the answer step actually needs.

Main risk
- The compression pass can accidentally delete a critical qualifier or number.

Judge focus
- Inspect whether evidence density improves without hiding the provenance of important facts.

## H25: Adaptive Stop/Continue Controller

Hypothesis
- Explicitly deciding whether to stop or continue retrieving at each iteration will outperform a fixed number of retrieval loops.

Implementation sketch
- Add a stop/continue decision after each retrieval step using the current evidence state.
- Start with a lightweight approximation of Stop-RAG's controller before considering any learned policy.
- Paper: [Stop-RAG: Value-Based Retrieval Control for Iterative RAG](https://arxiv.org/abs/2510.14337)

Expected upside
- This directly targets wasted retrieval loops and late-stage distraction.

Main risk
- Stopping too early is often more damaging than one extra retrieval step.

Judge focus
- Measure both accuracy and average iterations per question.

## H26: Q-Lambda Stopping Policy

Hypothesis
- A learned stop policy trained to value future retrieval benefit, rather than current confidence alone, will make better stopping decisions on multi-hop questions.

Implementation sketch
- Approximate the paper's Q(lambda) idea with an offline stop-policy scorer over partial retrieval traces.
- Use the scorer only for the stop/continue decision, leaving the rest of the pipeline unchanged.
- Paper: [Stop-RAG: Value-Based Retrieval Control for Iterative RAG](https://arxiv.org/abs/2510.14337)

Expected upside
- Value-based control should outperform simple confidence heuristics when more retrieval is useful but not yet obviously so.

Main risk
- This is higher effort and may be too data-hungry for the likely gain.

Judge focus
- Reject if the policy-training overhead is large relative to the end-to-end improvement.

## H27: Evidence-Only Stop State

Hypothesis
- For stopping decisions, using only the main question plus retrieved evidence may be sufficient, without feeding the full intermediate reasoning trace.

Implementation sketch
- Build the stop-state representation from the question and currently retrieved chunks only.
- Compare that against richer stop-state variants that also include intermediate search queries or draft answers.
- Paper: [Stop-RAG: Value-Based Retrieval Control for Iterative RAG](https://arxiv.org/abs/2510.14337)

Expected upside
- A simpler state may be more stable and cheaper while still capturing whether more evidence is needed.

Main risk
- The missing reasoning trace may hide whether the model has already resolved the hard part of the question.

Judge focus
- Prefer the simplest state that preserves decision quality.

## H28: Semantic-Exact Weighted Fusion

Hypothesis
- Letting the agent adaptively mix dense semantic search with exact lexical search will outperform a single fixed retrieval mode.

Implementation sketch
- Expose both semantic and exact retrieval, then let the prompt or control logic set fusion weights per query.
- Start with a small discrete set of fusion profiles instead of free-form weights.
- Paper: [Interact-RAG: Reason and Interact with the Corpus, Beyond Black-Box Retrieval](https://arxiv.org/abs/2510.27566)

Expected upside
- This captures the main benefit of Interact-RAG without requiring a full interaction engine.

Main risk
- Adaptive fusion adds another control surface that can oscillate or overfit.

Judge focus
- Check whether mixed-mode retrieval wins on both entity-heavy and semantic paraphrase questions.

## H29: Entity-Anchored Matching

Hypothesis
- When a key entity is clear, forcing an entity-anchored retrieval step will reduce distraction from semantically similar but wrong documents.

Implementation sketch
- Detect a candidate anchor entity from the question or current reasoning state.
- Run a focused entity match step before broader retrieval or reranking.
- Paper: [Interact-RAG: Reason and Interact with the Corpus, Beyond Black-Box Retrieval](https://arxiv.org/abs/2510.27566)

Expected upside
- Anchoring is especially attractive for finance and multi-entity questions where near-neighbor confusion is common.

Main risk
- Wrong entity extraction will sharply narrow retrieval to the wrong target.

Judge focus
- Inspect entity-disambiguation failures rather than only aggregate metrics.

## H30: Context Shaping Filters

Hypothesis
- Explicitly including, excluding, and resizing the current working set of documents will improve agentic retrieval over repeated blind top-`k` calls.

Implementation sketch
- Add simple controls for `include_docs`, `exclude_docs`, and retrieval-scale adjustment around the current candidate set.
- Keep the controls deterministic and lightweight so they are attributable in evaluation.
- Paper: [Interact-RAG: Reason and Interact with the Corpus, Beyond Black-Box Retrieval](https://arxiv.org/abs/2510.27566)

Expected upside
- This may reduce repeated retrieval noise without needing a new retriever.

Main risk
- Bad early filters can permanently hide the evidence needed later.

Judge focus
- Check whether the filters improve context quality without making the system fragile to early mistakes.

## H31: Planner-Reasoner-Executor Workflow

Hypothesis
- Separating high-level planning, adaptive reasoning, and concrete execution will outperform a single monolithic prompted agent on complex retrieval tasks.

Implementation sketch
- Use a prompt-only three-stage workflow: planner for decomposition, reasoner for step control, executor for concrete retrieval actions.
- Keep the stages narrow and explicit rather than asking one prompt to do everything.
- Paper: [Interact-RAG: Reason and Interact with the Corpus, Beyond Black-Box Retrieval](https://arxiv.org/abs/2510.27566)

Expected upside
- This should make behavior more legible and reduce tool-use drift.

Main risk
- Extra orchestration can add latency and brittle handoff errors.

Judge focus
- Look for cleaner action traces and better retrieval choices, not just longer reasoning.

## H32: Successful-Trajectory Agent Distillation

Hypothesis
- Fine-tuning or imitation on only successful retrieval trajectories will produce a more reliable agent than pure prompting alone.

Implementation sketch
- Treat this as a high-effort card: collect successful planner/action traces and distill them into a smaller policy if the lighter prompt-only cards win first.
- Mask retrieved content during training as in the paper if this path is ever pursued.
- Paper: [Interact-RAG: Reason and Interact with the Corpus, Beyond Black-Box Retrieval](https://arxiv.org/abs/2510.27566)

Expected upside
- Distillation could internalize the workflow so later inference is cheaper and more stable.

Main risk
- This likely exceeds the practical scope of the current repo and may not justify the engineering cost.

Judge focus
- Treat this as a strategic bet; require a much larger gain before accepting the added complexity.

### A-RAG Interface Design

## H33: Sentence-Boundary Chunking

Hypothesis
- Chunking near sentence boundaries instead of by raw window cuts will improve downstream retrieval and reduce broken evidence spans.

Implementation sketch
- Keep roughly the same chunk budget, but pack text by sentence-aware boundaries before falling back to raw slicing.
- Preserve mappings from sentences back to parent chunks for later retrieval stages.
- Paper: [A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces](https://arxiv.org/html/2602.03442v1#S3.SS1.SSS0.Px1)

Expected upside
- This is a low-risk structural improvement that aligns well with multi-stage retrieval.

Main risk
- Uneven chunk lengths may hurt embedding consistency or index balance.

Judge focus
- Compare both retrieval quality and chunk-count stability.

## H34: Sentence-Level Semantic Search

Hypothesis
- Searching over sentence embeddings and then aggregating back to chunks will outperform chunk-only dense retrieval on multi-hop questions.

Implementation sketch
- Embed sentences, retrieve the highest-scoring sentences for a query, then aggregate by parent chunk.
- Return chunk ids plus matched sentence snippets rather than raw whole-chunk scores.
- Paper: [A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces](https://arxiv.org/html/2602.03442v1#S3.SS2.SSS0.Px2)

Expected upside
- Fine-grained matching should surface the right chunk even when only one sentence is relevant.

Main risk
- Sentence-level indexing adds more objects and may create recall fragmentation.

Judge focus
- Verify that sentence retrieval improves evidence precision without exploding index size.

## H35: Runtime Keyword Search

Hypothesis
- A lightweight runtime keyword search path will recover exact entity and metric mentions that dense retrieval misses.

Implementation sketch
- Add an exact-match retrieval path at query time instead of relying on dense retrieval alone.
- Use it selectively for entity-heavy or terminology-sensitive questions.
- Paper: [A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces](https://arxiv.org/html/2602.03442v1#S3.SS1.SSS0.Px3)

Expected upside
- This should help with named entities, acronyms, and exact financial terms.

Main risk
- Pure lexical search can over-prioritize superficial matches.

Judge focus
- Check whether keyword wins come from true exact evidence rather than noisy term overlap.

## H36: Keyword-Weighted Snippet Return

Hypothesis
- Returning only keyword-bearing sentences as snippets, weighted toward longer and more specific keywords, will make lexical retrieval more usable for agents.

Implementation sketch
- Score exact matches using keyword frequency and keyword length.
- Return compact snippets made only of sentences containing matched keywords.
- Paper: [A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces](https://arxiv.org/html/2602.03442v1#S3.SS2.SSS0.Px1)

Expected upside
- Compact snippets can increase evidence density and make lexical results easier to judge.

Main risk
- Snippet extraction may omit nearby qualifiers that change the answer.

Judge focus
- Inspect whether snippet compaction preserves the decisive local context.

## H37: Chunk Read with Neighbor Access

Hypothesis
- Letting the agent explicitly open a chunk and optionally its adjacent chunks will outperform always injecting the full top-`k` set.

Implementation sketch
- Separate discovery from full reading: retrieval tools return ids and snippets, then a read tool fetches the full chunk only when asked.
- Allow optional adjacent-chunk reads for continuity around tables or split explanations.
- Paper: [A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces](https://arxiv.org/html/2602.03442v1#S3.SS2.SSS0.Px3)

Expected upside
- This can reduce prompt bloat while preserving the ability to recover local context.

Main risk
- Too many read decisions can slow the loop and create brittle control logic.

Judge focus
- Require a clear token-efficiency win, not just a different prompt shape.

## H38: ReAct Loop + Context Tracker

Hypothesis
- A simple one-tool-at-a-time ReAct loop combined with explicit tracking of already-read chunks will reduce redundant retrieval and improve evidence diversity.

Implementation sketch
- Keep tool use sequential and explicit rather than parallel or heavily orchestrated.
- Track read chunk ids and return a zero-cost notification when the agent tries to reread the same chunk.
- Paper: [A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces](https://arxiv.org/html/2602.03442v1#S3.SS3)

Expected upside
- This is a practical way to gain some agentic benefits without building a complex controller.

Main risk
- The one-tool loop may underperform more aggressive controllers on hard questions.

Judge focus
- Check whether redundant reads drop and whether the saved budget converts into useful exploration.

### A-RAG Related Methods

## H39: Query Rewriting

Hypothesis
- Rewriting the user question into a retrieval-optimized query before search will improve first-pass recall on under-specified questions.

Implementation sketch
- Add a lightweight query-rewrite step before retrieval, preserving the original question for answer generation.
- Prefer a deterministic or very small prompt so the change stays attributable.
- Paper source: [A-RAG related-work summary of query rewriting / RQ-RAG](https://arxiv.org/html/2602.03442v1#S2.SS1)

Expected upside
- A better query can solve many retrieval misses without touching the index.

Main risk
- Rewrites can drift semantically and over-narrow the search.

Judge focus
- Compare rewritten-query recall directly against answer accuracy.

## H40: Adaptive Retrieval Routing

Hypothesis
- Routing simple questions to cheap retrieval and harder questions to richer retrieval policies will improve cost-adjusted performance.

Implementation sketch
- Add a small complexity classifier or heuristic that chooses among retrieval modes, for example dense-only versus hybrid or iterative.
- Keep the routing decision observable so failures are diagnosable.
- Paper source: [A-RAG related-work summary of adaptive routing / Adaptive-RAG](https://arxiv.org/html/2602.03442v1#S2.SS1)

Expected upside
- This can concentrate expensive retrieval only where it is likely to matter.

Main risk
- Bad routing decisions create inconsistent behavior and hard-to-debug regressions.

Judge focus
- Evaluate both average cost and worst-case failure patterns.

## H41: Corrective Retrieval Evaluation

Hypothesis
- Scoring the retrieved set itself for likely usefulness before answer generation will help catch low-quality retrieval and trigger recovery behavior.

Implementation sketch
- Add a retrieval-quality check after search and before answer generation.
- If the set looks weak or contradictory, retry with a broader or rewritten query instead of proceeding blindly.
- Paper source: [A-RAG related-work summary of retrieval quality evaluation / CRAG](https://arxiv.org/html/2602.03442v1#S2.SS1)

Expected upside
- This targets retrieval failures directly instead of hoping the answer prompt will recover.

Main risk
- Quality checks can become just another noisy heuristic layered on top of retrieval.

Judge focus
- Look for fewer catastrophic retrieval misses, not only small average gains.

## H42: GraphRAG Community Retrieval

Hypothesis
- Building a graph-level view of the corpus and retrieving both local entity evidence and higher-level community summaries will improve multi-hop coverage.

Implementation sketch
- Construct a lightweight entity-relation or document-link graph, plus higher-level cluster summaries.
- Retrieve both node-local evidence and broader community context for multi-hop questions.
- Paper source: [A-RAG appendix summary of GraphRAG](https://arxiv.org/html/2602.03442v1#A2.I1.i1)

Expected upside
- Community-level retrieval may help when the answer needs both local facts and global context.

Main risk
- Graph construction can be expensive and fragile on noisy documents.

Judge focus
- Demand a real multi-hop gain before accepting graph-build complexity.

## H43: RAPTOR Recursive Summaries

Hypothesis
- Recursive bottom-up summarization of chunks into a tree will improve retrieval for broad or hierarchical questions.

Implementation sketch
- Summarize leaf chunks upward into a small hierarchy.
- Retrieve at multiple levels, then descend to the leaves only where needed.
- Paper source: [A-RAG related-work summary of RAPTOR](https://arxiv.org/html/2602.03442v1#S2.SS2)

Expected upside
- Hierarchical summaries can guide retrieval toward the right region of the corpus faster than flat search.

Main risk
- Summary compression may erase the exact number or phrase needed for the final answer.

Judge focus
- Check whether high-level summaries help discovery without hiding leaf-level evidence.

## H44: LightRAG Local-Global Search

Hypothesis
- Combining local graph retrieval with broader global search will outperform either narrow evidence lookup or broad search alone on compositional questions.

Implementation sketch
- Maintain both local neighborhood retrieval and a broader global graph or corpus search path.
- Fuse them differently depending on whether the question looks entity-centric or summary-centric.
- Paper source: [A-RAG related-work summary of LightRAG](https://arxiv.org/html/2602.03442v1#S2.SS2)

Expected upside
- This can widen coverage without giving up precise local grounding.

Main risk
- The system can overfetch context and lose the efficiency benefit.

Judge focus
- Prefer wins that come from better coverage, not just larger prompts.

## H45: HippoRAG2 Memory Walks

Hypothesis
- Personalized PageRank-style walks over a memory graph will improve single-step multi-hop retrieval compared with flat top-`k` search.

Implementation sketch
- Build a lightweight graph over entities or chunks and run graph-walk retrieval seeded by the question.
- Use the walk output as a candidate set for final chunk selection.
- Paper source: [A-RAG appendix summary of HippoRAG2](https://arxiv.org/html/2602.03442v1#A2.I1.i2)

Expected upside
- Graph walks can surface indirectly connected evidence that dense similarity misses.

Main risk
- Poor graph quality can make the walk confidently wrong.

Judge focus
- Inspect whether gains come from better path coverage rather than retrieval luck.

## H46: LinearRAG Hierarchical Entity Graph

Hypothesis
- Replacing heavy relation extraction with a simpler entity-centric hierarchical graph will capture some graph-RAG benefits at lower ingestion cost.

Implementation sketch
- Extract entities, build a simplified hierarchy, and use two-stage retrieval over the graph and original chunks.
- Keep graph construction intentionally lightweight.
- Paper source: [A-RAG appendix summary of LinearRAG](https://arxiv.org/html/2602.03442v1#A2.I1.i3)

Expected upside
- This may be the most practical graph-flavored experiment for the current codebase.

Main risk
- Dropping relation extraction may throw away exactly the structure needed for reasoning.

Judge focus
- Compare ingestion overhead directly against any retrieval gain.

## H47: FaithfulRAG Conflict Modeling

Hypothesis
- Explicitly modeling conflicts between retrieved evidence and the model's parametric beliefs will improve answer faithfulness when sources disagree.

Implementation sketch
- Add a conflict-detection pass that flags when retrieved evidence appears to contradict the model's first-pass answer or other retrieved chunks.
- Require the final answer to resolve or acknowledge the conflict explicitly.
- Paper source: [A-RAG appendix summary of FaithfulRAG](https://arxiv.org/html/2602.03442v1#A2.I1.i4)

Expected upside
- This directly targets a common failure mode in RAG: confident answers built on unresolved conflicts.

Main risk
- Conflict detection can be noisy and overly cautious.

Judge focus
- Inspect contradiction handling, not just average accuracy.

## H48: Self-RAG Reflection Controller

Hypothesis
- Adding explicit self-reflection steps about whether more evidence is needed and whether current evidence is trustworthy will improve agentic retrieval decisions.

Implementation sketch
- Approximate Self-RAG with structured reflection prompts after retrieval and before final answering.
- Restrict the reflection output to actionable decisions such as `retrieve_more`, `use_current_evidence`, `critique_answer`, or `answer_now`.
- Keep the implementation inference-only; do not depend on training special reflection tokens.
- Paper source: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)

Expected upside
- Structured reflection may improve both retrieval timing and answer caution.

Main risk
- The original paper learns reflection behavior during training, so a prompt-only approximation may underdeliver.

Judge focus
- Require reflections to change behavior in measurable ways, not just lengthen traces.

## H49: RA-ISF Iterative Self-Feedback

Hypothesis
- Iteratively critiquing and revising the current answer or retrieval state will improve multi-step QA over one-pass answer generation.

Implementation sketch
- After each answer draft, run a focused self-feedback pass that identifies missing evidence or unsupported claims.
- Use the feedback to trigger one more targeted retrieval or revision cycle.
- Paper source: [A-RAG related-work summary of RA-ISF](https://arxiv.org/html/2602.03442v1#S2.SS3)

Expected upside
- Self-feedback can recover from partial misses without needing a full new control policy.

Main risk
- Repeated feedback loops can spiral into latency without adding new information.

Judge focus
- Check whether revisions genuinely correct earlier mistakes instead of paraphrasing them.

## H50: MA-RAG Specialist Agents

Hypothesis
- Using specialized roles such as planner, extractor, and answerer will outperform a single generalist prompt on complex retrieval tasks.

Implementation sketch
- Approximate MA-RAG with a small set of narrow prompts that hand off a structured state rather than free-form text.
- Keep the specialist boundary explicit so the contribution of each role is testable.
- Paper source: [A-RAG appendix summary of MA-RAG](https://arxiv.org/html/2602.03442v1#A2.I1.i5)

Expected upside
- Specialization may reduce prompt overload and improve consistency.

Main risk
- Multi-agent scaffolding can add ceremony faster than it adds useful signal.

Judge focus
- Reject if the extra roles do not produce cleaner evidence handling or better answer quality.

## H51: RAGentA Filtering + Citations

Hypothesis
- Iterative document filtering plus citation-oriented answer formatting will improve both evidence quality and auditability.

Implementation sketch
- Add a filtering pass that trims the candidate set before final answering.
- Require the answer prompt to preserve source attributions for the evidence it actually uses.
- Paper source: [A-RAG appendix summary of RAGentA](https://arxiv.org/html/2602.03442v1#A2.I1.i6)

Expected upside
- Better filtering can improve answer quality, and citations make later judging easier.

Main risk
- Citation formatting can become superficial if the evidence filter is weak.

Judge focus
- Check whether cited evidence truly supports the answer, not just whether citations appear.

### PageIndex

## H52: Page-Level Vectorless Index

Hypothesis
- Building a page-level index with summaries and metadata, without vector embeddings, will improve long-document navigation when the answer lives in a small region of a large file.

Implementation sketch
- Represent each document as a structured page index rather than a flat chunk list.
- Route retrieval first at the page level, then only open the promising pages for finer-grained reading.
- Source article: [PageIndex: Vectorless, Human-Like RAG for Long Documents](https://dhrumilbhut.medium.com/pageindex-vectorless-human-like-rag-for-long-documents-092ddd56221c)

Expected upside
- This may be a better fit for long reports where page position and section locality matter more than semantic-nearest-neighbor search.

Main risk
- Without embeddings, recall can drop badly on paraphrased or indirect questions.

Judge focus
- Compare long-document recall and navigation cost, not only final accuracy.

## H53: Page-to-Passage Navigation

Hypothesis
- A hierarchical navigation flow that moves from document structure to page candidates to exact passages will outperform one-shot flat retrieval on long PDFs.

Implementation sketch
- Add a staged retrieval flow: identify candidate sections or pages, summarize them, then drill into exact passages only inside the shortlisted area.
- Keep the intermediate navigation trace visible so later agents can inspect where the answer was found.
- Source article: [PageIndex: Vectorless, Human-Like RAG for Long Documents](https://dhrumilbhut.medium.com/pageindex-vectorless-human-like-rag-for-long-documents-092ddd56221c)

Expected upside
- This mirrors how humans search long documents and may reduce wasted context on irrelevant pages.

Main risk
- Bad page selection early in the funnel can hide the answer completely.

Judge focus
- Inspect whether the staged funnel narrows correctly before rewarding answer gains.

### Retriever Model Variants

API-feasibility note
- H54-H59 are intended to be implementable with retrieval/index changes and model inference outputs only. They do not assume fine-tuning, gradient access, hidden states, or direct access to model weights.

## H54: BM25 Sparse Anchor Route

Hypothesis
- For rare entities, dates, quoted phrases, and literal metric names, a BM25-first sparse route will outperform the current generic hybrid order by anchoring retrieval on exact lexical overlap before dense similarity drifts.

Implementation sketch
- Add a retrieval mode that runs the existing Elasticsearch text search first and preserves at least a small fixed number of sparse hits before any later fusion or reranking.
- Trigger it only for obviously exact-match questions, or evaluate it as a clean ablation against the current hybrid path.
- Keep prompt formatting and answer generation unchanged so the effect stays attributable to sparse ranking.
- Paper source: [Am I on the Right Track? What Can Predicted Query Performance Tell Us about the Search Behaviour of Agentic RAG](https://arxiv.org/html/2507.10411v1#S5.SS1)

Expected upside
- This repo already has a BM25-style lexical leg, so testing a sparse-anchor route is low effort and easy to compare. The paper uses BM25 as its sparse baseline, making it a relevant control for exact-match slices.

Main risk
- BM25 was weaker than MonoT5 and E5 overall in the paper, so a global switch is unlikely to win.

Judge focus
- Require gains on entity-heavy, date-heavy, or quote-like questions rather than only on overall average.

## H55: MonoT5 Cross-Encoder Reranking

Hypothesis
- Re-ranking a BM25 candidate pool with MonoT5 will improve final evidence precision over the current lightweight sentence-similarity reranker.

Implementation sketch
- Retrieve a wider sparse candidate pool, for example top 50 or top 100.
- Score query-chunk pairs with MonoT5 and keep only the top final `k`.
- Limit MonoT5 to the reranking stage so indexing stays simple and the cost remains measurable.
- Paper source: [Am I on the Right Track? What Can Predicted Query Performance Tell Us about the Search Behaviour of Agentic RAG](https://arxiv.org/html/2507.10411v1#S5.SS1)

Expected upside
- In the paper, MonoT5 beats BM25 on both answer quality and average reasoning length, suggesting that stronger reranking can reduce noisy context.

Main risk
- Cross-encoder inference can be too slow for multi-iteration retrieval loops.

Judge focus
- Compare answer accuracy and retrieval precision against latency, not against accuracy alone.

## H56: E5 Dense Retriever Swap

Hypothesis
- Replacing the current embedding retriever with an E5-based dense retriever will improve semantic recall and reduce the number of retrieval iterations on paraphrased questions.

Implementation sketch
- Re-embed the corpus with a single E5 checkpoint and store those vectors in the existing dense index.
- Re-embed queries with the same E5 model and compare E5-only or E5-led retrieval against the current dense leg.
- Keep downstream reranking and prompting unchanged for the first pass so the retriever swap is isolated.
- Paper source: [Am I on the Right Track? What Can Predicted Query Performance Tell Us about the Search Behaviour of Agentic RAG](https://arxiv.org/html/2507.10411v1#S5.SS1)

Expected upside
- In the paper, E5 is the strongest retriever of the three tested and yields both the best answer quality and the fewest average iterations.

Main risk
- Re-ingestion cost is non-trivial, and E5 may underperform the current embedding model on domain-specific jargon without tuning.

Judge focus
- Check both retrieval recall and average iteration count, since E5’s reported upside is partly about shortening the reasoning loop.

## H57: First-Hop Premium Retrieval

Hypothesis
- Spending more retrieval budget on the first generated query, then using a cheaper retriever on later hops, will improve cost-adjusted answer quality because the first retrieval has the strongest influence on the final answer.

Implementation sketch
- On the first retrieval only, use a larger candidate pool and the strongest reranker or retriever available, such as MonoT5 or E5.
- On later iterations, fall back to the current cheaper hybrid path unless a low-confidence signal forces another premium retrieval.
- Measure both answer quality and total retrieval cost per question.
- Paper source: [Am I on the Right Track? What Can Predicted Query Performance Tell Us about the Search Behaviour of Agentic RAG](https://arxiv.org/html/2507.10411v1#S6)

Expected upside
- The paper concludes that better retrievers shorten reasoning and emphasizes the importance of the first retrieval iteration in final answer quality.

Main risk
- Later low-quality hops can still derail the answer, so front-loading quality may not be sufficient.

Judge focus
- Look for better cost-adjusted accuracy and fewer late-stage iterations.

## H58: Repeated Sub-Query Suppression

Hypothesis
- Detecting repeated or near-duplicate intermediate queries and forcing reformulation or diversification will reduce wasted retrieval iterations and improve evidence coverage.

Implementation sketch
- Track normalized generated sub-queries across iterations.
- If a new sub-query is identical or near-duplicate to an earlier one, either reject it, add the missing entities back in, or switch retriever mode before searching again.
- Log how often the suppression fires and whether the replacement query retrieves different evidence.
- Paper source: [Am I on the Right Track? What Can Predicted Query Performance Tell Us about the Search Behaviour of Agentic RAG](https://arxiv.org/html/2507.10411v1#A1.SS2)

Expected upside
- The paper’s appendix reports repeated identical sub-queries even in successful E5 runs, which suggests that current agentic retrieval often wastes iterations instead of broadening evidence.

Main risk
- Some repeated queries are legitimate when the system wants the same information under a different retriever or cutoff.

Judge focus
- Require fewer redundant searches and more distinct evidence, not just more rewritten queries.

## H59: KiRAG Triple-Bridge Retrieval

Hypothesis
- Triple-level iterative retrieval over extracted subject-relation-object facts will improve multi-hop recall when the missing evidence is relational rather than purely lexical.

Implementation sketch
- Build a side index of lightweight knowledge triples extracted from chunks during ingestion.
- During iterative retrieval, search triples first to find bridge facts, then fetch the source chunks behind the best triple matches.
- Merge triple-backed chunks with the normal chunk retriever and compare against the plain text-only pipeline.
- Paper source: [KiRAG: Knowledge-Driven Iterative Retriever for Enhancing Retrieval-Augmented Generation](https://arxiv.org/abs/2502.18397)

Expected upside
- KiRAG argues that triple-based iterative retrieval can bridge information gaps, adapt to evolving information needs, and reduce disruption from irrelevant documents.

Main risk
- Triple extraction noise can be severe on tables, messy HTML, or narrative passages where relations are implicit rather than explicit.

Judge focus
- Inspect multi-hop recall and whether the retrieved bridge facts are actually helpful to the final answer step.

### Token-Guard Related Baselines

## H60: Self-Reflection Hallucination Check

Hypothesis
- A dedicated self-reflection pass that checks whether the draft answer is supported by the retrieved evidence will reduce unsupported claims without needing a full retrieval redesign.

Implementation sketch
- After drafting an answer, prompt the model to mark unsupported, ambiguous, or contradictory claims against the current context.
- If unsupported content is detected, either revise once with the critique attached or trigger one targeted follow-up retrieval.
- Keep the reflection output short and schema-bound so it acts as a control signal instead of a free-form essay.
- Paper source: [Towards Mitigating Hallucination in Large Language Models via Self-Reflection](https://arxiv.org/abs/2310.06271)

Expected upside
- This gives a lightweight post-hoc faithfulness check that can catch answer errors even when retrieval itself was adequate.

Main risk
- Self-critique can rationalize the current answer instead of correcting it, adding latency without reducing hallucinations.

Judge focus
- Inspect unsupported-claim rate and correction quality, not just whether answers become longer or more cautious.

## H61: Self-Refine Answer Revision Loop

Hypothesis
- A short draft-feedback-rewrite loop will improve factual precision and completeness over a single-pass answer on context-heavy QA.

Implementation sketch
- Generate an initial answer, then ask the same model for concise feedback focused on omissions, unsupported claims, and contradictions.
- Run one or two refinement rounds that must act on that feedback while preserving grounded evidence.
- Keep this distinct from multi-agent critique by using one model and a tightly bounded revision loop.
- Paper source: [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)

Expected upside
- Self-Refine is cheap to prototype and may recover partially correct first drafts without changing retrieval or requiring extra models.

Main risk
- Refinement loops often paraphrase the same mistake, and repeated passes can drift away from the evidence.

Judge focus
- Verify that later drafts fix concrete factual errors or omissions rather than merely sounding cleaner.

## H62: Token-Guard API-Compatible Span Checking

Hypothesis
- An API-only approximation of Token-Guard, using logprobs and prompt-based span checks instead of hidden states, will reduce hallucinations more efficiently than regenerating the whole answer.

Implementation sketch
- Approximate Token-Guard with API-visible signals only: token logprobs if available, otherwise sentence- or clause-level self-check prompts and sampled agreement.
- Split the answer into short spans, score each span for support and logical consistency against the retrieved context, and regenerate only the weakest spans inside a local window.
- Allow at most one global retry after local repairs so cost stays bounded.
- Paper source: [Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](https://arxiv.org/html/2601.21969v2)

Expected upside
- Local repair can preserve the strong parts of an answer while focusing correction budget on the spans most likely to hallucinate.

Main risk
- The full Token-Guard method relies on token-level hidden-state and latent-space signals that many API-only models do not expose, so this approximation may be materially weaker or blocked by missing logprobs.

Judge focus
- Measure unsupported-span rate, edit locality, and cost per corrected answer, not only final exact-match accuracy.

### Newly Requested Papers

## H63: TreeQA Logic-Tree Retrieval

Hypothesis
- Decomposing multi-hop questions into a hierarchical logic tree of verifiable sub-questions, then retrieving evidence for each node, will improve reliability and interpretability over one-shot flat retrieval.

Implementation sketch
- Build a prompt-only logic tree where the root is the original question and children are simpler sub-questions.
- Retrieve supporting evidence per node from the existing corpus, then validate and propagate answers upward through the tree.
- Add a lightweight self-correction pass when a node lacks sufficient evidence or conflicts with sibling results.
- Paper source: [TreeQA: Enhanced LLM-RAG with logic tree reasoning for reliable and interpretable multi-hop question answering](https://doi.org/10.1016/j.knosys.2025.114526)

Expected upside
- This can turn opaque multi-hop retrieval into a stepwise evidence trail, making both retrieval errors and reasoning errors easier to localize.

Main risk
- Bad question decomposition can poison the whole tree and add substantial latency.

Judge focus
- Inspect node-level evidence support and whether the tree actually reduces multi-hop failure modes rather than just lengthening traces.

## H64: CottonBot Static + Live Evidence Router

Hypothesis
- Combining static document retrieval with explicit live-tool evidence for time-sensitive or sensor-like queries will outperform text-only RAG on actionable questions.

Implementation sketch
- Keep the normal document retriever for stable knowledge.
- Add a routing step that detects when the question depends on live numeric state or current external conditions and calls deterministic tools in addition to retrieval.
- Merge the static retrieved guidance and live tool outputs into one grounded response format.
- Paper source: [CottonBot: An AI-driven cotton farming assistant and irrigation advisor using LLM-RAG and agentic AI tools](https://doi.org/10.1016/j.atech.2025.101640)

Expected upside
- CottonBot shows that static corpus retrieval alone is insufficient when the answer must combine stored guidance with current external measurements.

Main risk
- Tool routing mistakes and heterogeneous evidence formats can make answers less coherent than text-only retrieval.

Judge focus
- Evaluate whether live-tool calls are triggered only when needed and whether the mixed evidence leads to measurably better grounded decisions.

## H65: Vul-RAG Knowledge-Card Retrieval

Hypothesis
- Retrieving distilled knowledge cards, not just raw passages, will improve specialized reasoning by surfacing root-cause and remediation structure that plain text retrieval often misses.

Implementation sketch
- During ingestion, distill each source document or example into a compact structured card such as `problem`, `cause`, `evidence`, and `resolution`.
- Retrieve over these cards first, then optionally fetch the underlying raw chunk for supporting detail.
- Compare card-first retrieval against direct raw-text retrieval on explanation-heavy questions.
- Paper source: [Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG](https://arxiv.org/abs/2406.11147)

Expected upside
- Vul-RAG’s core claim is that knowledge-level retrieval can surface the underlying causal structure more effectively than raw example matching.

Main risk
- Distilled cards can oversimplify or hallucinate structure during ingestion, causing retrieval to become confidently wrong.

Judge focus
- Check whether retrieved cards improve explanation quality and accuracy together, rather than merely producing cleaner-looking summaries.

## H66: T-RAG Hierarchical Table Retrieval

Hypothesis
- For table-heavy corpora, hierarchical memory indexing plus multi-stage retrieval will outperform flat chunk retrieval by preserving intra-table and inter-table structure.

Implementation sketch
- Build a staged table-aware index with at least table-level and row-or-cell-level representations.
- Retrieve candidate tables first, then drill down to the most relevant rows, cells, or linked tables before final answer generation.
- Add prompt formatting that preserves table relations rather than flattening everything into plain prose.
- Paper source: [RAG over Tables: Hierarchical Memory Index, Multi-Stage Retrieval, and Benchmarking](https://arxiv.org/abs/2504.01346)

Expected upside
- The T-RAG framework argues that table corpora need structure-aware retrieval rather than generic text chunk search.

Main risk
- The extra indexing and retrieval stages may add overhead without helping on corpora that are mostly narrative text.

Judge focus
- Evaluate table-answer recall, retrieval latency, and whether the retrieved context preserves the table relations needed for inference.

## H67: BioRAG Taxonomy-Augmented Retrieval

Hypothesis
- Augmenting retrieval with domain hierarchy metadata and iterative query decomposition will improve specialized scientific QA over generic vector search alone.

Implementation sketch
- Add domain taxonomy or hierarchy labels to documents and chunks during ingestion.
- Expand or rerank retrieval using those labels so semantically close but hierarchically relevant evidence gets promoted.
- For questions that likely need fresh or multi-step evidence, decompose the query and run iterative retrieval over both the index and any configured search source.
- Paper source: [BioRAG: A RAG-LLM Framework for Biological Question Reasoning](https://arxiv.org/abs/2408.01107)

Expected upside
- BioRAG shows that specialized QA can benefit from both domain-specific representation and explicit knowledge hierarchy instead of relying on generic similarity alone.

Main risk
- Poor taxonomy coverage or noisy tagging can become a brittle filter that hides relevant evidence.

Judge focus
- Check whether hierarchy-aware retrieval improves specialized recall on hard domain terms without sharply hurting general queries.

## H68: Coordinated Semantic Alignment + Evidence Constraints

Hypothesis
- Jointly improving query-evidence semantic alignment and constraining generation to the selected evidence will reduce semantic drift and unsupported claims better than optimizing retrieval and generation separately.

Implementation sketch
- Add a reranking stage that explicitly scores whether retrieved evidence matches the generation objective, not just topical similarity.
- Convert selected evidence into a stronger control signal during answering, for example by requiring claim-to-evidence alignment, citation slots, or evidence-bounded drafting.
- Compare this coordinated setup against the current looser handoff from retrieval to generation.
- Paper source: [Coordinated Semantic Alignment and Evidence Constraints for Retrieval-Augmented Generation with Large Language Models](https://arxiv.org/abs/2603.04647)

Expected upside
- The paper’s core claim is that semantic alignment and evidence constraints work better when modeled together, reducing both noisy retrieval effects and generation drift.

Main risk
- Over-constraining generation can make answers brittle, extractive, or unable to synthesize across multiple pieces of evidence.

Judge focus
- Measure groundedness and factual support rates together with answer completeness and fluency.

### Requested 2026-03 API-Compatible Additions

Repository note
- The cards below keep only techniques that are testable with API-accessible models and the current corpora. Fine-tuning-only variants, hidden-state-only controls, and logprobe-dependent methods are omitted or converted into prompt-level approximations.

## H69: AlignRAG Evidence-Grounded Critique Loop

Hypothesis
- A retrieval-aware critique pass that identifies evidence-misaligned reasoning and rewrites the answer will improve groundedness and multi-hop accuracy over single-pass RAG.

Implementation sketch
- Generate a first draft answer from the retrieved evidence.
- Ask a structured critic prompt to flag unsupported reasoning steps, missing bridge facts, and evidence misuse, then produce a short evidence-grounded revision plan.
- Run one or two revise passes using that plan before finalizing the answer.
- Paper source: [Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization](https://arxiv.org/html/2504.14858v4)

Expected upside
- AlignRAG shows that critique-guided test-time refinement can improve reasoning fidelity even when the retrieval set is unchanged.

Main risk
- The critique can become generic or self-affirming, adding cost without materially correcting the answer.

Judge focus
- Check whether revisions remove unsupported claims and fix missing bridge facts rather than merely restyling the answer.

## H70: Phase-Aware Misalignment Router

Hypothesis
- Classifying failures into retrieval relevance, query-evidence mapping, or evidence-synthesis errors and routing each case to a different recovery action will outperform a single generic retry rule.

Implementation sketch
- After the first draft, ask the critic to assign one dominant failure type: `relevance`, `mapping`, or `synthesis`.
- If the issue is `relevance`, expand or rerank retrieval; if `mapping`, generate a targeted bridge query; if `synthesis`, revise the answer without new retrieval.
- Compare this routed recovery policy against a generic one-size-fits-all self-reflection pass.
- Paper source: [Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization](https://arxiv.org/html/2504.14858v4)

Expected upside
- The paper frames RAG failure as multi-stage misalignment, so targeted recovery may fix more errors with fewer wasted retries.

Main risk
- Misclassification can send the system down the wrong recovery path and amplify failure.

Judge focus
- Compare retry count, latency, and whether each route actually fixes the failure mode it was assigned.

## H71: Dynamic Critique Stopping

Hypothesis
- A structured `GOOD` or `BAD` critic verdict can stop critique loops early and retain most of the gains of iterative refinement without fixed extra calls on every query.

Implementation sketch
- Require the critic output to begin with a bounded verdict token and one-sentence justification.
- Stop refinement when the verdict is `GOOD` or the groundedness score crosses a threshold; otherwise allow one additional revise pass.
- Log stop rates by benchmark family and question type.
- Paper source: [Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization](https://arxiv.org/html/2504.14858v4)

Expected upside
- AlignRAG-auto reports that dynamic stopping can preserve accuracy while reducing unnecessary refinement steps.

Main risk
- Poorly calibrated verdicts can halt too early on subtle errors or loop on already-good answers.

Judge focus
- Measure answer quality together with average calls per query, not quality alone.

## H72: Dual Reconstruction Answer Reranker

Hypothesis
- Among multiple answer candidates, the one that best reconstructs masked query constraints from the answer plus known query context will be more reliable than the highest-likelihood candidate.

Implementation sketch
- Sample two to four answer candidates for the same question.
- Decompose the question into known anchors and unknown target slots, then build dual prompts that reconstruct the unknown slots from each candidate answer plus the known anchors.
- Score candidates by reconstruction fidelity and use the best one, or abstain if none reconstruct cleanly.
- Paper source: [DuPO: Enabling Reliable LLM Self-Verification via Dual Preference Optimization](https://arxiv.org/abs/2508.14460)

Expected upside
- The paper reports strong gains from inference-time reranking, which makes this the most direct API-compatible DuPO derivative.

Main risk
- Reconstruction prompts can reward verbose candidates that leak clues without being truly correct.

Judge focus
- Compare candidate-selection win rate, final accuracy, and extra compute cost versus single-candidate decoding.

## H73: Known-Unknown Query Decomposition Checks

Hypothesis
- Explicitly decomposing each question into known and unknown components before retrieval and again during verification will preserve bridge constraints better than free-form query rewriting.

Implementation sketch
- Extract known anchors such as entities, time constraints, relation frame, and answer type, plus unknown target slots.
- Use the anchors to guide retrieval and use the target slots to define a post-answer consistency check that reconstructs what the answer was supposed to resolve.
- If one slot fails reconstruction, trigger a targeted follow-up retrieval only for that missing slot.
- Paper source: [DuPO: Enabling Reliable LLM Self-Verification via Dual Preference Optimization](https://arxiv.org/abs/2508.14460)

Expected upside
- DuPO’s generalized duality gives a concrete structure for self-verification instead of generic self-reflection.

Main risk
- Slot extraction can be brittle on comparison, yes-no, or heavily compositional questions.

Judge focus
- Inspect whether bridge facts, time constraints, and target entities are preserved across retrieval and answer generation.

## H74: EIRE Semantic Retrieval Profile

Hypothesis
- Extracting explicit entity, intent, relation, and evidence expectations from the query and scoring candidates against that profile will improve retrieval precision and groundedness.

Implementation sketch
- Use a short LLM parser to derive an `EIRE` profile from the user query.
- Score retrieved chunks by semantic agreement with those fields before final top-`k` selection.
- Use profile mismatch as a reranking penalty instead of relying only on dense or lexical similarity.
- Paper source: [SeCon-RAG: Stemming Hallucination in RAG Systems via Semantic and Conflict Filtering](https://arxiv.org/abs/2510.09710)

Expected upside
- SeCon-RAG argues that early semantic misalignment is a major driver of hallucinated downstream synthesis.

Main risk
- Profile extraction errors can suppress semantically correct but differently phrased evidence.

Judge focus
- Check false negatives on paraphrased evidence and whether retrieval precision improves enough to help the final answer.

## H75: EIRE-Gated Clean Pool Admission

Hypothesis
- Building a cleaner second-stage candidate pool by clustering retrieved chunks and retaining only EIRE-consistent clusters will beat flat top-`k` truncation on noisy or distractor-heavy questions.

Implementation sketch
- Retrieve a wider candidate pool.
- Cluster similar chunks or group them by document, then keep only clusters with strong EIRE alignment and reasonable source diversity.
- Feed the cleaned pool into the existing answer step without changing generation prompts.
- Paper source: [SeCon-RAG: Stemming Hallucination in RAG Systems via Semantic and Conflict Filtering](https://arxiv.org/abs/2510.09710)

Expected upside
- This tests the paper’s two-stage filtering idea in a way that fits the current pipeline.

Main risk
- Important minority evidence can sit in a small cluster and get dropped by the cleaner.

Judge focus
- Compare bridge-fact recall, evidence diversity, and contradiction rate in the final prompt.

## H76: Conflict-Free Evidence Filter

Hypothesis
- Removing or downweighting retrieved chunks that contradict the query’s semantic profile or the dominant evidence set will reduce hallucinated synthesis and improve abstention quality.

Implementation sketch
- After retrieval, compare candidate chunks for contradiction against the query profile and against each other.
- Suppress conflicting chunks or split them into competing evidence bundles that force abstention or targeted follow-up retrieval.
- Keep the final answer step blind to rejected chunks so contradictory evidence cannot leak back into generation.
- Paper source: [SeCon-RAG: Stemming Hallucination in RAG Systems via Semantic and Conflict Filtering](https://arxiv.org/abs/2510.09710)

Expected upside
- SeCon-RAG’s second stage directly targets conflict-induced hallucinations rather than only relevance.

Main risk
- Real ambiguity can be mistaken for noise, lowering answer completeness.

Judge focus
- Measure contradiction handling, abstention precision, and whether completeness drops on legitimately ambiguous questions.

## H77: Licensed Claim Emission

Hypothesis
- Requiring each answer claim to be licensed by a structured support fact or extracted triple before emission will sharply reduce false positives on factoid and multi-hop QA.

Implementation sketch
- Convert retrieved evidence into lightweight structured facts or triples during retrieval or prompt preprocessing.
- Extract atomic claims from the draft answer and validate each one against the structured evidence set.
- Emit only licensed claims, otherwise revise or abstain.
- Paper source: [Stemming Hallucination in Language Models Using a Licensing Oracle](https://arxiv.org/abs/2511.06073)

Expected upside
- The Licensing Oracle shows that deterministic validation can outperform plain RAG when the domain has enough structure to support licensing.

Main risk
- Fact extraction or normalization can miss valid support and over-trigger abstention.

Judge focus
- Track false-answer rate, abstention precision, and claim coverage instead of relying on EM alone.

## H78: Oracle-Backed Abstention Gate

Hypothesis
- A final abstention decision based on licensed-claim coverage will outperform confidence-only refusal heuristics when evidence is incomplete or contradictory.

Implementation sketch
- Require a minimum licensed-claim coverage threshold before the system is allowed to answer.
- If key claim slots remain unlicensed, return an explicit `INSUFFICIENT_EVIDENCE` response instead of a low-confidence guess.
- Tune thresholds separately for entity, binary, and comparison questions.
- Paper source: [Stemming Hallucination in Language Models Using a Licensing Oracle](https://arxiv.org/abs/2511.06073)

Expected upside
- The paper reports perfect abstention precision and zero false answers, which makes abstention quality the most interesting transferable idea.

Main risk
- A conservative gate can suppress partially answerable questions and reduce exact-match metrics.

Judge focus
- Evaluate false-answer rate and abstention precision alongside the usual answer metrics.

## H79: Clause-Level Acceptance and Repair

Hypothesis
- Scoring answer clauses by support, local coherence, and overall context alignment, then repairing only the mid-confidence clauses, will outperform whole-answer retries.

Implementation sketch
- Draft the answer as short clauses or sentences.
- Score each clause with prompt-based support and coherence checks, mirroring Token-Guard’s accept-refine-discard segment logic without hidden-state access.
- Accept strong clauses, locally repair middle-band clauses, and discard weak ones before final assembly.
- Paper source: [Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](https://arxiv.org/html/2601.21969v2)

Expected upside
- This preserves the good parts of an answer while focusing correction budget on the weakest factual spans.

Main risk
- Clause boundaries may not align with the true dependency structure of the answer.

Judge focus
- Compare unsupported-span rate, edit locality, and token cost rather than only final exact match.

## H80: Global Answer Chain Selection

Hypothesis
- Building the final answer from the most mutually coherent supported clauses across one or more drafts, with abstention when the global chain score stays low, will improve grounded multi-step answers.

Implementation sketch
- Sample two or three compact drafts or repaired clause sets.
- Re-rank clause chains by support coverage and inter-clause coherence.
- Return the best chain if it clears a threshold, otherwise abstain or trigger one targeted retrieval retry.
- Paper source: [Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](https://arxiv.org/html/2601.21969v2)

Expected upside
- Token-Guard’s global iteration stage suggests many hallucinations are chain-level, not isolated local errors.

Main risk
- Fragment recomposition can produce awkward or incomplete answers even when each clause is individually supported.

Judge focus
- Inspect logical consistency across multi-hop answers and the quality of abstentions, not just surface factuality.

## H81: Persistent Query State Contract

Hypothesis
- Carrying forward explicit per-thread constraints such as entities, time, metric, units, and source policy will improve MTRAG multi-turn accuracy by preventing retrieval drift across turns.

Implementation sketch
- Maintain a structured state object across turns and merge each new turn into it.
- Use the merged state to rewrite retrieval queries, bias reranking, and filter evidence before prompting.
- Keep state updates auditable so bad carryover can be analyzed quickly.
- Tool source: [ContextGuard](https://github.com/ahmedjawedaj/contextguard)

Expected upside
- ContextGuard is designed around state-contracted retrieval, which directly matches multi-turn benchmark failure modes.

Main risk
- Stale constraints can overconstrain follow-up turns that intentionally shift topic.

Judge focus
- Check whether later turns preserve the correct entities and timeframes without reducing flexibility on topic shifts.

## H82: Hard Constraint Evidence Gate

Hypothesis
- Rejecting retrieved chunks that violate persistent state constraints before answer generation will beat soft reranking alone on grounded multi-turn QA.

Implementation sketch
- Apply hard filters or large penalties for entity, time, or source-policy mismatches.
- Enforce lightweight diversity rules so one noisy source cannot monopolize the prompt.
- Log rejection reason codes for each gated chunk.
- Tool source: [ContextGuard](https://github.com/ahmedjawedaj/contextguard)

Expected upside
- ContextGuard’s gate explicitly aims to stop off-contract evidence before it contaminates the answer step.

Main risk
- Metadata extraction errors can incorrectly block the only useful chunk.

Judge focus
- Compare evidence precision, contradiction rate, and final answer accuracy, not retrieval recall alone.

## H83: Support + Counter-Evidence Retrieval

Hypothesis
- Planning both supportive and contradictory retrieval queries for each claim will reduce confirmation bias and improve robustness on comparison and conflict-heavy questions.

Implementation sketch
- Split the intended answer into one or more atomic claims.
- Retrieve supporting evidence and explicit counter-evidence in parallel for each claim.
- Feed both evidence types into the existing judge or answer-selection step before finalizing the response.
- Tool source: [ContextGuard](https://github.com/ahmedjawedaj/contextguard)

Expected upside
- ContextGuard treats counter-evidence retrieval as a first-class primitive instead of assuming retrieval should only confirm the current hypothesis.

Main risk
- Counter-evidence queries can add noise when the original evidence was already sufficient and unambiguous.

Judge focus
- Check whether contradiction handling improves without a large recall collapse or token explosion.

## H84: Coverage-Aware Claim Verdict Aggregation

Hypothesis
- Aggregating per-claim support and contradiction judgments into `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT`, or `MIXED` will improve calibration and abstention quality over a single scalar confidence score.

Implementation sketch
- Split the final answer into atomic claims.
- Score each claim against accepted evidence, then aggregate using coverage and contradiction counts.
- Use the aggregate label to decide whether to answer, abstain, or return a qualified answer with caveats.
- Tool source: [ContextGuard](https://github.com/ahmedjawedaj/contextguard)

Expected upside
- Claim-level aggregation is a better control signal for faithfulness-focused metrics than raw model confidence.

Main risk
- Claim splitting and aggregation heuristics can become another brittle layer in the pipeline.

Judge focus
- Evaluate calibration, abstention quality, and faithfulness, not just EM or F1.

