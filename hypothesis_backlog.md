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

## Summary

| id | name |
| --- | --- |
| H01 | Hybrid Vector + Text Retrieval |
| H02 | Document-Diversified Top-K |
| H03 | Table-Preserving FinQA Ingestion |
| H04 | Structure-Aware Chunk Boundaries |
| H05 | Adaptive Retrieval for Numeric Queries |
| H06 | Neighbor Chunk Expansion |
| H07 | Context Packing and Dedup |

## H01: Hybrid Vector + Text Retrieval

Hypothesis
- Combining dense vector retrieval with lexical text retrieval and fusing the rankings will improve FinQA recall for exact metric names, entity names, and numeric cues.

Implementation sketch
- Run both vector search and text search for the same query.
- Fuse the results with a simple reciprocal-rank or score-normalized merge.
- Deduplicate by chunk id before returning the final top `k`.

Expected upside
- The current code falls back to text search only on failure. This hypothesis makes keyword-sensitive retrieval first-class instead of accidental.

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
