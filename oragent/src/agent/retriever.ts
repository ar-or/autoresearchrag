import { OpenAIEmbeddings } from "@langchain/openai";
import { DynamicTool } from "@langchain/core/tools";
import { settings } from "../settings";
import type { RetrievedContext } from "../sessions/store";

const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });

async function esSearch(indexName: string, queryVector: number[], k: number): Promise<any[]> {
  const url = `${settings.elasticUrl}/${indexName}/_search`;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (settings.elasticApiKey) {
    headers["Authorization"] = `ApiKey ${settings.elasticApiKey}`;
  }
  const body = {
    size: k,
    query: {
      script_score: {
        query: { match_all: {} },
        script: {
          source: "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
          params: { query_vector: queryVector },
        },
      },
    },
    _source: { excludes: ["embedding"] },
  };
  const resp = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    // Fallback to simple text search if vector search fails
    return esTextSearch(indexName, "", k);
  }
  const data = await resp.json() as any;
  return data.hits?.hits ?? [];
}

async function esTextSearch(indexName: string, query: string, k: number): Promise<any[]> {
  const url = `${settings.elasticUrl}/${indexName}/_search`;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (settings.elasticApiKey) {
    headers["Authorization"] = `ApiKey ${settings.elasticApiKey}`;
  }
  const body = {
    size: k,
    query: query ? { multi_match: { query, fields: ["text", "title", "content"] } } : { match_all: {} },
  };
  const resp = await fetch(url, { method: "POST", headers, body: JSON.stringify(body) });
  if (!resp.ok) return [];
  const data = await resp.json() as any;
  return data.hits?.hits ?? [];
}

export async function retrieve(
  query: string,
  indexName?: string,
  k?: number,
): Promise<RetrievedContext[]> {
  const idx = indexName ?? settings.elasticIndex;
  const topK = k ?? settings.retrievalK;

  let hits: any[] = [];
  try {
    const vector = await embeddings.embedQuery(query);
    hits = await esSearch(idx, vector, topK);
  } catch {
    try {
      hits = await esTextSearch(idx, query, topK);
    } catch {
      return [];
    }
  }

  return hits.map((hit) => ({
    document_id: hit._source?.document_id ?? hit._source?.id ?? hit._id ?? "",
    text: hit._source?.text ?? hit._source?.content ?? hit._source?.pageContent ?? "",
    title: hit._source?.title ?? "",
    score: hit._score ?? 0,
  }));
}

export function createRetrievalTool(indexName?: string) {
  return new DynamicTool({
    name: "search_documents",
    description:
      "Search the knowledge base for relevant passages. Input should be a search query string.",
    func: async (query: string) => {
      try {
        const results = await retrieve(query, indexName);
        return JSON.stringify(
          results.map((r, i) => ({
            rank: i + 1,
            document_id: r.document_id,
            title: r.title,
            text: r.text,
            score: r.score,
          })),
        );
      } catch {
        return JSON.stringify([]);
      }
    },
  });
}
