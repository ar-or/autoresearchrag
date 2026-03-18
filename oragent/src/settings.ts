export const settings = {
  /** LLM model name used by the agent */
  model: process.env.ORAGENT_MODEL ?? "gpt-5-mini",

  /** System prompt for the agent */
  systemPrompt:
    process.env.ORAGENT_SYSTEM_PROMPT ??
    "You are a helpful AI assistant.",

  /** HTTP server port */
  port: Number(process.env.ORAGENT_PORT ?? 32522),

  /** Elasticsearch retrieval */
  elasticUrl: process.env.ES_HOST ?? "http://localhost:9200",
  elasticApiKey: process.env.ELASTIC_API_KEY ?? "",
  elasticIndex: process.env.ES_INDEX ?? "mtrag",
  retrievalK: Number(process.env.RETRIEVAL_K ?? 5),
} as const;
