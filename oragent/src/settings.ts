export const settings = {
  /** LLM model name used by the agent */
  model: process.env.ORAGENT_MODEL ?? "gpt-5-mini",

  /** System prompt for the agent */
  systemPrompt:
    process.env.ORAGENT_SYSTEM_PROMPT ??
    "You are a helpful AI assistant.",

  /** HTTP server port */
  port: Number(process.env.ORAGENT_PORT ?? 32522),
} as const;
