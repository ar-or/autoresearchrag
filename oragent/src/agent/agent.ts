import { createLangfuseHandler } from "../tracing/otel";
import type { ChatMessage, RetrievedContext } from "../sessions/store";
import { createDeepAgent } from "deepagents";
import { ChatOpenAI } from "@langchain/openai";
import { settings } from "../settings";
import { HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { createRetrievalTool, retrieve } from "./retriever";

function toLC(messages: ChatMessage[]) {
  return messages.map((m) =>
    m.role === "user" ? new HumanMessage(m.content) : new AIMessage(m.content),
  );
}

export interface TokenUsage {
  input_tokens: number;
  cached_tokens: number;
  output_tokens: number;
}

export interface AgentResult {
  content: string;
  contexts: RetrievedContext[];
  usage: TokenUsage;
}

export async function runAgent(
  sessionId: string,
  userMessage: string,
  history: ChatMessage[],
): Promise<AgentResult> {
  const langfuseHandler = createLangfuseHandler(sessionId);
  const retrievalTool = createRetrievalTool();

  // Always retrieve before the agent runs
  const contexts = await retrieve(userMessage);

  // Build context block for the agent prompt
  let augmentedMessage = userMessage;
  if (contexts.length > 0) {
    const contextBlock = contexts
      .map((c, i) => `[${i + 1}] ${c.title ? c.title + ": " : ""}${c.text}`)
      .join("\n\n");
    augmentedMessage =
      `Retrieved documents:\n${contextBlock}\n\nUser question: ${userMessage}`;
  }

  try {
    const llm = new ChatOpenAI({ model: settings.model });
    const agent = createDeepAgent({
      model: llm,
      systemPrompt: settings.systemPrompt,
      tools: [retrievalTool],
    });

    const messages = [
      ...toLC(history),
      new HumanMessage(augmentedMessage),
    ];

    const result = await agent.invoke(
      { messages },
      { callbacks: [langfuseHandler] },
    );

    // Collect any additional contexts from agent tool calls
    const additionalContexts: RetrievedContext[] = [];
    if (result?.messages && Array.isArray(result.messages)) {
      for (const msg of result.messages) {
        const name = msg?.name ?? msg?.kwargs?.name ?? "";
        if (msg instanceof ToolMessage || msg?.constructor?.name === "ToolMessage" || name === "search_documents") {
          try {
            const parsed = JSON.parse(msg.content as string);
            if (Array.isArray(parsed)) {
              for (const item of parsed) {
                if (item?.text) {
                  additionalContexts.push({
                    document_id: item.document_id ?? "",
                    text: item.text,
                    title: item.title ?? "",
                    score: item.score ?? 0,
                  });
                }
              }
            }
          } catch {}
        }
      }
    }

    // Accumulate token usage from all AI messages
    const usage: TokenUsage = { input_tokens: 0, cached_tokens: 0, output_tokens: 0 };
    let content: string;
    if (result?.messages && Array.isArray(result.messages)) {
      for (const msg of result.messages) {
        // LangChain stores usage in usage_metadata (snake_case) and response_metadata.tokenUsage (camelCase)
        const um = msg?.usage_metadata ?? msg?.kwargs?.usage_metadata;
        if (um) {
          usage.input_tokens += um.input_tokens ?? 0;
          usage.output_tokens += um.output_tokens ?? 0;
          usage.cached_tokens += um.input_token_details?.cache_read ?? 0;
        }
      }
      const lastAI = [...result.messages].reverse().find(
        (m: any) => m?.constructor?.name === "AIMessage" || m?.kwargs?.type === "ai" || m?.type === "ai",
      );
      content = lastAI?.content ?? lastAI?.kwargs?.content ?? JSON.stringify(result);
    } else if (typeof result === "string") {
      content = result;
    } else {
      content = result?.content ?? result?.output ?? JSON.stringify(result);
    }

    return { content, contexts: [...contexts, ...additionalContexts], usage };
  } finally {
    await langfuseHandler.shutdownAsync?.().catch(() => {});
  }
}
