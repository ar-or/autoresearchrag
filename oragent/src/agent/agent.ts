import { createLangfuseHandler } from "../tracing/otel";
import type { ChatMessage } from "../sessions/store";
import { createDeepAgent } from "deepagents";
import { ChatOpenAI } from "@langchain/openai";
import { settings } from "../settings";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";

function toLC(messages: ChatMessage[]) {
  return messages.map((m) =>
    m.role === "user" ? new HumanMessage(m.content) : new AIMessage(m.content),
  );
}

export async function runAgent(
  sessionId: string,
  userMessage: string,
  history: ChatMessage[],
): Promise<string> {
  const langfuseHandler = createLangfuseHandler(sessionId);

  try {
    const llm = new ChatOpenAI({ model: settings.model });
    const agent = createDeepAgent({
      model: llm,
      systemPrompt: settings.systemPrompt,
    });

    const messages = [
      ...toLC(history),
      new HumanMessage(userMessage),
    ];

    const result = await agent.invoke(
      { messages },
      { callbacks: [langfuseHandler] },
    );

    // DeepAgent returns { messages, todos, files } — extract last AI message
    let content: string;
    if (result?.messages && Array.isArray(result.messages)) {
      const lastAI = [...result.messages].reverse().find(
        (m: any) => m?.constructor?.name === "AIMessage" || m?.kwargs?.type === "ai" || m?.type === "ai",
      );
      content = lastAI?.content ?? lastAI?.kwargs?.content ?? JSON.stringify(result);
    } else if (typeof result === "string") {
      content = result;
    } else {
      content = result?.content ?? result?.output ?? JSON.stringify(result);
    }

    return content;
  } catch (err: any) {
    // Fallback: use @langchain/openai directly if DeepAgent fails
    console.error("DeepAgent error, falling back to ChatOpenAI:", err.message);
    const llm = new ChatOpenAI({ model: settings.model });
    const messages = [
      new SystemMessage(settings.systemPrompt),
      ...toLC(history),
      new HumanMessage(userMessage),
    ];
    const response = await llm.invoke(messages, { callbacks: [langfuseHandler] });
    const content =
      typeof response.content === "string"
        ? response.content
        : JSON.stringify(response.content);
    return content;
  } finally {
    await langfuseHandler.shutdownAsync?.().catch(() => {});
  }
}
