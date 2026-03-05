import { NodeSDK } from "@opentelemetry/sdk-node";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { CallbackHandler } from "@langfuse/langchain";

const baseUrl = process.env.LANGFUSE_HOST ?? "http://localhost:3000";

// Initialize OTEL SDK with Langfuse span processor
const sdk = new NodeSDK({
  spanProcessors: [
    new LangfuseSpanProcessor({
      baseUrl,
    }),
  ],
});
sdk.start();

/**
 * Create a Langfuse CallbackHandler for a given session.
 * Pass this to LangChain .invoke() calls to capture full LLM
 * input/output/tokens/cost automatically.
 */
export function createLangfuseHandler(sessionId: string) {
  return new CallbackHandler({
    sessionId,
    baseUrl,
  });
}
