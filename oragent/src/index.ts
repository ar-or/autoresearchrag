// Initialize OTEL tracing before anything else
import "./tracing/otel";

import { router } from "./router";
import { settings } from "./settings";

const server = Bun.serve({
  port: settings.port,
  fetch: router,
});

console.log(`oragent listening on http://localhost:${server.port} (model: ${settings.model})`);
