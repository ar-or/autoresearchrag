import { sessionStore } from "../sessions/store";
import { runAgent } from "../agent/agent";

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export async function createSession(): Promise<Response> {
  const session = sessionStore.createSession();
  return json({ session_id: session.id });
}

export async function listSessions(): Promise<Response> {
  const sessions = sessionStore.listSessions().map((s) => ({
    id: s.id,
    message_count: s.messages.length,
    createdAt: s.createdAt,
    updatedAt: s.updatedAt,
  }));
  return json({ sessions });
}

export async function getSession(id: string): Promise<Response> {
  const session = sessionStore.getSession(id);
  if (!session) return json({ error: "Session not found" }, 404);
  return json(session);
}

export async function deleteSession(id: string): Promise<Response> {
  const deleted = sessionStore.deleteSession(id);
  if (!deleted) return json({ error: "Session not found" }, 404);
  return json({ deleted: true });
}

export async function sendMessage(req: Request): Promise<Response> {
  const body = await req.json();
  const { session_id, message } = body as { session_id?: string; message?: string };

  if (!session_id || !message) {
    return json({ error: "session_id and message are required" }, 400);
  }

  const session = sessionStore.getSession(session_id);
  if (!session) return json({ error: "Session not found" }, 404);

  sessionStore.addMessage(session_id, {
    role: "user",
    content: message,
    timestamp: new Date().toISOString(),
  });

  const response = await runAgent(session_id, message, session.messages.slice(0, -1));

  sessionStore.addMessage(session_id, {
    role: "assistant",
    content: response,
    timestamp: new Date().toISOString(),
  });

  return json({ session_id, response });
}
