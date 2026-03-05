import { createSession, listSessions, getSession, deleteSession, sendMessage } from "./handlers/chat";

function corsHeaders(): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

function withCors(res: Response): Response {
  for (const [k, v] of Object.entries(corsHeaders())) {
    res.headers.set(k, v);
  }
  return res;
}

export async function router(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const { pathname } = url;
  const method = req.method;

  // CORS preflight
  if (method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders() });
  }

  try {
    let res: Response;

    if (method === "POST" && pathname === "/api/chat/session") {
      res = await createSession();
    } else if (method === "GET" && pathname === "/api/chat/sessions") {
      res = await listSessions();
    } else if (method === "POST" && pathname === "/api/chat/send-message") {
      res = await sendMessage(req);
    } else if (pathname.startsWith("/api/chat/session/")) {
      const id = pathname.slice("/api/chat/session/".length);
      if (method === "GET") {
        res = await getSession(id);
      } else if (method === "DELETE") {
        res = await deleteSession(id);
      } else {
        res = new Response(JSON.stringify({ error: "Method not allowed" }), {
          status: 405,
          headers: { "Content-Type": "application/json" },
        });
      }
    } else {
      res = new Response(JSON.stringify({ error: "Not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    }

    return withCors(res);
  } catch (err) {
    console.error("Request error:", err);
    return withCors(
      new Response(JSON.stringify({ error: "Internal server error" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }),
    );
  }
}
