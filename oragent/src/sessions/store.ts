export interface RetrievedContext {
  document_id: string;
  text: string;
  title: string;
  score: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  contexts?: RetrievedContext[];
}

export interface Session {
  id: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
}

declare var globalThis: {
  __oragent_sessions?: Map<string, Session>;
} & typeof global;

function getStore(): Map<string, Session> {
  if (!globalThis.__oragent_sessions) {
    globalThis.__oragent_sessions = new Map();
  }
  return globalThis.__oragent_sessions;
}

export class SessionStore {
  private store = getStore();

  createSession(): Session {
    const id = crypto.randomUUID();
    const now = new Date().toISOString();
    const session: Session = { id, messages: [], createdAt: now, updatedAt: now };
    this.store.set(id, session);
    return session;
  }

  getSession(id: string): Session | undefined {
    return this.store.get(id);
  }

  listSessions(): Session[] {
    return Array.from(this.store.values());
  }

  addMessage(sessionId: string, msg: ChatMessage): void {
    const session = this.store.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    session.messages.push(msg);
    session.updatedAt = new Date().toISOString();
  }

  deleteSession(id: string): boolean {
    return this.store.delete(id);
  }
}

export const sessionStore = new SessionStore();
