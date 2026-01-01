import React, { useMemo, useRef, useState } from "react";

type Msg = { role: "user" | "bot"; content: string };

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([
    {
      role: "bot",
      content: "Hi! Ask me about Promtior (services, founding, blog, etc.).",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatRef = useRef<HTMLDivElement | null>(null);

  const canSend = useMemo(
    () => input.trim().length > 0 && !loading,
    [input, loading]
  );

  function scrollToBottom() {
    requestAnimationFrame(() => {
      chatRef.current?.scrollTo({
        top: chatRef.current.scrollHeight,
        behavior: "smooth",
      });
    });
  }

  function pushMessage(msg: Msg) {
    setMessages((prev) => {
      const next = [...prev, msg];
      return next;
    });
    scrollToBottom();
  }

  async function send() {
    const q = input.trim();
    if (!q || loading) return;

    setInput("");
    pushMessage({ role: "user", content: q });
    setLoading(true);

    try {
      const res = await fetch("/rag/invoke", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // LangServe: simple string chain expects {"input": "..."}
        body: JSON.stringify({ input: q }),
      });

      if (!res.ok) {
        const text = await res.text();
        pushMessage({ role: "bot", content: `Error ${res.status}\n${text}` });
        return;
      }

      const data = await res.json();
      const out = data.output ?? JSON.stringify(data, null, 2);
      pushMessage({ role: "bot", content: out });
    } catch (e: any) {
      pushMessage({ role: "bot", content: `Error: ${String(e)}` });
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  }

  function clearChat() {
    setMessages([
      { role: "bot", content: "Cleared. Ask me something about Promtior." },
    ]);
    setInput("");
  }

  return (
    <div style={styles.page}>
      <div style={styles.wrap}>
        <header style={styles.header}>
          <div>
            <h1 style={styles.title}>Promtior RAG Chat</h1>
            <div style={styles.subtitle}>
              Frontend (Vite+React) + Backend (FastAPI/LangServe)
            </div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <a
              href="/rag/playground"
              target="_blank"
              rel="noreferrer"
              style={styles.linkBtn}
            >
              Playground
            </a>
            <button onClick={clearChat} style={styles.btn} disabled={loading}>
              Clear
            </button>
          </div>
        </header>

        <div ref={chatRef} style={styles.chat}>
          {messages.map((m, idx) => (
            <div
              key={idx}
              style={{
                ...styles.msg,
                ...(m.role === "user" ? styles.user : styles.bot),
              }}
            >
              {m.content}
            </div>
          ))}
          {loading && (
            <div style={{ ...styles.msg, ...styles.bot }}>Thinking…</div>
          )}
        </div>

        <div style={styles.row}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => (e.key === "Enter" ? send() : null)}
            placeholder="Ask something…"
            style={styles.input}
          />
          <button onClick={send} disabled={!canSend} style={styles.btn}>
            Send
          </button>
        </div>

        <div style={styles.hint}>
          Tip: This UI keeps short “memory” only in the browser (chat history).
          Backend remains single-turn RAG.
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: { minHeight: "100vh", background: "#0b0b0f", color: "#eee" },
  wrap: {
    maxWidth: 980,
    margin: "0 auto",
    padding: 24,
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  title: { margin: 0, fontSize: 20 },
  subtitle: { opacity: 0.75, fontSize: 12, marginTop: 4 },
  chat: {
    border: "1px solid #222",
    borderRadius: 12,
    padding: 12,
    background: "#111",
    height: "70vh",
    overflow: "auto",
  },
  msg: {
    padding: "10px 12px",
    margin: "10px 0",
    borderRadius: 10,
    maxWidth: "85%",
    whiteSpace: "pre-wrap",
    lineHeight: 1.35,
  },
  user: { background: "#1f2937", marginLeft: "auto" },
  bot: { background: "#111827", border: "1px solid #1f2937" },
  row: { display: "flex", gap: 10 },
  input: {
    flex: 1,
    padding: 12,
    borderRadius: 10,
    border: "1px solid #222",
    background: "#0f0f14",
    color: "#eee",
    outline: "none",
  },
  btn: {
    padding: "12px 14px",
    borderRadius: 10,
    border: "1px solid #222",
    background: "#18181b",
    color: "#eee",
    cursor: "pointer",
  },
  linkBtn: {
    padding: "12px 14px",
    borderRadius: 10,
    border: "1px solid #222",
    background: "#0f0f14",
    color: "#eee",
    textDecoration: "none",
    display: "inline-flex",
    alignItems: "center",
  },
  hint: { opacity: 0.7, fontSize: 12 },
};
