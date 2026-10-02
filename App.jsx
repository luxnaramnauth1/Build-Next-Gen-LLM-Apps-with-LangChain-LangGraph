import { useEffect, useRef, useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';

// One random session id per page load, so different runs can be told apart on the backend.
const SESSION_ID = crypto.randomUUID();
const transport = new DefaultChatTransport({ api: '/api/chat', body: { sessionId: SESSION_ID } });

const textOf = (m) => m.parts.filter((p) => p.type === 'text').map((p) => p.text).join('');
const timeOf = (d) => d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

export default function App() {
  const { messages, sendMessage, status } = useChat({ transport });
  const [input, setInput] = useState('');
  const [fb, setFb] = useState({});          // messageId -> { rating, flagged, comment, note, explainOpen, draft }
  const [stats, setStats] = useState(null);  // session stats from the backend
  const stamps = useRef({});                 // messageId -> Date first seen
  const bottom = useRef(null);

  const busy = status === 'submitted' || status === 'streaming';
  messages.forEach((m) => { stamps.current[m.id] ??= new Date(); });

  const loadStats = () =>
    fetch(`/api/feedback/stats?sessionId=${SESSION_ID}`).then((r) => r.json()).then((d) => setStats(d.session)).catch(() => {});

  useEffect(() => { loadStats(); }, []);
  useEffect(() => { if (status === 'ready') loadStats(); }, [status]);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const patch = (id, p) => setFb((s) => ({ ...s, [id]: { ...s[id], ...p } }));

  // Sends the FULL feedback state for one message (structured payload per the lab spec).
  async function submitFeedback(id, next) {
    const cur = { rating: null, flagged: false, comment: '', ...fb[id], ...next };
    patch(id, { ...next, note: 'Saving...' });
    try {
      const res = await fetch('/api/feedback', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messageId: id, rating: cur.rating, flagged: cur.flagged,
          comment: cur.comment || null, timestamp: new Date().toISOString(), sessionId: SESSION_ID,
        }),
      });
      if (!res.ok) throw new Error('bad status');
      const data = await res.json();
      setStats(data.stats);
      patch(id, { note: 'Feedback saved' });
    } catch {
      patch(id, { note: 'Try again' });
    }
  }

  const rate = (id, value) => submitFeedback(id, { rating: fb[id]?.rating === value ? null : value });
  const flag = (id) => submitFeedback(id, { flagged: !fb[id]?.flagged });
  const send = (text) => { if (text.trim() && !busy) { sendMessage({ text }); setInput(''); } };

  return (
    <div className="app">
      <main className="chat">
        <header><h1>Support Assistant</h1><span className="sub">Your feedback helps improve answers</span></header>

        <div className="list" role="log" aria-live="polite">
          {messages.length === 0 && (
            <div className="empty">
              Ask something, for example:
              <div className="chips">
                {['Can I change my subscription plan next month?', 'Can I get a refund?', 'Where do I find my invoice?'].map((q) => (
                  <button key={q} className="chip" onClick={() => send(q)}>{q}</button>))}
              </div>
            </div>)}

          {messages.map((m, i) => {
            const mine = m.role === 'user';
            const f = fb[m.id] || {};
            const streaming = !mine && busy && i === messages.length - 1;
            return (
              <div key={m.id} className={`row ${mine ? 'user' : 'bot'}`}>
                <div className="bubble">
                  <div className="meta">{mine ? 'You' : 'Assistant'} · {timeOf(stamps.current[m.id])}</div>
                  <div className="text">{textOf(m)}{streaming && <span className="cursor">▍</span>}</div>

                  {!mine && !streaming && (
                    <div className="fb" aria-label="Feedback controls">
                      <button className={`ic ${f.rating === 'up' ? 'on up' : ''}`} aria-pressed={f.rating === 'up'}
                              title="Helpful answer" aria-label="Thumbs up" onClick={() => rate(m.id, 'up')}>👍</button>
                      <button className={`ic ${f.rating === 'down' ? 'on down' : ''}`} aria-pressed={f.rating === 'down'}
                              title="Not helpful" aria-label="Thumbs down" onClick={() => rate(m.id, 'down')}>👎</button>
                      <button className={`txt ${f.flagged ? 'on flag' : ''}`} aria-pressed={!!f.flagged}
                              title="Report a wrong or unsafe answer" onClick={() => flag(m.id)}>⚑ {f.flagged ? 'Flagged' : "Something's wrong"}</button>
                      <button className="txt" title="Tell us what felt wrong" onClick={() => patch(m.id, { explainOpen: !f.explainOpen })}>Explain why</button>
                      <button className="txt" title="Ask the assistant to rephrase" disabled={busy}
                              onClick={() => send('Can you clarify that last answer?')}>Ask to clarify</button>
                      {f.note && <span className={`note ${f.note === 'Try again' ? 'err' : ''}`} role="status">{f.note}</span>}
                    </div>)}

                  {!mine && f.explainOpen && (
                    <div className="explain">
                      <textarea rows={2} maxLength={1000} placeholder="What was wrong or unclear?" value={f.draft ?? f.comment ?? ''}
                                onChange={(e) => patch(m.id, { draft: e.target.value })} />
                      <button className="send small" onClick={() => { submitFeedback(m.id, { comment: f.draft ?? '' }); patch(m.id, { explainOpen: false }); }}>Submit</button>
                    </div>)}
                </div>
              </div>);
          })}
          <div ref={bottom} />
        </div>

        <div className="composer">
          <input value={input} placeholder="Type your message and press Enter" disabled={busy}
                 onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && send(input)} />
          <button className="send" onClick={() => send(input)} disabled={busy || !input.trim()}>Send</button>
        </div>
      </main>

      <aside className="panel" aria-label="Feedback summary">
        <h2>Feedback Summary</h2>
        <dl>
          <dt>Total responses</dt><dd id="m-total">{stats?.total_responses ?? 0}</dd>
          <dt>Thumbs up</dt><dd id="m-up">{stats?.thumbs_up ?? 0}</dd>
          <dt>Thumbs down</dt><dd id="m-down">{stats?.thumbs_down ?? 0}</dd>
          <dt>Flagged</dt><dd id="m-flag">{stats?.flagged ?? 0}</dd>
          <dt>Comments</dt><dd>{stats?.comments ?? 0}</dd>
          <dt>Feedback rate</dt><dd>{Math.round((stats?.feedback_rate ?? 0) * 100)}%</dd>
        </dl>
        <p className="hint">Counts come from the backend, so they confirm your feedback was actually stored.</p>
        <p className="sid">Session {SESSION_ID.slice(0, 8)}</p>
      </aside>
    </div>
  );
}
