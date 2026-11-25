// AI chat endpoint for Emma. The API key lives only in Vercel environment
// variables (GEMINI_API_KEY); it is never sent to the browser.
const hits = new Map();

function rateLimited(ip) {
  const now = Date.now();
  const list = (hits.get(ip) || []).filter((t) => now - t < 60000);
  list.push(now);
  hits.set(ip, list);
  return list.length > 12;
}

module.exports = async (req, res) => {
  const key = (process.env.GEMINI_API_KEY || '').trim();
  if (req.method === 'GET') {
    return res.status(200).json({ enabled: !!key });
  }
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  if (!key) {
    return res.status(503).json({ error: 'AI chưa bật. Hãy đặt GEMINI_API_KEY trong Vercel.' });
  }
  const ip = String(req.headers['x-forwarded-for'] || 'unknown').split(',')[0].trim();
  if (rateLimited(ip)) {
    return res.status(429).json({ error: 'Bạn gửi hơi nhanh. Hãy chờ một chút nhé.' });
  }
  const body = typeof req.body === 'string' ? JSON.parse(req.body || '{}') : (req.body || {});
  const message = String(body.message || '').slice(0, 500).trim();
  if (!message) {
    return res.status(400).json({ error: 'Thiếu nội dung tin nhắn.' });
  }
  const history = Array.isArray(body.history) ? body.history.slice(-6) : [];
  const historyText = history
    .map((h) => `${h.role === 'model' ? 'Emma' : 'Learner'}: ${String(h.text || '').slice(0, 300)}`)
    .join('\n');
  const prompt = [
    'You are Emma, a friendly English tutor for a Vietnamese beginner (A1) who also learns math English.',
    'Reply in English only, in 1-2 very short and simple sentences, mostly present simple.',
    'If the learner makes a mistake, correct it gently in the reply, for example: "Good try. Say: I am fine."',
    'Keep the conversation going with one simple question when it fits.',
    historyText ? `Conversation so far:\n${historyText}` : '',
    `Learner: ${message}`,
    'Emma:',
  ].filter(Boolean).join('\n');

  const model = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
  try {
    const callModel = (m) => fetch(`https://generativelanguage.googleapis.com/v1beta/models/${m}:generateContent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
      body: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.4, maxOutputTokens: 120 },
      }),
    });
    let upstream = await callModel(model);
    if (upstream.status === 404 && model !== 'gemini-2.0-flash') {
      upstream = await callModel('gemini-2.0-flash');
    }
    if (!upstream.ok) {
      return res.status(502).json({ error: `AI đang bận (mã ${upstream.status}). Hãy thử lại sau nhé.` });
    }
    const data = await upstream.json();
    const reply = (data.candidates?.[0]?.content?.parts || []).map((p) => p.text || '').join('').trim();
    if (!reply) {
      return res.status(502).json({ error: 'AI chưa trả lời được. Hãy thử lại nhé.' });
    }
    return res.status(200).json({ reply: reply.slice(0, 600) });
  } catch (e) {
    return res.status(502).json({ error: 'AI đang bận. Hãy thử lại sau nhé.' });
  }
};
