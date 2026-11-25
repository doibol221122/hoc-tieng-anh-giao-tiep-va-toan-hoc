// AI endpoint for Emma. The API key lives only in Vercel environment
// variables (GEMINI_API_KEY); it is never sent to the browser.
const hits = new Map();
const MODEL_CACHE_MS = 10 * 60 * 1000;
let modelCache = { at: 0, candidates: [], working: null, bad: {} };

const MODES = {
  chat: {
    temperature: 0.35,
    maxOutputTokens: 500,
    instruction: 'Reply in English only, in 1-2 very short A1 sentences. Correct mistakes gently. Ask one simple question when it fits.',
  },
  correct: {
    temperature: 0.2,
    maxOutputTokens: 600,
    instruction: 'Correct the learner sentence. Output exactly two short lines: "Say: <correct English>" and "Why: <very short Vietnamese reason>". If it is already correct, repeat it after Say and write "Why: Câu này đúng rồi."',
  },
  explain: {
    temperature: 0.25,
    maxOutputTokens: 700,
    instruction: 'Explain the word or sentence for a Vietnamese A1 learner. Output three short lines: "Meaning: <Vietnamese meaning>", "Example: <simple English example>", "Say: <short English to repeat>".',
  },
  quiz: {
    temperature: 0.5,
    maxOutputTokens: 800,
    instruction: 'Make one very easy A1 English or math-English quiz from the learner topic. Output three short lines: "Question: ...", "A. ... B. ... C. ...", "Answer: ...".',
  },
};

function rateLimited(ip) {
  const now = Date.now();
  const list = (hits.get(ip) || []).filter((t) => now - t < 60000);
  list.push(now);
  hits.set(ip, list);
  return list.length > 20;
}

async function fetchWithTimeout(url, options, ms) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function preferredModels() {
  return [...new Set([
    process.env.GEMINI_MODEL,
    'gemini-2.5-flash',
    'gemini-flash-latest',
    'gemini-2.5-flash-lite',
    'gemini-flash-lite-latest',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-1.5-flash',
  ].filter(Boolean))];
}

function modelScore(name) {
  let score = 0;
  if (name.includes('flash')) score += 10;
  if (name.includes('2.5')) score += 4;
  if (name.includes('latest')) score += 3;
  if (name.includes('2.0')) score += 2;
  if (name.includes('pro')) score += 1;
  if (name.includes('lite')) score -= 1;
  if (name.includes('preview') || name.includes('exp')) score -= 3;
  return score;
}

async function getCandidates(key, force = false) {
  const now = Date.now();
  const preferred = preferredModels();
  if (!force && modelCache.working && now - modelCache.at < MODEL_CACHE_MS) {
    return [...new Set([modelCache.working, ...modelCache.candidates])].filter((m) => !(modelCache.bad[m] > now)).slice(0, 5);
  }
  if (!force && modelCache.candidates.length && now - modelCache.at < MODEL_CACHE_MS) {
    return modelCache.candidates.filter((m) => !(modelCache.bad[m] > now)).slice(0, 5);
  }
  let candidates = preferred.slice(0, 5);
  try {
    const listRes = await fetchWithTimeout('https://generativelanguage.googleapis.com/v1beta/models', {
      headers: { 'x-goog-api-key': key },
    }, 2500);
    if (listRes.ok) {
      const listData = await listRes.json();
      const available = (listData.models || [])
        .filter((m) => (m.supportedGenerationMethods || []).includes('generateContent'))
        .map((m) => String(m.name || '').replace(/^models\//, ''))
        .filter(Boolean);
      if (available.length) {
        const exact = preferred.filter((m) => available.includes(m));
        const scored = available
          .filter((m) => !exact.includes(m))
          .sort((a, b) => modelScore(b) - modelScore(a));
        candidates = [...new Set([...exact, ...scored, ...preferred])].filter((m) => !(modelCache.bad[m] > now)).slice(0, 5);
      }
    }
  } catch (e) { /* keep preferred candidates */ }
  modelCache = { at: now, candidates, working: force ? null : modelCache.working, bad: modelCache.bad };
  return candidates.slice(0, 5);
}

function buildContents(history, message) {
  const items = [];
  for (const h of history) {
    const role = h.role === 'model' ? 'model' : 'user';
    const text = String(h.text || '').slice(0, 300).trim();
    if (text) items.push({ role, text });
  }
  items.push({ role: 'user', text: message });
  const contents = [];
  for (const item of items) {
    if (!contents.length && item.role === 'model') continue;
    const last = contents[contents.length - 1];
    if (last && last.role === item.role) {
      last.parts[0].text += `\n${item.text}`;
    } else {
      contents.push({ role: item.role, parts: [{ text: item.text }] });
    }
  }
  return contents;
}

module.exports = async (req, res) => {
  const key = (process.env.GEMINI_API_KEY || '').trim();
  if (req.method === 'GET') {
    return res.status(200).json({ enabled: !!key, model: modelCache.working || null });
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
  const mode = MODES[body.mode] ? body.mode : 'chat';
  const cfg = MODES[mode];
  const history = Array.isArray(body.history) ? body.history.slice(-6) : [];
  const systemText = [
    'You are Emma, a friendly English tutor for a Vietnamese beginner (A1) who also learns math English.',
    'Use very simple words and mostly present simple. Be warm, clear, and brief.',
    cfg.instruction,
  ].join(' ');
  const payload = {
    systemInstruction: { parts: [{ text: systemText }] },
    contents: buildContents(history, message),
    generationConfig: {
      temperature: cfg.temperature,
      topP: 0.9,
      maxOutputTokens: cfg.maxOutputTokens,
      thinkingConfig: { thinkingBudget: 0 },
    },
  };

  const statuses = [];
  const attempted = new Set();
  const tryCandidates = async (candidates) => {
    for (const model of candidates) {
      if (attempted.has(model)) continue;
      attempted.add(model);
      try {
        const call = (body) => fetchWithTimeout(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
          body: JSON.stringify(body),
        }, 7000);
        let upstream = await call(payload);
        if (upstream.status === 400) {
          const retryPayload = JSON.parse(JSON.stringify(payload));
          delete retryPayload.generationConfig.thinkingConfig;
          upstream = await call(retryPayload);
        }
        if (!upstream.ok) {
          statuses.push(`${model}:${upstream.status}`);
          if (upstream.status === 404 || upstream.status === 400) modelCache.bad[model] = Date.now() + MODEL_CACHE_MS;
          if (upstream.status === 503) modelCache.bad[model] = Date.now() + 60000;
          if (modelCache.working === model && upstream.status === 404) modelCache.working = null;
          continue;
        }
        const data = await upstream.json();
        const reply = (data.candidates?.[0]?.content?.parts || []).map((p) => p.text || '').join('').trim();
        if (reply) {
          delete modelCache.bad[model];
          modelCache = { at: Date.now(), candidates: modelCache.candidates, working: model, bad: modelCache.bad };
          return { reply: reply.slice(0, 700), model, mode };
        }
        statuses.push(`${model}:empty`);
      } catch (e) {
        statuses.push(`${model}:${e && e.name === 'AbortError' ? 'timeout' : 'network'}`);
      }
    }
    return null;
  };

  let result = await tryCandidates(await getCandidates(key));
  if (!result && statuses.some((s) => s.endsWith(':404'))) {
    result = await tryCandidates(await getCandidates(key, true));
  }
  if (result) return res.status(200).json(result);
  return res.status(502).json({ error: `AI đang bận (${statuses.join(', ')}). Hãy thử lại sau nhé.` });
};
