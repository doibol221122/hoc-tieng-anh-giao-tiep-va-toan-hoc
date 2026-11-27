// Progress sync endpoint for the English app.
// Supabase credentials live only in Vercel environment variables:
// SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY. The service-role key is
// used server-side only and is never sent to the browser.
const hits = new Map();

function rateLimited(ip) {
  const now = Date.now();
  const list = (hits.get(ip) || []).filter((t) => now - t < 60000);
  list.push(now);
  hits.set(ip, list);
  return list.length > 60;
}

function config() {
  const url = (process.env.SUPABASE_URL || '').replace(/\/$/, '');
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || '';
  return { url, key, enabled: Boolean(url && key) };
}

function validDeviceId(value) {
  return typeof value === 'string' && /^[A-Za-z0-9_-]{8,80}$/.test(value);
}

function cleanKnown(value) {
  if (!Array.isArray(value)) return null;
  const out = [];
  const seen = new Set();
  for (const item of value) {
    const n = Number(item);
    if (!Number.isInteger(n) || n < 0 || n > 10000 || seen.has(n)) continue;
    seen.add(n);
    out.push(n);
    if (out.length > 5000) return null;
  }
  return out.sort((a, b) => a - b);
}

async function supabaseFetch(path, options = {}) {
  const { url, key } = config();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    return await fetch(`${url}${path}`, {
      ...options,
      signal: controller.signal,
      headers: {
        apikey: key,
        Authorization: `Bearer ${key}`,
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
    });
  } finally {
    clearTimeout(timer);
  }
}

module.exports = async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  const ip = String(req.headers['x-forwarded-for'] || req.socket?.remoteAddress || 'unknown').split(',')[0].trim();
  if (rateLimited(ip)) {
    return res.status(429).json({ enabled: true, error: 'Bạn thao tác hơi nhanh. Hãy chờ một chút nhé.' });
  }

  const { enabled } = config();
  if (!enabled) {
    return res.status(200).json({ enabled: false, error: 'Supabase chưa bật. Tiến độ vẫn lưu trên máy này.' });
  }

  if (req.method === 'GET') {
    const deviceId = req.query?.deviceId;
    if (!validDeviceId(deviceId)) return res.status(400).json({ enabled: true, error: 'Mã thiết bị chưa đúng.' });
    const response = await supabaseFetch(`/rest/v1/english_app_progress?device_id=eq.${encodeURIComponent(deviceId)}&select=device_id,xp,known,speech_mode,updated_at&limit=1`);
    if (!response.ok) return res.status(502).json({ enabled: true, error: 'Chưa đọc được tiến độ từ Supabase.' });
    const rows = await response.json();
    const row = rows[0];
    return res.status(200).json({
      enabled: true,
      progress: row ? { xp: row.xp, known: row.known, speechMode: row.speech_mode, updatedAt: row.updated_at } : null,
    });
  }

  if (req.method === 'POST') {
    const body = req.body || {};
    const deviceId = body.deviceId;
    const xp = Number(body.xp);
    const known = cleanKnown(body.known);
    const speechMode = ['slow', 'normal', 'fast'].includes(body.speechMode) ? body.speechMode : 'slow';
    if (!validDeviceId(deviceId) || !Number.isInteger(xp) || xp < 0 || xp > 1000000 || !known) {
      return res.status(400).json({ enabled: true, error: 'Dữ liệu tiến độ chưa đúng.' });
    }
    const response = await supabaseFetch('/rest/v1/english_app_progress?on_conflict=device_id', {
      method: 'POST',
      headers: { Prefer: 'resolution=merge-duplicates,return=representation' },
      body: JSON.stringify({ device_id: deviceId, xp, known, speech_mode: speechMode, updated_at: new Date().toISOString() }),
    });
    if (!response.ok) return res.status(502).json({ enabled: true, error: 'Chưa lưu được tiến độ lên Supabase.' });
    const rows = await response.json();
    const row = rows[0] || {};
    return res.status(200).json({ enabled: true, saved: true, updatedAt: row.updated_at || null });
  }

  return res.status(405).json({ enabled: true, error: 'Phương thức chưa được hỗ trợ.' });
};
