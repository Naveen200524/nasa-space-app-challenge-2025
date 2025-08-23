export class ApiClient {
  constructor(base = "http://127.0.0.1:5000") {
    this.base = base.replace(/\/$/, "");
    this.timeoutMs = 12000; // conservative default
    // Simple in-memory cache and backoff for production readiness
    this._cache = new Map(); // key -> { ts, data }
    this._cacheTTLms = 5 * 60 * 1000; // 5 minutes
    this._cooldownUntil = 0;
    this._backoffMs = 5000; // start at 5s
    this._maxBackoffMs = 60000; // cap at 60s
  }

  // Internal: timeout wrapper for fetch
  async _fetch(url, options = {}) {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      return res;
    } finally {
      clearTimeout(t);
    }
  }

  _now() { return Date.now(); }
  _cacheGet(key) {
    const v = this._cache.get(key);
    if (!v) return null;
    if (this._now() - v.ts > this._cacheTTLms) { this._cache.delete(key); return null; }
    return v.data;
  }
  _cacheSet(key, data) { this._cache.set(key, { ts: this._now(), data }); }
  _inCooldown() { return this._now() < this._cooldownUntil; }
  _noteFailure() {
    this._cooldownUntil = this._now() + this._backoffMs;
    this._backoffMs = Math.min(this._backoffMs * 2, this._maxBackoffMs);
  }
  _noteSuccess() { this._cooldownUntil = 0; this._backoffMs = 5000; }

  // Detect endpoint: accepts a FormData instance
  // Optional cacheKey enables basic response caching and request de-duping by callers
  async detect(formData, { cacheKey } = {}) {
    if (cacheKey) {
      const cached = this._cacheGet(cacheKey);
      if (cached) return cached;
    }
    if (this._inCooldown()) throw new Error('backend_cooldown');

    try {
      const res = await this._fetch(`${this.base}/detect`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Detect failed: ${res.status}`);
      const json = await res.json();
      if (cacheKey) this._cacheSet(cacheKey, json);
      this._noteSuccess();
      return json;
    } catch (e) {
      this._noteFailure();
      throw e;
    }
  }

  // High-level helpers
  async health() {
    const res = await this._fetch(`${this.base}/health`);
    if (!res.ok) throw new Error(`health ${res.status}`);
    return res.json();
  }

  async earthquakesRecent({ magnitude = 4.0, hours = 24, sources = 'usgs,emsc' } = {}) {
    const params = new URLSearchParams({ magnitude, hours, sources });
    const res = await this._fetch(`${this.base}/earthquakes/recent?${params.toString()}`);
    if (!res.ok) throw new Error(`recent ${res.status}`);
    return res.json();
  }

  // Convenience: IRIS fetch via /detect
  async detectIris({ network, station, channel, starttime, endtime, planet = 'earth' }, { cacheKey } = {}) {
    const fd = new FormData();
    fd.append('source', 'iris');
    fd.append('network', network);
    fd.append('station', station);
    fd.append('channel', channel);
    fd.append('starttime', starttime);
    fd.append('endtime', endtime);
    fd.append('planet', planet);
    return this.detect(fd, { cacheKey });
  }

  // Convenience: NASA PDS search via /detect
  async detectPdsSearch({ mission = 'insight', instrument = 'SEIS', planet = 'mars' } = {}, { cacheKey } = {}) {
    const fd = new FormData();
    fd.append('source', 'pds_search');
    fd.append('mission', mission);
    fd.append('instrument', instrument);
    fd.append('planet', planet);
    return this.detect(fd, { cacheKey });
  }
}

