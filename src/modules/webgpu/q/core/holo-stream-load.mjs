// core/holo-stream-load.mjs — H1 of HOLO-BITNET-CANONICAL-INSTANT: load a model from ONE `.holo`
// (packed by forge/holo-kappa-pack.mjs) instead of 330 loose b/<κ>.gz blocks. Returns the SAME
// { manifest, fetchTensor, info } contract as loadKappaObject, PLUS { tokenizerBytes } (embedded),
// so core/loader.js can drop it in and createQvacGPU / the KV path stay untouched.
//
// WHY: the drawer + q-chat cold-load = 330 HTTP requests to HF (latency-bound, the #1 felt defect —
// DRAWER-TRUTH T2). The `.holo` is ONE Range-streamable file: header (1 req) → bodies by absolute
// offset, coalesced + prefetched, each block BLAKE3-verified over its STORED (gzip) bytes BEFORE
// gunzip (Law L5, cheapest). WARM: the whole file persists to OPFS on first load; later boots read
// it locally with re-verify (never trusted). Cold falls back to the parts path in loader.js on any
// throw (fail-soft). Bodies are byte-identical to the parts blocks, so output is byte-identical.
//
// W1 (HOLO-Q-WAFER): the cold path serves reads PROGRESSIVELY while the single GET is still in
// flight — read(s,l) resolves the moment byte s+l has arrived, and the pack order is FIRST-USE
// order, so the engine builds + uploads weights PIPELINED BEHIND the download (the model "plays
// like video"). Before W1 the cold boot paid download → build → prefill strictly in sequence;
// now openHoloKappa returns at ~header-arrival and engine-ready ≈ last-byte-arrival (zero tail).
// Honest physics: the FIRST TOKEN still needs every layer once (a transformer forward pass walks
// all layers), so W1 collapses the post-download tail to ~0 — it does not beat the wire.
import { reshapeTensor, buildEngineManifest } from "../holo-load2bit.mjs";

const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
async function gunzip(u8) { const ds = new DecompressionStream("gzip"); const w = ds.writable.getWriter(); w.write(u8); w.close(); return new Uint8Array(await new Response(ds.readable).arrayBuffer()); }

// ── OPFS single-file warm store: the whole .holo, keyed by URL, re-verified against header on read ──
const OPFS_DIR = "holo-model-files";
const fname = (url) => String(url).replace(/[^a-zA-Z0-9._-]/g, "_").slice(-140);
async function opfsFile(url) {
  try {
    if (typeof navigator === "undefined" || !navigator.storage || !navigator.storage.getDirectory) return null;
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(OPFS_DIR, { create: true });
    return await dir.getFileHandle(fname(url) + ".holo", { create: true });
  } catch { return null; }
}

export async function openHoloKappa(url, opts = {}) {
  const onProgress = opts.onProgress || (() => {});
  const range = async (reader, s, l) => reader(s, l);

  // W3 (Q-WAFER) ORIGIN RACING: every block is content-addressed, so origins are interchangeable
  // BY CONSTRUCTION — the mirror list is just "places the same bytes live". We race the mirrors'
  // first response, stream from the winner, and rotate the Range fallback through survivors when
  // any origin stalls mid-flight. Zero servers: every origin is dumb static hosting.
  const urls = [url, ...(opts.mirrors || [])].filter(Boolean);
  let originIdx = 0;                                          // the origin currently believed fastest

  // Prefer a fully-resident OPFS copy (warm path); else HTTP Range (cold). Both expose read(off,len).
  // Warm key stays the PRIMARY url — mirrors serve identical bytes, one OPFS copy is the truth.
  let fh = await opfsFile(url), warmFile = null;
  if (fh) { try { const f = await fh.getFile(); if (f.size > 0) warmFile = f; } catch {} }

  const httpRead = async (s, l) => {
    let lastErr;
    for (let k = 0; k < urls.length; k++) {
      const u = urls[(originIdx + k) % urls.length];
      try {
        const r = await fetch(u, { headers: { Range: `bytes=${s}-${s + l - 1}` } });
        if (!(r.ok || r.status === 206)) throw new Error("holo range " + r.status + " @" + s);
        originIdx = (originIdx + k) % urls.length;            // stick with the origin that worked
        return new Uint8Array(await r.arrayBuffer());
      } catch (e) { lastErr = e; }
    }
    throw lastErr;
  };

  // ── W1 PROGRESSIVE COLD STREAM ──────────────────────────────────────────────────────────────
  // ONE plain GET starts IMMEDIATELY (before the header is even parsed — the header rides the
  // first chunk, so cold = exactly 1 request). Bytes land in wholeBuf; read(s,l) resolves as soon
  // as byte s+l has arrived (bytes below `got` are immutable — subarrays are final). Pack order is
  // first-use order, so the engine's sequential fetchTensor calls pipeline behind the wire.
  // Fail-soft is preserved: a host without plain GET (or a mid-stream error) wakes all waiters and
  // every un-arrived read falls back to the per-Range path — same behavior as before W1.
  let wholeBuf = null, got = 0, streamDone = !!warmFile, streamErr = null, knownTotal = 0;
  const waiters = [];
  const wake = () => {
    for (let i = waiters.length - 1; i >= 0; i--) {
      const w = waiters[i];
      if (got >= w.need) { waiters.splice(i, 1); w.resolve(); }
      else if (streamErr) { waiters.splice(i, 1); w.reject(streamErr); }
      else if (streamDone) { waiters.splice(i, 1); w.resolve(); }   // truncated: read() rechecks + falls back
    }
  };
  const arrived = (need) => new Promise((resolve, reject) => {
    if (got >= need || streamDone) return resolve();
    if (streamErr) return reject(streamErr);
    waiters.push({ need, resolve, reject });
  });
  if (!warmFile) {
    (async () => {
      // Race every origin's FIRST RESPONSE (headers = the honest "who is fastest" signal), stream
      // the body from the winner, abort the losers. A blackhole origin can't hang the race: the
      // collar aborts everything still pending once the timeout passes (then per-Range fail-soft).
      let resp;
      if (urls.length === 1) {
        resp = await fetch(url);                              // one origin — no race to run
      } else {
        const ctls = urls.map(() => new AbortController());
        const collar = setTimeout(() => ctls.forEach((c) => { try { c.abort(); } catch {} }), opts.raceTimeoutMs || 30000);
        const attempts = urls.map((u, i) => (async () => {
          const r = await fetch(u, { signal: ctls[i].signal });
          if (!r.ok) throw new Error("holo GET " + r.status + " @" + u);
          return { r, i };
        })());
        try {
          const win = await Promise.any(attempts);
          resp = win.r; originIdx = win.i;                    // Range fallback starts at the proven-fast origin
          for (let i = 0; i < ctls.length; i++) if (i !== win.i) { try { ctls[i].abort(); } catch {} }
        } finally { clearTimeout(collar); }
        attempts.forEach((p) => p.catch(() => {}));           // losers reject on abort — never unhandled
      }
      if (!resp.ok) throw new Error("holo GET " + resp.status);
      const clen = Number(resp.headers.get("content-length")) || 0;
      let cap = clen || (64 << 20);
      wholeBuf = new Uint8Array(cap);
      const tot = () => knownTotal || clen || 0;
      if (resp.body && resp.body.getReader) {
        const rd = resp.body.getReader();
        for (;;) {
          const { done, value } = await rd.read(); if (done) break;
          if (got + value.length > cap) {                     // no content-length: grow (old subarrays stay valid views)
            cap = Math.max(cap * 2, got + value.length);
            const nb = new Uint8Array(cap); nb.set(wholeBuf.subarray(0, got)); wholeBuf = nb;
          }
          wholeBuf.set(value, got); got += value.length; onProgress(got, tot()); wake();
        }
      } else { const all = new Uint8Array(await resp.arrayBuffer()); wholeBuf = all; got = all.length; onProgress(got, tot()); }
      streamDone = true; wake();
      // persist the COMPLETE file to OPFS for the next visit (fire-and-forget; warm re-verifies anyway)
      if (fh && opts.persist !== false && got > 0) {
        try { const w = await fh.createWritable(); await w.write(wholeBuf.subarray(0, got)); await w.close(); } catch {}
      }
    })().catch((e) => {
      streamErr = e; wake();
      try { console.warn("[holo] whole-file stream failed, per-range fallback:", e && e.message || e); } catch {}
    });
  }
  const read = warmFile
    ? async (s, l) => new Uint8Array(await warmFile.slice(s, s + l).arrayBuffer())
    : async (s, l) => {
        if (got >= s + l) return wholeBuf.subarray(s, s + l);
        try { await arrived(s + l); } catch {}
        if (got >= s + l) return wholeBuf.subarray(s, s + l);
        return httpRead(s, l);                                // stream failed/truncated → per-range fail-soft
      };

  // header: magic(6) + u32 len + JSON
  const head = await read(0, 10);
  if (new TextDecoder().decode(head.slice(0, 6)) !== "HOLK1\n") throw new Error("bad .holo magic");
  const hlen = new DataView(head.buffer, head.byteOffset).getUint32(6, true);
  const H = JSON.parse(new TextDecoder().decode(await read(10, hlen)));
  if (H.format !== "holo-kappa/1") throw new Error("unknown .holo format " + H.format);

  // canonical BLAKE3 verifier (κ = blake3 over STORED gz bytes). Falls back to no-verify only if wasm
  // is unavailable — but the whole point is L5, so require it when present (it always is in the app).
  let b3 = null;
  try { const fn = (await import("../pkg/holospaces_web.js")).kappa; if (fn(new Uint8Array([1])).startsWith("blake3:")) b3 = fn; } catch {}

  const kvcRecs = (H.sealed && H.sealed.kvc) || [];
  const total = 10 + hlen + H.tokenizer.len + kvcRecs.reduce((a, r) => a + r.len, 0) + H.dir.reduce((a, r) => a + r.len, 0);
  knownTotal = total;                                          // pump progress now reports the true total

  // reconstruct the raw manifest (top-level fields + tensors dict) for buildEngineManifest
  const man = { ...H.manifest };
  const normRecs = {};
  const dirByName = {};
  for (const rec of H.dir) { normRecs[rec.name] = { N: rec.N, K: rec.K, fmt: rec.fmt, ...(rec.s !== undefined ? { s: rec.s } : {}) }; dirByName[rec.name] = rec; }
  man.tensors = normRecs;

  // e8 LUT (if any) — a named dir entry; fetch+verify like a block
  const getBody = async (rec) => {
    const gz = await read(rec.off, rec.len);
    if (b3) { const got = b3(gz); if (got !== rec.kappa) throw new Error("BLAKE3 κ MISMATCH " + rec.name + " " + rec.kappa.slice(0, 20)); }
    return await gunzip(gz);
  };
  let e8lutData;
  if (man.e8lut) { const r = H.dir.find((d) => d.kappa === man.e8lut || d.name === "e8lut"); if (r) { const b = await getBody(r); e8lutData = new Float32Array(b.buffer, b.byteOffset, 2048); } }

  const manifest = buildEngineManifest(man, normRecs, e8lutData);

  // W1 honest marks: at what % of the wire each pipeline event happened (rides ?stats + witnesses)
  const marks = { firstTensorPct: null, tensors: 0 };
  const pct = () => (streamDone || warmFile) ? 100 : (knownTotal ? Math.min(100, Math.round(got / knownTotal * 100)) : 0);

  const fetchTensor = async (name) => {
    const rec = dirByName[name]; if (!rec) return new Uint8Array(0);
    const body = await getBody(rec);
    if (marks.firstTensorPct == null) marks.firstTensorPct = pct();
    marks.tensors++;
    return reshapeTensor({ N: rec.N, K: rec.K, fmt: rec.fmt, ...(rec.s !== undefined ? { s: rec.s } : {}) }, body);
  };

  // embedded tokenizer bytes (the loader uses these directly → no separate HF header fetch)
  const tokenizerBytes = await read(H.tokenizer.off, H.tokenizer.len);
  man.source = url;                                            // informational; tokenizer comes from tokenizerBytes

  // W2 (Q-WAFER) SEALED KV COMMONS: persona-prefill K/V minted once on a real GPU, shipped in the
  // wafer — a FRESH device restores the persona state instead of computing it. Entries are keyed by
  // (layout, nIds, idsSha); bodies ride right after the tokenizer (early bytes). kvc κ = sha256
  // TRANSPORT axis over the stored gz (this section is packed without the blake3 manifest); verified
  // with crypto.subtle before gunzip — same L5 shape, refuse on mismatch.
  const sealed = kvcRecs.length ? {
    kvc: kvcRecs.map((rec) => ({
      layout: rec.layout, L: rec.L, nIds: rec.nIds, idsSha: rec.idsSha,
      get: async () => {
        const gz = await read(rec.off, rec.len);
        if (String(rec.kappa || "").startsWith("sha256:")) {
          const got = "sha256:" + hex(await crypto.subtle.digest("SHA-256", gz));
          if (got !== rec.kappa) throw new Error("kvc κ MISMATCH " + rec.layout);
        }
        return await gunzip(gz);
      },
    })),
  } : null;

  const progress = () => ({ got: warmFile ? total : got, total, done: streamDone, pct: pct(), marks });
  return { manifest, fetchTensor, info: man, tokenizerBytes, warm: !!warmFile, progress, sealed };
}
