// core/parts-cache.mjs — WARM WEIGHTS (W1 of HOLO-INSTANT-STRANGER-PROMPT.md): the fetched +
// decompressed tensor parts of a κ-object model persist in OPFS after the FIRST load, so every
// later visit skips the network AND the gunzip entirely — disk → κ re-verify → GPU. Measured
// motive: the warm resident phase was partsMs 4.4 s (serial fetch+gunzip; concurrency already
// A/B'd and LOST — contention-bound, see qvac-gpu.js QLOAD note) vs gpuMs 0.25 s.
// Sealed like kv-commons: each part carries its content κ, re-derived on read; a mismatch purges
// the entry and falls back to the canonical fetch path (Law L5 — warm is never trusted).
import { kappa } from "../pkg/holospaces_web.js";

const enc = new TextEncoder();
const safe = (s) => String(s).replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 120);

export async function openPartsCache(modelKey) {
  try {
    if (typeof navigator === "undefined" || !navigator.storage || !navigator.storage.getDirectory) return null;
    const root = await navigator.storage.getDirectory();
    const dirName = "holo-q-parts_" + String(kappa(enc.encode(String(modelKey)))).replace(/[^a-zA-Z0-9]/g, "").slice(-24);
    const dir = await root.getDirectoryHandle(dirName, { create: true });
    let hits = 0, misses = 0;
    const api = {
      stats: () => ({ hits, misses }),
      async get(name) {
        try {
          const bytes = new Uint8Array(await (await (await dir.getFileHandle(safe(name) + ".bin")).getFile()).arrayBuffer());
          const want = await (await (await dir.getFileHandle(safe(name) + ".k")).getFile()).text();
          if (String(kappa(bytes)) !== want) {                      // L5: corrupt/tampered → purge + miss
            try { await dir.removeEntry(safe(name) + ".bin"); await dir.removeEntry(safe(name) + ".k"); } catch {}
            misses++; return null;
          }
          hits++; return bytes;
        } catch { misses++; return null; }
      },
      put(name, bytes) {                                            // fire-and-forget; failure just means a re-fetch next visit
        (async () => {
          try {
            const k = String(kappa(bytes));
            const bf = await dir.getFileHandle(safe(name) + ".bin", { create: true });
            const bw = await bf.createWritable(); await bw.write(bytes); await bw.close();
            const kf = await dir.getFileHandle(safe(name) + ".k", { create: true });
            const kw = await kf.createWritable(); await kw.write(k); await kw.close();
          } catch {}
        })();
      },
    };
    return api;
  } catch { return null; }
}
