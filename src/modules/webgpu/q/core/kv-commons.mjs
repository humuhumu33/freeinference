// core/kv-commons.mjs — the DURABLE KV COMMONS store (I1 of HOLO-INSTANT-EVERY-USER-PROMPT.md).
// A prefix's K/V state, dumped once by qvac-gpu's dumpState(), persists in OPFS as a κ-verified
// object so the NEXT visit restores the persona in tens of milliseconds instead of re-prefilling.
// Law L5 on load: the bytes must re-derive to the recorded content κ — a mismatch (corruption,
// partial write, tamper) purges the entry and reports a miss, and the caller re-prefills cleanly.
// Engine-agnostic by design: keys and layouts come from the caller, so the gguf engine can share
// this store unchanged when the two brains unify (D8).

import { kappa } from "../pkg/holospaces_web.js";

const DIR = "holo-kv-commons";
const enc = new TextEncoder();

async function dir() {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(DIR, { create: true });
}
const fname = (key) => String(key).replace(/[^a-zA-Z0-9._-]/g, "_");

// key = κ(formatVer | modelκ | layout | prefix token ids) — computed by the engine; hashed here so
// arbitrary-length prefixes become a fixed filename and the ids never appear in storage names.
export function commonsKey(parts) {
  try { return String(kappa(enc.encode(parts.join("|")))).replace(/[^a-zA-Z0-9]/g, "").slice(-56); }
  catch { return null; }
}

export async function save(key, bytes, meta) {
  const d = await dir();
  const contentK = String(kappa(bytes));
  const bf = await d.getFileHandle(fname(key) + ".bin", { create: true });
  const bw = await bf.createWritable(); await bw.write(bytes); await bw.close();
  const mf = await d.getFileHandle(fname(key) + ".json", { create: true });
  const mw = await mf.createWritable(); await mw.write(JSON.stringify({ ...meta, contentK, ts: Date.now() })); await mw.close();
  return true;
}

export async function load(key) {
  const d = await dir();
  let meta, bytes;
  try {
    meta = JSON.parse(await (await (await d.getFileHandle(fname(key) + ".json")).getFile()).text());
    bytes = new Uint8Array(await (await (await d.getFileHandle(fname(key) + ".bin")).getFile()).arrayBuffer());
  } catch { return null; }                                       // miss — never an error
  if (String(kappa(bytes)) !== meta.contentK) {                  // L5: refuse + purge on tamper
    try { await d.removeEntry(fname(key) + ".bin"); await d.removeEntry(fname(key) + ".json"); } catch {}
    return null;
  }
  return { bytes, meta };
}

export async function drop(key) {
  try { const d = await dir(); await d.removeEntry(fname(key) + ".bin"); await d.removeEntry(fname(key) + ".json"); return true; }
  catch { return false; }
}
