"""Independent re-derivation of a model's root κ from bytes on disk.

Nothing here comes from the engine: tensors are cut out of model.safetensors
by the header's data_offsets, hashed with plain BLAKE3, and the root is the
BLAKE3 of the canonical manifest JSON the engine documents. Every κ is then
looked up in the κ store and its bytes re-hashed. A single changed byte
anywhere changes an address and this script says where.
"""
import json, os, struct, sys
from blake3 import blake3

model = sys.argv[1]
store = os.path.join(os.path.dirname(model.rstrip("/\\")), ".kappa-store")
receipt = json.load(open(os.path.join(model, ".holo-ingest-receipt.json"), encoding="utf-8"))

def kappa(b): return "blake3:" + blake3(b).hexdigest()
def store_path(k): return os.path.join(store, k.replace(":", "-") + ".bin")

# 1. tensors: cut by the safetensors header, hash, compare with the receipt
with open(os.path.join(model, "model.safetensors"), "rb") as f:
    hl = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hl))
    data_start = 8 + hl
    mismatched, missing, store_bad = [], [], []
    kappas = {}
    for name, k in zip(receipt["keys"], receipt["kappas"]):
        s, e = header[name]["data_offsets"]
        f.seek(data_start + s); k2 = kappa(f.read(e - s)); kappas[name] = k2
        if k2 != k: mismatched.append(name)
        p = store_path(k)
        if not os.path.exists(p): missing.append(name)
        elif kappa(open(p, "rb").read()) != k: store_bad.append(name)
print(f"tensors: {len(receipt['keys'])} re-hashed from safetensors; mismatches {len(mismatched)}; missing in store {len(missing)}; store bytes wrong {len(store_bad)}")

# 2. config, tokenizer, tokenizer_config
for fname, key in [("config.json", "config"), ("tokenizer.json", "tokenizer"), ("tokenizer_config.json", "tokenizer_config")]:
    k2 = kappa(open(os.path.join(model, fname), "rb").read())
    print(f"{key}: {'same' if k2 == receipt[key] else 'DIFFERENT'} {k2[:24]}…")

# 3. derived int8 artifacts in the store
q_missing = [q for q in receipt["quant"] if not os.path.exists(store_path(q["artifact"]))]
q_bad = [q for q in receipt["quant"] if os.path.exists(store_path(q["artifact"])) and kappa(open(store_path(q["artifact"]), "rb").read()) != q["artifact"]]
print(f"int8 artifacts: {len(receipt['quant'])}; missing {len(q_missing)}; bytes wrong {len(q_bad)}")

# 4. the root: canonical manifest, both key orders, against the receipt and the store
tensors = [{"name": n, "kappa": k, "shape": s} for n, k, s in zip(receipt["keys"], receipt["kappas"], receipt["shapes"])]
tier = [{"wide": q["key"], "artifact": q["artifact"], "out": q["out"], "in": q["in"]} for q in sorted(receipt["quant"], key=lambda q: q["key"])]
doc = {"kind": "hologram-live/model-root", "version": 1, "config": receipt["config"], "tokenizer": receipt["tokenizer"],
       "tokenizer_config": receipt["tokenizer_config"], "tensors": tensors, "int8_tier": tier}
def canon(o, sort):
    return json.dumps(o, separators=(",", ":"), sort_keys=sort, ensure_ascii=False).encode("utf-8")
for label, sort in [("sorted keys (serde_json default)", True), ("insertion order", False)]:
    b = canon(doc, sort); k = kappa(b)
    print(f"root via {label}: {k[:24]}… {'== receipt root' if k == receipt['root'] else '!= receipt root'}")
    if k == receipt["root"]:
        p = store_path(k)
        print(f"  store holds the root object: {os.path.exists(p)}; its bytes hash to the root: {os.path.exists(p) and kappa(open(p,'rb').read()) == k}; byte identical to this re-derivation: {os.path.exists(p) and open(p,'rb').read() == b}")
print("receipt root:", receipt["root"])
