"""Run the official SimpleMem pipeline on passage dialogue entries."""
import json, os, sys, time
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla", os.environ.get("SIMPLEMEM_SOURCE", BASE + "/vendor/SimpleMem")):
    if p not in sys.path:
        sys.path.append(p)
import requests

SHIM = os.environ.get("SHIM_URL", "http://127.0.0.1:8123")
N = int(os.environ.get("SM_N", "20"))
OUT = os.environ.get("SM_OUT", BASE + "/sota/results/simplemem_subset.jsonl")
MAN = BASE + "/sota/data/manifest_500.jsonl"
READER = os.environ.get("SM_READER", "http://127.0.0.1:8123/v1/chat/completions")


def strip_html(t):
    import re
    return re.sub(r"\s+", " ", t or "").strip()


def reader_answer(tok, context, question):
    sys_p = ("You answer questions using ONLY the context provided. Reply with the shortest possible "
             "answer phrase. If the context does not contain enough information, reply with exactly UNANSWERABLE.")
    r = requests.post(READER, json={"model": "local", "messages": [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": "Context:\n%s\n\nQuestion: %s\n\nAnswer:" % (context, question)}],
        "temperature": 0.0, "max_tokens": 24}, timeout=600)
    return r.json()["choices"][0]["message"]["content"].strip(), r.json()["usage"]


def main():
    sys.path.append(os.environ.get("SIMPLEMEM_SOURCE", BASE + "/vendor/SimpleMem"))
    from simplemem import create
    OFF = int(os.environ.get("SM_OFFSET", "0"))
    rows = [json.loads(l) for l in open(MAN)][OFF:OFF + N]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE + "/models/Qwen3.5-4B")
    out = []
    for i, r in enumerate(rows):
        t0 = time.time()
        mem = create(mode="text", clear_db=True)
        import re as _re
        ts = "2025-01-01T00:00:00"
        for p in sorted(r["paragraphs"], key=lambda x: x["idx"]):
            text = strip_html(p["paragraph_text"])
            turns = [t.strip() for t in _re.split(r"(?<=[.!?])\s+", text) if len(t.strip()) > 25]
            if os.environ.get("SM_TURNS", "sentence") == "sentence" and turns:
                for t in turns:
                    mem.add_dialogue(p["title"], t, ts)
            else:
                mem.add_dialogue(p["title"], text, ts)
        mem.finalize()
        q = r["question"]

        pred = mem.ask(q) or ""
        try:
            n_mem = len(mem.get_all_memories())
        except Exception:
            n_mem = -1
        sys.path.append(BASE + "/sota/harness")
        import scoring
        golds = [r.get("answer")]
        sc = scoring.score_answer(pred, golds)
        out.append({"id": r["id"], "hop": r.get("hop"), "question": q, "pred": pred,
                    "gold": r.get("answer"), "em": float(sc.get("em", 0)), "f1": float(sc.get("f1", 0)),
                    "n_memories": n_mem, "elapsed": round(time.time() - t0, 1)})
        print("[sm] %d/%d n_mem=%d pred=%r" % (i + 1, N, out[-1]["n_memories"], pred[:40]), flush=True)
        try:
            mem.close()
        except Exception:
            pass
    with open(OUT, "w") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    import statistics
    print("[sm] done", len(out), "EM %.3f" % statistics.mean(x["em"] for x in out),
          "mean elapsed %.1fs" % statistics.mean(x["elapsed"] for x in out))


if __name__ == "__main__":
    main()
