import sys, json, time, urllib.request
vid, question = sys.argv[1], sys.argv[2]
def post(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=1800)
conv = json.load(post(f"http://localhost:8010/api/videos/{vid}/conversations"))
t0 = time.time(); ev = None
with post(f"http://localhost:8010/api/conversations/{conv['id']}/messages", {"content": question}) as resp:
    for raw in resp:
        line = raw.decode().strip()
        if line.startswith("event:"): ev = line[6:].strip()
        elif line.startswith("data:"):
            d = json.loads(line[5:]); el = f"{time.time()-t0:6.1f}s"
            if ev == "plan": print(el, "PLAN", d["text"])
            elif ev == "tool_call": print(el, "CALL", d["step"]["tier"], d["step"]["label"], json.dumps(d["step"]["args"]))
            elif ev == "tool_result": print(el, "  ->", d["step"]["status"], (d["step"]["summary"] or "")[:200], "| overlay" if d["step"]["overlay_url"] else "")
            elif ev == "final":
                a = d["message"]["answer"]
                print(el, f"ANSWER [{a['verdict']} {a['confidence']}] {a['answer']}")
                for e in a["evidence"]: print(f"       evidence {e['t_start']}-{e['t_end']}: {e['reason']}")
                print("       coverage:", a["coverage"])
            elif ev == "error": print(el, "ERROR", d)
