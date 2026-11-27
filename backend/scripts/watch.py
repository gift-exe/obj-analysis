import sys, json, time, urllib.request
vid = sys.argv[1]
t0 = time.time(); ev = None; last = {}
with urllib.request.urlopen(f"http://localhost:8010/api/videos/{vid}/events", timeout=900) as resp:
    for raw in resp:
        line = raw.decode().strip()
        if line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:"):
            d = json.loads(line[5:])
            for s in d.get("steps", []):
                k = (s["status"], s["detail"])
                if last.get(s["step"]) != k:
                    last[s["step"]] = k
                    print(f"{time.time()-t0:6.1f}s {s['step']:<11} {s['status']:<8} {s['detail'] or ''} {(s['error'] or '')[:300]}", flush=True)
            if ev == "done":
                print("DONE", d["video"]["status"], d["video"]["error"]); break
