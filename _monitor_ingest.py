import json
import requests

url = "http://127.0.0.1:5050/api/ingest/stream"

with requests.get(url, stream=True, timeout=3600) as resp:
    resp.raise_for_status()
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        if not raw.startswith("data: "):
            continue
        evt = json.loads(raw[6:])
        stage = evt.get("stage")
        status = evt.get("status")
        print(f"{stage}: {status}")
        if stage == "Pipeline" and status in {"done", "error"}:
            print(json.dumps(evt, indent=2))
            break
