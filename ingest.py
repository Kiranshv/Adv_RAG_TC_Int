import argparse
from pathlib import Path

import pandas as pd
import requests


def main():
    parser = argparse.ArgumentParser(description="CLI ingestion helper for Advanced RAG Explorer")
    parser.add_argument("file", help="CSV/XLSX file path")
    parser.add_argument("--text-cols", default="title,steps,expected,tags")
    parser.add_argument("--meta-cols", default="id,jira_id,priority,module")
    parser.add_argument("--server", default="http://127.0.0.1:5050")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("rb") as f:
        upload = requests.post(
            f"{args.server}/api/upload",
            files={"file": (file_path.name, f)},
            timeout=180,
        )
    upload.raise_for_status()

    text_cols = [x.strip() for x in args.text_cols.split(",") if x.strip()]
    meta_cols = [x.strip() for x in args.meta_cols.split(",") if x.strip()]

    sel = requests.post(
        f"{args.server}/api/selection",
        json={"text_columns": text_cols, "meta_columns": meta_cols},
        timeout=30,
    )
    sel.raise_for_status()

    start = requests.post(f"{args.server}/api/ingest/start", timeout=30)
    start.raise_for_status()

    print("Ingestion triggered successfully.")
    print("Open /ingest UI for live SSE logs.")


if __name__ == "__main__":
    main()
