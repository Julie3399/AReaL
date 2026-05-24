#!/usr/bin/env python3
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    out_dir = Path("/home/ubuntu/data/asearcher_local_rag_full")
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "inclusionAI/ASearcher-Local-Knowledge"
    for filename in ["wiki_corpus.jsonl", "wiki_webpages.jsonl"]:
        print(f"downloading {filename}", flush=True)
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(out_dir),
        )
        print(f"downloaded {filename}: {path}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
