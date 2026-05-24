#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"<failed: {exc}>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:11451")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--log", default="/tmp/sglang_11451.log")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()

    print("== processes ==")
    print(run(["bash", "-lc", "ps -ef | rg 'sglang|launch_server|11451' | rg -v rg || true"]).strip())
    print("== ports ==")
    print(run(["bash", "-lc", "ss -ltnp | rg ':11451' || true"]).strip())

    log_path = Path(args.log)
    if log_path.exists():
        print("== log tail ==")
        print(run(["bash", "-lc", f"tail -80 {args.log!r}"]).strip())

    health_url = args.url.rstrip("/") + "/health"
    try:
        r = requests.get(health_url, timeout=3)
        print(f"health: {r.status_code}")
        r.raise_for_status()
    except Exception as exc:
        print(f"health failed: {exc!r}")
        return 1

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        venv_python = Path("/home/ubuntu/.venv/bin/python")
        if venv_python.exists() and Path(sys.executable) != venv_python:
            os.execv(str(venv_python), [str(venv_python), *sys.argv])
        raise

    try:

        tok = AutoTokenizer.from_pretrained(args.model)
        ids = tok.apply_chat_template(
            [{"role": "user", "content": "2+2=? Answer briefly."}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    except Exception as exc:
        print(f"tokenizer failed: {exc!r}")
        return 1

    payload = {
        "input_ids": ids,
        "image_data": [],
        "sampling_params": {
            "max_new_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "skip_special_tokens": True,
        },
        "return_logprob": True,
        "stream": False,
    }
    try:
        r = requests.post(args.url.rstrip("/") + "/generate", json=payload, timeout=args.timeout)
        print(f"generate: {r.status_code}")
        print(r.text[:1000])
        r.raise_for_status()
        out = r.json()
        tokens = out.get("output_ids") or out.get("output_tokens") or []
        print("decoded:", tok.decode(tokens))
    except Exception as exc:
        print(f"generate failed: {exc!r}")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
