#!/usr/bin/env python3
"""Build a compressed FAISS index for the full ASearcher local corpus.

The upstream ASearcher `utils/index_builder.py` builds a Flat index by holding
all embeddings in memory. That is not practical for the full local knowledge
base on a single 70GB-RAM workstation. This script streams the JSONL corpus,
encodes documents in batches with e5-base-v2, trains an IVFPQ index on an
initial sample, and adds embeddings incrementally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def iter_contents(path: Path):
    with path.open("r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)["contents"]


def pooling(pooler_output, last_hidden_state, attention_mask, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pooling_method == "cls":
        return last_hidden_state[:, 0]
    if pooling_method == "pooler":
        return pooler_output
    raise ValueError(f"unsupported pooling method: {pooling_method}")


class E5Encoder:
    def __init__(self, model_dir: str, max_length: int, use_fp16: bool):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).eval().cuda()
        if use_fp16:
            self.model = self.model.half()
        self.max_length = max_length

    @torch.no_grad()
    def encode_passages(self, docs: list[str]) -> np.ndarray:
        texts = [f"passage: {doc}" for doc in docs]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        output = self.model(**inputs, return_dict=True)
        emb = pooling(output.pooler_output, output.last_hidden_state, inputs["attention_mask"], "mean")
        emb = torch.nn.functional.normalize(emb, dim=-1)
        arr = emb.detach().float().cpu().numpy().astype(np.float32, order="C")
        del inputs, output, emb
        torch.cuda.empty_cache()
        return arr


def read_training_docs(corpus_path: Path, n_train: int) -> list[str]:
    docs = []
    for doc in iter_contents(corpus_path):
        docs.append(doc)
        if len(docs) >= n_train:
            break
    return docs


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", default="/home/ubuntu/data/asearcher_local_rag_full/wiki_corpus.jsonl")
    parser.add_argument("--model-dir", default="/home/ubuntu/models/e5-base-v2")
    parser.add_argument("--save-dir", default="/home/ubuntu/data/asearcher_local_rag_full/e5.ivfpq.index")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--nlist", type=int, default=16384)
    parser.add_argument("--m", type=int, default=96)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--use-fp16", action="store_true", default=True)
    parser.add_argument("--save-every", type=int, default=250000)
    args = parser.parse_args()

    corpus_path = Path(args.corpus_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    index_path = save_dir / "e5_IVFPQ.index"

    encoder = E5Encoder(args.model_dir, args.max_length, args.use_fp16)
    dim = encoder.model.config.hidden_size

    print(f"reading {args.train_size} docs for IVFPQ training", flush=True)
    train_docs = read_training_docs(corpus_path, args.train_size)
    train_embs = []
    for start in tqdm(range(0, len(train_docs), args.batch_size), desc="encode train"):
        train_embs.append(encoder.encode_passages(train_docs[start : start + args.batch_size]))
    train_emb = np.concatenate(train_embs, axis=0)

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, args.nlist, args.m, args.nbits, faiss.METRIC_INNER_PRODUCT)
    print("training index", flush=True)
    index.train(train_emb)
    del train_emb, train_embs, train_docs

    total = count_lines(corpus_path)
    added = 0
    batch = []
    for doc in tqdm(iter_contents(corpus_path), total=total, desc="add corpus"):
        batch.append(doc)
        if len(batch) >= args.batch_size:
            embs = encoder.encode_passages(batch)
            index.add(embs)
            added += len(batch)
            batch = []
            if added % args.save_every < args.batch_size:
                faiss.write_index(index, str(index_path))
                print(f"checkpoint saved after {added} docs", flush=True)
    if batch:
        embs = encoder.encode_passages(batch)
        index.add(embs)
        added += len(batch)
    faiss.write_index(index, str(index_path))
    print(f"done, added={added}, index={index_path}", flush=True)


if __name__ == "__main__":
    main()
