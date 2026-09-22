# -*- coding: utf-8 -*-
"""用 data/uploads/ 根目录的文件全量重建索引。

与 app.py 的 rebuild() 等价（同一套 config / splitter / embed_store），
只是不经过 Streamlit。manifest 格式与 app.py:ingest_file 保持一致。
"""
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\A\WorkBuddy\2026-09-18-14-19-51\DocMind_")
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from core import embed_store  # noqa: E402
from splitter import split_document  # noqa: E402


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    t0 = time.time()
    files = sorted(
        p for p in config.UPLOAD_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in config.ALLOWED_EXTENSIONS
    )
    print("待入库文件 %d 个 | CHUNK_SIZE=%d OVERLAP=%d" %
          (len(files), config.CHUNK_SIZE, config.OVERLAP), flush=True)
    print("版本 %s" % config.DOCMIND_VERSION, flush=True)

    print("\n[1/3] reset_index() ...", flush=True)
    embed_store.reset_index()
    print("      done  %.1fs" % (time.time() - t0), flush=True)

    print("[2/3] 逐文件 切分 + 向量化", flush=True)
    manifest = {}
    total_chunks = 0
    total_chars = 0
    t_load = time.time()
    for p in files:
        t1 = time.time()
        try:
            chunks = split_document(
                str(p), chunk_size=config.CHUNK_SIZE, chunk_overlap=config.OVERLAP)
        except Exception as e:  # noqa: BLE001
            print("      [ERR] %-46s %s: %s" % (p.name, type(e).__name__, e), flush=True)
            continue
        chunks = [c for c in chunks if c and c.strip()]
        if not chunks:
            print("      [EMPTY] %s" % p.name, flush=True)
            continue
        digest = file_sha1(p)
        docs = [
            {
                # 与 app.py:ingest_file 保持一致：块内容前带来源文件名
                "content": f"《{p.name}》\n{c}",
                "metadata": {
                    "source": p.name,
                    "chunk_index": i,
                    "file_sha1": digest,
                    "ingested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            }
            for i, c in enumerate(chunks)
        ]
        embed_store.add_documents(docs)
        manifest[p.name] = {
            "sha1": digest,
            "chunks": len(docs),
            "size": p.stat().st_size,
            "ingested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_size": config.CHUNK_SIZE,
            "overlap": config.OVERLAP,
        }
        total_chunks += len(docs)
        total_chars += sum(len(c) for c in chunks)
        lens = [len(c) for c in chunks]
        print("      %-46s %4d 块  %6.1fs  字数 min/avg/max = %d/%d/%d" %
              (p.name, len(docs), time.time() - t1, min(lens), sum(lens) // len(lens), max(lens)),
              flush=True)

    print("\n[3/3] 写 manifest ...", flush=True)
    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("入库完成：%d 个文件，%d 块，%d 字（约 %.1f 万字），总耗时 %.1fs" %
          (len(manifest), total_chunks, total_chars, total_chars / 10000, time.time() - t0))
    print("collection 实际条数 =", embed_store.get_store().collection.count())
    print("=" * 78)


if __name__ == "__main__":
    main()
