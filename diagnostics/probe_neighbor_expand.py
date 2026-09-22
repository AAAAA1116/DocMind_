# -*- coding: utf-8 -*-
"""邻块扩展（neighbor expansion）的收益上限 —— 零 LLM。

背景
----
切分对齐被否掉了（切点对齐率 19.6%→78.6%，但"答案同块率"一动不动）。
这引出下一个假设：**答案不是被切分劈开的，而是本来就在相邻的条款里**。
若成立，"取块时顺带取 chunk_index±1"是一剂廉价解药 ——
它只改 `rag_chain` 的取块逻辑，**不需要重建索引**。

做法
----
跑真实混合检索（k = TOP_K），对比两种口径的要点覆盖率：
  now = 召回的 K 个块
  exp = 召回的 K 个块 + 它们在**同一份文档**里的 chunk_index±1 邻居
另带上界参照：把每个召回块的邻域扩到 ±2，看还有没有余量。

判据沿用 scan_topk.py 的口径（归一化空白 + 可达性过滤），保证可比。
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent / 'DocMind_'
sys.path.insert(0, str(ROOT))

import config                              # noqa: E402
from core import embed_store, hybrid       # noqa: E402

_WS = re.compile(r'[\s\u3000\u200b]+')
_SPLIT = re.compile(r'[；。！？\n]|（[一二三四五六七八九十]+）|\([0-9]+\)|[①-⑩]')
KEY_MIN, KEY_WIN = 6, 12


def norm(t: str) -> str:
    return _WS.sub('', t)


def key_hit(key: str, text: str) -> bool:
    key, text = norm(key), norm(text)
    if len(key) < KEY_MIN:
        return key in text
    if key in text:
        return True
    if len(key) <= KEY_WIN:
        return False
    return any(key[i:i + KEY_WIN] in text for i in range(len(key) - KEY_WIN + 1))


def split_keys(gt: str) -> List[str]:
    parts = [p.strip(' ，、;；') for p in _SPLIT.split(gt)]
    return [p for p in parts if len(p) >= 4]


def coverage(keys: List[str], chunks: List[str]) -> float:
    if not keys:
        return 0.0
    return sum(1 for k in keys if any(key_hit(k, c) for c in chunks)) / len(keys)


def main() -> None:
    t0 = time.time()
    docs = embed_store.all_documents()
    by_idx: Dict[tuple, str] = {}
    max_idx: Dict[str, int] = defaultdict(int)
    for d in docs:
        md = d.get('metadata') or {}
        s, i = md.get('source'), md.get('chunk_index')
        if s is None or i is None:
            continue
        by_idx[(s, i)] = d.get('content', '')
        max_idx[s] = max(max_idx[s], i)
    corpus = norm('\n'.join(d.get('content', '') for d in docs))
    print('语料 %d 块 / %d 份文档（加载 %.0fs，未调用任何大模型）' % (
        len(docs), len(max_idx), time.time() - t0))
    print()

    ds = [json.loads(l) for l in
          (ROOT / 'eval/dataset.jsonl').read_text(encoding='utf-8').strip().splitlines()]
    K = config.TOP_K

    print('=' * 104)
    print('邻块扩展收益上限（K = TOP_K = %d，混合检索，零 LLM）' % K)
    print('  %-5s %-6s %4s %4s | %8s %8s %8s | %s' %
          ('id', 'type', '要点', '可达', '现状', '扩展±1', '上界±2', '题面'))
    print('  ' + '-' * 100)

    agg = []
    for r in ds:
        if r.get('type') == '负例':
            continue
        keys = [k for k in split_keys(r.get('ground_truth') or '') if key_hit(k, corpus)]
        if not keys:
            print('  %-5s %-6s    —    — |   n/a（可达要点为 0）' % (r['id'], r.get('type')))
            continue

        hits = hybrid.hybrid_search(r['question'], k=K)
        now, exp1, exp2 = [], [], []
        picked = []
        for rec in hits:
            doc = rec.get('doc') or {}
            md = doc.get('metadata') or {}
            s, i = md.get('source'), md.get('chunk_index')
            now.append(doc.get('content', ''))
            picked.append((s, i))
        for s, i in picked:
            for j in (i - 1, i + 1):
                if i is not None and 0 <= j <= max_idx.get(s, 0):
                    if (s, j) in by_idx:
                        exp1.append(by_idx[(s, j)])
            for j in (i - 2, i + 2):
                if i is not None and 0 <= j <= max_idx.get(s, 0):
                    if (s, j) in by_idx:
                        exp2.append(by_idx[(s, j)])

        c_now = coverage(keys, now)
        c_e1 = coverage(keys, now + exp1)
        c_e2 = coverage(keys, now + exp2)
        agg.append((c_now, c_e1, c_e2, len(keys)))
        mark = '  ← 有提升' if c_e1 > c_now + 1e-9 else ''
        print('  %-5s %-6s %4d %4d | %8.3f %8.3f %8.3f |%s %s' % (
            r['id'], r.get('type'), len(split_keys(r.get('ground_truth') or '')), len(keys),
            c_now, c_e1, c_e2, mark, (r.get('question') or '')[:26]))
    print('  ' + '-' * 100)

    n = len(agg)
    m0 = sum(a[0] for a in agg) / n
    m1 = sum(a[1] for a in agg) / n
    m2 = sum(a[2] for a in agg) / n
    print('  %d 条可评估正例的**平均要点覆盖率**：' % n)
    print('    现状（%d 块）        %.4f' % (K, m0))
    print('    扩展 ±1（约 %d 块）  %.4f   （%+.4f）' % (K * 3, m1, m1 - m0))
    print('    上界 ±2（约 %d 块）  %.4f   （%+.4f）' % (K * 5, m2, m2 - m0))
    better = [(a[2] - a[0], i) for i, a in enumerate(agg) if a[1] > a[0] + 1e-9]
    print('    ±1 有提升的题数：%d/%d' % (len(better), n))
    print()
    print('  判据口径与 scan_topk.py 完全一致（归一化空白 + 可达性过滤），数字可直接比。')
    print('  总耗时 %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
