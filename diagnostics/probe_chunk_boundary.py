# -*- coding: utf-8 -*-
"""块边界 vs 条款边界 —— 诊断「块级召回失败」（q05/q21/q26）的根因。

零 LLM、零检索、零编码：只读 ChromaDB 里已入库的块 + 重新切一遍原文，
不加载 embedding 模型。目的是回答一个二选一的问题：

    答案没被召回，是因为「排序没把它排上来」，
    还是因为「切分时把它劈开了，两半各自都不像一个完整答案」？

判据（都是可数的）：
  1. 块首对齐率 —— 有多少块是从「第X条」这类条款边界开始的
  2. 每块完整条款数 —— 一个 500 字块里塞了几个「第X条」标记
  3. 答案跨块度 —— 关键答案段被几个块分掉
"""

import json
import re
import sys
import time
import pathlib
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent / 'DocMind_'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))

import config                                    # noqa: E402
from core import embed_store                     # noqa: E402

ARTICLE = re.compile(r'第[一二三四五六七八九十百零〇\d]{1,6}条')
CNUM = r'[一二三四五六七八九十百零〇\d]{1,6}'
HEAD_PREFIX = re.compile(r'^《[^》]+》\s*')


def norm(t: str) -> str:
    return re.sub(r'\s+', '', t)


def content_of(d) -> str:
    for k in ('content', 'document', 'text'):
        v = d.get(k)
        if isinstance(v, str) and v:
            return v
    return ''


def source_of(d) -> str:
    m = d.get('metadata') or {}
    return m.get('source') or m.get('file') or ''


def main() -> None:
    t0 = time.time()
    docs = embed_store.all_documents()
    print('=' * 98)
    print('【0】数据结构')
    print('  总块数 = %d   （读取耗时 %.1fs —— 未加载 embedding 模型）' % (len(docs), time.time() - t0))
    print('  第一项 keys     =', list(docs[0].keys()))
    print('  第一项 metadata =', docs[0].get('metadata'))
    print()

    by_file = defaultdict(list)
    for d in docs:
        by_file[source_of(d)].append(content_of(d))

    print('=' * 98)
    print('【1】块边界 vs 条款边界')
    print('  %-40s %5s %7s %9s %9s %9s' %
          ('文件（来源）', '块数', '条款块', '块首对齐', '块尾未完', '每块条款'))
    print('  ' + '-' * 88)
    tot = Counter()
    for src in sorted(by_file):
        chunks = by_file[src]
        art_n = heads = tails = arts = 0
        for c in chunks:
            body = HEAD_PREFIX.sub('', c)
            marks = list(ARTICLE.finditer(body))
            if marks:
                art_n += 1
                arts += len(marks)
                # 块首对齐：第一个条款标记出现在块的开头 12 字内
                if marks[0].start() <= 12:
                    heads += 1
                # 块尾未完：最后一个条款标记之后还有 40 字以上正文（这条可能被截断）
                if len(body) - marks[-1].end() >= 40:
                    tails += 1
        tot['n'] += len(chunks)
        tot['art_n'] += art_n
        tot['heads'] += heads
        tot['tails'] += tails
        tot['arts'] += arts
        print('  %-40s %5d %7d %8.1f%% %8.1f%% %9.2f' % (
            src[:40], len(chunks), art_n,
            100 * heads / max(art_n, 1), 100 * tails / max(art_n, 1),
            arts / max(len(chunks), 1)))
    print('  ' + '-' * 88)
    print('  %-40s %5d %7d %8.1f%% %8.1f%% %9.2f' % (
        '【合计】', tot['n'], tot['art_n'],
        100 * tot['heads'] / max(tot['art_n'], 1),
        100 * tot['tails'] / max(tot['art_n'], 1),
        tot['arts'] / max(tot['n'], 1)))
    print()
    print('  块首对齐 = 块是从条款开头切的（否则是从条款中段切进来的）')
    print('  块尾未完 = 最后一个条款标记后还有 40+ 字 —— 该条款很可能被推到下一块')
    print()

    # ---------------------------------------------------------------- 逐题
    print('=' * 98)
    print('【2】每条题的答案散落在几个块里（用 ground_truth 的字面要点定位）')
    ds = pathlib.Path('eval/dataset.jsonl').read_text(encoding='utf-8').strip().splitlines()
    print('  %-5s %-6s %5s %6s %6s   %s' % ('id', 'type', '要点', '命中块', '跨档', '题面'))
    print('  ' + '-' * 94)
    for line in ds:
        r = json.loads(line)
        if r.get('type') == '负例':
            continue
        gt = norm(r.get('ground_truth') or '')
        gold = norm('\n'.join(content_of(d) for d in docs))
        keys = [k for k in re.split(r'[，、；;。]', gt) if len(k) >= 8]
        if not keys:
            continue
        hit_chunks = []
        for i, c in enumerate(docs):
            cb = norm(content_of(c))
            n = sum(1 for k in keys if k[:12] in cb or (len(k) > 20 and any(k[j:j + 12] in cb for j in range(0, len(k) - 12, 4))))
            if n:
                hit_chunks.append((i, n))
        top = sorted(hit_chunks, key=lambda x: -x[1])[:6]
        docs_hit = {source_of(docs[i]) for i, _ in top}
        print('  %-5s %-6s %5d %6d %6d   %s' % (
            r['id'], r.get('type'), len(keys), len(hit_chunks), len(docs_hit),
            (r.get('question') or '')[:40]))
    print()


if __name__ == '__main__':
    main()
