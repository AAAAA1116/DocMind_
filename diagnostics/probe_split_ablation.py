# -*- coding: utf-8 -*-
"""切分粒度消融模拟：固定窗口 vs 条款边界对齐（仓库外的纯模拟，不改任何源码）。

回答一个问题：**把「第X条」提升为最高优先级切点，能换来多少？**

三个可数量化的收益/代价：
  1. 切点对齐率       —— 切点是否落在条款边界（无歧义；这是条款版的核心卖点）
  2. 答案同块率       —— 每条题的要点能否全落进同一个块（"能答全"的必要条件）
  3. 块数 / 块长      —— 代价（块数翻倍会拉大索引、让 BM25 的 avgdl 失真）

⚠️ 判据自检（写这版脚本时才想通）：
   **「块首」不等于「切点」。** overlap=80 会让每块开头的 80 字是上一块的尾巴，
   所以"块首 12 字内有没有条款标记"根本测不出切点落在哪里 —— 上一版用块首对齐率
   推"88% 的块边界落在条款内部"是不严谨的。这里改用切点对齐率（由 chunk 长度
   与 overlap 反推切点绝对位置，并用切片回验），才是无歧义的。

现状一律走仓库真实代码 `splitter.split_text`；条款版在本地实现（同结构的 while 循环，
只多一级"条款边界"，失败时回退原三级分隔符）。
零 LLM、零编码、零检索。
"""

import json
import re
import sys
import time
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent / 'DocMind_'
sys.path.insert(0, str(ROOT))

import config                                          # noqa: E402
from splitter import split_text, _find_split_pos       # noqa: E402
from loader import load_document                       # noqa: E402

CACHE = pathlib.Path(__file__).resolve().parent / 'paper_text_cache.json'

ARTICLE = re.compile(r'第[一二三四五六七八九十百零〇\d]{1,6}条')
KEY_WIN, KEY_MIN, KEY_SPLIT_MIN = 12, 4, 8
HEAD_PREFIX = re.compile(r'^《[^》]+》\s*')


def norm(t: str) -> str:
    return re.sub(r'\s+', '', t)


# ---------------------------------------------------------------- 条款版切分
def _find_split_pos_article(window: str, chunk_size: int, min_pos: int) -> int:
    """先试「条款边界」（新的最高优先级），不满足下限则回退原三级分隔符。

    切点取条款标记的**起始位置**（不是标记之后）—— 这样下一块从「第X条」开头；
    原三级分隔符相反，是把标记留在上一块末尾。
    """
    best = -1
    for m in ARTICLE.finditer(window[:chunk_size]):
        if m.start() == 0:
            continue                       # 窗口开头就是条款标记：上一块已切在这
        nxt = ARTICLE.search(window, m.end())
        if nxt is not None and nxt.start() - m.end() < 20:
            continue                       # 紧跟下一条 = 多半是目录行，不当切点
        best = max(best, m.start())
    if best >= min_pos:
        return best
    return _find_split_pos(window, chunk_size, min_pos)


def split_by_article(text: str, chunk_size: int = 500, chunk_overlap: int = 80):
    """与 split_text 同结构，只把切点查找换成 _find_split_pos_article。"""
    if not text:
        return []
    chunk_size = int(chunk_size)
    overlap = max(0, min(int(chunk_overlap), chunk_size // 2))
    if len(text) <= chunk_size:
        return [text]

    chunks, total, start = [], len(text), 0
    min_pos = overlap + 1
    while start < total:
        end = start + chunk_size
        if end >= total:
            tail = text[start:]
            if tail.strip() and len(tail) > overlap:
                chunks.append(tail)
            break
        cut = _find_split_pos_article(text[start:end], chunk_size, min_pos)
        piece = text[start:start + cut]
        if piece.strip():
            chunks.append(piece)
        nxt = start + cut - overlap
        start = nxt if nxt > start else start + cut
    return chunks


# ---------------------------------------------------------------- 定位与判据
def locate(chunks, text: str, overlap: int):
    """反推每块的 (start, end) 绝对偏移，并用切片回验。失败返回 None。"""
    pos, out = 0, []
    for c in chunks:
        if text[pos:pos + len(c)] != c:
            j = text.find(c, max(0, pos - 300))
            if j == -1:
                return None
            pos = j
        out.append((pos, pos + len(c)))
        pos = pos + len(c) - overlap
    return out


def cut_align_rate(text: str, chunks, overlap: int):
    """切点对齐率：有多少个切点之后紧跟「第X条」。分母 = 切点数（块数 - 1）。"""
    loc = locate(chunks, text, overlap)
    if not loc:
        return None, 0, 0
    ok = tot = 0
    for _, e in loc[:-1]:
        tot += 1
        if ARTICLE.match(re.sub(r'^\s+', '', text[e:e + 12])):
            ok += 1
    return (ok / tot if tot else 0.0), ok, tot


def head_align_rate(chunks) -> float:
    """参考指标：块首 12 字内含条款标记的比例（受 overlap 污染，只作对照）。"""
    has = ok = 0
    for c in chunks:
        body = HEAD_PREFIX.sub('', c)
        m = ARTICLE.search(body)
        if m is None:
            continue
        has += 1
        if m.start() <= 12:
            ok += 1
    return ok / has if has else 0.0


def key_hit(key: str, text: str) -> bool:
    if len(key) < KEY_MIN:
        return key in text
    if key in text:
        return True
    if len(key) <= KEY_WIN:
        return False
    return any(key[i:i + KEY_WIN] in text for i in range(len(key) - KEY_WIN + 1))


def main() -> None:
    t0 = time.time()
    if CACHE.exists():
        texts = json.loads(CACHE.read_text(encoding='utf-8'))
        print('文本缓存命中 %s（%d 份）' % (CACHE.name, len(texts)))
    else:
        texts = {}
        for p in sorted((ROOT / 'data/uploads').glob('*.pdf')):
            texts[p.name] = load_document(str(p))
            print('  已加载 %-42s %7d 字' % (p.name, len(texts[p.name])))
            sys.stdout.flush()
        CACHE.write_text(json.dumps(texts, ensure_ascii=False), encoding='utf-8')
        print('已写入缓存 %s' % CACHE.name)
    print()

    cs, ov = config.CHUNK_SIZE, config.OVERLAP
    print('=' * 104)
    print('【1】结构：切点是否落在条款边界（CHUNK_SIZE=%d  OVERLAP=%d）' % (cs, ov))
    print('  %-40s %13s %14s %10s %11s' % ('文件', '块数 现→条', '块长 现→条', '切点对齐', '块首含条(参)'))
    print('  ' + '-' * 100)

    store = {}
    tot = Counter()
    for name in sorted(texts):
        raw = texts[name]
        cn_raw = split_text(raw, cs, ov)
        ca_raw = split_by_article(raw, cs, ov)
        cn = [f'《{name}》\n{c}' for c in cn_raw]
        ca = [f'《{name}》\n{c}' for c in ca_raw]
        store[name] = (raw, cn, ca)
        rn, okn, tn = cut_align_rate(raw, cn_raw, ov)
        ra, oka, ta = cut_align_rate(raw, ca_raw, ov)
        tot['n'] += len(cn)
        tot['a'] += len(ca)
        tot['okn'] += okn
        tot['tn'] += tn
        tot['oka'] += oka
        tot['ta'] += ta
        print('  %-40s %5d → %-5d %6.0f → %-6.0f %9.1f%% %9.1f%% → %.1f%%' % (
            name[:40], len(cn), len(ca),
            sum(len(c) for c in cn) / len(cn), sum(len(c) for c in ca) / len(ca),
            100 * (rn or 0), 100 * (ra or 0), 100 * head_align_rate(ca)))
    print('  ' + '-' * 100)
    print('  合计块数 %d → %d（%+d，%+.1f%%）' % (
        tot['n'], tot['a'], tot['a'] - tot['n'], 100 * (tot['a'] - tot['n']) / tot['n']))
    print('  切点对齐率：现状 %d/%d = %.1f%%  →  条款版 %d/%d = %.1f%%' % (
        tot['okn'], tot['tn'], 100 * tot['okn'] / max(tot['tn'], 1),
        tot['oka'], tot['ta'], 100 * tot['oka'] / max(tot['ta'], 1)))
    print()

    # ------------------------------------------------------------ 答案同块率
    ds = pathlib.Path(ROOT / 'eval/dataset.jsonl').read_text(encoding='utf-8').strip().splitlines()
    chunks_now = [c for name in store for c in store[name][1]]
    chunks_art = [c for name in store for c in store[name][2]]
    chunks_now_n = [norm(c) for c in chunks_now]
    chunks_art_n = [norm(c) for c in chunks_art]
    all_raw = norm(''.join(store[name][0] for name in store))

    print('=' * 104)
    print('【2】每条题的答案能否落进同一个块（"检索一次就能答全"的必要条件）')
    print('  %-5s %-6s %4s %4s | %8s %8s | %8s %8s' %
          ('id', 'type', '要点', '可达', '现状覆盖', '条款覆盖', '现状同块', '条款同块'))
    print('  ' + '-' * 100)

    agg = []
    for line in ds:
        r = json.loads(line)
        if r.get('type') == '负例':
            continue
        gt = norm(r.get('ground_truth') or '')
        keys = [k for k in re.split(r'[，、；;。]', gt) if len(k) >= KEY_SPLIT_MIN]
        reach = [k for k in keys if key_hit(k, all_raw)]
        if not reach:
            print('  %-5s %-6s %4d %4d |  n/a —— 可达要点为 0，无法用字面判据评估' % (
                r['id'], r.get('type'), len(keys), 0))
            continue

        def cover(chs):
            best, hit = 0.0, 0
            for cb in chs:
                n = sum(1 for k in reach if key_hit(k, cb))
                if n > best:
                    best, hit = n, n
            return best / len(reach), hit

        cn_, hn = cover(chunks_now_n)
        ca_, ha = cover(chunks_art_n)
        agg.append((cn_, ca_, len(reach), hn, ha))
        flag = '  ← 从"只能答一半"变"能答全"' if ha == len(reach) and hn < len(reach) else ''
        print('  %-5s %-6s %4d %4d | %8.3f %8.3f | %5d/%-3d %5d/%-3d%s' % (
            r['id'], r.get('type'), len(keys), len(reach), cn_, ca_,
            hn, len(reach), ha, len(reach), flag))
    print('  ' + '-' * 100)

    n = len(agg)
    m_now = sum(a[0] for a in agg) / n
    m_art = sum(a[1] for a in agg) / n
    full_now = sum(1 for a in agg if a[3] == a[2])
    full_art = sum(1 for a in agg if a[4] == a[2])
    print('  %d 条可评估正例的**平均最佳单块覆盖率**：%.4f → %.4f（%+.4f）' % (
        n, m_now, m_art, m_art - m_now))
    print('  **要点全部落进同一块**的题数：%d/%d → %d/%d' % (full_now, n, full_art, n))
    gain = sum(1 for a in agg if a[1] > a[0] + 1e-9)
    loss = sum(1 for a in agg if a[0] > a[1] + 1e-9)
    print('  变好 %d 条 / 变平 %d 条 / 变差 %d 条' % (gain, n - gain - loss, loss))
    print()
    print('  总耗时 %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
