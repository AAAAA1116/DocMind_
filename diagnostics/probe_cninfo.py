# -*- coding: utf-8 -*-
"""探测巨潮公告查询接口：能不能搜到「公司章程」这类纯制度条款文本。"""
import json
import urllib.request
import urllib.parse

BASE = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}


def query(searchkey, column="szse", pagesize=30, sedate="", category=""):
    payload = {
        "pageNum": 1, "pageSize": pagesize, "column": column,
        "tabName": "fulltext", "plate": "", "stock": "", "searchkey": searchkey,
        "secid": "", "category": category, "trade": "", "seDate": sedate,
        "sortName": "", "sortType": "", "isHLtitle": "true",
    }
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(BASE, data=data, headers=HDR)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def show(tag, **kw):
    print("=" * 78)
    print("[%s]  params=%s" % (tag, {k: v for k, v in kw.items() if v}))
    try:
        j = query(**kw)
    except Exception as e:
        print("  ERROR:", e)
        return
    print("  totalRecordNum =", j.get("totalRecordNum"))
    for a in (j.get("announcements") or [])[:12]:
        print("  %-8s %-8s %6sKB  %s" % (a.get("secCode"), a.get("secName"),
                                         a.get("adjunctSize"), (a.get("announcementTitle") or "")[:44]))
        print("           https://static.cninfo.com.cn/%s" % a.get("adjunctUrl"))


show("searchkey=公司章程", searchkey="公司章程", sedate="2026-01-01~2026-09-20")
show("searchkey=内部控制制度", searchkey="内部控制制度", sedate="2026-01-01~2026-09-20")
show("searchkey=信息披露管理制度", searchkey="信息披露管理制度", sedate="2026-01-01~2026-09-20")
show("category=category_zf_szsh", category="category_zf_szsh", sedate="2026-01-01~2026-09-20")
