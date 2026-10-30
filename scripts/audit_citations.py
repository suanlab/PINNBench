#!/usr/bin/env python3
"""
Check every entry of a BibTeX file against a primary bibliographic record.

This is the mechanical half of the citation standard: it catches fabricated
entries and wrong authors, years, venues, volumes or pages.  It does *not*
check whether the paper's prose says something the cited work actually says
-- that half needs a reader with the cited text open (see the standard in the
docstring of ``main``).

Rules it enforces, each learned from a failure in an earlier audit:

* An entry is verified only against a primary record: the DOI registry
  (Crossref) for entries carrying a ``doi`` field, or the publisher/venue page
  named in its ``url`` field.  Search results are never evidence -- a
  bibliographic *search* for two JCP papers here returned their SSRN preprint
  records, and an LLM-summarised web page called the same NeurIPS paper both
  "Poster" and "Spotlight Poster".
* Therefore every journal or proceedings entry must carry a ``doi``; entries
  without one (ICLR, AISTATS, IJCAI, JMLR, older NeurIPS) must carry a ``url``
  that resolves.
* Authors are compared in order, by surname, after Unicode/TeX-accent
  normalisation (``Piovar{\\v{c}}i`` == ``Piovarci``).
* The year compared is the print year of the cited version.  Crossref's
  ``issued`` is often the online-first date, so ``published-print`` wins when
  present (JCP vol. 474 is 2023 in print, 2022 online).

Usage:
    python scripts/audit_citations.py docs/paper_neurips_camera/references.bib

Exit code 0 when every entry passes, 1 otherwise.  Makes network requests to
api.crossref.org and to each entry's url; no credentials or personal data are
sent.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from typing import Any

_TEX_ACCENT = re.compile(r"\\[`'^\"~=.uvHtcdbk]\s*\{?(\w)\}?")


def _plain(s: str) -> str:
    s = _TEX_ACCENT.sub(r"\1", s or "")
    s = s.replace("{", "").replace("}", "").replace("\\&", "&")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", _plain(s).lower()).split()


def _surname(name: str) -> str:
    name = _plain(name).strip()
    last = name.split(",")[0] if "," in name else name.split()[-1]
    return "".join(_words(last))


def _balanced(text: str, i: int) -> tuple[str, int]:
    """Return the brace-balanced group starting at text[i] == '{'."""
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j], j + 1
        j += 1
    raise ValueError(f"unbalanced braces at offset {i}")


def parse_bib(text: str) -> list[dict[str, Any]]:
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        body, _ = _balanced(text, m.end() - len(m.group(0)) + m.group(0).index("{"))
        fields: dict[str, str] = {}
        for f in re.finditer(r"(\w+)\s*=\s*\{", body):
            val, _ = _balanced(body, f.end() - 1)
            fields.setdefault(f.group(1).lower(), val)
        entries.append({"type": m.group(1).lower(), "key": m.group(2), **fields})
    return entries


def _get_json(url: str) -> dict[str, Any] | None:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(2 * (attempt + 1))
    return None


def _fetch_text(url: str) -> tuple[int, str]:
    """Return (HTTP status, page text with tags stripped)."""
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read().decode("utf-8", errors="replace")
            return r.status, html.unescape(re.sub(r"<[^>]+>", " ", raw))
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, TimeoutError):
        return 0, ""


def _openreview_root(forum: str, title: str) -> dict[str, Any] | None:
    """Fetch the root note of an OpenReview forum.

    Forum pages are rendered client-side, so their HTML carries neither title
    nor authors.  The search API does, but it also returns DBLP-imported copies
    of the same paper (e.g. a "CoRR 2020" record beside the "ICLR 2021 Poster"
    one), so the match is made on the forum id taken from the url -- never on
    the title alone.
    """
    q = urllib.request.quote(_plain(title))
    data = _get_json(f"https://api.openreview.net/notes/search?term={q}&limit=25") or {}
    for n in data.get("notes", []):
        if n.get("id") == forum == n.get("forum"):
            c = n.get("content", {})
            val = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in c.items()}
            return val
    return None


def _year(msg: dict[str, Any]) -> str | None:
    for k in ("published-print", "issued", "published-online"):
        parts = (msg.get(k) or {}).get("date-parts") or [[None]]
        if parts[0][0]:
            return str(parts[0][0])
    return None


def check(entry: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return (field, status, detail) rows; status is PASS, FAIL or MANUAL."""
    out: list[tuple[str, str, str]] = []
    doi = entry.get("doi")
    if not doi:
        url = entry.get("url")
        if not url:
            return [("source", "FAIL", "no doi and no url: nothing primary to check against")]
        m = re.search(r"openreview\.net/forum\?id=([\w-]+)", url)
        if m:
            note = _openreview_root(m.group(1), entry.get("title", ""))
            if note is None:
                return [("url", "FAIL", f"no OpenReview root note for forum {m.group(1)}")]
            a, b = set(_words(entry.get("title", ""))), set(_words(note.get("title", "")))
            sim = len(a & b) / max(1, len(a | b))
            out.append(("title", "PASS" if sim >= 0.85 else "FAIL", f"jaccard {sim:.2f}"))
            bib_auth = [_surname(x) for x in re.split(r"\s+and\s+", entry.get("author", "")) if x]
            src_auth = [_surname(x) for x in note.get("authors") or []]
            out.append(("authors", "PASS" if bib_auth == src_auth else "FAIL",
                        f"{len(src_auth)} in order" if bib_auth == src_auth else f"{src_auth}"))
            venue = str(note.get("venue") or "")
            ok = bool(entry.get("year")) and entry["year"] in venue
            out.append(("venue", "PASS" if ok else "FAIL", f"OpenReview venue: {venue!r}"))
            return out
        # HTTP 200 is not evidence: IJCAI serves a 200 index page for any
        # nonexistent abstract id.  The page must actually carry the paper.
        code, text = _fetch_text(url)
        if code != 200:
            return [("url", "FAIL", f"HTTP {code} {url}")]
        page = set(_words(text))
        title = _words(entry.get("title", ""))
        cover = sum(w in page for w in title) / max(1, len(title))
        out.append(("title", "PASS" if cover >= 0.9 else "FAIL",
                    f"{cover:.0%} of title words on the page | {url}"))
        names = [_surname(x) for x in re.split(r"\s+and\s+", entry.get("author", "")) if x.strip()]
        missing = [n for n in names if n not in page]
        out.append(("authors", "FAIL" if missing else "PASS",
                    f"missing on page: {missing}" if missing else f"{len(names)} on page"))
        out.append(("venue", "MANUAL", "no DOI: venue/year/pages are not machine-checked"))
        return out

    rec = _get_json(f"https://api.crossref.org/works/{urllib.request.quote(doi)}")
    if not rec:
        return [("doi", "FAIL", f"{doi} does not resolve in Crossref")]
    msg = rec["message"]
    if msg.get("type") in {"posted-content"} or "ssrn" in doi.lower():
        out.append(("doi", "FAIL", f"{doi} is a preprint record, not the cited version"))

    title = (msg.get("title") or [""])[0]
    a, b = set(_words(entry.get("title", ""))), set(_words(title))
    sim = len(a & b) / max(1, len(a | b))
    out.append(("title", "PASS" if sim >= 0.85 else "FAIL", f"jaccard {sim:.2f} | {title[:70]}"))

    bib_auth = [_surname(x) for x in re.split(r"\s+and\s+", entry.get("author", "")) if x.strip()]
    src_auth = ["".join(_words(x.get("family", ""))) for x in msg.get("author", [])]
    if bib_auth == src_auth:
        out.append(("authors", "PASS", f"{len(bib_auth)} in order"))
    elif set(bib_auth) == set(src_auth):
        out.append(("authors", "FAIL", f"same names, different order: {src_auth}"))
    else:
        # hyphenated surnames are often split by registries (Braga-Neto -> Neto)
        loose = len(bib_auth) == len(src_auth) and all(
            s.endswith(r) or r.endswith(s) for r, s in zip(src_auth, bib_auth, strict=True)
        )
        detail = f"bib {bib_auth} | registry {src_auth}"
        out.append(("authors", "PASS" if loose else "FAIL", detail))

    y = _year(msg)
    detail = f"bib {entry.get('year')} | print {y}"
    out.append(("year", "PASS" if y == entry.get("year") else "FAIL", detail))

    pages = msg.get("page") or msg.get("article-number")
    for field, src in (("volume", msg.get("volume")), ("pages", pages)):
        mine = (entry.get(field) or "").replace("--", "-")
        if not mine:
            continue
        ok = src is None or mine == src or mine.split("-")[0] == str(src).split("-")[0]
        out.append((field, "PASS" if ok else "FAIL", f"bib {mine} | registry {src}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bib")
    args = ap.parse_args()

    with open(args.bib, encoding="utf-8") as fh:
        entries = parse_bib(fh.read())
    failed = manual = 0
    for e in entries:
        rows = check(e)
        worst = "FAIL" if any(s == "FAIL" for _, s, _ in rows) else (
            "MANUAL" if any(s == "MANUAL" for _, s, _ in rows) else "PASS")
        failed += worst == "FAIL"
        manual += worst == "MANUAL"
        print(f"{worst:<7}{e['key']}")
        for field, status, detail in rows:
            if status != "PASS" or worst != "PASS":
                print(f"         {status:<7}{field:<8}{detail}")
        time.sleep(0.3)
    print(f"\n{len(entries)} entries: {len(entries) - failed - manual} pass, "
          f"{manual} need a manual check of the url page, {failed} fail")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
